"""Browser-backed Qwen Realtime voice sessions for the Web channel."""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jiuwenswarm.common.config import get_config
from jiuwenswarm.voice.models import (
    VoiceCommand,
    reconcile_voice_command_with_transcript,
    reconcile_supplement_with_team_info,
)
from jiuwenswarm.voice.qwen_realtime import (
    LEADER_BRIEF_REPORT_PREFIX,
    LEADER_REPORT_PREFIX,
    QwenRealtimeConfig,
    build_session_update,
    build_voice_instructions,
)

logger = logging.getLogger(__name__)

VoiceEventSender = Callable[[str, dict[str, Any]], Awaitable[None]]


def _context_value(value: Any, max_chars: int) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def normalize_voice_team_info(value: Any) -> dict[str, Any]:
    """Keep member identity plus the minimal task directory needed for targeting."""
    if not isinstance(value, dict):
        return {"team_id": "", "members": [], "tasks": []}
    team_id = _context_value(value.get("team_id"), 128)
    raw_members = value.get("members")
    members_by_id: dict[str, dict[str, str]] = {}
    if isinstance(raw_members, list):
        for item in raw_members[:64]:
            if not isinstance(item, dict):
                continue
            member_id = _context_value(item.get("member_id"), 100)
            if not member_id:
                continue
            member: dict[str, str] = {"member_id": member_id}
            name = _context_value(item.get("name"), 100)
            role = _context_value(item.get("role") or item.get("mode"), 60)
            if name:
                member["name"] = name
            if role:
                member["role"] = role
            members_by_id[member_id] = member
    raw_tasks = value.get("tasks")
    tasks_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(raw_tasks, list):
        for index, item in enumerate(raw_tasks[:64], start=1):
            if not isinstance(item, dict):
                continue
            task_id = _context_value(item.get("task_id"), 128)
            if not task_id:
                continue
            title = _context_value(
                item.get("title") or item.get("name") or item.get("content"),
                160,
            )
            status = _context_value(item.get("status"), 40) or "unknown"
            try:
                created_order = max(1, int(item.get("created_order")))
            except (TypeError, ValueError):
                created_order = index
            tasks_by_id[task_id] = {
                "task_id": task_id,
                "title": title or "未命名任务",
                "status": status,
                "created_order": created_order,
            }
    tasks = sorted(
        tasks_by_id.values(),
        key=lambda item: (item["created_order"], item["task_id"]),
    )
    return {
        "team_id": team_id,
        "members": list(members_by_id.values()),
        "tasks": tasks,
    }


@dataclass(slots=True)
class _PendingVoiceCommand:
    call_id: str
    command: VoiceCommand


@dataclass(slots=True)
class _InterruptedTurn:
    transcript: str
    attempts: int = 0
    # Tool calls Qwen had fully emitted before the next utterance cancelled
    # its response. They are dispatched as-is instead of asking Qwen again.
    commands: list[_PendingVoiceCommand] = field(default_factory=list)


# Qwen's server VAD cancels the in-flight response as soon as the next
# utterance starts. When the user chains two requests with almost no pause the
# first response dies before its tool call is emitted and that request is
# silently lost. Such turns are replayed once the session is idle again.
_INTERRUPTED_TURN_MAX_ATTEMPTS = 2
_INTERRUPTED_TURN_PROMPT = (
    "[系统补发，非用户新发言] 用户刚才说的这句话在你调用工具之前被下一句语音打断，尚未执行：\n"
    "「{transcript}」\n"
    "请现在只针对这一句调用对应的工具，不要重复处理已经执行过的其他请求；"
    "如果用户后面的话已经更正或取消了这句，就不要调用工具。确认语只说一句。"
)


class WebQwenVoiceSession:
    """One Qwen full-duplex connection backed by browser PCM input/output."""

    def __init__(
        self,
        config: QwenRealtimeConfig,
        send_event: VoiceEventSender,
        leader_reply_mode: str = "full",
        team_info: dict[str, Any] | None = None,
    ) -> None:
        normalized_reply_mode = leader_reply_mode.strip().lower()
        if normalized_reply_mode not in {"full", "brief"}:
            raise ValueError("leader reply mode must be 'full' or 'brief'")
        self._config = config
        self._send_event = send_event
        self._leader_reply_mode = normalized_reply_mode
        self._team_info = normalize_voice_team_info(team_info)
        self._websocket: Any = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._leader_reply_queue: asyncio.Queue[str] = asyncio.Queue()
        self._leader_task: asyncio.Task[None] | None = None
        self._leader_response_requested = False
        self._active_origin: str | None = None
        self._user_turn_active = False
        self._drop_audio_until_done = False
        self._pending_requests: list[_PendingVoiceCommand] = []
        self._completed_call_ids: set[str] = set()
        self._last_transcript = ""
        # Input audio item of the utterance currently being spoken/answered.
        # Late transcripts of an older item must not be attributed to it.
        self._turn_item_id: str | None = None
        # Input audio item a user-turn response answers. server_vad may start
        # the next utterance before that response is even created, so the
        # speech_started item alone cannot identify it.
        self._committed_item_id: str | None = None
        self._response_item_id: str | None = None
        self._interrupted_item_id: str | None = None
        self._response_interrupted_by_speech = False
        self._orphan_item_ids: set[str] = set()
        self._item_transcripts: dict[str, str] = {}
        self._replay_queue: asyncio.Queue[_InterruptedTurn] = asyncio.Queue()
        self._replay_task: asyncio.Task[None] | None = None
        self._replay_requested: _InterruptedTurn | None = None
        # A replay taken off the queue but still waiting for the idle slot.
        self._replay_waiting = False
        self._active_replay: _InterruptedTurn | None = None
        # Leader reports and replayed turns each create one response and wait
        # for it; serialize them so they never race for the same idle slot.
        self._response_slot = asyncio.Lock()

    async def connect(self) -> None:
        import websockets

        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        header_key = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )
        self._websocket = await websockets.connect(
            self._config.url,
            **{header_key: headers},
        )
        await self._send(
            build_session_update(self._config, team_info=self._team_info)
        )
        await asyncio.wait_for(self._wait_for_session_updated(), timeout=15)
        self._receiver_task = asyncio.create_task(
            self._receive_events(),
            name="web-qwen-voice-receiver",
        )
        self._leader_task = asyncio.create_task(
            self._speak_leader_replies(),
            name="web-qwen-leader-speech",
        )
        self._replay_task = asyncio.create_task(
            self._replay_interrupted_turns(),
            name="web-qwen-interrupted-turn-replay",
        )
        await self._send_event("voice.status", {"state": "listening"})

    async def append_audio(self, audio_base64: str) -> None:
        if not audio_base64:
            return
        # 100 ms of 16 kHz mono PCM16 is 3,200 bytes. Allow several chunks in
        # one browser frame while rejecting accidental/unbounded payloads.
        if len(audio_base64) > 128_000:
            raise ValueError("audio chunk is too large")
        try:
            base64.b64decode(audio_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("audio must be valid base64 PCM") from exc
        await self._send({"type": "input_audio_buffer.append", "audio": audio_base64})

    async def enqueue_leader_reply(self, text: str) -> None:
        normalized = text.strip()
        if normalized:
            await self._leader_reply_queue.put(normalized)

    async def update_team_info(self, team_info: Any) -> bool:
        """Replace Qwen's identity-only roster without producing a response."""
        normalized = normalize_voice_team_info(team_info)
        if normalized == self._team_info:
            return False
        self._team_info = normalized
        if self._websocket is not None:
            await self._send(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": build_voice_instructions(self._team_info),
                    },
                }
            )
        return True

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._websocket is None:
            raise RuntimeError("voice session is not connected")
        async with self._send_lock:
            await self._websocket.send(json.dumps(payload, ensure_ascii=False))

    async def _wait_for_session_updated(self) -> None:
        while True:
            event = json.loads(await self._websocket.recv())
            event_type = str(event.get("type") or "")
            if event_type == "session.updated":
                return
            if event_type == "error":
                error = event.get("error")
                message = error.get("message") if isinstance(error, dict) else error
                raise RuntimeError(str(message or "Qwen voice session setup failed"))

    async def _receive_events(self) -> None:
        try:
            async for raw in self._websocket:
                event = json.loads(raw)
                event_type = str(event.get("type") or "")
                if event_type == "response.created":
                    self._idle.clear()
                    self._response_item_id = None
                    if self._leader_response_requested:
                        self._active_origin = "leader-report"
                        self._leader_response_requested = False
                    elif self._replay_requested is not None:
                        self._active_origin = "replay-turn"
                        self._active_replay = self._replay_requested
                        self._replay_requested = None
                    else:
                        self._active_origin = "user-turn"
                        self._response_item_id = (
                            self._committed_item_id or self._turn_item_id
                        )
                    self._drop_audio_until_done = False
                    self._response_interrupted_by_speech = False
                elif event_type == "response.audio.delta":
                    audio = str(event.get("delta") or "")
                    if audio and not self._drop_audio_until_done:
                        await self._send_event(
                            "voice.audio",
                            {"audio": audio, "sample_rate": 24_000},
                        )
                elif event_type == "input_audio_buffer.speech_started":
                    interrupted = self._active_origin is not None
                    if interrupted:
                        self._response_interrupted_by_speech = True
                        if self._active_origin == "user-turn":
                            self._interrupted_item_id = self._turn_item_id
                    self._turn_item_id = str(event.get("item_id") or "") or None
                    self._idle.clear()
                    self._user_turn_active = True
                    self._drop_audio_until_done = interrupted
                    self._last_transcript = ""
                    await self._send_event(
                        "voice.interrupted",
                        {"interrupted": interrupted},
                    )
                    await self._send_event("voice.status", {"state": "listening"})
                elif event_type == "input_audio_buffer.speech_stopped":
                    await self._send_event("voice.status", {"state": "thinking"})
                    if event.get("reason") == "turn_invalid":
                        self._user_turn_active = False
                        if self._active_origin is None:
                            self._idle.set()
                        # No response follows an invalid turn; release any
                        # speech-start pause the frontend took for it.
                        await self._notify_turn_without_command("turn_invalid")
                elif event_type == "input_audio_buffer.committed":
                    self._committed_item_id = str(event.get("item_id") or "") or None
                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    transcript = str(event.get("transcript") or "").strip()
                    item_id = str(event.get("item_id") or "") or None
                    if item_id and transcript:
                        self._remember_item_transcript(item_id, transcript)
                    if item_id and self._turn_item_id and item_id != self._turn_item_id:
                        # Late transcript of an earlier utterance that arrived
                        # after the next one started: it must not become the
                        # current turn's text or user bubble.
                        logger.info(
                            "[VoiceTrace] late transcript item_id=%s orphan=%s transcript=%s",
                            item_id,
                            item_id in self._orphan_item_ids,
                            transcript[:120],
                        )
                        if item_id in self._orphan_item_ids and transcript:
                            self._orphan_item_ids.discard(item_id)
                            self._enqueue_replay(_InterruptedTurn(transcript))
                        continue
                    self._last_transcript = transcript
                    if self._last_transcript:
                        await self._send_event(
                            "voice.transcript",
                            {"text": self._last_transcript},
                        )
                elif event_type in {
                    "response.audio_transcript.done",
                    "response.text.done",
                }:
                    transcript = str(
                        event.get("transcript") or event.get("text") or ""
                    ).strip()
                    if transcript:
                        await self._send_event(
                            "voice.assistant_transcript",
                            {
                                "text": transcript,
                                "origin": self._active_origin or "user-turn",
                            },
                        )
                elif event_type == "response.function_call_arguments.done":
                    if self._active_origin != "leader-report":
                        try:
                            self._record_function_call(event)
                        except (TypeError, ValueError) as exc:
                            logger.warning(
                                "Invalid Qwen voice function call", exc_info=True
                            )
                            await self._send_event(
                                "voice.error",
                                {"message": f"Invalid voice request: {exc}"},
                            )
                elif event_type == "response.done":
                    response = event.get("response")
                    status = (
                        response.get("status") if isinstance(response, dict) else None
                    )
                    origin = self._active_origin
                    interrupted_by_speech = self._response_interrupted_by_speech
                    replay = self._active_replay
                    response_item_id = self._response_item_id
                    # The next utterance began before this response was created
                    # (speech_started raced ahead of response.created), so the
                    # barge-in flag was never set, yet server_vad still
                    # cancels it and the next turn is the live one.
                    superseded = bool(
                        origin == "user-turn"
                        and response_item_id
                        and self._turn_item_id
                        and response_item_id != self._turn_item_id
                    )
                    logger.info(
                        "[VoiceTrace] response.done origin=%s status=%s pending_calls=%d "
                        "interrupted_by_speech=%s superseded=%s transcript=%s",
                        origin,
                        status,
                        len(self._pending_requests),
                        interrupted_by_speech,
                        superseded,
                        (replay.transcript if replay else self._last_transcript)[:120],
                    )
                    self._active_origin = None
                    self._active_replay = None
                    self._response_item_id = None
                    self._response_interrupted_by_speech = False
                    self._drop_audio_until_done = False
                    if origin != "leader-report":
                        if status == "cancelled":
                            emitted = list(self._pending_requests)
                            self._pending_requests.clear()
                            if emitted and (interrupted_by_speech or superseded):
                                # The call was complete before the next speech
                                # cut the response off; only the spoken tail was
                                # lost. Re-asking Qwen here tends to yield no
                                # tool, since it already confirmed the request.
                                transcript = (
                                    replay.transcript
                                    if replay is not None
                                    else self._item_transcripts.get(
                                        response_item_id
                                        or self._interrupted_item_id
                                        or "",
                                        "",
                                    )
                                )
                                self._enqueue_replay(
                                    _InterruptedTurn(transcript, commands=emitted)
                                )
                            elif origin == "user-turn" and (
                                interrupted_by_speech or superseded
                            ):
                                orphan_item_id = (
                                    response_item_id or self._interrupted_item_id
                                )
                                if orphan_item_id:
                                    self._mark_orphan_item(orphan_item_id)
                            elif (
                                origin == "replay-turn"
                                and interrupted_by_speech
                                and replay is not None
                            ):
                                replay.attempts += 1
                                # The live turn that cut it off reports the
                                # outcome, even if this replay is dropped.
                                self._enqueue_replay(replay)
                        else:
                            tool_selected = bool(self._pending_requests)
                            logger.info(
                                "[VoiceTrace] turn completed: %s",
                                "tool selected" if tool_selected else "no tool selected",
                            )
                            await self._flush_voice_requests(replay)
                            if not tool_selected and not superseded:
                                await self._notify_turn_without_command(
                                    "replay-no-tool" if replay else "no-tool"
                                )
                        self._interrupted_item_id = None
                    # A response cancelled by new speech ends while the user is
                    # still talking; that next turn is still active.
                    if (
                        origin != "leader-report"
                        and not interrupted_by_speech
                        and not superseded
                    ):
                        self._user_turn_active = False
                    if not self._user_turn_active:
                        self._idle.set()
                    await self._send_event("voice.status", {"state": "listening"})
                elif event_type == "error":
                    error = event.get("error")
                    message = error.get("message") if isinstance(error, dict) else error
                    await self._send_event(
                        "voice.error", {"message": str(message or "Qwen voice error")}
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Web Qwen voice receiver stopped", exc_info=True)
            # ``closed`` tells the browser this session is gone for good, so it
            # tears it down and the next microphone start opens a fresh one
            # instead of streaming audio into a dead connection.
            await self._send_event("voice.error", {"message": str(exc), "closed": True})
            return
        logger.warning("Web Qwen voice connection closed by the server")
        await self._send_event(
            "voice.error",
            {"message": "Realtime voice connection closed", "closed": True},
        )

    def _record_function_call(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id") or event.get("item_id") or "").strip()
        name = str(event.get("name") or "").strip()
        if not call_id or call_id in self._completed_call_ids:
            return
        arguments = event.get("arguments") or "{}"
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed, dict):
            raise ValueError("voice function arguments must be an object")
        self._pending_requests.append(
            _PendingVoiceCommand(call_id, VoiceCommand.from_tool_call(name, parsed))
        )
        logger.info(
            "[VoiceTrace] function_call call_id=%s name=%s arg_keys=%s pending=%d",
            call_id,
            name,
            sorted(parsed)[:16],
            len(self._pending_requests),
        )

    async def _notify_turn_without_command(self, reason: str) -> None:
        """Tell the frontend a voice turn ended without dispatching anything.

        The frontend pauses a running Team the moment speech starts; only a
        dispatched command resumes it. A turn that selects no tool (filler,
        noise, an utterance split mid-sentence) would otherwise leave the Team
        paused and the voice turn spinner running. A queued replay still owns
        the outcome, so stay silent until it has run.
        """
        if (
            self._replay_waiting
            or self._replay_requested is not None
            or not self._replay_queue.empty()
        ):
            return
        logger.info("[VoiceTrace] turn ended without command reason=%s", reason)
        await self._send_event("voice.turn_completed", {"tool_selected": False, "reason": reason})

    def _remember_item_transcript(self, item_id: str, transcript: str) -> None:
        self._item_transcripts[item_id] = transcript
        while len(self._item_transcripts) > 16:
            self._item_transcripts.pop(next(iter(self._item_transcripts)))

    def _mark_orphan_item(self, item_id: str) -> None:
        """Replay an utterance whose response died before dispatching it."""
        transcript = self._item_transcripts.get(item_id)
        if transcript:
            self._enqueue_replay(_InterruptedTurn(transcript))
        else:
            # Its transcription usually lands after the next speech_started.
            self._orphan_item_ids.add(item_id)
            while len(self._orphan_item_ids) > 16:
                self._orphan_item_ids.pop()

    def _enqueue_replay(self, turn: _InterruptedTurn) -> None:
        if turn.attempts >= _INTERRUPTED_TURN_MAX_ATTEMPTS:
            logger.warning(
                "[VoiceTrace] interrupted turn dropped after %d attempts transcript=%s",
                turn.attempts,
                turn.transcript[:120],
            )
            return
        logger.info(
            "[VoiceTrace] interrupted turn queued for replay attempts=%d transcript=%s",
            turn.attempts,
            turn.transcript[:120],
        )
        self._replay_queue.put_nowait(turn)

    async def _flush_voice_requests(self, replay: _InterruptedTurn | None = None) -> None:
        if replay is not None and replay.commands:
            pending = list(replay.commands)
        else:
            pending = list(self._pending_requests)
            self._pending_requests.clear()
        if not pending:
            return
        transcript = replay.transcript if replay is not None else self._last_transcript

        # Reconcile each command independently.  The transcript is used only
        # for the cancellation safety rule; it must never change another
        # command's action or target.
        reconciled: list[tuple[str, VoiceCommand]] = []
        for request in pending:
            if request.call_id in self._completed_call_ids:
                continue
            command = reconcile_voice_command_with_transcript(
                request.command,
                transcript,
            )
            command = reconcile_supplement_with_team_info(command, self._team_info)
            reconciled.append((request.call_id, command))

        if not reconciled:
            return

        batch_call_ids = [call_id for call_id, _ in reconciled]
        commands = tuple(command for _, command in reconciled)
        # One voice turn collapses to one chat.send so a multi-intent turn is a
        # single Leader request (the typed-text path also sends one frame per
        # utterance). Multiple serial chat.send frames 0.1s apart trip the
        # Gateway's same-session cancel-then-start rule, dropping the first.
        # All intents render through render_voice_batch into a unified shape
        # (one header + ``意图N 动作：`` blocks + one posture footer) so the
        # Leader can split them reliably instead of seeing welded heterogeneous
        # templates and dropping the second.
        first_call_id = batch_call_ids[0]
        first_command = commands[0]
        instruction = first_command.instruction
        fallback_text = (
            instruction.instruction
            if instruction is not None
            else first_command.query or first_command.reason or first_command.name
        )
        display_text = transcript or fallback_text
        batch_dispatch_text = VoiceCommand.render_voice_batch(
            commands, first_call_id
        )

        await self._send_event(
            "voice.command_batch",
            {
                "call_ids": batch_call_ids,
                "count": len(batch_call_ids),
                "text": display_text,
                "dispatch_text": batch_dispatch_text,
                "name": first_command.name,
                # Replayed turns arrive outside any live user turn; the
                # frontend must not bind them to the current speech state.
                "replayed": replay is not None,
                "commands": [
                    {
                        "call_id": call_id,
                        "name": command.name,
                        "reason": command.reason,
                        "query": command.query,
                        "target_task_ids": list(command.target_task_ids),
                        "summary": command.instruction.summary if command.instruction else "",
                        "instruction": command.instruction.instruction if command.instruction else "",
                    }
                    for call_id, command in reconciled
                ],
            },
        )

        # Each call_id still needs its own function_call_output to satisfy
        # Qwen's turn contract, but they all describe the same single
        # dispatch the frontend will perform.
        for call_id, command in reconciled:
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(
                            {
                                "ok": True,
                                "command_emitted": command.name,
                                "execution": "pending_frontend_dispatch",
                                "batched_with": [c for c in batch_call_ids if c != call_id],
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            )
            self._completed_call_ids.add(call_id)


    async def _speak_leader_replies(self) -> None:
        while True:
            text = await self._leader_reply_queue.get()
            try:
                async with self._response_slot:
                    await self._idle.wait()
                    report_prefix = (
                        LEADER_BRIEF_REPORT_PREFIX
                        if self._leader_reply_mode == "brief"
                        else LEADER_REPORT_PREFIX
                    )
                    await self._send(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"{report_prefix}{text}",
                                    }
                                ],
                            },
                        }
                    )
                    self._leader_response_requested = True
                    self._idle.clear()
                    await self._send(
                        {
                            "type": "response.create",
                            "response": {"modalities": ["audio", "text"]},
                        }
                    )
                    await self._idle.wait()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to inject Team Leader report")
                self._leader_response_requested = False
                self._release_idle_after_failed_injection()
            finally:
                self._leader_reply_queue.task_done()

    async def _replay_interrupted_turns(self) -> None:
        while True:
            turn = await self._replay_queue.get()
            self._replay_waiting = True
            try:
                async with self._response_slot:
                    await self._idle.wait()
                    if turn.commands:
                        logger.info(
                            "[VoiceTrace] dispatching tool calls of interrupted turn "
                            "call_ids=%s transcript=%s",
                            [request.call_id for request in turn.commands],
                            turn.transcript[:120],
                        )
                        await self._flush_voice_requests(turn)
                        continue
                    logger.info(
                        "[VoiceTrace] replaying interrupted turn attempts=%d transcript=%s",
                        turn.attempts,
                        turn.transcript[:120],
                    )
                    await self._send(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": _INTERRUPTED_TURN_PROMPT.format(
                                            transcript=turn.transcript
                                        ),
                                    }
                                ],
                            },
                        }
                    )
                    self._replay_requested = turn
                    self._replay_waiting = False
                    self._idle.clear()
                    await self._send(
                        {
                            "type": "response.create",
                            "response": {"modalities": ["audio", "text"]},
                        }
                    )
                    await self._idle.wait()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to replay interrupted voice turn")
                self._replay_requested = None
                self._release_idle_after_failed_injection()
            finally:
                self._replay_waiting = False
                self._replay_queue.task_done()

    def _release_idle_after_failed_injection(self) -> None:
        """Undo the ``_idle.clear()`` of an injection whose response never came.

        Without it the next queued report or replay waits on ``_idle`` forever.
        """
        if self._active_origin is None and not self._user_turn_active:
            self._idle.set()

    async def close(self) -> None:
        tasks = [self._receiver_task, self._leader_task, self._replay_task]
        for task in tasks:
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in tasks if task is not None),
            return_exceptions=True,
        )
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None


@dataclass(slots=True)
class _ManagedSession:
    owner_ws_id: int
    session: WebQwenVoiceSession


class WebVoiceSessionManager:
    """Own browser voice sessions and isolate them by WebSocket + chat session."""

    def __init__(self, channel: Any) -> None:
        self._channel = channel
        self._sessions: dict[tuple[int, str], _ManagedSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _config_from_env(params: dict[str, Any]) -> QwenRealtimeConfig:
        api_key = str(os.getenv("DASHSCOPE_API_KEY") or "").strip()
        workspace_id = str(os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip()
        if not api_key or not workspace_id:
            raise RuntimeError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID must be set"
            )
        return QwenRealtimeConfig(
            api_key=api_key,
            workspace_id=workspace_id,
            model=str(params.get("model") or "qwen-audio-3.0-realtime-plus"),
            voice=str(params.get("voice") or "longanqian"),
            turn_detection=str(params.get("turn_detection") or "server_vad"),
            silence_duration_ms=int(params.get("silence_duration_ms") or 800),
        )

    @staticmethod
    def _reply_mode_from_config(params: dict[str, Any]) -> str:
        config = get_config() or {}
        voice_config = config.get("voice")
        configured_mode = (
            voice_config.get("reply_mode")
            if isinstance(voice_config, dict)
            else None
        )
        reply_mode = str(
            params.get("reply_mode")
            or configured_mode
            or os.getenv("JIUWENSWARM_VOICE_REPLY_MODE")
            or "full"
        ).strip().lower()
        if reply_mode not in {"full", "brief"}:
            raise RuntimeError(
                "voice.reply_mode must be 'full' or 'brief' in config.yaml"
            )
        return reply_mode

    async def start(
        self,
        ws: Any,
        session_id: str,
        params: dict[str, Any],
    ) -> None:
        key = (id(ws), session_id)
        await self.stop(ws, session_id)

        async def send_event(event: str, payload: dict[str, Any]) -> None:
            await self._channel.send_event(
                ws,
                event,
                {**payload, "session_id": session_id},
            )

        session = WebQwenVoiceSession(
            self._config_from_env(params),
            send_event,
            leader_reply_mode=self._reply_mode_from_config(params),
            team_info=params.get("team_info"),
        )
        try:
            await session.connect()
        except Exception:
            await session.close()
            raise
        async with self._lock:
            self._sessions[key] = _ManagedSession(id(ws), session)

    async def append_audio(self, ws: Any, session_id: str, audio: str) -> None:
        managed = self._sessions.get((id(ws), session_id))
        if managed is None:
            raise RuntimeError("voice session is not started")
        await managed.session.append_audio(audio)

    async def leader_reply(self, ws: Any, session_id: str, text: str) -> None:
        managed = self._sessions.get((id(ws), session_id))
        if managed is not None:
            await managed.session.enqueue_leader_reply(text)

    async def update_team_info(
        self,
        ws: Any,
        session_id: str,
        team_info: Any,
    ) -> bool:
        managed = self._sessions.get((id(ws), session_id))
        if managed is None:
            raise RuntimeError("voice session is not started")
        return await managed.session.update_team_info(team_info)

    async def stop(self, ws: Any, session_id: str) -> None:
        key = (id(ws), session_id)
        async with self._lock:
            managed = self._sessions.pop(key, None)
        if managed is not None:
            await managed.session.close()

    async def cleanup_ws(self, ws: Any, _session_ids: Any = None) -> None:
        ws_id = id(ws)
        async with self._lock:
            keys = [key for key in self._sessions if key[0] == ws_id]
            sessions = [self._sessions.pop(key).session for key in keys]
        await asyncio.gather(*(session.close() for session in sessions))
