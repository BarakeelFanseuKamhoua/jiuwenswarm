"""Submit normalized voice requests through JiuwenSwarm's existing Gateway."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from jiuwenswarm.cli.gateway_client import GatewayClient
from jiuwenswarm.voice.models import (
    ASK_LEADER_TOOL,
    CANCEL_TASK_TOOL,
    GET_TASK_STATUS_TOOL,
    RESUME_TASK_TOOL,
    SUBMIT_TASK_TOOL,
    SUPPLEMENT_TASK_TOOL,
    LeaderInstruction,
    VoiceCommand,
)

logger = logging.getLogger(__name__)

LeaderReplyHandler = Callable[[str], Awaitable[None]]
_LEADER_ROLES = frozenset({"leader", "team-leader", "team_leader", "assistant"})
_NON_LEADER_ROLES = frozenset({"teammate", "member", "human_agent", "human-agent"})
_LEADER_MEMBER_NAMES = frozenset({"leader", "team-leader", "team_leader"})
LEADER_SPEECH_MAX_CHARS = 1200


def _normalized_identity(value: Any) -> str:
    return str(value or "").strip().casefold()


def _reply_stream_key(payload: dict[str, Any]) -> str:
    return str(
        payload.get("rid")
        or payload.get("request_id")
        or payload.get("message_id")
        or "__legacy__"
    )


def _visible_request_id(payload: dict[str, Any]) -> str:
    return str(payload.get("request_id") or "").strip().removeprefix("voice-")


def is_leader_reply_payload(payload: dict[str, Any]) -> bool:
    """Return whether a Gateway chat event is attributable to Team Leader."""
    role = _normalized_identity(payload.get("role"))
    if role in _NON_LEADER_ROLES:
        return False
    if role and role not in _LEADER_ROLES:
        return False

    member_name = _normalized_identity(
        payload.get("member_name") or payload.get("source_member")
    )
    if member_name and member_name not in _LEADER_MEMBER_NAMES:
        return False
    return True


def leader_reply_for_speech(
    payload: dict[str, Any],
    *,
    fallback_content: str = "",
    max_chars: int = LEADER_SPEECH_MAX_CHARS,
) -> str | None:
    """Select and bound one complete user-facing Leader report."""
    if not is_leader_reply_payload(payload) or payload.get("is_complete") is False:
        return None
    content = str(payload.get("content") or fallback_content).strip()
    if not content:
        return None
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip() + "……详细内容较长，请在终端或网页中查看。"
    return content


def build_chat_send_frame(
    *,
    voice_turn_id: str,
    session_id: str,
    instruction: LeaderInstruction,
    mode: str,
    cwd: str,
    project_dir: str,
) -> dict[str, Any]:
    """Build the same ``chat.send`` envelope used by existing clients."""
    content = instruction.render_for_leader(voice_turn_id)
    return build_chat_text_frame(
        voice_turn_id=voice_turn_id,
        session_id=session_id,
        content=content,
        mode=mode,
        cwd=cwd,
        project_dir=project_dir,
    )


def build_chat_text_frame(
    *,
    voice_turn_id: str,
    session_id: str,
    content: str,
    mode: str,
    cwd: str,
    project_dir: str,
) -> dict[str, Any]:
    """Build ``chat.send`` with the user's original conversational text."""
    return {
        "type": "req",
        "id": f"voice-{voice_turn_id}",
        "method": "chat.send",
        "is_stream": True,
        "params": {
            "session_id": session_id,
            "content": content,
            "query": content,
            "mode": mode,
            "source": "voice",
            "cwd": cwd,
            "project_dir": project_dir,
            "trusted_dirs": [project_dir],
            "supports_user_interaction": False,
            "agent_ref": {"mode": mode, "id": "default"},
        },
    }


def build_chat_interrupt_frame(
    *,
    voice_turn_id: str,
    session_id: str,
    command: VoiceCommand,
    mode: str,
    cwd: str,
    project_dir: str,
) -> dict[str, Any]:
    """Build the existing ``chat.interrupt`` envelope for a voice control."""
    intent = command.name.removesuffix("_task")
    params: dict[str, Any] = {
        "session_id": session_id,
        "intent": intent,
        "mode": mode,
        "source": "voice",
        "cwd": cwd,
        "project_dir": project_dir,
        "trusted_dirs": [project_dir],
    }
    if mode == "team":
        params["team"] = True
    if command.name == SUPPLEMENT_TASK_TOOL:
        params["new_input"] = command.dispatch_text(voice_turn_id)
    return {
        "type": "req",
        "id": f"voice-{voice_turn_id}",
        "method": "chat.interrupt",
        "is_stream": False,
        "params": params,
    }


class LeaderGatewayBridge:
    """Long-lived Gateway connection for Leader-only voice interaction."""

    def __init__(
        self,
        *,
        gateway_url: str,
        session_id: str,
        mode: str = "team",
        cwd: str | None = None,
        project_dir: str | None = None,
        client: GatewayClient | None = None,
        leader_reply_handler: LeaderReplyHandler | None = None,
    ) -> None:
        resolved_cwd = str(Path(cwd or ".").resolve())
        self._session_id = session_id
        self._mode = mode
        self._cwd = resolved_cwd
        self._project_dir = str(Path(project_dir or resolved_cwd).resolve())
        self._client = client or GatewayClient(gateway_url)
        self._leader_reply_handler = leader_reply_handler
        self._receiver_task: asyncio.Task[None] | None = None
        self._executed_turn_ids: set[str] = set()

    def set_leader_reply_handler(self, handler: LeaderReplyHandler | None) -> None:
        self._leader_reply_handler = handler

    # TeamVoiceInterface names.  Keep ``submit``/``execute`` as the historic
    # API while exposing explicit Team-facing verbs for new integrations.
    async def submit_instruction(
        self, voice_turn_id: str, instruction: LeaderInstruction
    ) -> dict[str, Any]:
        return await self.submit(voice_turn_id, instruction)

    async def dispatch_command(
        self, voice_turn_id: str, command: VoiceCommand
    ) -> dict[str, Any]:
        return await self.execute(voice_turn_id, command)

    async def connect(self) -> None:
        await self._client.connect()
        self._receiver_task = asyncio.create_task(
            self._receive_events(),
            name="voice-leader-gateway-receiver",
        )

    async def submit(
        self,
        voice_turn_id: str,
        instruction: LeaderInstruction,
    ) -> dict[str, Any]:
        """Submit exactly once and always route to Team Leader."""
        if voice_turn_id in self._executed_turn_ids:
            return {"ok": True, "duplicate": True, "voice_turn_id": voice_turn_id}

        frame = build_chat_send_frame(
            voice_turn_id=voice_turn_id,
            session_id=self._session_id,
            instruction=instruction,
            mode=self._mode,
            cwd=self._cwd,
            project_dir=self._project_dir,
        )
        await self._client.send_request(frame)
        self._executed_turn_ids.add(voice_turn_id)
        logger.info("Voice request %s submitted to Team Leader", voice_turn_id)
        return {"ok": True, "duplicate": False, "voice_turn_id": voice_turn_id}

    async def execute(
        self,
        voice_turn_id: str,
        command: VoiceCommand,
    ) -> dict[str, Any]:
        """Map one voice command to the Gateway's existing chat methods."""
        if voice_turn_id in self._executed_turn_ids:
            return {
                "ok": True,
                "duplicate": True,
                "voice_turn_id": voice_turn_id,
                "command": command.name,
            }

        if command.name == SUBMIT_TASK_TOOL:
            if command.instruction is None:
                raise ValueError("submit_task requires an instruction")
            return {
                **await self.submit(voice_turn_id, command.instruction),
                "command": command.name,
            }

        if command.name == ASK_LEADER_TOOL:
            frame = build_chat_text_frame(
                voice_turn_id=voice_turn_id,
                session_id=self._session_id,
                content=command.query,
                mode=self._mode,
                cwd=self._cwd,
                project_dir=self._project_dir,
            )
            await self._client.send_request(frame)
            self._executed_turn_ids.add(voice_turn_id)
            logger.info("Voice question %s sent to Team Leader", voice_turn_id)
            return {
                "ok": True,
                "duplicate": False,
                "voice_turn_id": voice_turn_id,
                "command": command.name,
            }

        if command.name in {
            SUPPLEMENT_TASK_TOOL,
            CANCEL_TASK_TOOL,
            GET_TASK_STATUS_TOOL,
        }:
            frame = build_chat_text_frame(
                voice_turn_id=voice_turn_id,
                session_id=self._session_id,
                content=command.dispatch_text(voice_turn_id),
                mode=self._mode,
                cwd=self._cwd,
                project_dir=self._project_dir,
            )
            await self._client.send_request(frame)
            self._executed_turn_ids.add(voice_turn_id)
            logger.info(
                "Targeted voice command %s sent to Team Leader: %s targets=%s",
                voice_turn_id,
                command.name,
                command.target_task_ids,
            )
            return {
                "ok": True,
                "duplicate": False,
                "voice_turn_id": voice_turn_id,
                "command": command.name,
            }

        if command.name == RESUME_TASK_TOOL and self._mode == "team":
            instruction = LeaderInstruction(
                summary="恢复暂停的团队任务",
                instruction=command.dispatch_text(voice_turn_id),
                constraints=("保持原有目标、已完成工作和用户约束",),
            )
            return {
                **await self.submit(voice_turn_id, instruction),
                "command": command.name,
            }

        frame = build_chat_interrupt_frame(
            voice_turn_id=voice_turn_id,
            session_id=self._session_id,
            command=command,
            mode=self._mode,
            cwd=self._cwd,
            project_dir=self._project_dir,
        )
        await self._client.send_request(frame)
        self._executed_turn_ids.add(voice_turn_id)
        logger.info("Voice command %s sent: %s", voice_turn_id, command.name)
        return {
            "ok": True,
            "duplicate": False,
            "voice_turn_id": voice_turn_id,
            "command": command.name,
        }

    async def execute_batch(
        self,
        voice_turn_id: str,
        commands: tuple[VoiceCommand, ...],
    ) -> dict[str, Any]:
        if not commands:
            return {"ok": False, "error": "empty voice command batch"}
        if voice_turn_id in self._executed_turn_ids:
            return {"ok": True, "duplicate": True, "voice_turn_id": voice_turn_id}
        frame = build_chat_text_frame(
            voice_turn_id=voice_turn_id,
            session_id=self._session_id,
            content=VoiceCommand.render_voice_batch(commands, voice_turn_id),
            mode=self._mode,
            cwd=self._cwd,
            project_dir=self._project_dir,
        )
        await self._client.send_request(frame)
        self._executed_turn_ids.add(voice_turn_id)
        return {"ok": True, "duplicate": False, "voice_turn_id": voice_turn_id, "batched": True}

    async def _receive_events(self) -> None:
        """Drain Gateway events and forward only complete Leader reports."""
        reply_delta_parts: dict[str, list[str]] = {}
        try:
            while True:
                data = await self._client.recv()
                if data.get("type") != "event":
                    continue
                event_type = str(data.get("event") or "")
                payload = data.get("payload")
                if not isinstance(payload, dict):
                    payload = {}
                content = str(payload.get("content") or "")
                if event_type == "chat.delta" and content:
                    reply_delta_parts.setdefault(_reply_stream_key(payload), []).append(
                        content
                    )
                elif event_type == "chat.final":
                    fallback = "".join(
                        reply_delta_parts.pop(_reply_stream_key(payload), [])
                    )
                    terminal_reply = content.strip() or fallback.strip()
                    if terminal_reply:
                        label = "Team Leader" if is_leader_reply_payload(payload) else str(
                            payload.get("member_name")
                            or payload.get("source_member")
                            or "团队成员"
                        ).strip()
                        request_label = _visible_request_id(payload)
                        if request_label:
                            label = f"{label} | {request_label}"
                        print(f"\n[{label}] {terminal_reply}", flush=True)

                    spoken_reply = leader_reply_for_speech(
                        payload,
                        fallback_content=fallback,
                    )
                    if spoken_reply and self._leader_reply_handler is not None:
                        # The handler only queues text. Qwen Realtime performs
                        # generation and playback on the microphone socket.
                        await self._leader_reply_handler(spoken_reply)
                elif event_type == "chat.error":
                    error = payload.get("error") or payload.get("message") or "unknown error"
                    print(f"\n[Team Leader error] {error}", flush=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Gateway event receiver stopped", exc_info=True)

    async def close(self) -> None:
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver_task
            self._receiver_task = None
        await self._client.close()
