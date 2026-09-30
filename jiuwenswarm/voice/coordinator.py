"""Order Qwen acknowledgements before dispatching work to the team leader."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from jiuwenswarm.voice.models import (
    VoiceCommand,
    reconcile_voice_command_with_transcript,
)

logger = logging.getLogger(__name__)

ToolOutputWriter = Callable[[str, dict[str, Any]], Awaitable[None]]
FallbackAcknowledger = Callable[[str], Awaitable[None]]


class VoiceCommandDispatcher(Protocol):
    async def execute(
        self,
        voice_turn_id: str,
        command: VoiceCommand,
    ) -> dict[str, Any]: ...

    async def execute_batch(
        self,
        voice_turn_id: str,
        commands: tuple[VoiceCommand, ...],
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class _PendingCall:
    call_id: str
    name: str
    command: VoiceCommand | None
    error: str | None = None


class VoiceDispatchCoordinator:
    """Hold tool calls until Qwen has produced the spoken acknowledgement."""

    def __init__(self, dispatcher: VoiceCommandDispatcher) -> None:
        self._dispatcher = dispatcher
        self._pending: dict[str, _PendingCall] = {}
        self._completed_call_ids: set[str] = set()
        self._ack_completed = False
        self._last_user_transcript = ""

    def record_user_transcript(self, transcript: str) -> None:
        self._last_user_transcript = transcript.strip()

    def mark_ack_completed(self, transcript: str) -> None:
        if transcript.strip():
            self._ack_completed = True

    def record_function_call(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id") or event.get("item_id") or "").strip()
        name = str(event.get("name") or "").strip()
        if not call_id:
            logger.warning("Ignoring Qwen function call without call_id")
            return
        if call_id in self._completed_call_ids or call_id in self._pending:
            return

        raw_arguments = event.get("arguments", "{}")
        try:
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments or "{}")
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise ValueError("arguments must be a JSON object")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            command = VoiceCommand.from_tool_call(name, arguments)
            error = None
        except (ValueError, json.JSONDecodeError) as exc:
            command = None
            error = str(exc)
        self._pending[call_id] = _PendingCall(
            call_id=call_id,
            name=name,
            command=command,
            error=error,
        )

    def cancel_pending(self) -> None:
        """Discard calls from an interrupted or cancelled Qwen response."""
        self._pending.clear()
        self._ack_completed = False
        self._last_user_transcript = ""

    async def flush(
        self,
        *,
        write_tool_output: ToolOutputWriter,
        fallback_acknowledge: FallbackAcknowledger,
    ) -> None:
        """Dispatch pending calls after acknowledgement, with exactly-once semantics."""
        calls = list(self._pending.values())
        self._pending.clear()
        if not calls:
            self._ack_completed = False
            return

        if not self._ack_completed:
            await fallback_acknowledge(
                "收到，我会按照你的要求处理。"
            )

        # Reconcile the complete turn before batching so transcript safety
        # rules (for example an affirmative continuation overriding a mistaken
        # cancel_task) apply to the command actually sent to the backend.
        reconciled: dict[str, VoiceCommand] = {}
        for call in calls:
            if call.command is not None and not call.error:
                reconciled[call.call_id] = reconcile_voice_command_with_transcript(
                    call.command, self._last_user_transcript
                )
        valid = [call for call in calls if call.call_id in reconciled]
        batch_result: dict[str, Any] | None = None
        execute_batch = getattr(self._dispatcher, "execute_batch", None)
        if len(valid) > 1 and callable(execute_batch):
            try:
                batch_result = await execute_batch(
                    valid[0].call_id,
                    tuple(reconciled[call.call_id] for call in valid),
                )
            except Exception as exc:
                logger.exception("Voice batch dispatch failed")
                batch_result = {"ok": False, "error": str(exc)}

        for call in calls:
            if call.call_id in self._completed_call_ids:
                continue
            if call.error or call.command is None:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": call.error or "invalid call",
                }
            else:
                try:
                    command = reconciled[call.call_id]
                    result = (
                        {**batch_result, "voice_turn_id": call.call_id, "batched": True}
                        if batch_result is not None
                        else await self._dispatcher.execute(call.call_id, command)
                    )
                except Exception as exc:
                    logger.exception("Failed to execute voice command %s", call.call_id)
                    result = {"ok": False, "error": str(exc)}
            await write_tool_output(call.call_id, result)
            self._completed_call_ids.add(call.call_id)
        self._ack_completed = False
        self._last_user_transcript = ""
