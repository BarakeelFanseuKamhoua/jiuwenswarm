# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

"""Read-only event mirroring from a TUI voice session to Web clients."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceMirrorBinding:
    """One browser websocket observing one TUI session."""

    voice_session_id: str
    web_session_id: str
    user_id: str | None = None


class VoiceMirrorRegistry:
    """Keep websocket-scoped mirror bindings and copy outbound TUI events.

    The registry deliberately mirrors only already-produced event frames. It never
    forwards a request to the agent runtime, so binding a browser cannot execute a
    task for a second time.
    """

    def __init__(self, web_channel: Any) -> None:
        self._web_channel = web_channel
        self._bindings_by_voice: dict[str, dict[int, tuple[Any, VoiceMirrorBinding]]] = {}
        self._voice_by_ws: dict[int, str] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_session_id(value: Any, field: str) -> str:
        session_id = str(value or "").strip()
        if not session_id:
            raise ValueError(f"{field} is required")
        if len(session_id) > 256:
            raise ValueError(f"{field} is too long")
        return session_id

    async def bind(
        self,
        ws: Any,
        *,
        voice_session_id: Any,
        web_session_id: Any,
        user_id: str | None = None,
    ) -> VoiceMirrorBinding:
        voice_sid = self._normalize_session_id(voice_session_id, "voice_session_id")
        web_sid = self._normalize_session_id(web_session_id, "session_id")
        binding = VoiceMirrorBinding(
            voice_session_id=voice_sid,
            web_session_id=web_sid,
            user_id=str(user_id).strip() if user_id else None,
        )
        ws_key = id(ws)
        async with self._lock:
            self._remove_ws_locked(ws_key)
            self._bindings_by_voice.setdefault(voice_sid, {})[ws_key] = (ws, binding)
            self._voice_by_ws[ws_key] = voice_sid
        logger.info(
            "[VoiceMirror] bound voice_session_id=%s -> web_session_id=%s",
            voice_sid,
            web_sid,
        )
        return binding

    async def unbind(self, ws: Any) -> VoiceMirrorBinding | None:
        ws_key = id(ws)
        async with self._lock:
            removed = self._remove_ws_locked(ws_key)
        if removed is not None:
            logger.info(
                "[VoiceMirror] unbound voice_session_id=%s -> web_session_id=%s",
                removed.voice_session_id,
                removed.web_session_id,
            )
        return removed

    async def cleanup_ws(self, ws: Any, _session_ids: Any = None) -> None:
        """WebChannel disconnect hook."""
        await self.unbind(ws)

    def _remove_ws_locked(self, ws_key: int) -> VoiceMirrorBinding | None:
        voice_sid = self._voice_by_ws.pop(ws_key, None)
        if voice_sid is None:
            return None
        entries = self._bindings_by_voice.get(voice_sid)
        if entries is None:
            return None
        item = entries.pop(ws_key, None)
        if not entries:
            self._bindings_by_voice.pop(voice_sid, None)
        return item[1] if item else None

    async def mirror(self, msg: Any, routing_target: Any = None) -> int:
        """Copy one TUI event to every browser observing its source session."""
        if getattr(msg, "type", None) == "res":
            return 0
        voice_sid = str(getattr(msg, "session_id", "") or "").strip()
        if not voice_sid:
            return 0
        async with self._lock:
            bindings = list(self._bindings_by_voice.get(voice_sid, {}).values())
        if not bindings:
            return 0

        frame = self._web_channel.serialize_event_frame(msg, routing_target)
        event = str(frame.get("event") or "chat.final")
        source_payload = frame.get("payload")
        source_payload = source_payload if isinstance(source_payload, dict) else {}
        delivered = 0
        stale_ws: list[Any] = []
        for ws, binding in bindings:
            if bool(getattr(ws, "closed", False)):
                stale_ws.append(ws)
                continue
            message_user_id = str(getattr(msg, "user_id", "") or "").strip()
            if binding.user_id and message_user_id and binding.user_id != message_user_id:
                logger.warning(
                    "[VoiceMirror] rejected cross-user event voice_session_id=%s",
                    voice_sid,
                )
                continue
            payload = {
                **source_payload,
                "session_id": binding.web_session_id,
                "voice_mirror": {
                    "read_only": True,
                    "origin_channel": "tui",
                    "origin_session_id": voice_sid,
                },
            }
            await self._web_channel.send_event(ws, event, payload)
            delivered += 1

        for ws in stale_ws:
            await self.unbind(ws)
        return delivered
