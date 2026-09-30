# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Recognise requests issued by the web realtime-voice path.

Voice barge-in needs stricter pause / follow-up semantics than typed input
(AgentCore ``voice=True``); every other request keeps the default behavior.
"""

from __future__ import annotations

from typing import Any


def is_voice_request_params(params: Any) -> bool:
    """Whether request params come from the realtime-voice path.

    The frontend marks the barge-in pause with ``voice: true``; spoken
    commands always carry the transcribed ``voice_display_text``.
    """
    if not isinstance(params, dict):
        return False
    if params.get("voice") is True:
        return True
    voice_display = params.get("voice_display_text")
    return isinstance(voice_display, str) and bool(voice_display.strip())


def is_voice_members_pause_params(params: Any) -> bool:
    """Whether params mark the spoken members-only pause.

    ``voice_pause_members`` rides on the chat.send that parks the teammates
    but keeps the leader live (``_apply_voice_team_pause_state``); the flag
    is set by the web frontend's spoken-pause dispatch and by nothing else,
    so the stream layer can treat the round as answered once the leader's
    own final has been yielded.
    """
    if not isinstance(params, dict):
        return False
    return params.get("voice_pause_members") is True
