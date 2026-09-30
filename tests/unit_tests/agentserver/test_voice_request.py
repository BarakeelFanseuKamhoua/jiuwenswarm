# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the realtime-voice request param recognisers."""

from __future__ import annotations

from jiuwenswarm.server.runtime.agent_adapter.voice_request import (
    is_voice_members_pause_params,
    is_voice_request_params,
)


class TestIsVoiceMembersPauseParams:
    def test_true_only_for_explicit_flag(self) -> None:
        assert is_voice_members_pause_params(
            {"mode": "team", "voice_pause_members": True}
        ) is True

    def test_false_for_flag_off(self) -> None:
        assert is_voice_members_pause_params(
            {"mode": "team", "voice_pause_members": False}
        ) is False

    def test_false_for_voice_request_without_flag(self) -> None:
        # 仅带语音标记、未带成员级暂停标记的请求不能触发静默窗口逻辑。
        assert (
            is_voice_members_pause_params(
                {"mode": "team", "voice": True, "voice_display_text": "把报表任务先停一停。"}
            )
            is False
        )

    def test_false_for_non_dict(self) -> None:
        assert is_voice_members_pause_params(None) is False
        assert is_voice_members_pause_params("voice_pause_members") is False


class TestIsVoiceRequestParams:
    def test_voice_flag_marks_request(self) -> None:
        assert is_voice_request_params({"voice": True}) is True

    def test_display_text_marks_request(self) -> None:
        assert (
            is_voice_request_params({"voice_display_text": "把报表任务先停一停。"}) is True
        )

    def test_blank_display_text_does_not_mark(self) -> None:
        assert is_voice_request_params({"voice_display_text": "   "}) is False

    def test_plain_params_are_not_voice(self) -> None:
        assert is_voice_request_params({"mode": "team"}) is False
        assert is_voice_request_params(None) is False
