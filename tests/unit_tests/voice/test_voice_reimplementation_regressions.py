from __future__ import annotations

from typing import Any

import pytest

from jiuwenswarm.server.runtime.agent_adapter.interface import _history_user_content
from jiuwenswarm.voice.coordinator import VoiceDispatchCoordinator
from jiuwenswarm.voice.models import (
    LeaderInstruction,
    VoiceCommand,
    SUBMIT_TASK_TOOL,
    SUPPLEMENT_TASK_TOOL,
    ASK_LEADER_TOOL,
    reconcile_voice_command_with_transcript,
    reconcile_supplement_with_team_info,
)
from jiuwenswarm.voice.qwen_realtime import build_session_update, QwenRealtimeConfig


def _submit_args(summary: str) -> dict[str, Any]:
    return {"summary": summary, "instruction": f"执行：{summary}"}


def test_voice_prompt_separates_execution_from_team_question() -> None:
    tools = {
        item["function"]["name"]: item["function"]["description"]
        for item in build_session_update(
            QwenRealtimeConfig(api_key="secret", workspace_id="workspace")
        )["session"]["tools"]
    }
    assert "submit_task" in tools["submit_task"]
    assert "新的执行" in tools["submit_task"]
    assert "外部访问" in tools["submit_task"]
    assert "ask_leader" in tools["ask_leader"]
    assert "状态" in tools["ask_leader"]
    assert "只读取后端已有上下文" in tools["ask_leader"]


def test_voice_batch_is_machine_marked_and_keeps_each_intent() -> None:
    first = VoiceCommand.from_tool_call("submit_task", _submit_args("查询今日金价"))
    second = VoiceCommand.from_tool_call("submit_task", _submit_args("查询武汉天气"))
    rendered = VoiceCommand.render_voice_batch((first, second), "turn-1")
    assert "[VOICE_DISPATCH_MACHINE_ONLY]" in rendered
    assert "意图1" in rendered and "意图2" in rendered
    assert "查询今日金价" in rendered and "查询武汉天气" in rendered


def test_history_prefers_voice_transcript_over_machine_dispatch() -> None:
    dispatch = "[语音请求 turn-1]\n意图1 动作：新建任务\n完整要求：机器调度模板"
    assert _history_user_content(
        {"voice_display_text": "再查一下今天的金价"}, dispatch
    ) == "再查一下今天的金价"


class _BatchDispatcher:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, tuple[VoiceCommand, ...]]] = []
        self.single_calls: list[str] = []

    async def execute(self, voice_turn_id: str, command: VoiceCommand) -> dict[str, Any]:
        self.single_calls.append(voice_turn_id)
        return {"ok": True, "single": True}

    async def execute_batch(
        self, voice_turn_id: str, commands: tuple[VoiceCommand, ...]
    ) -> dict[str, Any]:
        self.batch_calls.append((voice_turn_id, commands))
        return {"ok": True, "batch": True}


@pytest.mark.asyncio
async def test_batch_dispatch_is_called_once_for_multiple_function_calls() -> None:
    dispatcher = _BatchDispatcher()
    coordinator = VoiceDispatchCoordinator(dispatcher)
    outputs: list[tuple[str, dict[str, Any]]] = []
    coordinator.record_function_call(
        {"call_id": "call-1", "name": "submit_task", "arguments": _submit_args("金价")}
    )
    coordinator.record_function_call(
        {"call_id": "call-2", "name": "submit_task", "arguments": _submit_args("天气")}
    )
    await coordinator.flush(
        write_tool_output=lambda call_id, result: _append(outputs, call_id, result),
        fallback_acknowledge=lambda _message: _noop(),
    )
    assert len(dispatcher.batch_calls) == 1
    assert dispatcher.single_calls == []
    assert [call_id for call_id, _ in outputs] == ["call-1", "call-2"]
    assert all(result["batch"] is True for _, result in outputs)


async def _append(
    outputs: list[tuple[str, dict[str, Any]]], call_id: str, result: dict[str, Any]
) -> None:
    outputs.append((call_id, result))


async def _noop() -> None:
    return None


def test_supplement_requires_existing_target_ids() -> None:
    with pytest.raises(ValueError, match="target_task_ids"):
        VoiceCommand.from_tool_call(
            SUPPLEMENT_TASK_TOOL,
            {"instruction": "改成银价", "target_task_ids": []},
        )


def test_revision_action_is_rendered_as_modify_task() -> None:
    command = VoiceCommand.from_tool_call(
        SUPPLEMENT_TASK_TOOL,
        {"instruction": "把杭州改成武汉", "target_task_ids": ["weather-1"]},
    )
    assert "动作：修改任务" in command.dispatch_text("turn-1")
    supplement_schema = next(
        item["function"]
        for item in build_session_update(
            QwenRealtimeConfig(api_key="secret", workspace_id="workspace")
        )["session"]["tools"]
        if item["function"]["name"] == SUPPLEMENT_TASK_TOOL
    )
    assert "不创建独立新目标" in supplement_schema["description"]
    assert supplement_schema["parameters"]["required"] == [
        "instruction",
        "target_task_ids",
    ]


def test_submit_task_is_not_reclassified_by_another_clause_in_transcript() -> None:
    command = VoiceCommand.from_tool_call(
        SUBMIT_TASK_TOOL,
        {"summary": "查询 NBA 赛程", "instruction": "获取最新 NBA 赛程"},
    )
    assert reconcile_voice_command_with_transcript(
        command,
        "查询 NBA 赛程，然后把黄金价格改成白银价格",
    ) == command


def test_ask_leader_reconciliation_does_not_contain_domain_specific_rules() -> None:
    command = VoiceCommand.from_tool_call(
        ASK_LEADER_TOOL, {"question": "请解释当前团队的决策"}
    )
    assert reconcile_voice_command_with_transcript(command, command.query) == command


def test_revision_without_ids_does_not_guess_existing_task() -> None:
    command = VoiceCommand(
        name=SUPPLEMENT_TASK_TOOL,
        instruction=LeaderInstruction(
            summary="调整天气", instruction="把杭州天气改成武汉天气"
        ),
    )
    fixed = reconcile_supplement_with_team_info(
        command,
        {"tasks": [{"task_id": "weather-1", "title": "查询杭州天气"}]},
    )
    assert fixed.name == SUPPLEMENT_TASK_TOOL
    assert fixed.target_task_ids == ()


def test_partial_target_ids_are_not_silently_dropped() -> None:
    command = VoiceCommand(
        name=SUPPLEMENT_TASK_TOOL,
        instruction=LeaderInstruction(summary="修改任务", instruction="修改价格范围"),
        target_task_ids=("known-task", "missing-task"),
    )
    fixed = reconcile_supplement_with_team_info(
        command,
        {"tasks": [{"task_id": "known-task", "title": "价格查询"}]},
    )
    assert fixed.target_task_ids == ("known-task", "missing-task")


def test_unresolved_supplement_is_never_relabelled_as_new_submission() -> None:
    command = VoiceCommand(
        name=SUPPLEMENT_TASK_TOOL,
        instruction=LeaderInstruction(summary="修改目标", instruction="修改已有目标"),
        target_task_ids=("missing-task",),
    )
    fixed = reconcile_supplement_with_team_info(
        command,
        {"tasks": [{"task_id": "other-task", "title": "另一个目标"}]},
    )
    assert fixed.name == SUPPLEMENT_TASK_TOOL
    assert fixed.target_task_ids == ("missing-task",)
