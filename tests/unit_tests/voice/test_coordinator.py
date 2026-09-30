from __future__ import annotations

import json
from typing import Any

import pytest

from jiuwenswarm.voice.coordinator import VoiceDispatchCoordinator
from jiuwenswarm.voice.models import (
    CANCEL_TASK_TOOL,
    RESUME_TASK_TOOL,
    LeaderInstruction,
    VoiceCommand,
    reconcile_voice_command_with_transcript,
)


class _Dispatcher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, VoiceCommand]] = []

    async def execute(
        self,
        voice_turn_id: str,
        command: VoiceCommand,
    ) -> dict[str, Any]:
        self.events.append("execute")
        self.calls.append((voice_turn_id, command))
        return {"ok": True, "voice_turn_id": voice_turn_id}


class _BatchDispatcher(_Dispatcher):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.batch_calls: list[tuple[str, tuple[VoiceCommand, ...]]] = []

    async def execute_batch(
        self, voice_turn_id: str, commands: tuple[VoiceCommand, ...]
    ) -> dict[str, Any]:
        self.batch_calls.append((voice_turn_id, commands))
        self.events.append("execute_batch")
        return {"ok": True, "batch": True}


def _function_call(
    call_id: str = "call-1",
    *,
    name: str = "submit_task",
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(
            {
                "summary": "检查测试",
                "instruction": "重新运行 PR 中失败的测试并报告结果",
                "constraints": ["先不要修改代码"],
                "expected_output": "测试结论",
                # Old/model-invented routing data is intentionally ignored by
                # the Leader-only data model.
                "target_member": "writer",
            },
            ensure_ascii=False,
        ),
    }


def test_tool_arguments_create_leader_only_instruction() -> None:
    instruction = LeaderInstruction.from_arguments(
        {
            "summary": "普通任务",
            "instruction": "检查当前状态",
            "target_member": "writer",
        }
    )

    assert instruction.summary == "普通任务"
    assert not hasattr(instruction, "target_member")


@pytest.mark.parametrize(
    "transcript",
    [
        "不用，继续执行",
        "不用了，接着做",
        "没关系，不要停",
    ],
)
def test_positive_continuation_blocks_accidental_cancel(transcript: str) -> None:
    command = VoiceCommand(
        name=CANCEL_TASK_TOOL,
        reason="用户要求停止",
        target_task_ids=("task_001",),
    )

    reconciled = reconcile_voice_command_with_transcript(command, transcript)

    assert reconciled.name == RESUME_TASK_TOOL
    assert reconciled.target_task_ids == ()


@pytest.mark.parametrize(
    "transcript",
    [
        "不要继续执行",
        "取消 task_001，然后继续第二个任务",
        "这个任务不要了",
    ],
)
def test_explicit_cancel_is_not_rewritten_as_resume(transcript: str) -> None:
    command = VoiceCommand(
        name=CANCEL_TASK_TOOL,
        target_task_ids=("task_001",),
    )

    assert reconcile_voice_command_with_transcript(command, transcript) == command


@pytest.mark.asyncio
async def test_acknowledgement_happens_before_submission_and_only_once() -> None:
    events: list[str] = []
    dispatcher = _Dispatcher(events)
    coordinator = VoiceDispatchCoordinator(dispatcher)
    outputs: list[tuple[str, dict[str, Any]]] = []

    async def write_output(call_id: str, output: dict[str, Any]) -> None:
        events.append("tool_output")
        outputs.append((call_id, output))

    async def fallback_ack(message: str) -> None:
        events.append("fallback_ack")

    coordinator.record_function_call(_function_call())
    coordinator.mark_ack_completed("收到，我会交给 Team Leader。")
    events.append("ack")
    await coordinator.flush(
        write_tool_output=write_output,
        fallback_acknowledge=fallback_ack,
    )
    coordinator.record_function_call(_function_call())
    await coordinator.flush(
        write_tool_output=write_output,
        fallback_acknowledge=fallback_ack,
    )

    assert events == ["ack", "execute", "tool_output"]
    assert len(dispatcher.calls) == 1
    assert dispatcher.calls[0][1].instruction is not None
    assert dispatcher.calls[0][1].instruction.constraints == ("先不要修改代码",)
    assert outputs == [("call-1", {"ok": True, "voice_turn_id": "call-1"})]


@pytest.mark.asyncio
async def test_tool_only_response_gets_fallback_ack_before_submission() -> None:
    events: list[str] = []
    coordinator = VoiceDispatchCoordinator(_Dispatcher(events))

    async def write_output(_call_id: str, _output: dict[str, Any]) -> None:
        events.append("tool_output")

    async def fallback_ack(message: str) -> None:
        assert "收到" in message
        events.append("fallback_ack")

    coordinator.record_function_call(_function_call(name="submit_to_leader"))
    await coordinator.flush(
        write_tool_output=write_output,
        fallback_acknowledge=fallback_ack,
    )
    assert events == ["fallback_ack", "execute", "tool_output"]


@pytest.mark.asyncio
async def test_cancelled_response_is_not_submitted() -> None:
    events: list[str] = []
    dispatcher = _Dispatcher(events)
    coordinator = VoiceDispatchCoordinator(dispatcher)
    coordinator.record_function_call(_function_call())
    coordinator.cancel_pending()
    await coordinator.flush(
        write_tool_output=lambda _call_id, _output: _noop(),
        fallback_acknowledge=lambda _message: _noop(),
    )
    assert events == []
    assert dispatcher.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["pause_task", "resume_task", "cancel_task", "get_task_status", "ask_leader"],
)
async def test_control_tools_are_dispatched(name: str) -> None:
    events: list[str] = []
    dispatcher = _Dispatcher(events)
    coordinator = VoiceDispatchCoordinator(dispatcher)
    arguments = {"query": "做到哪了"}
    if name == "cancel_task":
        arguments["target_task_ids"] = ["task-a"]
    coordinator.record_function_call(
        {
            "call_id": f"call-{name}",
            "name": name,
            "arguments": json.dumps(arguments),
        }
    )
    await coordinator.flush(
        write_tool_output=lambda _call_id, _output: _noop(),
        fallback_acknowledge=lambda _message: _noop(),
    )

    assert dispatcher.calls[0][1].name == name
    if name == "cancel_task":
        assert dispatcher.calls[0][1].target_task_ids == ("task-a",)


@pytest.mark.asyncio
async def test_multiple_calls_use_one_batch_dispatch_and_keep_each_ack() -> None:
    events: list[str] = []
    dispatcher = _BatchDispatcher(events)
    coordinator = VoiceDispatchCoordinator(dispatcher)
    outputs: list[tuple[str, dict[str, Any]]] = []

    coordinator.record_function_call(_function_call("call-1"))
    coordinator.record_function_call(_function_call("call-2"))
    await coordinator.flush(
        write_tool_output=lambda call_id, result: _record_output(outputs, call_id, result),
        fallback_acknowledge=lambda _message: _noop(),
    )

    assert len(dispatcher.batch_calls) == 1
    assert dispatcher.calls == []
    assert [call_id for call_id, _ in outputs] == ["call-1", "call-2"]
    assert all(result["batch"] is True for _, result in outputs)


async def _record_output(
    outputs: list[tuple[str, dict[str, Any]]], call_id: str, result: dict[str, Any]
) -> None:
    outputs.append((call_id, result))


async def _noop() -> None:
    return None
