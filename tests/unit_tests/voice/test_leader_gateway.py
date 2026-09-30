import asyncio
from typing import Any

import pytest

from jiuwenswarm.voice.leader_gateway import (
    LeaderGatewayBridge,
    build_chat_interrupt_frame,
    build_chat_send_frame,
    build_chat_text_frame,
    is_leader_reply_payload,
    leader_reply_for_speech,
)
from jiuwenswarm.voice.models import LeaderInstruction, VoiceCommand


class _QueueGatewayClient:
    def __init__(self) -> None:
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []

    async def recv(self) -> dict[str, Any]:
        return await self.events.get()

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send_request(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)


def test_build_chat_send_frame_always_targets_team_leader() -> None:
    frame = build_chat_send_frame(
        voice_turn_id="turn-42",
        session_id="session-7",
        instruction=LeaderInstruction(
            summary="修复测试",
            instruction="重新运行 PR 测试并处理失败",
            constraints=("不要合并 PR",),
            expected_output="测试报告",
        ),
        mode="team",
        cwd="/tmp/project",
        project_dir="/tmp/project",
    )

    assert frame["method"] == "chat.send"
    assert frame["params"]["agent_ref"] == {"mode": "team", "id": "default"}
    assert not frame["params"]["content"].startswith("@")
    assert "[语音请求 turn-42]" in frame["params"]["content"]
    assert "不要合并 PR" in frame["params"]["content"]


def test_build_chat_text_frame_preserves_original_question() -> None:
    frame = build_chat_text_frame(
        voice_turn_id="turn-question",
        session_id="session-7",
        content="这个方案为什么这样设计？",
        mode="team",
        cwd="/tmp/project",
        project_dir="/tmp/project",
    )

    assert frame["method"] == "chat.send"
    assert frame["params"]["content"] == "这个方案为什么这样设计？"
    assert frame["params"]["query"] == frame["params"]["content"]


def test_leader_reply_filter_rejects_all_member_output() -> None:
    assert is_leader_reply_payload({"role": "leader", "content": "完成"})
    assert not is_leader_reply_payload(
        {"role": "teammate", "member_name": "writer", "content": "成员汇报"}
    )
    assert leader_reply_for_speech(
        {"role": "teammate", "member_name": "writer", "content": "成员汇报"}
    ) is None


def test_leader_reply_filter_uses_delta_fallback_and_bounds_long_text() -> None:
    assert leader_reply_for_speech(
        {"role": "leader", "content": ""},
        fallback_content="来自 delta 的完整回复",
    ) == "来自 delta 的完整回复"
    assert leader_reply_for_speech(
        {"role": "leader", "content": "一二三四五六"},
        max_chars=4,
    ) == "一二三四……详细内容较长，请在终端或网页中查看。"


@pytest.mark.asyncio
async def test_submit_is_exactly_once_and_has_no_member_target() -> None:
    client = _QueueGatewayClient()
    bridge = LeaderGatewayBridge(
        gateway_url="ws://unused",
        session_id="voice-test",
        client=client,  # type: ignore[arg-type]
    )
    instruction = LeaderInstruction(summary="测试", instruction="运行测试")

    first = await bridge.submit("call-1", instruction)
    second = await bridge.submit("call-1", instruction)

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert len(client.sent) == 1
    assert client.sent[0]["params"]["agent_ref"]["id"] == "default"


def test_build_chat_interrupt_frame_maps_voice_control() -> None:
    frame = build_chat_interrupt_frame(
        voice_turn_id="turn-pause",
        session_id="session-7",
        command=VoiceCommand(name="pause_task", reason="用户要确认"),
        mode="team",
        cwd="/tmp/project",
        project_dir="/tmp/project",
    )

    assert frame["method"] == "chat.interrupt"
    assert frame["params"]["intent"] == "pause"
    assert frame["params"]["team"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_name", "expected_method", "expected_intent"),
    [
        ("pause_task", "chat.interrupt", "pause"),
        ("resume_task", "chat.send", None),
        ("cancel_task", "chat.send", None),
        ("get_task_status", "chat.send", None),
        ("ask_leader", "chat.send", None),
    ],
)
async def test_execute_maps_voice_commands_to_gateway(
    command_name: str,
    expected_method: str,
    expected_intent: str | None,
) -> None:
    client = _QueueGatewayClient()
    bridge = LeaderGatewayBridge(
        gateway_url="ws://unused",
        session_id="voice-test",
        client=client,  # type: ignore[arg-type]
    )
    command = VoiceCommand.from_tool_call(
        command_name,
        {
            "query": "现在做到哪了",
            "question": "这个方案为什么这样设计？",
            "reason": "测试",
            "target_task_ids": ["task-a"],
        },
    )

    result = await bridge.execute(f"call-{command_name}", command)

    assert result["ok"] is True
    assert client.sent[0]["method"] == expected_method
    if expected_intent is not None:
        assert client.sent[0]["params"]["intent"] == expected_intent
    if command_name == "ask_leader":
        assert client.sent[0]["params"]["content"] == "这个方案为什么这样设计？"
    if command_name == "cancel_task":
        content = client.sent[0]["params"]["content"]
        assert "目标任务：task-a" in content
        assert "不得停止整个 Team" in content


@pytest.mark.asyncio
async def test_supplement_execute_sends_new_input() -> None:
    client = _QueueGatewayClient()
    bridge = LeaderGatewayBridge(
        gateway_url="ws://unused",
        session_id="voice-test",
        client=client,  # type: ignore[arg-type]
    )
    command = VoiceCommand.from_tool_call(
        "supplement_task",
        {
            "instruction": "把当前实现改成 React",
            "target_task_ids": ["task-a"],
        },
    )

    await bridge.execute("call-supplement", command)

    assert client.sent[0]["method"] == "chat.send"
    assert "改成 React" in client.sent[0]["params"]["content"]
    assert "目标任务：task-a" in client.sent[0]["params"]["content"]


@pytest.mark.asyncio
async def test_gateway_forwards_only_complete_leader_reply() -> None:
    client = _QueueGatewayClient()
    spoken: list[str] = []
    received = asyncio.Event()

    async def handle_reply(text: str) -> None:
        spoken.append(text)
        received.set()

    bridge = LeaderGatewayBridge(
        gateway_url="ws://unused",
        session_id="voice-test",
        client=client,  # type: ignore[arg-type]
        leader_reply_handler=handle_reply,
    )
    receiver = asyncio.create_task(bridge._receive_events())
    try:
        await client.events.put(
            {
                "type": "event",
                "event": "chat.final",
                "payload": {
                    "role": "teammate",
                    "member_name": "writer",
                    "content": "成员内部输出",
                },
            }
        )
        await client.events.put(
            {
                "type": "event",
                "event": "chat.delta",
                "payload": {"rid": 1, "role": "leader", "content": "任务"},
            }
        )
        await client.events.put(
            {
                "type": "event",
                "event": "chat.final",
                "payload": {"rid": 1, "role": "leader", "content": "已完成"},
            }
        )
        await asyncio.wait_for(received.wait(), timeout=1)
        assert spoken == ["已完成"]
    finally:
        receiver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receiver
