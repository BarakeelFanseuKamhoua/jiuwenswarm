# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""流被取消时消费循环补发 cancelled 终止帧的回归测试。

契约：每个流式请求恰有一个 is_complete=True 终止帧。chat.interrupt 等
取消场景下，原实现只留痕+log+raise，流静默死亡，客户端只能靠超时收尾。
"""

from __future__ import annotations

import asyncio

import pytest

from jiuwenswarm.common.e2a.wire_codec import encode_agent_chunk_for_wire
from jiuwenswarm.common.schema.agent import AgentRequest, AgentResponseChunk
from jiuwenswarm.server.runtime.agent_adapter import interface as interface_module


def _request(request_id: str) -> AgentRequest:
    return AgentRequest(
        request_id=request_id,
        channel_id="tui",
        session_id=f"sess-{request_id}",
        params={"query": "hello", "mode": "agent"},
        is_stream=True,
    )


def _delta(request_id: str, content: str) -> AgentResponseChunk:
    return AgentResponseChunk(
        request_id=request_id,
        channel_id="tui",
        payload={"event_type": "chat.delta", "content": content},
        is_complete=False,
    )


def _terminal(request_id: str) -> AgentResponseChunk:
    return AgentResponseChunk(
        request_id=request_id,
        channel_id="tui",
        payload={"is_complete": True},
        is_complete=True,
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    monkeypatch.setattr(
        interface_module.JiuWenSwarm,
        "_ensure_adapter",
        lambda *_args, **_kwargs: adapter,
    )
    monkeypatch.setattr(
        interface_module,
        "get_config",
        lambda: {"preferred_language": "zh", "memory": {"mode": "disabled"}},
    )
    monkeypatch.setattr(interface_module, "get_memory_mode", lambda _config: "disabled")
    monkeypatch.setattr(interface_module, "append_history_record", lambda **_kwargs: None)
    monkeypatch.setattr(
        interface_module, "_schedule_symphony_session_feedback", lambda *_args: None
    )


def _cancelled_chunks(chunks) -> list:
    return [
        c for c in chunks
        if isinstance(c.payload, dict) and c.payload.get("code") == "cancelled"
    ]


@pytest.mark.asyncio
async def test_consumer_cancellation_emits_cancelled_terminal_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """流静默期被取消：消费方在 CancelledError 前收到唯一 cancelled 终止帧。"""
    started = asyncio.Event()
    hang_forever = asyncio.Event()

    class HangingAdapter:
        @staticmethod
        async def process_message_stream_impl(*_args, **_kwargs):
            yield _delta("req-cancel", "x")
            yield _delta("req-cancel", "y")
            started.set()
            await hang_forever.wait()

    _patch_common(monkeypatch, HangingAdapter)
    swarm = interface_module.JiuWenSwarm()
    chunks: list[AgentResponseChunk] = []

    async def consume() -> None:
        async for chunk in swarm.process_message_stream(_request("req-cancel")):
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    finals = [c for c in chunks if c.is_complete]
    cancelled = _cancelled_chunks(chunks)
    assert len(finals) == 1
    assert finals[0] is cancelled[0]
    assert cancelled[0].payload["event_type"] == "chat.error"
    assert [c.payload.get("event_type") for c in chunks[:2]] == ["chat.delta", "chat.delta"]


@pytest.mark.asyncio
async def test_cancellation_after_terminal_does_not_double_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """终止帧已发出后再取消：不得补发第二个终止帧。"""
    started = asyncio.Event()
    hang_forever = asyncio.Event()

    class TerminalThenHangingAdapter:
        @staticmethod
        async def process_message_stream_impl(*_args, **_kwargs):
            yield _delta("req-nodouble", "x")
            yield _terminal("req-nodouble")
            started.set()
            await hang_forever.wait()

    _patch_common(monkeypatch, TerminalThenHangingAdapter)
    swarm = interface_module.JiuWenSwarm()
    chunks: list[AgentResponseChunk] = []

    async def consume() -> None:
        async for chunk in swarm.process_message_stream(_request("req-nodouble")):
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len([c for c in chunks if c.is_complete]) == 1
    assert _cancelled_chunks(chunks) == []


@pytest.mark.asyncio
async def test_normal_completion_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正常完成路径不出现 cancelled 帧。"""

    class CompletingAdapter:
        @staticmethod
        async def process_message_stream_impl(*_args, **_kwargs):
            yield _delta("req-normal", "answer")
            yield AgentResponseChunk(
                request_id="req-normal",
                channel_id="tui",
                payload={"event_type": "chat.final", "content": "answer"},
                is_complete=False,
            )
            yield _terminal("req-normal")

    _patch_common(monkeypatch, CompletingAdapter)
    swarm = interface_module.JiuWenSwarm()
    chunks = [
        chunk async for chunk in swarm.process_message_stream(_request("req-normal"))
    ]

    assert chunks[-1].is_complete is True
    assert _cancelled_chunks(chunks) == []


def test_cancelled_terminal_chunk_encodes_as_final_error_envelope() -> None:
    """补发帧经现成信封映射成为合法终止帧（is_final + status=failed + e2a.error）。"""
    chunk = AgentResponseChunk(
        request_id="req-env",
        channel_id="tui",
        payload={"event_type": "chat.error", "code": "cancelled", "error": "任务已取消"},
        is_complete=True,
    )
    wire = encode_agent_chunk_for_wire(chunk, response_id="req-env", sequence=2)

    assert wire["is_final"] is True
    assert wire["status"] == "failed"
    assert wire["response_kind"] == "e2a.error"
    assert wire["body"]["details"]["code"] == "cancelled"


@pytest.mark.asyncio
async def test_team_followup_deferred_stream_emits_single_terminal_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """follow-up 受理短流（deferred 占位）：全流恰有一个终止帧（facade 收尾）。

    2026-10-08 专家团 HITL 续跑事故：team_helpers 自发终帧 + facade 收尾帧 =
    双终止帧上线，桌面端 deferred 守卫一次性消费后被第二帧穿透误判回合完成
    （计费 FINISH 提前收口、会话上报链 final 断链）。team_helpers 已不再自发
    终帧，本用例在 facade 出口钉死「恰一个终止帧」。
    """

    class FollowupAckAdapter:
        @staticmethod
        async def process_message_stream_impl(*_args, **_kwargs):
            # team_helpers follow-up 受理分支的真实产出形态：只有 deferred 占位帧
            yield AgentResponseChunk(
                request_id="req-followup-ack",
                channel_id="desktop",
                payload={
                    "event_type": "chat.processing_status_deferred",
                    "session_id": "sess-req-followup-ack",
                },
                is_complete=False,
            )

    _patch_common(monkeypatch, FollowupAckAdapter)
    swarm = interface_module.JiuWenSwarm()
    chunks = [
        chunk async for chunk in swarm.process_message_stream(_request("req-followup-ack"))
    ]

    assert chunks[0].payload == {
        "event_type": "chat.processing_status_deferred",
        "session_id": "sess-req-followup-ack",
    }
    assert chunks[0].is_complete is False
    finals = [c for c in chunks if c.is_complete]
    assert len(finals) == 1
