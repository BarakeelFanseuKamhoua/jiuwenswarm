import asyncio
import base64
import json
from typing import Any

import pytest

from jiuwenswarm.voice.qwen_realtime import (
    LEADER_BRIEF_REPORT_PREFIX,
    LEADER_REPORT_PREFIX,
    QwenRealtimeConfig,
)
from jiuwenswarm.voice import web_session
from jiuwenswarm.voice.web_session import (
    WebQwenVoiceSession,
    WebVoiceSessionManager,
    normalize_voice_team_info,
)


def _config() -> QwenRealtimeConfig:
    return QwenRealtimeConfig(api_key="secret-value", workspace_id="workspace-123")


def test_reply_mode_reads_voice_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JIUWENSWARM_VOICE_REPLY_MODE", raising=False)
    monkeypatch.setattr(
        web_session,
        "get_config",
        lambda: {"voice": {"reply_mode": "brief"}},
    )

    assert WebVoiceSessionManager._reply_mode_from_config({}) == "brief"


def test_reply_mode_request_parameter_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        web_session,
        "get_config",
        lambda: {"voice": {"reply_mode": "full"}},
    )

    assert (
        WebVoiceSessionManager._reply_mode_from_config({"reply_mode": "brief"})
        == "brief"
    )


class _WebSocket:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = events or []
        self.sent: list[dict[str, Any]] = []

    def __aiter__(self):
        events = iter(self.events)

        async def _next():
            try:
                return json.dumps(next(events), ensure_ascii=False)
            except StopIteration:
                raise StopAsyncIteration

        return type(
            "_Iterator",
            (),
            {"__aiter__": lambda self: self, "__anext__": lambda self: _next()},
        )()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))


def test_team_info_normalization_keeps_minimal_task_directory() -> None:
    normalized = normalize_voice_team_info(
        {
            "team_id": "content-forge",
            "members": [
                {
                    "member_id": "writer",
                    "name": "撰稿人",
                    "role": "teammate",
                    "status": "busy",
                    "execution_status": "running",
                }
            ],
            "tasks": [
                {
                    "task_id": "task-a",
                    "title": "创建 A 文件",
                    "status": "in_progress",
                    "content": "不应进入语音上下文的完整任务内容",
                    "assignee": "writer",
                }
            ],
        }
    )

    assert normalized == {
        "team_id": "content-forge",
        "members": [
            {"member_id": "writer", "name": "撰稿人", "role": "teammate"}
        ],
        "tasks": [
            {
                "task_id": "task-a",
                "title": "创建 A 文件",
                "status": "in_progress",
                "created_order": 1,
            }
        ],
    }


def test_voice_display_text_is_sent_as_user_text_separately_from_dispatch() -> None:
    from jiuwenswarm.server.runtime.agent_adapter.interface import _history_user_content

    dispatch = "[语音请求 turn-1]\n意图1 动作：新建任务\n完整要求：机器模板"
    assert _history_user_content(
        {"voice_display_text": "再查一下今天的金价"}, dispatch
    ) == "再查一下今天的金价"


@pytest.mark.asyncio
async def test_team_info_update_refreshes_instructions_without_response() -> None:
    async def send_event(_name: str, _payload: dict[str, Any]) -> None:
        return None

    websocket = _WebSocket()
    session = WebQwenVoiceSession(_config(), send_event, team_info={"members": []})
    session._websocket = websocket
    team_info = {
        "team_id": "content-forge",
        "members": [
            {"member_id": "writer", "name": "撰稿人", "role": "teammate"}
        ],
    }

    assert await session.update_team_info(team_info) is True
    assert await session.update_team_info(team_info) is False
    assert len(websocket.sent) == 1
    assert websocket.sent[0]["type"] == "session.update"
    assert "成员ID：writer" in websocket.sent[0]["session"]["instructions"]
    assert not any(item.get("type") == "response.create" for item in websocket.sent)


@pytest.mark.asyncio
async def test_submit_function_call_becomes_structured_web_voice_command() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "生成一个只含 OK 的文本",
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-1",
                "arguments": json.dumps(
                    {
                        "summary": "生成 OK 文本",
                        "instruction": "生成一个只含 OK 的文本",
                        "constraints": ["无换行"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    request = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert request["call_ids"] == ["call-1"]
    assert request["count"] == 1
    assert request["name"] == "submit_task"
    assert request["text"] == "生成一个只含 OK 的文本"
    assert request["commands"][0]["call_id"] == "call-1"
    assert request["commands"][0]["instruction"] == "生成一个只含 OK 的文本"
    assert request["commands"][0]["summary"] == "生成 OK 文本"
    # The single-intent batch collapses to a unified request header + one
    # intent block so the Leader sees the same shape as a multi-intent turn.
    assert "意图1 动作：新建任务" in request["dispatch_text"]
    assert "生成一个只含 OK 的文本" in request["dispatch_text"]
    assert "- 无换行" in request["dispatch_text"]
    assert websocket.sent[-1]["item"]["type"] == "function_call_output"
    assert json.loads(websocket.sent[-1]["item"]["output"])["command_emitted"] == "submit_task"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("pause_task", {"reason": "用户需要确认"}),
        ("resume_task", {}),
        (
            "cancel_task",
            {"reason": "用户明确放弃", "target_task_ids": ["task-a"]},
        ),
        ("get_task_status", {"query": "现在做到哪了"}),
        ("ask_leader", {"question": "这个方案为什么这样设计？"}),
    ],
)
async def test_control_function_calls_become_named_web_commands(
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "测试控制命令",
            },
            {
                "type": "response.function_call_arguments.done",
                "name": tool_name,
                "call_id": f"call-{tool_name}",
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    batch = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert batch["name"] == tool_name
    assert batch["text"] == "测试控制命令"
    assert batch["call_ids"] == [f"call-{tool_name}"]
    assert len(batch["commands"]) == 1
    assert batch["commands"][0]["name"] == tool_name
    if tool_name == "cancel_task":
        assert batch["commands"][0]["target_task_ids"] == ["task-a"]


@pytest.mark.asyncio
async def test_multi_call_turn_emits_batch_boundary_and_all_commands() -> None:
    """A turn with several function calls is wrapped in batch start/end events."""
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "查武汉天气 并查NBA赛程 还查agent进展",
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-1",
                "arguments": json.dumps(
                    {
                        "summary": "查武汉天气",
                        "instruction": "查询武汉当前天气并报告",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-2",
                "arguments": json.dumps(
                    {
                        "summary": "查NBA赛程",
                        "instruction": "查询今日NBA赛程并报告",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-3",
                "arguments": json.dumps(
                    {
                        "summary": "查agent进展",
                        "instruction": "查询AI agent最新进展并报告",
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    names = [name for name, _ in sent_events]
    # voice.transcript precedes the batch event; voice.status may follow it.
    batch_idx = names.index("voice.command_batch")
    # The fake socket then ends, which reports the connection as closed.
    trailing = sent_events[batch_idx + 1:]
    assert all(
        name == "voice.status" or (name == "voice.error" and payload.get("closed"))
        for name, payload in trailing
    )

    batch = sent_events[batch_idx][1]
    assert batch["count"] == 3
    assert batch["call_ids"] == ["call-1", "call-2", "call-3"]
    # A multi-intent turn collapses to ONE chat.send body: one request header
    # plus three unified intent blocks plus one posture footer. The previous
    # welded-template shape (per-action headers, repeated dispatch posture)
    # made the Leader drop the second intent; this shape must not regress.
    dispatch_text = batch["dispatch_text"]
    assert dispatch_text.count("[语音请求") == 1
    assert dispatch_text.count("意图1 动作：") == 1
    assert dispatch_text.count("意图2 动作：") == 1
    assert dispatch_text.count("意图3 动作：") == 1
    assert "查武汉天气" in dispatch_text
    assert "查NBA赛程" in dispatch_text
    assert "查agent进展" in dispatch_text
    # The dispatch posture appears exactly once (not repeated per intent).
    assert dispatch_text.count("不要重复寒暄") == 1

    assert [command["call_id"] for command in batch["commands"]] == [
        "call-1",
        "call-2",
        "call-3",
    ]
    assert [command["summary"] for command in batch["commands"]] == [
        "查武汉天气",
        "查NBA赛程",
        "查agent进展",
    ]

    # Each call still gets its own function_call_output back to Qwen.
    outputs = [
        json.loads(frame["item"]["output"])
        for frame in websocket.sent
        if frame.get("item", {}).get("type") == "function_call_output"
    ]
    assert len(outputs) == 3
    assert all(output["ok"] for output in outputs)


@pytest.mark.asyncio
async def test_revision_and_new_request_keep_independent_actions_and_targets() -> None:
    """A revision clause must not turn a separate new request into a revision."""
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "把黄金价格改成白银价格，然后查询 NBA 赛程",
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "supplement_task",
                "call_id": "call-silver",
                "arguments": json.dumps(
                    {
                        "summary": "白银价格",
                        "instruction": "查询今天白银价格",
                        "target_task_ids": ["task-gold-price"],
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-nba",
                "arguments": json.dumps(
                    {
                        "summary": "NBA 赛程",
                        "instruction": "查询今天或近期 NBA 赛程",
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(
        _config(),
        send_event,
        team_info={
            "tasks": [{"task_id": "task-gold-price", "title": "黄金价格"}],
        },
    )
    session._websocket = websocket

    await session._receive_events()

    batch = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert [(item["name"], item["target_task_ids"]) for item in batch["commands"]] == [
        ("supplement_task", ["task-gold-price"]),
        ("submit_task", []),
    ]


@pytest.mark.asyncio
async def test_continue_transcript_overrides_qwen_cancel_function_call() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "不用，继续执行",
            },
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "cancel_task",
                "call_id": "call-continue-guard",
                "arguments": json.dumps(
                    {
                        "target_task_ids": ["task_001"],
                        "reason": "用户要求停止",
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    batch = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert batch["name"] == "resume_task"
    assert batch["commands"][0]["name"] == "resume_task"
    assert batch["commands"][0]["target_task_ids"] == []
    assert batch["commands"][0]["reason"] == "用户明确要求继续执行"


@pytest.mark.asyncio
async def test_supplement_command_preserves_structured_adjustment() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "supplement_task",
                "call_id": "call-supplement",
                "arguments": json.dumps(
                    {
                        "instruction": "把当前页面改成 React",
                        "constraints": ["保留已完成的样式"],
                        "target_task_ids": ["task-a"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(
        _config(),
        send_event,
        team_info={
            "tasks": [
                {
                    "task_id": "task-a",
                    "title": "改造当前页面",
                    "status": "in_progress",
                }
            ]
        },
    )
    session._websocket = websocket

    await session._receive_events()

    batch = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert batch["name"] == "supplement_task"
    assert batch["commands"][0]["target_task_ids"] == ["task-a"]
    assert "把当前页面改成 React" in batch["dispatch_text"]
    assert "- 保留已完成的样式" in batch["dispatch_text"]
    assert "目标任务：task-a" in batch["dispatch_text"]


@pytest.mark.asyncio
async def test_supplement_before_task_creation_keeps_revision_intent() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "supplement_task",
                "call_id": "call-before-task-created",
                "arguments": json.dumps(
                    {
                        "summary": "调整文本字数",
                        "instruction": "生成一个准确达到300字的文本",
                        "target_task_ids": ["task_001"],
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(
        _config(),
        send_event,
        team_info={"tasks": []},
    )
    session._websocket = websocket

    await session._receive_events()

    batch = next(payload for name, payload in sent_events if name == "voice.command_batch")
    assert batch["name"] == "supplement_task"
    assert batch["commands"][0]["target_task_ids"] == ["task_001"]
    assert "生成一个准确达到300字的文本" in batch["dispatch_text"]
    # The explicit revision target is preserved while the roster catches up.
    assert "目标任务：task_001" in batch["dispatch_text"]


@pytest.mark.asyncio
async def test_interruption_stops_forwarding_old_audio() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    pcm = base64.b64encode(b"pcm").decode("ascii")
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "response.audio.delta", "delta": pcm},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.audio.delta", "delta": pcm},
            {"type": "response.done", "response": {"status": "cancelled"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket
    session._leader_response_requested = True
    session._idle.clear()

    await session._receive_events()

    assert len([name for name, _ in sent_events if name == "voice.audio"]) == 1
    interrupted = next(
        payload for name, payload in sent_events if name == "voice.interrupted"
    )
    assert interrupted["interrupted"] is True
    assert session._user_turn_active is True
    assert not session._idle.is_set()


@pytest.mark.asyncio
async def test_turn_cut_off_before_tool_call_is_replayed_after_next_turn() -> None:
    # A revision is followed within ~100ms by a new request; server VAD
    # cancels the first response before its tool call, and the first late
    # transcript must neither become the second turn's bubble nor be lost.
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    revision = "把北京天气的查询改成上海天气"
    request = "另外帮我查一下明天的航班信息。"
    websocket = _WebSocket(
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-revision"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-revision"},
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started", "item_id": "item-request"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-revision",
                "transcript": revision,
            },
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-request"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-request",
                "transcript": request,
            },
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-request",
                "arguments": json.dumps(
                    {"summary": "查询航班信息", "instruction": "查询明天的航班信息"},
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    transcripts = [p["text"] for name, p in sent_events if name == "voice.transcript"]
    assert transcripts == [request]
    batches = [p for name, p in sent_events if name == "voice.command_batch"]
    assert [(b["text"], b["replayed"]) for b in batches] == [(request, False)]
    assert session._idle.is_set()
    assert session._replay_queue.qsize() == 1

    replay_task = asyncio.create_task(session._replay_interrupted_turns())
    try:
        for _ in range(10):
            if websocket.sent and websocket.sent[-1]["type"] == "response.create":
                break
            await asyncio.sleep(0)
        assert websocket.sent[-1]["type"] == "response.create"
        assert revision in websocket.sent[-2]["item"]["content"][0]["text"]

        websocket.events = [
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "supplement_task",
                "call_id": "call-revision",
                "arguments": json.dumps(
                    {
                        "target_task_ids": ["task-weather"],
                        "instruction": "天气查询改为上海",
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
        await session._receive_events()
        await asyncio.wait_for(session._replay_queue.join(), timeout=1)
    finally:
        replay_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replay_task

    replayed = [p for name, p in sent_events if name == "voice.command_batch"][-1]
    assert replayed["replayed"] is True
    assert replayed["text"] == revision
    assert replayed["name"] == "supplement_task"


@pytest.mark.asyncio
async def test_tool_call_emitted_before_cancel_is_dispatched_without_reasking() -> None:
    # The tool call is complete when the next utterance cancels the response;
    # only the spoken tail is cut. Asking Qwen again tends to yield no tool,
    # so the emitted call itself is dispatched once the session is idle.
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    revision = "把会议纪要的收件人改成产品组"
    request = "再整理一份本周的周报。"
    websocket = _WebSocket(
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-revision"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-revision"},
            {"type": "input_audio_buffer.committed", "item_id": "item-revision"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-revision",
                "transcript": revision,
            },
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "supplement_task",
                "call_id": "call-revision",
                "arguments": json.dumps(
                    {
                        "target_task_ids": ["task-minutes"],
                        "summary": "修改收件人",
                        "instruction": "会议纪要改为发给产品组",
                    },
                    ensure_ascii=False,
                ),
            },
            {"type": "input_audio_buffer.speech_started", "item_id": "item-request"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-request"},
            {"type": "input_audio_buffer.committed", "item_id": "item-request"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-request",
                "transcript": request,
            },
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_task",
                "call_id": "call-request",
                "arguments": json.dumps(
                    {"summary": "整理周报", "instruction": "整理本周的周报"},
                    ensure_ascii=False,
                ),
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    batches = [p for name, p in sent_events if name == "voice.command_batch"]
    assert [(b["text"], b["replayed"]) for b in batches] == [(request, False)]
    assert session._replay_queue.qsize() == 1

    replay_task = asyncio.create_task(session._replay_interrupted_turns())
    try:
        await asyncio.wait_for(session._replay_queue.join(), timeout=1)
    finally:
        replay_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replay_task

    assert not any(item["type"] == "response.create" for item in websocket.sent)
    dispatched = [p for name, p in sent_events if name == "voice.command_batch"][-1]
    assert dispatched["replayed"] is True
    assert dispatched["text"] == revision
    assert dispatched["call_ids"] == ["call-revision"]
    assert dispatched["commands"][0]["target_task_ids"] == ["task-minutes"]
    outputs = [
        item["item"]["call_id"]
        for item in websocket.sent
        if item["type"] == "conversation.item.create"
        and item["item"]["type"] == "function_call_output"
    ]
    assert outputs == ["call-request", "call-revision"]


@pytest.mark.asyncio
async def test_turn_without_tool_reports_completion_to_frontend() -> None:
    # The frontend paused the Team at speech start; only this event tells it
    # that no command will follow to resume it.
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-1"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-1"},
            {"type": "input_audio_buffer.committed", "item_id": "item-1"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "好的，谢谢。",
            },
            {"type": "response.created"},
            {"type": "response.done", "response": {"status": "completed"}},
            {"type": "input_audio_buffer.speech_started", "item_id": "item-2"},
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": "item-2",
                "reason": "turn_invalid",
            },
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    completed = [p for name, p in sent_events if name == "voice.turn_completed"]
    assert [p["reason"] for p in completed] == ["no-tool", "turn_invalid"]
    assert all(p["tool_selected"] is False for p in completed)
    assert session._idle.is_set()


@pytest.mark.asyncio
async def test_fragment_cancelled_before_speech_flag_is_replayed() -> None:
    # One request split by a pause into two utterances. The second
    # speech_started arrives before the first response.created, so the first
    # response is cancelled without the barge-in flag; it must still be
    # replayed, and the fragment turn that selects no tool must not report
    # completion while that replay is pending.
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    first = "帮我查询一下"
    websocket = _WebSocket(
        [
            {"type": "input_audio_buffer.speech_started", "item_id": "item-a"},
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-a"},
            {"type": "input_audio_buffer.committed", "item_id": "item-a"},
            {"type": "input_audio_buffer.speech_started", "item_id": "item-b"},
            {"type": "response.created"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-a",
                "transcript": first,
            },
            {"type": "input_audio_buffer.speech_stopped", "item_id": "item-b"},
            {"type": "input_audio_buffer.committed", "item_id": "item-b"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-b",
                "transcript": "上海明天的天气。",
            },
            {"type": "response.created"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    assert session._replay_queue.qsize() == 1
    # The queued replay still owns the outcome of this exchange.
    assert not [p for name, p in sent_events if name == "voice.turn_completed"]
    assert session._idle.is_set()

    replay_task = asyncio.create_task(session._replay_interrupted_turns())
    try:
        for _ in range(10):
            if websocket.sent and websocket.sent[-1]["type"] == "response.create":
                break
            await asyncio.sleep(0)
        assert first in websocket.sent[-2]["item"]["content"][0]["text"]

        websocket.events = [
            {"type": "response.created"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
        await session._receive_events()
        await asyncio.wait_for(session._replay_queue.join(), timeout=1)
    finally:
        replay_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replay_task

    completed = [p for name, p in sent_events if name == "voice.turn_completed"]
    assert [p["reason"] for p in completed] == ["replay-no-tool"]


@pytest.mark.asyncio
async def test_leader_reply_uses_same_qwen_socket() -> None:
    async def send_event(_name: str, _payload: dict[str, Any]) -> None:
        return None

    websocket = _WebSocket()
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket
    await session.enqueue_leader_reply("任务已完成。")

    task = asyncio.create_task(session._speak_leader_replies())
    try:
        for _ in range(10):
            if len(websocket.sent) == 2:
                break
            await asyncio.sleep(0)

        assert websocket.sent[0]["item"]["content"][0]["text"] == (
            f"{LEADER_REPORT_PREFIX}任务已完成。"
        )
        assert websocket.sent[1] == {
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]},
        }
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class _FlakyWebSocket(_WebSocket):
    """Fails the first ``response.create`` send, as a transient socket error."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    async def send(self, message: str) -> None:
        if not self.failed and json.loads(message).get("type") == "response.create":
            self.failed = True
            raise ConnectionError("transient send failure")
        await super().send(message)


async def _wait_for_sent(websocket: _WebSocket, count: int) -> None:
    for _ in range(50):
        if len(websocket.sent) >= count:
            return
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_leader_reply_loop_survives_a_failed_send() -> None:
    async def send_event(_name: str, _payload: dict[str, Any]) -> None:
        return None

    websocket = _FlakyWebSocket()
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket
    await session.enqueue_leader_reply("第一份报告已生成。")
    await session.enqueue_leader_reply("第二份报告已生成。")

    task = asyncio.create_task(session._speak_leader_replies())
    try:
        await _wait_for_sent(websocket, 3)

        assert [item["type"] for item in websocket.sent] == [
            "conversation.item.create",
            "conversation.item.create",
            "response.create",
        ]
        assert websocket.sent[1]["item"]["content"][0]["text"] == (
            f"{LEADER_REPORT_PREFIX}第二份报告已生成。"
        )
        assert session._leader_response_requested is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_replay_loop_survives_a_failed_send() -> None:
    async def send_event(_name: str, _payload: dict[str, Any]) -> None:
        return None

    websocket = _FlakyWebSocket()
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket
    session._enqueue_replay(web_session._InterruptedTurn("帮我整理本周的会议纪要"))
    session._enqueue_replay(web_session._InterruptedTurn("再汇总一下项目进度"))

    task = asyncio.create_task(session._replay_interrupted_turns())
    try:
        await _wait_for_sent(websocket, 3)

        assert [item["type"] for item in websocket.sent] == [
            "conversation.item.create",
            "conversation.item.create",
            "response.create",
        ]
        assert "再汇总一下项目进度" in websocket.sent[1]["item"]["content"][0]["text"]
        assert session._replay_requested is not None
        assert session._replay_requested.transcript == "再汇总一下项目进度"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_brief_leader_reply_uses_summary_prompt_on_same_socket() -> None:
    async def send_event(_name: str, _payload: dict[str, Any]) -> None:
        return None

    websocket = _WebSocket()
    session = WebQwenVoiceSession(
        _config(),
        send_event,
        leader_reply_mode="brief",
    )
    session._websocket = websocket
    await session.enqueue_leader_reply("任务已完成，完整细节保留在网页中。")

    task = asyncio.create_task(session._speak_leader_replies())
    try:
        for _ in range(10):
            if len(websocket.sent) == 2:
                break
            await asyncio.sleep(0)

        assert websocket.sent[0]["item"]["content"][0]["text"] == (
            f"{LEADER_BRIEF_REPORT_PREFIX}任务已完成，完整细节保留在网页中。"
        )
        assert websocket.sent[1] == {
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]},
        }
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_invalid_function_arguments_report_error_without_killing_receiver() -> (
    None
):
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "name": "submit_voice_request",
                "call_id": "call-bad",
                "arguments": "not-json",
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = websocket

    await session._receive_events()

    assert any(name == "voice.error" for name, _ in sent_events)
    assert not any(name == "voice.command" for name, _ in sent_events)
    assert session._idle.is_set()


@pytest.mark.asyncio
async def test_server_closing_the_connection_reports_a_closed_session() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = _WebSocket([])

    await session._receive_events()

    assert sent_events[-1][0] == "voice.error"
    assert sent_events[-1][1]["closed"] is True


@pytest.mark.asyncio
async def test_lost_connection_reports_a_closed_session() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    class _BrokenWebSocket(_WebSocket):
        def __aiter__(self):
            async def _next():
                raise ConnectionError("connection dropped")

            return type(
                "_Iterator",
                (),
                {"__aiter__": lambda self: self, "__anext__": lambda self: _next()},
            )()

    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = _BrokenWebSocket()

    await session._receive_events()

    assert sent_events == [
        ("voice.error", {"message": "connection dropped", "closed": True})
    ]


@pytest.mark.asyncio
async def test_provider_error_event_keeps_the_session_open() -> None:
    sent_events: list[tuple[str, dict[str, Any]]] = []

    async def send_event(name: str, payload: dict[str, Any]) -> None:
        sent_events.append((name, payload))

    session = WebQwenVoiceSession(_config(), send_event)
    session._websocket = _WebSocket(
        [{"type": "error", "error": {"message": "bad request"}}]
    )

    await session._receive_events()

    assert sent_events[0] == ("voice.error", {"message": "bad request"})
