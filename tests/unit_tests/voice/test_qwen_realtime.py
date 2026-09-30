import asyncio
import base64
import json
from typing import Any

import pytest

from jiuwenswarm.voice.qwen_realtime import (
    LEADER_REPORT_PREFIX,
    QwenRealtimeConfig,
    QwenRealtimeVoiceClient,
    build_session_update,
    build_voice_instructions,
)


def _config(**overrides: object) -> QwenRealtimeConfig:
    values: dict[str, object] = {
        "api_key": "secret-value",
        "workspace_id": "workspace-123",
    }
    values.update(overrides)
    return QwenRealtimeConfig(**values)  # type: ignore[arg-type]


def test_session_update_enables_all_voice_control_tools() -> None:
    event = build_session_update(_config())
    session = event["session"]

    assert session["modalities"] == ["text", "audio"]
    assert session["turn_detection"]["type"] == "server_vad"
    tool_names = [tool["function"]["name"] for tool in session["tools"]]
    assert tool_names == [
        "submit_task",
        "supplement_task",
        "pause_task",
        "resume_task",
        "cancel_task",
        "get_task_status",
        "ask_leader",
    ]
    parameters = session["tools"][0]["function"]["parameters"]
    assert "target_member" not in parameters["properties"]
    assert "execution_mode" not in parameters["properties"]
    assert "tasks" not in parameters["properties"]
    tools_by_name = {
        tool["function"]["name"]: tool["function"] for tool in session["tools"]
    }
    assert "target_task_ids" in tools_by_name["supplement_task"]["parameters"][
        "required"
    ]
    assert "target_task_ids" in tools_by_name["cancel_task"]["parameters"][
        "required"
    ]
    assert "target_task_ids" in tools_by_name["get_task_status"]["parameters"][
        "properties"
    ]
    assert "target_task_ids" not in tools_by_name["pause_task"]["parameters"][
        "properties"
    ]
    assert "target_task_ids" not in tools_by_name["resume_task"]["parameters"][
        "properties"
    ]
    assert "Team Leader 汇报" in session["instructions"]
    assert "禁止把当前用户称为“用户”" in session["instructions"]
    assert "以 Team Leader 本人的身份" in session["instructions"]
    assert "你就是 JiuwenSwarm 团队的 Team Leader 本人" in session["instructions"]
    assert "不是把请求转交给另一个 Leader" in session["instructions"]
    assert "不得根据语音上下文猜测任务已完成" in session["instructions"]
    assert "是否需要后端权威数据" in session["instructions"]
    ask_leader_schema = session["tools"][-1]["function"]
    assert "上下文足以回答时不要调用" in ask_leader_schema["description"]
    assert "secret-value" not in str(event)


def test_voice_tool_descriptions_distinguish_external_work_from_team_questions() -> None:
    tools = {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in build_session_update(_config())["session"]["tools"]
    }
    assert "新的执行" in tools["submit_task"]
    assert "外部访问" in tools["submit_task"]
    assert "状态" in tools["ask_leader"]
    assert "只读取后端已有上下文" in tools["ask_leader"]


def test_voice_batch_has_machine_only_boundary() -> None:
    from jiuwenswarm.voice.models import VoiceCommand

    first = VoiceCommand.from_tool_call("submit_task", {"summary": "金价", "instruction": "查询金价"})
    second = VoiceCommand.from_tool_call("submit_task", {"summary": "天气", "instruction": "查询武汉天气"})
    rendered = VoiceCommand.render_voice_batch((first, second), "turn-1")
    assert "[VOICE_DISPATCH_MACHINE_ONLY]" in rendered
    assert "意图1" in rendered and "意图2" in rendered


def test_smart_turn_configuration() -> None:
    event = build_session_update(_config(turn_detection="smart_turn"))
    assert event["session"]["turn_detection"] == {"type": "smart_turn"}


def test_team_directory_contains_member_identity_and_minimal_task_identity() -> None:
    instructions = build_voice_instructions(
        {
            "team_id": "content-forge",
            "members": [
                {
                    "member_id": "content-writer",
                    "name": "撰稿人",
                    "role": "teammate",
                    "status": "busy-secret-status",
                    "execution_status": "running-secret-status",
                }
            ],
            "tasks": [
                {
                    "task_id": "task-a",
                    "title": "创建 A 文件",
                    "status": "in_progress",
                    "created_order": 1,
                    "content": "secret-task-content",
                    "assignee": "secret-assignee",
                }
            ],
        }
    )

    assert "团队：content-forge" in instructions
    assert "成员ID：content-writer；名称：撰稿人；角色：teammate" in instructions
    assert "busy-secret-status" not in instructions
    assert "running-secret-status" not in instructions
    assert "创建顺序：1；任务ID：task-a；标题：创建 A 文件；状态：in_progress" in instructions
    assert "secret-task-content" not in instructions
    assert "secret-assignee" not in instructions


def test_workspace_id_is_used_only_in_websocket_host() -> None:
    config = _config()
    assert config.url.startswith("wss://workspace-123.cn-beijing.maas.aliyuncs.com/")
    assert "secret-value" not in config.url


class _Coordinator:
    def __init__(self) -> None:
        self.acknowledgements: list[str] = []
        self.cancelled = 0
        self.calls: list[dict[str, Any]] = []
        self.user_transcripts: list[str] = []

    def record_user_transcript(self, transcript: str) -> None:
        self.user_transcripts.append(transcript)

    def mark_ack_completed(self, transcript: str) -> None:
        self.acknowledgements.append(transcript)

    def record_function_call(self, event: dict[str, Any]) -> None:
        self.calls.append(event)

    def cancel_pending(self) -> None:
        self.cancelled += 1

    async def flush(self, **_: Any) -> None:
        return None


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


@pytest.mark.asyncio
async def test_first_speech_does_not_claim_it_interrupted_playback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    websocket = _WebSocket(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped", "reason": "turn_invalid"},
        ]
    )
    client = QwenRealtimeVoiceClient(  # type: ignore[arg-type]
        _config(),
        _Coordinator(),
    )

    await client._receive_events(websocket)

    output = capsys.readouterr().out
    assert "[麦克风] 检测到说话\n" in output
    assert "已打断" not in output


@pytest.mark.asyncio
async def test_main_receiver_streams_audio_and_prints_transcription(
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = base64.b64encode(b"pcm").decode("ascii")
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "response.audio.delta", "delta": audio},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "测试显示",
            },
            {"type": "response.done", "response": {"status": "cancelled"}},
        ]
    )
    coordinator = _Coordinator()
    client = QwenRealtimeVoiceClient(_config(), coordinator)  # type: ignore[arg-type]

    await asyncio.wait_for(client._receive_events(websocket), timeout=1)

    # Speech start discards PCM that belonged to the interrupted response.
    assert client._audio_output_queue.empty()
    assert coordinator.cancelled == 1
    assert coordinator.user_transcripts == ["测试显示"]
    output = capsys.readouterr().out
    assert "已打断语音入口播报" in output
    assert "[你] 测试显示" in output


@pytest.mark.asyncio
async def test_leader_reply_is_injected_and_spoken_on_same_websocket() -> None:
    websocket = _WebSocket()
    client = QwenRealtimeVoiceClient(  # type: ignore[arg-type]
        _config(),
        _Coordinator(),
    )
    await client.enqueue_leader_reply("任务完成，测试通过。")
    task = asyncio.create_task(client._speak_leader_replies(websocket))
    try:
        for _ in range(10):
            if len(websocket.sent) == 2:
                break
            await asyncio.sleep(0)

        assert websocket.sent[0] == {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{LEADER_REPORT_PREFIX}任务完成，测试通过。",
                    }
                ],
            },
        }
        assert websocket.sent[1] == {
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]},
        }
        assert client._leader_response_requested is True
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_leader_report_never_dispatches_function_call_again() -> None:
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {
                "type": "response.function_call_arguments.done",
                "call_id": "unexpected",
            },
            {
                "type": "response.audio_transcript.done",
                "transcript": "任务完成。",
            },
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    coordinator = _Coordinator()
    client = QwenRealtimeVoiceClient(_config(), coordinator)  # type: ignore[arg-type]
    client._leader_response_requested = True
    client._realtime_idle.clear()

    await client._receive_events(websocket)

    assert coordinator.calls == []
    assert coordinator.acknowledgements == []
    assert client._realtime_idle.is_set()


@pytest.mark.asyncio
async def test_user_speech_releases_interrupted_leader_queue_on_cancelled_done() -> None:
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
        ]
    )
    coordinator = _Coordinator()
    client = QwenRealtimeVoiceClient(_config(), coordinator)  # type: ignore[arg-type]
    client._leader_response_requested = True
    client._realtime_idle.clear()

    await client._receive_events(websocket)

    assert coordinator.cancelled == 0
    # The cancelled response belonged to the interrupted Leader report. The
    # new user turn is still active, so another queued report must not inject.
    assert not client._realtime_idle.is_set()
    assert client._user_turn_active is True


@pytest.mark.asyncio
async def test_interrupting_leader_report_has_specific_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
        ]
    )
    client = QwenRealtimeVoiceClient(  # type: ignore[arg-type]
        _config(),
        _Coordinator(),
    )
    client._leader_response_requested = True
    client._realtime_idle.clear()

    await client._receive_events(websocket)

    assert "已打断Leader 播报" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_new_user_response_done_releases_queue_after_leader_interruption() -> None:
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
            {"type": "response.created"},
            {"type": "response.done", "response": {"status": "completed"}},
        ]
    )
    coordinator = _Coordinator()
    client = QwenRealtimeVoiceClient(_config(), coordinator)  # type: ignore[arg-type]
    client._leader_response_requested = True
    client._realtime_idle.clear()

    await client._receive_events(websocket)

    assert coordinator.cancelled == 0
    assert client._user_turn_active is False
    assert client._realtime_idle.is_set()


@pytest.mark.asyncio
async def test_interrupting_voice_ack_keeps_new_user_turn_busy() -> None:
    websocket = _WebSocket(
        [
            {"type": "response.created"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "response.done", "response": {"status": "cancelled"}},
        ]
    )
    coordinator = _Coordinator()
    client = QwenRealtimeVoiceClient(_config(), coordinator)  # type: ignore[arg-type]

    await client._receive_events(websocket)

    assert coordinator.cancelled == 1
    assert client._user_turn_active is True
    assert not client._realtime_idle.is_set()
