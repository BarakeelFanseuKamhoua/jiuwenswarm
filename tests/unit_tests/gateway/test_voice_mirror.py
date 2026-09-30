# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import time

import pytest

from jiuwenswarm.common.schema.message import EventType, Message
from jiuwenswarm.gateway.channel_manager.tui.tui_channel import TuiChannel
from jiuwenswarm.gateway.voice_mirror import VoiceMirrorRegistry


class FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False


class FakeWebChannel:
    def __init__(self) -> None:
        self.events: list[tuple[FakeWebSocket, str, dict]] = []

    @staticmethod
    def serialize_event_frame(msg, _routing_target=None):
        return {
            "type": "event",
            "event": msg.event_type.value,
            "payload": {**msg.payload, "session_id": msg.session_id},
        }

    async def send_event(self, ws, event, payload):
        self.events.append((ws, event, payload))


def make_team_task(*, user_id: str | None = None) -> Message:
    return Message(
        id="evt-1",
        type="event",
        channel_id="tui",
        session_id="voice-leader-test",
        params={},
        timestamp=time.time(),
        ok=True,
        user_id=user_id,
        payload={"task_id": "write-test", "status": "in_progress"},
        event_type=EventType.TEAM_TASK,
        agent_ref={"mode": "team", "id": "writer"},
    )


@pytest.mark.asyncio
async def test_mirror_retargets_original_team_event_without_executing_request():
    web_channel = FakeWebChannel()
    registry = VoiceMirrorRegistry(web_channel)
    ws = FakeWebSocket()
    await registry.bind(
        ws,
        voice_session_id="voice-leader-test",
        web_session_id="web-observer",
    )

    delivered = await registry.mirror(make_team_task())

    assert delivered == 1
    assert len(web_channel.events) == 1
    target_ws, event, payload = web_channel.events[0]
    assert target_ws is ws
    assert event == "team.task"
    assert payload["session_id"] == "web-observer"
    assert payload["task_id"] == "write-test"
    assert payload["voice_mirror"] == {
        "read_only": True,
        "origin_channel": "tui",
        "origin_session_id": "voice-leader-test",
    }


@pytest.mark.asyncio
async def test_rebinding_one_websocket_replaces_previous_source():
    web_channel = FakeWebChannel()
    registry = VoiceMirrorRegistry(web_channel)
    ws = FakeWebSocket()
    await registry.bind(ws, voice_session_id="old", web_session_id="web-observer")
    await registry.bind(
        ws,
        voice_session_id="voice-leader-test",
        web_session_id="web-observer",
    )

    old_event = make_team_task()
    old_event.session_id = "old"
    assert await registry.mirror(old_event) == 0
    assert await registry.mirror(make_team_task()) == 1


@pytest.mark.asyncio
async def test_unbind_and_disconnect_cleanup_stop_delivery():
    web_channel = FakeWebChannel()
    registry = VoiceMirrorRegistry(web_channel)
    ws = FakeWebSocket()
    await registry.bind(
        ws,
        voice_session_id="voice-leader-test",
        web_session_id="web-observer",
    )
    await registry.cleanup_ws(ws, {"web-observer"})

    assert await registry.mirror(make_team_task()) == 0
    assert web_channel.events == []


@pytest.mark.asyncio
async def test_known_different_users_cannot_receive_mirrored_event():
    web_channel = FakeWebChannel()
    registry = VoiceMirrorRegistry(web_channel)
    ws = FakeWebSocket()
    await registry.bind(
        ws,
        voice_session_id="voice-leader-test",
        web_session_id="web-observer",
        user_id="web-user",
    )

    assert await registry.mirror(make_team_task(user_id="voice-user")) == 0
    assert web_channel.events == []


@pytest.mark.asyncio
async def test_response_frames_are_never_mirrored():
    web_channel = FakeWebChannel()
    registry = VoiceMirrorRegistry(web_channel)
    ws = FakeWebSocket()
    await registry.bind(
        ws,
        voice_session_id="voice-leader-test",
        web_session_id="web-observer",
    )
    response = make_team_task()
    response.type = "res"

    assert await registry.mirror(response) == 0
    assert web_channel.events == []


@pytest.mark.asyncio
async def test_tui_channel_mirrors_even_without_a_connected_terminal_client():
    class MirrorProbe:
        def __init__(self) -> None:
            self.messages: list[Message] = []

        async def mirror(self, msg, _routing_target=None):
            self.messages.append(msg)

    channel = TuiChannel()
    probe = MirrorProbe()
    channel.set_voice_mirror_registry(probe)
    message = make_team_task()

    await channel.send(message)

    assert probe.messages == [message]
