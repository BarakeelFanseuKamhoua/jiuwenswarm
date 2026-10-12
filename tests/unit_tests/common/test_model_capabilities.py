"""Declared model inputs must control native image reads without changing core."""

import base64
import uuid
from types import SimpleNamespace

import pytest

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.runner import Runner
from openjiuwen.harness import create_deep_agent
from openjiuwen.harness.rails import SysOperationRail, SkillUseRail
from openjiuwen.harness.schema.config import VisionModelConfig
from openjiuwen.core.common.exception.codes import StatusCode


def _agent(tmp_path, name="opaque", owner="capability-main", *, vision=True, native=None):
    model = Model(
        ModelClientConfig(client_provider="OpenAI", api_key="offline", api_base="http://example.invalid/v1"),
        ModelRequestConfig(model=name),
    )
    agent = create_deep_agent(
        model=model, card=AgentCard(id=f"{owner}-{uuid.uuid4().hex}", name=owner, description="test"),
        workspace=str(tmp_path), auto_create_workspace=False,
        add_general_purpose_agent=False, enable_task_loop=False,
        enable_task_planning=False, enable_skill_discovery=False,
        vision_model_config=VisionModelConfig(api_key="offline", base_url="http://example.invalid/v1", model="vision") if vision else None,
        enable_read_image_multimodal=native,
    )
    rail = SysOperationRail()
    rail.set_sys_operation(agent.deep_config.sys_operation)
    rail.init(agent)
    skill = SkillUseRail(str(tmp_path / "skills"), include_tools=False)
    skill.set_sys_operation(agent.deep_config.sys_operation)
    skill.init(agent)
    return agent


def _tool(agent, name):
    return Runner.resource_mgr.get_tool(agent.ability_manager.get(name).id)


@pytest.mark.parametrize("raw,want", [
    ([" TEXT ", "image", "Image", "", 42], ("text", "image")),
    (" IMAGE ", ("image",)), (None, None), ([], None), ({"image": True}, None),
])
def test_normalize_input(raw, want):
    from jiuwenswarm.common.model_capabilities import normalize_model_input
    assert normalize_model_input(raw) == want


def test_request_capability_only_applies_to_matching_actual_model(tmp_path):
    from jiuwenswarm.common.model_capabilities import resolve_image_support
    agent = _agent(tmp_path)
    model = agent.deep_config.model
    assert resolve_image_support(model, {"model_name": "opaque", "model_input": ["text", "image"]}) is True
    assert resolve_image_support(model, {"model_name": "opaque", "model_input": ["text"]}) is False
    assert resolve_image_support(model, {"model_name": "other", "model_input": ["image"]}) is None
    assert resolve_image_support(model, {"model_name": "opaque"}) is None


def test_config_capabilities_match_endpoint_and_model(tmp_path):
    from jiuwenswarm.common.model_capabilities import resolve_image_support
    agent = _agent(tmp_path)
    config = {"models": {"defaults": [
        {"input": ["text"], "model_client_config": {"model_name": "opaque", "api_base": "http://other.invalid/v1"}},
        {"input": ["image"], "model_client_config": {"model_name": "opaque", "api_base": "http://example.invalid/v1"}},
    ]}}
    assert resolve_image_support(agent.deep_config.model, {}, config) is True


def test_bound_request_does_not_enable_same_name_at_another_endpoint(tmp_path):
    from jiuwenswarm.common.model_capabilities import resolve_image_support
    agent = _agent(tmp_path)
    request = {
        "model_name": "opaque", "model_input": ["image"],
        "model_endpoint": {"api_base": "http://other.invalid/v1", "client_provider": "OpenAI"},
    }
    assert resolve_image_support(agent.deep_config.model, request) is None


@pytest.mark.asyncio
async def test_live_image_read_switch_and_missing_field_restore_legacy(tmp_path):
    from jiuwenswarm.common.model_capabilities import ImageCapabilityPolicy
    first = _agent(tmp_path)
    second = _agent(tmp_path, owner="capability-other")
    request = {"model_name": "opaque", "model_input": ["text", "image"]}
    policy = ImageCapabilityPolicy(lambda: request)
    read = _tool(first, "read_file")
    skill = _tool(first, "skill_tool")
    raw = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jq/QAAAAASUVORK5CYII=")

    class FS:
        async def read_file(self, _path, mode):
            return SimpleNamespace(code=StatusCode.SUCCESS.code, data=SimpleNamespace(content=raw))

    read.operation = SimpleNamespace(fs=lambda: FS())
    assert policy.apply(first) is True
    assert len((await read._read_image("fixture.png", "opaque"))["multimodal"]) == 1
    assert skill.enable_read_image_multimodal is True
    assert first.ability_manager.get("visual_question_answering") is not None
    assert _tool(second, "read_file").enable_image_multimodal is False
    request["model_input"] = ["text"]
    assert policy.apply(first) is False
    assert (await read._read_image("fixture.png", "opaque"))["multimodal"] == []
    request["model_input"] = ["image"]
    policy.apply(first)
    request.pop("model_input")
    assert policy.apply(first) is None
    assert read.enable_image_multimodal is False
    assert skill.enable_read_image_multimodal is False
    assert first.deep_config.enable_read_image_multimodal is False


def test_explicit_disable_wins_and_policy_does_not_modify_model_parameters(tmp_path):
    from jiuwenswarm.common.model_capabilities import ImageCapabilityPolicy
    agent = _agent(tmp_path)
    before = agent.deep_config.model.model_config.model_dump()
    policy = ImageCapabilityPolicy(
        lambda: {"model_name": "opaque", "model_input": ["image"]},
        lambda: {"react": {"enable_read_image_multimodal": False}},
    )
    assert policy.apply(agent) is False
    assert agent._jiuwenswarm_model_image_support is True
    assert _tool(agent, "read_file").enable_image_multimodal is False
    assert agent.deep_config.model.model_config.model_dump() == before


@pytest.mark.asyncio
async def test_stream_rail_applies_declared_support_before_context_filter(tmp_path):
    from openjiuwen.core.foundation.llm import UserMessage
    from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
    agent = _agent(tmp_path)
    request = {"model_name": "opaque", "model_input": ["image"]}
    rail = JiuSwarmStreamEventRail()
    rail.configure_image_capabilities(lambda: request)
    rail.init(agent)
    parts = [{"type": "text", "text": "image"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    message = UserMessage(content=list(parts))
    class Context:
        def get_messages(self):
            return [message]
    ctx = SimpleNamespace(session=None, inputs=SimpleNamespace(tools=[]), context=Context(), extra={})
    await rail.before_model_call(ctx)
    assert message.content == parts
    assert _tool(agent, "read_file").enable_image_multimodal is True
    request["model_input"] = ["text"]
    await rail.before_model_call(ctx)
    assert message.content == "image"
    assert _tool(agent, "read_file").enable_image_multimodal is False


def test_declared_support_overrides_name_heuristic_and_uses_effective_override(tmp_path):
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter
    from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
    from jiuwenswarm.common.schema.agent import AgentRequest
    adapter = object.__new__(JiuWenSwarmDeepAdapter)
    adapter._instance = _agent(tmp_path, name="base-text")
    adapter._stream_event_rail = JiuSwarmStreamEventRail()
    adapter._stream_event_rail.init(adapter._instance)
    adapter._vision_model_config = object()
    base = adapter._instance.deep_config.model
    request = AgentRequest(request_id="capability-override", params={"model_name": "glm-4", "model_input": ["image"]})
    effective = adapter._apply_model_to_react_agent(base, model_name_override="glm-4", request=request)
    assert adapter._native_image_input_enabled({}, effective) is True
    assert effective.model_config.model_name == "glm-4"
    assert base.model_config.model_name == "base-text"


def test_attachment_rail_uses_current_policy_instead_of_frozen_vision_setting():
    from jiuwenswarm.agents.harness.common.rails.multimodal_image_rail import MultimodalImageRail
    rail = MultimodalImageRail(enable_image_multimodal=False)
    agent = SimpleNamespace(_jiuwenswarm_declared_image_support=True)
    rail.init(agent)
    assert rail._read_image_multimodal_enabled() is True


def test_attachment_rail_reconfigure_preserves_legacy_restore_state(tmp_path):
    from jiuwenswarm.agents.harness.common.rails.multimodal_image_rail import MultimodalImageRail
    agent = _agent(tmp_path)
    rail = MultimodalImageRail()
    rail.configure_image_capabilities(lambda: {"model_name": "opaque", "model_input": ["image"]})
    rail.init(agent)
    assert _tool(agent, "read_file").enable_image_multimodal is True
    rail.configure_image_capabilities(lambda: {"model_name": "opaque"})
    rail._sync_image_capabilities()
    assert agent.deep_config.enable_read_image_multimodal is False
    assert _tool(agent, "read_file").enable_image_multimodal is False


def test_team_metadata_uses_current_params_and_clears_stale_capabilities():
    from jiuwenswarm.common.schema.agent import AgentRequest
    from jiuwenswarm.server.runtime.agent_adapter.team_helpers import build_team_request_metadata
    request = AgentRequest(request_id="team-capability", channel_id="web", params={"model_name": "opaque", "model_input": [" IMAGE "]})
    metadata = build_team_request_metadata(request)
    assert metadata["model_capabilities"] == {"model_name": "opaque", "model_input": ["image"]}
    legacy = AgentRequest(request_id="legacy-team", channel_id="web", params={"model_name": "opaque"}, metadata=metadata)
    assert build_team_request_metadata(legacy)["model_capabilities"] == {"model_name": "opaque"}


@pytest.mark.asyncio
async def test_active_team_is_available_for_capability_updates_without_trace(monkeypatch):
    from jiuwenswarm.agents.harness.team.team_manager import TeamManager
    team = SimpleNamespace(_configurator=SimpleNamespace(ctx=SimpleNamespace(team_spec=object())))

    class Pool:
        async def get(self, team_name):
            assert team_name == "demo-team"
            return SimpleNamespace(agent=team)

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.team_manager._runner_team_runtime_manager",
        lambda _: SimpleNamespace(pool=Pool()),
    )
    manager = TeamManager()
    assert await manager._refresh_active_team_trace("sid-1", "demo-team", {}) is team


@pytest.mark.asyncio
async def test_team_provider_applies_per_model_policy_and_refreshes_reused_members(tmp_path, monkeypatch):
    from jiuwenswarm.agents.harness.team.team_manager import TeamManager
    from jiuwenswarm.agents.swarm.providers.builtin_rails import _build_stream_event_rail
    from jiuwenswarm.agents.swarm.context import SwarmBuildContext
    manager = TeamManager()
    monkeypatch.setattr("jiuwenswarm.agents.harness.team.team_manager.get_team_manager", lambda *_: manager)
    context = SwarmBuildContext(session_id="capability-team", request_metadata={
        "model_capabilities": {"model_name": "opaque", "model_input": ["image"]},
    })
    main = _agent(tmp_path, owner="team-main")
    specialist = _agent(tmp_path, name="specialist", owner="team-specialist")
    main_rail = _build_stream_event_rail({}, context)
    specialist_rail = _build_stream_event_rail({}, context)
    main_rail.init(main)
    specialist_rail.init(specialist)
    assert _tool(main, "read_file").enable_image_multimodal is True
    assert _tool(specialist, "read_file").enable_image_multimodal is False
    # The next round's metadata is authoritative, even when input is omitted.
    manager.update_model_capabilities("capability-team", {"model_capabilities": {"model_name": "opaque"}})
    ctx = SimpleNamespace(session=None, inputs=SimpleNamespace(), context=None, extra={})
    await main_rail.before_model_call(ctx)
    assert _tool(main, "read_file").enable_image_multimodal is False
    assert manager.get_model_capabilities("other-team") is None


@pytest.mark.asyncio
async def test_text_model_clears_previous_attachment_window_mutator(tmp_path):
    from jiuwenswarm.agents.harness.common.prompt.user_prompt_builder import ensure_multimodal_image_window_mutator
    from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
    agent = _agent(tmp_path)
    rail = JiuSwarmStreamEventRail()
    rail.configure_image_capabilities(lambda: {"model_name": "opaque", "model_input": ["text"]})
    rail.init(agent)
    context = SimpleNamespace(_window_mutators=[], get_messages=lambda: [])
    assert ensure_multimodal_image_window_mutator(context, [{"path": "old.png", "mime_type": "image/png"}])
    ctx = SimpleNamespace(session=None, inputs=SimpleNamespace(tools=[]), context=context, extra={})
    await rail.before_model_call(ctx)
    assert context._window_mutators == []


def test_hot_reload_restores_original_flags_on_reused_tools(tmp_path):
    import copy
    from jiuwenswarm.common.model_capabilities import ImageCapabilityPolicy
    agent = _agent(tmp_path)
    request = {"model_name": "opaque", "model_input": ["image"]}
    policy = ImageCapabilityPolicy(lambda: request)
    policy.apply(agent)
    read = _tool(agent, "read_file")
    replacement = copy.copy(agent.deep_config)
    replacement.enable_read_image_multimodal = False
    agent._deep_config = replacement
    policy.apply(agent)
    request.pop("model_input")
    policy.apply(agent)
    assert read.enable_image_multimodal is False
    assert _tool(agent, "skill_tool").enable_read_image_multimodal is False


def test_missing_field_restores_legacy_native_enabled_mode(tmp_path):
    from jiuwenswarm.common.model_capabilities import ImageCapabilityPolicy
    agent = _agent(tmp_path, vision=False, native=True)
    request = {"model_name": "opaque", "model_input": ["text"]}
    policy = ImageCapabilityPolicy(lambda: request)
    policy.apply(agent)
    assert _tool(agent, "read_file").enable_image_multimodal is False
    request.pop("model_input")
    policy.apply(agent)
    assert _tool(agent, "read_file").enable_image_multimodal is True
    assert agent.deep_config.enable_read_image_multimodal is True


def test_subagent_is_seeded_before_probe_and_does_not_rebind_parent_rail(tmp_path):
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import _jws_install_subagent_runtime_patch
    from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
    from jiuwenswarm.agents.harness.common.rails.multimodal_image_rail import MultimodalImageRail
    _jws_install_subagent_runtime_patch()
    source = _agent(tmp_path)
    request = {"model_name": "opaque", "model_input": ["image"]}
    stream = JiuSwarmStreamEventRail()
    stream.configure_image_capabilities(lambda: request)
    image = MultimodalImageRail()
    parent = create_deep_agent(
        model=source.deep_config.model, workspace=str(tmp_path / "parent"),
        auto_create_workspace=False, add_general_purpose_agent=True,
        enable_read_image_multimodal=False, enable_skill_discovery=False,
        rails=[stream, image],
    )
    stream.init(parent)
    image.init(parent)
    child = parent.create_subagent("general-purpose", "child")
    assert child.deep_config.enable_read_image_multimodal is True
    child_stream = child.find_rails_by_type((JiuSwarmStreamEventRail,))[0]
    assert child_stream is not stream
    child_stream.init(child)
    assert stream._deep_agent is parent
    request["model_input"] = ["text"]
    child_stream._sync_image_capabilities()
    assert child.deep_config.enable_read_image_multimodal is False
    assert parent.deep_config.enable_read_image_multimodal is True


def test_mixed_subagent_rails_preserve_owned_binding_and_restore_state(tmp_path):
    from jiuwenswarm.server.runtime.agent_adapter.interface_deep import _jws_install_subagent_runtime_patch
    from jiuwenswarm.agents.harness.common.rails.stream_event_rail import JiuSwarmStreamEventRail
    from jiuwenswarm.agents.harness.common.rails.multimodal_image_rail import MultimodalImageRail
    _jws_install_subagent_runtime_patch()
    parent = _agent(tmp_path / "parent", owner="mixed-parent")
    child = _agent(tmp_path / "child", owner="mixed-child")
    request = {"model_name": "opaque", "model_input": ["image"]}
    owned = MultimodalImageRail()
    owned.configure_image_capabilities(lambda: request)
    child.add_rail(owned)
    owned.init(child)
    original_policy = owned._image_capability_policy
    shared = JiuSwarmStreamEventRail()
    parent.add_rail(shared)
    shared.init(parent)
    child.add_rail(shared)
    parent.deep_config.subagents = [child]
    result = parent.create_subagent("mixed-child", "mixed-subsession")
    assert result is child
    assert owned._deep_agent is child
    assert owned._image_capability_policy is original_policy
    assert shared._deep_agent is parent
    request.pop("model_input")
    owned._sync_image_capabilities()
    assert _tool(child, "read_file").enable_image_multimodal is False
