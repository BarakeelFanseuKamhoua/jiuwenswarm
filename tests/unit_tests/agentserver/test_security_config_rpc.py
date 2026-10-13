"""Unified security RPC reports individual runtime apply outcomes."""

from copy import deepcopy

import pytest

from jiuwenswarm.common.config import dump_yaml_round_trip, load_yaml_round_trip
from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.common.security_config_service import SecurityConfigService
from jiuwenswarm.server.security_config_rpc import (
    apply_trusted_cloud_policy,
    dispatch_security_config_request,
    resume_managed_security_on_start,
)


class Manager:
    def __init__(self):
        self.configs = []

    async def reload_agents_config(self, config, env):
        self.configs.append(deepcopy(config))


@pytest.fixture
def service(tmp_path):
    path = tmp_path / "config.yaml"
    dump_yaml_round_trip(path, {
        "model": "untouched", "permissions": {"enabled": True, "tools": {"bash": "allow"}},
        "sandbox": {"type": "jiuwenbox", "enabled": False},
    })
    return SecurityConfigService(path, builtin_reader=lambda: {"permissions": {}})


@pytest.mark.asyncio
async def test_initialize_and_update_apply_without_sandbox_restart(service, monkeypatch):
    manager = Manager()
    published = []
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda config: published.append(deepcopy(config)),
    )

    async def unexpected_restart(params):
        pytest.fail("tool-only update must not restart sandbox")

    first = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=unexpected_restart, service=service,
    )
    assert first["applied"] and first["apply"]["targets"]["sandbox"] == "not_required"
    second = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
        {"module": "tool", "value": {"bash": "deny"}, "expected_revision": first["apply"]["revision"]},
        agent_manager=manager, restart_sandboxes=unexpected_restart, service=service,
    )
    assert second["applied"]
    assert manager.configs[-1]["permissions"]["tools"]["bash"] == "deny"
    assert published[-1]["permissions"]["composition"]["revision"] == second["apply"]["revision"]
    assert service.get_candidate()["apply"]["status"] == "applied"


@pytest.mark.asyncio
async def test_builtin_get_does_not_publish_or_restart(service):
    service.initialize_from_live()

    async def unexpected_restart(_params):
        pytest.fail("read-only builtin query must not restart")

    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_BUILTIN_GET, {"module": "tool"},
        agent_manager=Manager(), restart_sandboxes=unexpected_restart, service=service,
    )
    assert result == {"module": "tool", "enabled": True, "value": {}}
    with pytest.raises(ValueError, match="requires module only"):
        await dispatch_security_config_request(
            ReqMethod.SECURITY_BUILTIN_GET, {"module": "tool", "other": 1},
            agent_manager=Manager(), restart_sandboxes=unexpected_restart, service=service,
        )


@pytest.mark.asyncio
async def test_startup_replays_pending_sources_before_ready(service, monkeypatch):
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def no_restart(_params):
        pytest.fail("sandbox is disabled")

    assert await resume_managed_security_on_start(
        agent_manager=Manager(), restart_sandboxes=no_restart, service=service,
    ) is None
    service.initialize_from_live()
    manager = Manager()
    state = await resume_managed_security_on_start(
        agent_manager=manager, restart_sandboxes=no_restart, service=service,
    )
    assert state["applied"] and manager.configs
    already_applied = Manager()
    await resume_managed_security_on_start(
        agent_manager=already_applied, restart_sandboxes=no_restart, service=service,
    )
    assert not already_applied.configs
    service.update_local("tool", {"bash": "deny"})
    resumed = Manager()
    state = await resume_managed_security_on_start(
        agent_manager=resumed, restart_sandboxes=no_restart, service=service,
    )
    assert state["applied"]
    assert resumed.configs[-1]["permissions"]["tools"]["bash"] == "deny"


@pytest.mark.asyncio
async def test_startup_does_not_announce_ready_after_apply_failure(service, monkeypatch):
    service.initialize_from_live()

    def fail(_config):
        raise RuntimeError("host exit unavailable")

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        fail,
    )

    async def no_restart(_params):
        pytest.fail("sandbox is disabled")

    with pytest.raises(RuntimeError, match="not ready"):
        await resume_managed_security_on_start(
            agent_manager=Manager(), restart_sandboxes=no_restart, service=service,
        )
    assert service.get_candidate()["apply"]["status"] == "failed"


@pytest.mark.asyncio
async def test_startup_rejects_old_core_even_for_applied_state(service, monkeypatch):
    from openjiuwen.harness.security.permission_engine import core

    service.initialize_from_live()
    revision = service.publish_candidate()["apply"]["revision"]
    service.record_apply(revision, {"rail": "applied", "host_exit": "applied", "sandbox": "not_required"})
    monkeypatch.setattr(core, "COMPOSITION_SCHEMA_VERSION", 0)
    with pytest.raises(RuntimeError, match="does not support"):
        await resume_managed_security_on_start(
            agent_manager=Manager(), restart_sandboxes=None, service=service,
        )


@pytest.mark.asyncio
async def test_host_exit_failure_stays_saved_but_not_applied(service, monkeypatch):
    def fail(_config):
        raise RuntimeError("host exit unavailable")

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        fail,
    )

    async def unexpected_restart(_params):
        pytest.fail("sandbox disabled")

    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=Manager(),
        restart_sandboxes=unexpected_restart, service=service,
    )
    assert result["saved"] and not result["applied"]
    assert "host exit unavailable" in result["apply"]["targets"]["host_exit"]
    assert result["apply"]["status"] == "failed"


@pytest.mark.asyncio
async def test_old_core_reports_saved_but_not_published(service, monkeypatch):
    from openjiuwen.harness.security.permission_engine import core

    monkeypatch.setattr(core, "COMPOSITION_SCHEMA_VERSION", 0)

    async def unexpected_restart(_params):
        pytest.fail("incompatible core must not apply")

    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=Manager(),
        restart_sandboxes=unexpected_restart, service=service,
    )
    assert result["saved"] and not result["applied"]
    assert "does not support" in result["apply"]["targets"]["rail"]
    assert "composition" not in load_yaml_round_trip(service.config_path)["permissions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", ["https://example.org/path", "*.example.org"])
async def test_unrepresentable_sandbox_rule_is_not_published(service, monkeypatch, pattern):
    manager = Manager()
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def unexpected_restart(_params):
        pytest.fail("unrepresentable policy must not restart the sandbox")

    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=unexpected_restart, service=service,
    )
    live = load_yaml_round_trip(service.config_path)
    live["sandbox"]["enabled"] = True
    dump_yaml_round_trip(service.config_path, live)
    before = service.config_path.read_bytes()
    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
        {"module": "network", "value": {pattern: "deny"}},
        agent_manager=manager, restart_sandboxes=unexpected_restart, service=service,
    )
    assert result["saved"] and not result["applied"]
    assert "UNSUPPORTED_COVERAGE" in result["apply"]["targets"]["sandbox"]
    assert service.config_path.read_bytes() == before
    assert len(manager.configs) == 1


@pytest.mark.asyncio
async def test_external_sandbox_network_is_pending(service, monkeypatch):
    manager = Manager()
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def restart(_params):
        return {"network_status": "externally_managed"}

    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=restart, service=service,
    )
    live = load_yaml_round_trip(service.config_path)
    live["sandbox"]["enabled"] = True
    dump_yaml_round_trip(service.config_path, live)
    monkeypatch.setattr(
        "jiuwenswarm.common.net_guard_config.sync_net_guard_to_sandbox",
        lambda: {"skipped": [], "urls": {"example.org": "deny"}},
    )
    monkeypatch.setattr(
        "jiuwenswarm.common.file_guard_config.sync_file_guard_to_sandbox",
        lambda: {"skipped": [], "files": []},
    )
    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
        {"module": "network", "value": {"example.org": "deny"}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    assert result["saved"] and not result["applied"]
    assert result["apply"]["targets"]["sandbox"] == "external_pending"


@pytest.mark.asyncio
async def test_trusted_cloud_policy_uses_same_apply_path(service, monkeypatch):
    manager = Manager()
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def unexpected_restart(_params):
        pytest.fail("sandbox disabled")

    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=unexpected_restart, service=service,
    )
    result = await apply_trusted_cloud_policy(
        1, {"tools": {"bash": "deny"}}, agent_manager=manager,
        restart_sandboxes=unexpected_restart, service=service,
    )
    assert result["applied"]
    assert service.get_candidate()["cloud"]["policy_version"] == 1
    assert manager.configs[-1]["permissions"]["tools"]["bash"] == "deny"


@pytest.mark.asyncio
async def test_trusted_cloud_policy_duplicate_skips_apply_but_failed_apply_retries(service, monkeypatch):
    manager = Manager()
    attempts = []

    def publish(_config):
        attempts.append("host_exit")
        if len(attempts) == 2:
            raise RuntimeError("temporary failure")

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        publish,
    )

    async def no_restart(_params):
        pytest.fail("sandbox is disabled")

    initialized = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    assert initialized["applied"]
    failed = await apply_trusted_cloud_policy(
        1, {"tools": {"bash": "deny"}}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    assert not failed["applied"]
    assert len(manager.configs) == 2
    retried = await apply_trusted_cloud_policy(
        1, {"tools": {"bash": "deny"}}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    assert retried["applied"]
    assert retried["apply"]["revision"] == failed["apply"]["revision"]
    assert len(manager.configs) == 3
    repeated = await apply_trusted_cloud_policy(
        1, {"tools": {"bash": "deny"}}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    assert repeated == retried
    assert len(manager.configs) == 3


@pytest.mark.asyncio
async def test_disabled_cloud_source_saves_policy_without_enforcing_until_enabled(service, monkeypatch):
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def no_restart(_params):
        pytest.fail("sandbox is disabled")

    manager = Manager()
    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    disabled = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_SETTINGS_UPDATE,
        {"patch": {"policy_sources": [
            {"id": "user", "enabled": True},
            {"id": "cloud", "enabled": False},
            {"id": "builtin", "enabled": True},
        ]}},
        agent_manager=manager, restart_sandboxes=no_restart, service=service,
    )
    assert disabled["applied"]
    received = await apply_trusted_cloud_policy(
        1, {"tools": {"cloud_only": "deny"}}, agent_manager=manager,
        restart_sandboxes=no_restart, service=service,
    )
    assert received["applied"]
    assert service.cloud_path.is_file()
    assert "cloud_only" not in manager.configs[-1]["permissions"]["tools"]
    enabled = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_SETTINGS_UPDATE,
        {"patch": {"policy_sources": [
            {"id": "user", "enabled": True},
            {"id": "cloud", "enabled": True},
            {"id": "builtin", "enabled": True},
        ]}},
        agent_manager=manager, restart_sandboxes=no_restart, service=service,
    )
    assert enabled["applied"]
    assert manager.configs[-1]["permissions"]["tools"]["cloud_only"] == "deny"


@pytest.mark.asyncio
async def test_network_source_uses_native_sandbox_sync(service, monkeypatch):
    from jiuwenswarm.common import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_YAML_PATH", service.config_path)
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )
    manager = Manager()
    restarts = []

    async def restart(params):
        restarts.append(params)
        return {"status": "applied"}

    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=restart, service=service,
    )
    live = load_yaml_round_trip(service.config_path)
    live["sandbox"]["enabled"] = True
    dump_yaml_round_trip(service.config_path, live)
    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_MODULE_UPDATE,
        {"module": "network", "patch": {"enabled": True}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
        {"module": "network", "value": {"example.org": "deny"}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    assert result["applied"]
    live = load_yaml_round_trip(service.config_path)
    assert live["permissions"]["net_guard"]["urls"] == {"example.org": "deny"}
    assert live["sandbox"]["urls"] == {"example.org": "deny"}
    assert restarts


@pytest.mark.asyncio
async def test_cloud_network_source_switch_updates_sandbox_projection(service, monkeypatch):
    from jiuwenswarm.common import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_YAML_PATH", service.config_path)
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )
    restarts = []

    async def restart(params):
        restarts.append(params)
        return {"status": "applied"}

    manager = Manager()
    initialized = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=manager,
        restart_sandboxes=restart, service=service,
    )
    assert initialized["applied"]
    live = load_yaml_round_trip(service.config_path)
    live["sandbox"]["enabled"] = True
    dump_yaml_round_trip(service.config_path, live)
    enabled = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_MODULE_UPDATE,
        {"module": "network", "patch": {"enabled": True}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    assert enabled["applied"]
    restarts_before_cloud = len(restarts)
    cloud = await apply_trusted_cloud_policy(
        1, {"net_guard": {"urls": {"blocked.example.org": "deny"}}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    assert cloud["applied"]
    assert load_yaml_round_trip(service.config_path)["sandbox"]["urls"] == {
        "blocked.example.org": "deny",
    }
    disabled = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_SETTINGS_UPDATE,
        {"patch": {"policy_sources": [
            {"id": "user", "enabled": True},
            {"id": "cloud", "enabled": False},
            {"id": "builtin", "enabled": True},
        ]}},
        agent_manager=manager, restart_sandboxes=restart, service=service,
    )
    assert disabled["applied"]
    assert load_yaml_round_trip(service.config_path)["sandbox"]["urls"] == {}
    assert len(restarts) == restarts_before_cloud + 2


@pytest.mark.asyncio
async def test_independent_sandbox_urls_are_not_overwritten(service, monkeypatch):
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc.publish_host_exit_policy_from_config",
        lambda _config: None,
    )

    async def unexpected_restart(_params):
        pytest.fail("independently managed sandbox URLs must not be replaced")

    await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=Manager(),
        restart_sandboxes=unexpected_restart, service=service,
    )
    live = load_yaml_round_trip(service.config_path)
    live["sandbox"].update(enabled=True, urls={"manual.example": "deny"})
    dump_yaml_round_trip(service.config_path, live)
    before = service.config_path.read_bytes()
    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
        {"module": "network", "value": {"example.org": "deny"}},
        agent_manager=Manager(), restart_sandboxes=unexpected_restart, service=service,
    )
    assert not result["applied"]
    assert "independently managed" in result["apply"]["targets"]["sandbox"]
    assert service.config_path.read_bytes() == before


@pytest.mark.asyncio
async def test_cloud_domain_deny_reaches_real_core_host_exit(service):
    from openjiuwen.harness.security.outbound import policy as host_exit

    async def no_restart(_params):
        pytest.fail("sandbox is disabled")

    host_exit.reset_host_exit_policy()
    try:
        initialized = await dispatch_security_config_request(
            ReqMethod.SECURITY_CONFIG_INITIALIZE, {}, agent_manager=Manager(),
            restart_sandboxes=no_restart, service=service,
        )
        assert initialized["applied"]
        enabled = await dispatch_security_config_request(
            ReqMethod.SECURITY_CONFIG_MODULE_UPDATE,
            {"module": "network", "patch": {"enabled": True}},
            agent_manager=Manager(), restart_sandboxes=no_restart, service=service,
        )
        assert enabled["applied"]
        cloud = await apply_trusted_cloud_policy(
            1, {"net_guard": {"urls": {"blocked.example.org": "deny"}}},
            agent_manager=Manager(), restart_sandboxes=no_restart, service=service,
        )
        assert cloud["applied"]
        with pytest.raises(host_exit.OutboundBlockedError):
            host_exit.check_outbound_url("https://blocked.example.org/resource")
    finally:
        host_exit.reset_host_exit_policy()
