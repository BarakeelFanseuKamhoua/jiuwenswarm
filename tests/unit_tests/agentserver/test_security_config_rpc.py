"""Unified security RPC reports individual runtime apply outcomes."""

from copy import deepcopy

import pytest

from jiuwenswarm.common.config import dump_yaml_round_trip, load_yaml_round_trip
from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.common.security_config_service import SecurityConfigService
from jiuwenswarm.server.security_config_rpc import apply_trusted_cloud_policy, dispatch_security_config_request


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
