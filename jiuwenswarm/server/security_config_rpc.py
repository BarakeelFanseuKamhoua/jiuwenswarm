"""AgentServer adapter for the security configuration facade."""

from typing import Any

from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.common.security_config_service import SecurityConfigService


_METHODS = frozenset({
    ReqMethod.SECURITY_CONFIG_GET,
    ReqMethod.SECURITY_CONFIG_INITIALIZE,
    ReqMethod.SECURITY_CONFIG_SETTINGS_UPDATE,
    ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE,
    ReqMethod.SECURITY_CONFIG_MODULE_UPDATE,
    ReqMethod.SECURITY_CONFIG_APPLY,
    ReqMethod.SECURITY_BUILTIN_GET,
})


def is_security_config_method(method: ReqMethod) -> bool:
    return method in _METHODS


def _check_sandbox_coverage(permissions: dict[str, Any], live: dict[str, Any]) -> None:
    """Reject unrepresentable changes before publishing only one enforcement plane."""
    from jiuwenswarm.common.behavior_security import runtime_permission_config
    from jiuwenswarm.common.file_guard_config import project_file_guard_to_sandbox, sandbox_file_buckets
    from jiuwenswarm.common.net_guard_config import project_net_guard_to_sandbox

    previous = live.get("permissions") or {}
    file_changed = previous.get("file_guard") != permissions.get("file_guard")
    net_changed = previous.get("net_guard") != permissions.get("net_guard")
    sandbox = live.get("sandbox") or {}
    if not sandbox.get("enabled") or not (file_changed or net_changed):
        return
    if sandbox.get("type", "jiuwenbox") != "jiuwenbox":
        raise ValueError("UNSUPPORTED_COVERAGE sandbox type")
    runtime = runtime_permission_config({**live, "permissions": permissions})
    if not runtime.get("enabled"):
        raise ValueError("UNSUPPORTED_COVERAGE permissions disabled for sandbox sync")
    if file_changed:
        previous_files, previous_skipped = project_file_guard_to_sandbox(previous.get("file_guard") or {})
        if previous_skipped or (sandbox.get("files") or []) != previous_files:
            raise ValueError("UNSUPPORTED_COVERAGE existing sandbox files are independently managed")
        guard = permissions.get("file_guard") or {}
        if guard.get("enabled") is False:
            raise ValueError("UNSUPPORTED_COVERAGE disabled FileGuard cannot replace sandbox rules")
        try:
            files, skipped = project_file_guard_to_sandbox(guard)
        except NotImplementedError as exc:
            raise ValueError(f"UNSUPPORTED_COVERAGE file: {exc}") from exc
        if skipped:
            raise ValueError(f"UNSUPPORTED_COVERAGE file: {skipped}")
        sandbox_file_buckets(files)
    if net_changed:
        previous_network, previous_skipped = project_net_guard_to_sandbox(previous.get("net_guard") or {})
        previous_urls = {domain: "allow" for domain in previous_network["allow_domains"]}
        previous_urls.update({domain: "deny" for domain in previous_network["deny_domains"]})
        if previous_skipped or (sandbox.get("urls") or {}) != previous_urls:
            raise ValueError("UNSUPPORTED_COVERAGE existing sandbox URLs are independently managed")
        guard = permissions.get("net_guard") or {}
        if guard.get("enabled") is False:
            raise ValueError("UNSUPPORTED_COVERAGE disabled NetGuard cannot replace sandbox rules")
        if any("*" in pattern or "?" in pattern for pattern in (guard.get("urls") or {})):
            raise ValueError("UNSUPPORTED_COVERAGE wildcard domains differ between Rail and sandbox")
        _, skipped = project_net_guard_to_sandbox(guard)
        if skipped:
            raise ValueError(f"UNSUPPORTED_COVERAGE network: {skipped}")


async def dispatch_security_config_request(
    method: ReqMethod, params: dict[str, Any], *, agent_manager: Any,
    restart_sandboxes: Any, service: SecurityConfigService | None = None,
) -> dict[str, Any]:
    """Save and apply a revision; never report success for an unacknowledged target."""
    service = service or SecurityConfigService()
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if method == ReqMethod.SECURITY_CONFIG_GET:
        if params:
            raise ValueError("security.config.get accepts no parameters")
        return service.get_candidate()
    if method == ReqMethod.SECURITY_BUILTIN_GET:
        if set(params) != {"module"}:
            raise ValueError("security.builtin.get requires module only")
        return service.get_builtin(params["module"])

    expected = params.get("expected_revision")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("expected_revision must be a string")
    if method == ReqMethod.SECURITY_CONFIG_INITIALIZE:
        if params:
            raise ValueError("security.config.initialize accepts no parameters")
        state = service.initialize_from_live()
    elif method == ReqMethod.SECURITY_CONFIG_SETTINGS_UPDATE:
        state = service.update_settings(params.get("patch"), expected_revision=expected)
    elif method == ReqMethod.SECURITY_CONFIG_LOCAL_UPDATE:
        state = service.update_local(params.get("module"), params.get("value"), expected_revision=expected)
    elif method == ReqMethod.SECURITY_CONFIG_MODULE_UPDATE:
        state = service.update_module_settings(params.get("module"), params.get("patch"), expected_revision=expected)
    elif method == ReqMethod.SECURITY_CONFIG_APPLY:
        state = service.get_candidate()["apply"]
        if expected is not None and state["revision"] != expected:
            raise ValueError("VERSION_CONFLICT")
    else:
        raise ValueError(f"unsupported security config method: {method}")

    revision = state["revision"]
    targets = {"rail": "pending", "host_exit": "pending", "sandbox": "pending"}
    try:
        published = service.publish_candidate(
            expected_revision=revision, validator=_check_sandbox_coverage,
        )
    except Exception as exc:
        if str(exc) == "VERSION_CONFLICT":
            raise
        targets["rail"] = f"failed: publication: {exc}"
        targets["host_exit"] = "not_published"
        targets["sandbox"] = f"not_published: {exc}"
        result = service.record_apply(revision, targets)
        return {"saved": True, "applied": False, "apply": result}
    config = published["config"]
    previous = published["previous"]
    before = previous.get("permissions") or {}
    after = config.get("permissions") or {}
    sandbox = config.get("sandbox") or {}
    file_changed = before.get("file_guard") != after.get("file_guard")
    net_changed = before.get("net_guard") != after.get("net_guard")

    from jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc import (
        publish_host_exit_policy_from_config,
    )

    try:
        publish_host_exit_policy_from_config(config)
        targets["host_exit"] = "applied"
    except Exception as exc:
        targets["host_exit"] = f"failed: {exc}"
    try:
        await agent_manager.reload_agents_config(config, None)
        targets["rail"] = "applied"
    except Exception as exc:
        targets["rail"] = f"failed: {exc}"

    if not sandbox.get("enabled") or not (file_changed or net_changed):
        targets["sandbox"] = "not_required"
    elif sandbox.get("type", "jiuwenbox") != "jiuwenbox":
        targets["sandbox"] = "unsupported: sandbox type"
    else:
        try:
            if file_changed:
                from jiuwenswarm.common.file_guard_config import sync_file_guard_to_sandbox

                file_result = sync_file_guard_to_sandbox()
                if file_result["skipped"]:
                    raise ValueError(f"UNSUPPORTED_COVERAGE file: {file_result['skipped']}")
            if net_changed:
                from jiuwenswarm.common.net_guard_config import sync_net_guard_to_sandbox

                net_result = sync_net_guard_to_sandbox()
                if net_result["skipped"]:
                    raise ValueError(f"UNSUPPORTED_COVERAGE network: {net_result['skipped']}")
            restarted = await restart_sandboxes({})
            if restarted.get("network_status") == "externally_managed" and net_changed:
                targets["sandbox"] = "external_pending"
            else:
                targets["sandbox"] = "applied"
        except Exception as exc:
            targets["sandbox"] = f"failed: {exc}"

    result = service.record_apply(revision, targets)
    return {"saved": True, "applied": result["applied"], "apply": result}


async def apply_trusted_cloud_policy(
    policy_version: int, permissions: dict[str, Any], *, agent_manager: Any,
    restart_sandboxes: Any, service: SecurityConfigService | None = None,
) -> dict[str, Any]:
    """Trusted cloud ingress only; intentionally absent from public ReqMethod."""
    service = service or SecurityConfigService()
    state = service.replace_cloud_policy(policy_version, permissions)
    if state["applied"]:
        return {"saved": True, "applied": True, "apply": state}
    return await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_APPLY,
        {"expected_revision": state["revision"]},
        agent_manager=agent_manager,
        restart_sandboxes=restart_sandboxes,
        service=service,
    )


async def resume_managed_security_on_start(
    *, agent_manager: Any, restart_sandboxes: Any,
    service: SecurityConfigService | None = None,
) -> dict[str, Any] | None:
    """Restore the latest source revision before AgentServer accepts requests."""
    service = service or SecurityConfigService()
    if not service.is_initialized():
        return None
    from openjiuwen.harness.security.permission_engine import core as permission_core

    if getattr(permission_core, "COMPOSITION_SCHEMA_VERSION", None) != 1:
        raise RuntimeError("installed core does not support security composition v1")
    state = service.get_candidate()["apply"]
    if state["applied"]:
        return state
    result = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_APPLY,
        {"expected_revision": state["revision"]},
        agent_manager=agent_manager,
        restart_sandboxes=restart_sandboxes,
        service=service,
    )
    if not result["applied"]:
        raise RuntimeError(f"managed security configuration is not ready: {result['apply']['targets']}")
    return result["apply"]
