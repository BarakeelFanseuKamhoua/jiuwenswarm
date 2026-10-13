"""Keep existing permissions RPC shapes while a managed instance uses source files."""

import uuid
from copy import deepcopy
from typing import Any

from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.common.security_config_service import SecurityConfigService
from jiuwenswarm.server.security_config_rpc import dispatch_security_config_request


_READS = frozenset({
    ReqMethod.PERMISSIONS_FILE_GUARD_GET,
    ReqMethod.PERMISSIONS_SHELL_GUARD_GET,
    ReqMethod.PERMISSIONS_NET_GUARD_GET,
    ReqMethod.PERMISSIONS_TOOLS_GET,
    ReqMethod.PERMISSIONS_RULES_GET,
    ReqMethod.PERMISSIONS_APPROVAL_OVERRIDES_GET,
})
_WRITES = frozenset({
    ReqMethod.PERMISSIONS_FILE_GUARD_UPDATE,
    ReqMethod.PERMISSIONS_SHELL_GUARD_UPDATE,
    ReqMethod.PERMISSIONS_NET_GUARD_SET,
    ReqMethod.PERMISSIONS_TOOLS_SET,
    ReqMethod.PERMISSIONS_TOOLS_UPDATE,
    ReqMethod.PERMISSIONS_TOOLS_DELETE,
    ReqMethod.PERMISSIONS_RULES_CREATE,
    ReqMethod.PERMISSIONS_RULES_UPDATE,
    ReqMethod.PERMISSIONS_RULES_DELETE,
    ReqMethod.PERMISSIONS_APPROVAL_OVERRIDES_DELETE,
})


def is_managed_permissions_method(method: ReqMethod) -> bool:
    return method in _READS | _WRITES


def _saved_sections(service: SecurityConfigService) -> tuple[dict[str, Any], dict[str, Any]]:
    from jiuwenswarm.common.security_config_compose import validate_settings, validate_source

    settings = validate_settings(service._read_yaml(service.settings_path))
    local = validate_source(service._read_yaml(service.local_path))
    return settings.get("permissions") or {}, local.get("permissions") or {}


def _read(method: ReqMethod, service: SecurityConfigService) -> dict[str, Any]:
    settings, local = _saved_sections(service)
    if method == ReqMethod.PERMISSIONS_FILE_GUARD_GET:
        guard = deepcopy(settings.get("file_guard") or {})
        guard["paths"] = deepcopy((local.get("file_guard") or {}).get("paths") or [])
        return {"file_guard": guard}
    if method == ReqMethod.PERMISSIONS_SHELL_GUARD_GET:
        from openjiuwen.harness.security.permission_engine.toolguard.builtin_rules import load_package_command_rules
        from openjiuwen.harness.security.permission_engine.toolguard.tool_categories import shell_tools_from_config

        names = shell_tools_from_config(settings) | {"shell"}
        rules = [deepcopy(rule) for rule in local.get("rules") or []
                 if rule.get("tools") and all(tool in names for tool in rule["tools"])]
        return {
            "shell_guard": {"builtin_rules_enabled": (settings.get("shell_guard") or {}).get("builtin_rules_enabled", True)},
            "rules": rules,
            "builtin_rules": deepcopy(load_package_command_rules()),
        }
    if method == ReqMethod.PERMISSIONS_NET_GUARD_GET:
        from jiuwenswarm.agents.harness.common.rails.permissions.permissions_config_rpc import _net_guard_payload

        result = _net_guard_payload()
        result["net_guard"] = deepcopy(settings.get("net_guard") or {})
        result["net_guard"]["urls"] = deepcopy((local.get("net_guard") or {}).get("urls") or {})
        return result
    if method == ReqMethod.PERMISSIONS_TOOLS_GET:
        return {"tools": deepcopy(local.get("tools") or {})}
    if method == ReqMethod.PERMISSIONS_RULES_GET:
        return {"rules": deepcopy(local.get("rules") or [])}
    if method == ReqMethod.PERMISSIONS_APPROVAL_OVERRIDES_GET:
        return {"approval_overrides": deepcopy(local.get("approval_overrides") or [])}
    raise ValueError(f"unsupported permissions read method: {method}")


def _edit(method: ReqMethod, params: dict[str, Any], settings: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    from jiuwenswarm.common.config import (
        _RULE_MUTABLE_KEYS,
        _normalize_rule_severity_action,
        _normalize_rule_tools,
        _validate_rule_regex,
        _validate_tools_map,
    )
    from jiuwenswarm.common.file_guard_config import patch_file_guard
    from jiuwenswarm.common.shell_guard_config import patch_shell_guard
    from jiuwenswarm.agents.harness.common.rails.permissions.permissions_persist import normalize_net_guard_patch

    saved = local.setdefault("permissions", {})
    if method == ReqMethod.PERMISSIONS_FILE_GUARD_UPDATE:
        patch = params.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("patch must be an object")
        if "paths" in patch:
            patch_file_guard(local, {"paths": patch["paths"]})
        rest = {key: value for key, value in patch.items() if key != "paths"}
        patch_file_guard(settings, rest)
        guard = deepcopy(settings.get("permissions", {}).get("file_guard") or {})
        guard["paths"] = deepcopy((saved.get("file_guard") or {}).get("paths") or [])
        return {"file_guard": guard}
    if method == ReqMethod.PERMISSIONS_SHELL_GUARD_UPDATE:
        patch_shell_guard(settings, params.get("patch"))
        return {"shell_guard": deepcopy(settings["permissions"]["shell_guard"])}
    if method == ReqMethod.PERMISSIONS_NET_GUARD_SET:
        patch = params.get("net_guard") if isinstance(params.get("net_guard"), dict) else params
        normalized = normalize_net_guard_patch(patch)
        if "urls" in normalized:
            saved.setdefault("net_guard", {})["urls"] = normalized.pop("urls")
        settings.setdefault("permissions", {}).setdefault("net_guard", {}).update(normalized)
        guard = deepcopy(settings["permissions"]["net_guard"])
        guard["urls"] = deepcopy((saved.get("net_guard") or {}).get("urls") or {})
        return {"net_guard": guard}
    if method == ReqMethod.PERMISSIONS_TOOLS_SET:
        saved["tools"] = _validate_tools_map(params.get("tools"))
        return {"ok": True}
    if method in {ReqMethod.PERMISSIONS_TOOLS_UPDATE, ReqMethod.PERMISSIONS_TOOLS_DELETE}:
        name = str(params.get("tool") or params.get("name") or "").strip()
        if not name:
            raise ValueError("tool is required")
        tools = saved.setdefault("tools", {})
        if method == ReqMethod.PERMISSIONS_TOOLS_DELETE:
            if name not in tools:
                raise LookupError("tool not found in permissions.tools")
            del tools[name]
        else:
            if "level" not in params:
                raise ValueError("level is required")
            tools[name] = _validate_tools_map({name: params["level"]})[name]
        return {"tools": deepcopy(tools)}
    if method in {ReqMethod.PERMISSIONS_RULES_CREATE, ReqMethod.PERMISSIONS_RULES_UPDATE, ReqMethod.PERMISSIONS_RULES_DELETE}:
        rules = saved.setdefault("rules", [])
        if method == ReqMethod.PERMISSIONS_RULES_CREATE:
            rule = params.get("rule")
            if not isinstance(rule, dict) or "layer" in rule:
                raise ValueError("rule must be an object without layer")
            rid = str(rule.get("id") or "").strip() or f"ui_rule_{uuid.uuid4().hex[:12]}"
            if any(item.get("id") == rid for item in rules):
                raise ValueError(f"rule id already exists: {rid}")
            stored = {"id": rid, **{key: deepcopy(value) for key, value in rule.items()
                                    if key in _RULE_MUTABLE_KEYS and value is not None}}
            if "tools" not in stored or "pattern" not in stored:
                raise ValueError("tools and pattern are required")
            stored["tools"] = _normalize_rule_tools(stored["tools"])
            stored["pattern"] = str(stored["pattern"]).strip()
            _normalize_rule_severity_action(stored)
            _validate_rule_regex(stored)
            rules.append(stored)
            return {"rule": deepcopy(stored)}
        rid = str(params.get("id") or "").strip()
        if not rid:
            raise ValueError("id is required")
        index = next((index for index, item in enumerate(rules) if item.get("id") == rid), None)
        if index is None:
            raise LookupError(f"rule not found: {rid}")
        if method == ReqMethod.PERMISSIONS_RULES_DELETE:
            del rules[index]
            return {"ok": True}
        patch = params.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("patch must be an object")
        stored = deepcopy(rules[index])
        for key, value in patch.items():
            if key in _RULE_MUTABLE_KEYS:
                if value is None:
                    stored.pop(key, None)
                else:
                    stored[key] = deepcopy(value)
        if "tools" in stored:
            stored["tools"] = _normalize_rule_tools(stored["tools"])
        if "pattern" in stored:
            stored["pattern"] = str(stored["pattern"]).strip()
        _normalize_rule_severity_action(stored)
        _validate_rule_regex(stored)
        rules[index] = stored
        return {"rule": deepcopy(stored)}
    if method == ReqMethod.PERMISSIONS_APPROVAL_OVERRIDES_DELETE:
        rid = str(params.get("id") or "").strip()
        if not rid:
            raise ValueError("id is required")
        overrides = saved.setdefault("approval_overrides", [])
        index = next((index for index, item in enumerate(overrides) if item.get("id") == rid), None)
        if index is None:
            raise LookupError("approval_override not found")
        del overrides[index]
        return {"ok": True}
    raise ValueError(f"unsupported permissions write method: {method}")


async def dispatch_managed_permissions_request(
    method: ReqMethod, params: dict[str, Any], *, agent_manager: Any,
    restart_sandboxes: Any, service: SecurityConfigService,
) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    if method in _READS:
        return _read(method, service)
    if method not in _WRITES:
        raise ValueError(f"unsupported permissions method: {method}")
    response, state = service.edit_native(lambda settings, local: _edit(method, params, settings, local))
    applied = await dispatch_security_config_request(
        ReqMethod.SECURITY_CONFIG_APPLY,
        {"expected_revision": state["revision"]},
        agent_manager=agent_manager,
        restart_sandboxes=restart_sandboxes,
        service=service,
    )
    if method == ReqMethod.PERMISSIONS_NET_GUARD_SET:
        response = _read(ReqMethod.PERMISSIONS_NET_GUARD_GET, service)
    return {**response, **applied}
