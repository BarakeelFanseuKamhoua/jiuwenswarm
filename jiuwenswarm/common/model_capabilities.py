"""Swarm-owned native image policy; model inputs never become LLM parameters."""

from collections.abc import Callable, Mapping
from typing import Any


def normalize_model_input(raw: Any) -> tuple[str, ...] | None:
    values = raw if isinstance(raw, (list, tuple)) else [raw] if isinstance(raw, str) else []
    modes = tuple(dict.fromkeys(value.strip().lower() for value in values if isinstance(value, str) and value.strip()))
    return modes or None


def model_identity(model: Any) -> tuple[str, str, str]:
    client = getattr(model, "model_client_config", None)
    request = getattr(model, "model_config", None)
    return (
        str(getattr(request, "model_name", "") or "").strip(),
        str(getattr(client, "api_base", "") or "").rstrip("/"),
        str(getattr(client, "client_provider", "") or ""),
    )


def resolve_image_support(model: Any, request: Mapping | None = None, config: Mapping | None = None) -> bool | None:
    """Match declarations to the actual model, including endpoint for YAML entries."""
    name, base, provider = model_identity(model)
    request = request or {}
    modes = normalize_model_input(request.get("model_input"))
    endpoint = request.get("model_endpoint") or {}
    endpoint_matches = isinstance(endpoint, Mapping) and (
        (not endpoint.get("api_base") or str(endpoint["api_base"]).rstrip("/") == base)
        and (not endpoint.get("client_provider") or str(endpoint["client_provider"]) == provider)
    )
    if name and name == str(request.get("model_name") or "").strip() and modes and endpoint_matches:
        return "image" in modes
    models = (config or {}).get("models") or {}
    if not isinstance(models, Mapping):
        return None
    entries = models.get("defaults") or [models.get("default") or {}]
    matches = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, Mapping):
            continue
        client = entry.get("model_client_config") or {}
        if not isinstance(client, Mapping) or str(client.get("model_name") or "").strip() != name:
            continue
        if client.get("api_base") and str(client["api_base"]).rstrip("/") != base:
            continue
        if client.get("client_provider") and str(client["client_provider"]) != provider:
            continue
        modes = normalize_model_input(entry.get("input"))
        matches.append("image" in modes if modes else None)
    return matches[0] if matches and all(value == matches[0] for value in matches) else None


class ImageCapabilityPolicy:
    """Refresh only this Agent's native image state; unknown restores core's policy."""

    def __init__(
        self,
        request_source: Callable[[], Mapping | None] | None = None,
        config_source: Callable[[], Mapping | None] | None = None,
    ):
        self.request_source = request_source or (lambda: None)
        self.config_source = config_source or (lambda: None)
        self._config: Any = None
        self._original: dict[tuple[int, str], tuple[Any, Any]] = {}

    def _set(self, obj: Any, attr: str, enabled: bool | None) -> None:
        key = (id(obj), attr)
        if enabled is not None:
            if key not in self._original:
                self._original[key] = (obj, getattr(obj, attr))
            setattr(obj, attr, enabled)
        elif key in self._original:
            saved, original = self._original.pop(key)
            setattr(saved, attr, original)

    def apply(self, agent: Any) -> bool | None:
        config = getattr(agent, "deep_config", None) or getattr(agent, "_deep_config", None)
        if config is None:
            return None
        if config is not self._config:
            # Reconfiguration can replace DeepAgentConfig while retaining rail-owned
            # tools. Preserve their original flags instead of capturing our override.
            self._original = {key: value for key, value in self._original.items() if value[0] is not self._config}
            self._config = config
        source = self.config_source() or {}
        react = source.get("react") or {}
        enabled = resolve_image_support(getattr(config, "model", None), self.request_source(), source)
        setattr(agent, "_jiuwenswarm_model_image_support", enabled)
        if isinstance(react, Mapping) and react.get("enable_read_image_multimodal") is False:
            enabled = False
        self._set(config, "enable_read_image_multimodal", enabled)
        setattr(agent, "_jiuwenswarm_declared_image_support", enabled)
        manager = getattr(agent, "ability_manager", None)
        live_ids = {id(config)}
        get_card = getattr(manager, "get", None)
        if callable(get_card):
            from openjiuwen.core.runner import Runner
            for name, attr in (("read_file", "enable_image_multimodal"), ("skill_tool", "enable_read_image_multimodal")):
                card = get_card(name)
                tool = Runner.resource_mgr.get_tool(card.id) if card else None
                if tool is not None and hasattr(tool, attr):
                    live_ids.add(id(tool))
                    self._set(tool, attr, enabled)
        self._original = {key: value for key, value in self._original.items() if key[0] in live_ids}
        return enabled
