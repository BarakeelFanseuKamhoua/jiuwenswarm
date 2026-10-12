# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Swarm-owned team rail declarations.

Most rails take no construction params and use a class ``builder``. The stream
rail uses a factory to bind current session model capabilities. Only swarm-owned
rails live here; the generic openjiuwen rails (``sys_operation`` /
``task_planning`` / ``security`` / ``heartbeat`` / ...) are declared and
registered by openjiuwen itself and referenced by their bare names from
``config_specs``.

Declarations run at import time (the module-level ``harness_element`` calls
populate the catalog). ``registry`` imports this module so the declarations are
present before ``register_from_catalog`` runs.
"""

from __future__ import annotations

from typing import Any

from openjiuwen.agent_teams.harness.manifest import (
    ConstructionInput,
    ElementKind,
    context_field,
    harness_element,
)

from jiuwenswarm.agents.harness.common.rails.avatar_rail import AvatarPromptRail
from jiuwenswarm.agents.harness.common.rails.response_prompt_rail import (
    ResponsePromptRail,
)
from jiuwenswarm.agents.harness.common.rails.stream_event_rail import (
    JiuSwarmStreamEventRail,
)

# No-parameter swarm-owned rail type names; namespaced under "swarm.".
RESPONSE_PROMPT = "swarm.response_prompt"
STREAM_EVENT = "swarm.stream_event"
AVATAR_PROMPT = "swarm.avatar_prompt"


class ResponsePromptInput(ConstructionInput):
    """Construction inputs for the response-format prompt rail."""

    channel: str = context_field(
        attr="channel",
        default="default",
        description="Resolved channel key.",
    )


@harness_element(
    kind=ElementKind.RAIL,
    name=RESPONSE_PROMPT,
    description="Appends the response-format prompt segment before each model call.",
)
def _build_response_prompt_rail(
    params: dict[str, Any],
    context: Any,
) -> ResponsePromptRail:
    inp = ResponsePromptInput.resolve(params, context)
    rail = ResponsePromptRail()
    rail.set_channel(inp.channel)
    return rail


@harness_element(
    kind=ElementKind.RAIL,
    name=STREAM_EVENT,
    description="Emits JiuSwarm streaming events across the member's lifecycle.",
)
def _build_stream_event_rail(params: dict[str, Any], context: Any) -> JiuSwarmStreamEventRail:
    """Resolve image declarations against each member's actual runtime model."""
    from jiuwenswarm.agents.harness.team.team_manager import get_team_manager
    from jiuwenswarm.common.config import get_config

    session_id = getattr(context, "session_id", "")
    manager = get_team_manager()

    def request_source():
        current = manager.get_model_capabilities(session_id)
        if current is not None:
            return current
        return (getattr(context, "request_metadata", None) or {}).get("model_capabilities")

    rail = JiuSwarmStreamEventRail()
    rail.configure_image_capabilities(request_source, get_config)
    return rail

harness_element(
    kind=ElementKind.RAIL,
    name=AVATAR_PROMPT,
    description="Injects per-request digital-avatar prompt sections.",
    builder=AvatarPromptRail,
)

__all__ = [
    "RESPONSE_PROMPT",
    "STREAM_EVENT",
    "AVATAR_PROMPT",
]
