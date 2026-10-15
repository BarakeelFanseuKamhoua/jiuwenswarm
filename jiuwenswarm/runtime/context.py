# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Task-local access to the Runtime currently executing an agent request."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jiuwenswarm.runtime.service import AgentRuntime
    from jiuwenswarm.server.runtime.agent_manager import AgentManager


@dataclass(frozen=True, slots=True)
class RuntimeExecutionContext:
    """Runtime ownership inherited by child asyncio tasks for one execution."""

    runtime: AgentRuntime
    agent_manager: AgentManager


_CURRENT_RUNTIME_CONTEXT: ContextVar[RuntimeExecutionContext | None] = ContextVar(
    "jiuwenswarm_runtime_execution_context",
    default=None,
)

# Request-scoped Manager selected by AgentRuntime._tenant_request_scope.
# Task-local so concurrent requests on one Runtime do not race on an instance field.
_REQUEST_AGENT_MANAGER: ContextVar[AgentManager | None] = ContextVar(
    "jiuwenswarm_request_agent_manager",
    default=None,
)


def set_runtime_context(
    runtime: AgentRuntime,
    agent_manager: AgentManager,
) -> Token[RuntimeExecutionContext | None]:
    """Bind one Runtime execution and return the token required for reset."""

    return _CURRENT_RUNTIME_CONTEXT.set(
        RuntimeExecutionContext(runtime=runtime, agent_manager=agent_manager)
    )


def reset_runtime_context(token: Token[RuntimeExecutionContext | None]) -> None:
    """Restore the previous task-local Runtime execution."""

    _CURRENT_RUNTIME_CONTEXT.reset(token)


def get_runtime_context() -> RuntimeExecutionContext | None:
    """Return the current task-local Runtime execution, if one is active."""

    return _CURRENT_RUNTIME_CONTEXT.get()


def bind_request_agent_manager(
    agent_manager: AgentManager,
) -> Token[AgentManager | None]:
    """Bind the AgentManager for the current tenant request (task-local)."""

    return _REQUEST_AGENT_MANAGER.set(agent_manager)


def reset_request_agent_manager(token: Token[AgentManager | None]) -> None:
    """Restore the previous request-scoped AgentManager binding."""

    _REQUEST_AGENT_MANAGER.reset(token)


def get_request_agent_manager() -> AgentManager | None:
    """Return the Manager bound by ``_tenant_request_scope``, if any."""

    return _REQUEST_AGENT_MANAGER.get()


def get_current_runtime() -> AgentRuntime | None:
    context = get_runtime_context()
    return context.runtime if context is not None else None


def get_current_agent_manager() -> AgentManager | None:
    """Return the request-scoped Manager, else the execution-slice Manager."""

    request_manager = get_request_agent_manager()
    if request_manager is not None:
        return request_manager
    context = get_runtime_context()
    return context.agent_manager if context is not None else None


__all__ = [
    "RuntimeExecutionContext",
    "bind_request_agent_manager",
    "get_current_agent_manager",
    "get_current_runtime",
    "get_request_agent_manager",
    "get_runtime_context",
    "reset_request_agent_manager",
    "reset_runtime_context",
    "set_runtime_context",
]
