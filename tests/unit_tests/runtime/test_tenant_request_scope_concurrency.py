# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression: concurrent _tenant_request_scope must not race on _agent_manager."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jiuwenswarm.common.schema.agent import AgentRequest
from jiuwenswarm.runtime import AgentRuntime
from jiuwenswarm.runtime.context import get_request_agent_manager


class _NamedManager:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.asyncio
async def test_concurrent_tenant_scopes_do_not_overwrite_instance_manager(
    monkeypatch,
) -> None:
    default = _NamedManager("default")
    manager_a = _NamedManager("tenant-a")
    manager_b = _NamedManager("tenant-b")
    by_key = {"tenant-a": manager_a, "tenant-b": manager_b}

    runtime = AgentRuntime(agent_manager=default, initializer=AsyncMock())

    pool = SimpleNamespace(
        extract_tenant_key=lambda request: request.metadata["tenant_key"],
        get_agent_manager=AsyncMock(side_effect=lambda key: by_key[key]),
        _agent_wrappers=SimpleNamespace(touch_if_same=AsyncMock()),
    )
    monkeypatch.setattr(
        "jiuwenswarm.runtime.service.TenantAgentPool.get_instance",
        staticmethod(lambda: pool),
    )
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.tenant_context.TenantContext.bind",
        lambda _key: SimpleNamespace(reset=lambda: None),
    )

    seen: dict[str, list[str]] = {"a": [], "b": []}

    async def _run(label: str, tenant_key: str, expected: _NamedManager) -> None:
        request = AgentRequest(
            request_id=f"req-{label}",
            channel_id="web",
            session_id=f"sess-{label}",
            metadata={"tenant_key": tenant_key},
        )
        async with runtime._tenant_request_scope(request) as manager:
            assert manager is expected
            assert runtime._agent_manager is default
            assert runtime.agent_manager is expected
            assert get_request_agent_manager() is expected
            # Yield so the peer can enter the scope while we are still bound.
            await asyncio.sleep(0)
            seen[label].append(runtime.agent_manager.name)
            assert runtime.agent_manager is expected
            assert runtime._agent_manager is default

    await asyncio.gather(
        _run("a", "tenant-a", manager_a),
        _run("b", "tenant-b", manager_b),
    )

    assert seen["a"] == ["tenant-a"]
    assert seen["b"] == ["tenant-b"]
    assert runtime._agent_manager is default
    assert get_request_agent_manager() is None
