"""Windows sandbox failures must prevent a successful backend handshake."""

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwenswarm.server import agent_ws_server as server
from jiuwenswarm.server.sandbox import jiuwenbox_runner as runner_module


@pytest.mark.asyncio
async def test_failed_bootstrap_prevents_listener_and_sets_failed_gate(monkeypatch):
    from jiuwenswarm.agents.harness.common import auto_harness
    monkeypatch.setattr(auto_harness, "reset_harness_packages_state", Mock())
    error = RuntimeError("UAC 已取消")
    owner = SimpleNamespace(
        _server=None, _desktop_listenerless=False,
        _startup_complete=asyncio.Event(), _startup_error=None,
        _bootstrap_internal_jiuwenbox=AsyncMock(side_effect=error),
    )
    with pytest.raises(RuntimeError, match="UAC 已取消"):
        await server.AgentWebSocketServer.start(owner, listen_tcp=False)
    assert owner._startup_error is error
    assert owner._startup_complete.is_set()
    assert owner._desktop_listenerless is False


@pytest.mark.asyncio
async def test_disabled_sandbox_skips_privileged_setup(monkeypatch):
    monkeypatch.setattr(server, "get_sandbox_runtime", lambda: {"enabled": False})
    build = AsyncMock(side_effect=AssertionError("unexpected setup"))
    monkeypatch.setattr(server, "_build_sandbox_env", build)
    await server.AgentWebSocketServer._bootstrap_internal_jiuwenbox(SimpleNamespace())
    build.assert_not_called()


@pytest.mark.asyncio
async def test_connection_waits_for_setup_and_failed_setup_never_acks(monkeypatch):
    owner = server.AgentWebSocketServer.__new__(server.AgentWebSocketServer)
    owner.begin_startup()
    transport = SimpleNamespace(close=AsyncMock(), remote_address="test")
    send = AsyncMock()
    monkeypatch.setattr(server, "send_wire_payload", send)
    connection = asyncio.create_task(owner.run_connection(transport))
    await asyncio.sleep(0)
    assert not connection.done()
    send.assert_not_called()
    owner._startup_error = RuntimeError("UAC 已取消")
    owner._startup_complete.set()
    with pytest.raises(RuntimeError, match="UAC 已取消"):
        await connection
    send.assert_not_called()
    transport.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ready_setup_allows_ack(monkeypatch):
    owner = server.AgentWebSocketServer.__new__(server.AgentWebSocketServer)
    owner.begin_startup()
    owner._startup_complete.set()
    owner._session_stream_tasks = {}
    owner._agent_manager = SimpleNamespace(cancel_all_inflight_work=AsyncMock())
    owner._stop_scheduler = AsyncMock()
    owner._unbind_channel_conns = Mock()
    owner._clear_ws_acp_client_capabilities = Mock()
    transport = SimpleNamespace(close=AsyncMock(), recv_text=AsyncMock(return_value=None), remote_address="test")
    send = AsyncMock()
    monkeypatch.setattr(server, "send_wire_payload", send)
    await owner.run_connection(transport)
    assert send.await_args_list[0].args[1]["event"] == "connection.ack"


@pytest.mark.asyncio
@pytest.mark.parametrize("supported,expected", [(False, False), (True, True)])
async def test_health_requires_windows_capability(monkeypatch, supported, expected):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = SimpleNamespace(status_code=200, json=lambda: {
        "status": "ok", "windows_supported": supported,
    })
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda **kwargs: client)
    runner = runner_module.JiuwenBoxRunner()
    assert await runner.health_check() is expected


@pytest.mark.asyncio
async def test_failed_setup_never_spawns_box_or_job(monkeypatch, tmp_path):
    runner = runner_module.JiuwenBoxRunner()
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1")
    monkeypatch.setattr(runner_module, "_cleanup_stale_win_proxy_ports", Mock())
    monkeypatch.setattr(runner_module, "_apply_soft_delete_env", Mock())
    monkeypatch.setattr(runner, "_prepare_windows_setup", AsyncMock(side_effect=RuntimeError("授权已取消")))
    spawn = AsyncMock()
    bind = Mock()
    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(runner, "_bind_win_job", bind)
    with pytest.raises(RuntimeError, match="授权已取消"):
        await runner.ensure_running(policy_path=policy)
    spawn.assert_not_called()
    bind.assert_not_called()
    assert runner.process is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exit", "timeout"])
async def test_setup_helper_failure_is_explicit(monkeypatch, failure):
    runner = runner_module.JiuwenBoxRunner()
    result = SimpleNamespace(returncode=1, stderr="UAC 已取消".encode())
    run = Mock(return_value=result)
    if failure == "timeout":
        run.side_effect = subprocess.TimeoutExpired("setup", 180)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="初始化"):
        await runner._prepare_windows_setup({"PYTHONPATH": "local-source"}, None)
    assert run.call_args.kwargs["env"] == {"PYTHONPATH": "local-source"}
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL
