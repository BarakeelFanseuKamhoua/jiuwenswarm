"""Policy reload must preserve the Windows runner's bootstrap environment."""

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jiuwenswarm.server.sandbox import jiuwenbox_runner as runner_module


@pytest.mark.asyncio
async def test_stop_preserves_pumps_until_process_exit(monkeypatch):
    runner = runner_module.JiuwenBoxRunner()
    exited = asyncio.Event()
    output = []

    async def pump():
        await exited.wait()
        output.append("shutdown complete")

    async def wait():
        assert not runner._stdout_pump_task.cancelled()
        exited.set()
        return 0

    runner.process = SimpleNamespace(pid=123, returncode=None, terminate=Mock(), wait=wait)
    runner.owns_process = True
    runner._stdout_pump_task = asyncio.create_task(pump())
    monkeypatch.setattr(runner_module, "sys", SimpleNamespace(platform="linux"))
    await runner.stop()
    assert output == ["shutdown complete"]
    assert runner._stdout_pump_task is None


def test_port_bind_failure_preserves_error(monkeypatch):
    from jiuwenswarm.server.sandbox import port_util
    error = OSError("permission denied")
    error.winerror = 10013
    sock = Mock()
    sock.bind.side_effect = error
    monkeypatch.setattr(port_util.socket, "socket", Mock(return_value=sock))
    warning = Mock()
    monkeypatch.setattr(port_util.logger, "warning", warning)
    assert not port_util.is_tcp_port_bindable("127.0.0.1", 8321)
    assert warning.call_args.args[-1] == 10013
    assert "cannot bind" in warning.call_args.args[0]
    sock.close.assert_called_once()


@pytest.mark.asyncio
async def test_stop_timeout_kills_before_draining_shutdown_logs(monkeypatch):
    import ctypes

    runner = runner_module.JiuwenBoxRunner()
    exited = asyncio.Event()
    output = []

    async def pump():
        await exited.wait()
        output.append("shutdown tail")

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: SimpleNamespace(
        GenerateConsoleCtrlEvent=lambda *a: False,
    ), raising=False)
    proc = SimpleNamespace(pid=123, returncode=None, terminate=Mock(),
                           wait=AsyncMock(side_effect=[asyncio.TimeoutError(), 0]),
                           kill=Mock(side_effect=exited.set))
    runner.process = proc
    runner.owns_process = True
    runner._stdout_pump_task = asyncio.create_task(pump())
    await runner.stop()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert output == ["shutdown tail"]
    assert runner.process is None


@pytest.mark.asyncio
async def test_policy_reload_retains_bootstrap_env_until_explicitly_replaced(tmp_path, monkeypatch):
    runner = runner_module.JiuwenBoxRunner()
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1", encoding="utf-8")
    monkeypatch.setattr(runner_module, "_cleanup_stale_win_proxy_ports", lambda: None)
    monkeypatch.setattr(runner_module, "_resolve_jiuwenbox_src_dir", lambda: None)
    monkeypatch.setattr(runner, "_register_atexit_once", Mock())
    monkeypatch.setattr(runner, "_bind_win_job", Mock())
    monkeypatch.setattr(runner, "_wait_until_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(runner, "_pump_stream", AsyncMock())
    monkeypatch.delenv("JIUWENBOX_RUNNER_PYTHON", raising=False)
    monkeypatch.delenv("JIUWENBOX_SKILLS_DIR", raising=False)
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "must-not-be-forwarded")

    async def stop():
        runner.process = None
        runner.owns_process = False
        runner.spawned_policy_path = None
        runner._spawned_policy_fingerprint = None

    monkeypatch.setattr(runner, "_stop_no_lock", AsyncMock(side_effect=stop))
    spawn = AsyncMock(side_effect=lambda *args, **kwargs: SimpleNamespace(
        returncode=None, stdout=None, stderr=None,
    ))
    monkeypatch.setattr(runner_module.asyncio, "create_subprocess_exec", spawn)
    bootstrap_env = {
        "JIUWENBOX_RUNNER_PYTHON": "C:/Python312/python.exe",
        "JIUWENBOX_SKILLS_DIR": "C:/workspace/skills",
    }
    assert await runner.ensure_running(policy_path=policy, extra_env=bootstrap_env)
    # The caller must not be able to mutate retained startup parameters.
    bootstrap_env["JIUWENBOX_RUNNER_PYTHON"] = "C:/unexpected/python.exe"
    policy.write_text("version: 2", encoding="utf-8")
    assert await runner.ensure_running(policy_path=policy)
    assert spawn.await_count == 2
    for call in spawn.await_args_list:
        env = call.kwargs["env"]
        assert env["JIUWENBOX_RUNNER_PYTHON"] == "C:/Python312/python.exe"
        assert env["JIUWENBOX_SKILLS_DIR"] == "C:/workspace/skills"
        assert env["JIUWENBOX_POLICY_PATH"] == str(policy)
        assert "UNRELATED_TEST_SECRET" not in env

    policy.write_text("version: 3", encoding="utf-8")
    assert await runner.ensure_running(policy_path=policy, extra_env={})
    assert "JIUWENBOX_RUNNER_PYTHON" not in spawn.await_args.kwargs["env"]
    assert "JIUWENBOX_SKILLS_DIR" not in spawn.await_args.kwargs["env"]
    await asyncio.sleep(0)  # Complete mocked stdout/stderr drain tasks.


_HARD_KILLED_AGENT = """
import asyncio, sys
from jiuwenswarm.server.sandbox.jiuwenbox_runner import JiuwenBoxRunner

async def main():
    runner = JiuwenBoxRunner()
    runner.process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)")
    runner._bind_win_job()
    print(runner.process.pid if runner._win_job else "unbound", flush=True)
    await asyncio.sleep(60)

asyncio.run(main())
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object")
def test_hard_killed_agent_server_takes_box_server_down():
    import ctypes

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    agent = subprocess.Popen([sys.executable, "-c", _HARD_KILLED_AGENT], stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = agent.stdout.readline().strip()
        assert line.isdigit(), line
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        child = kernel32.OpenProcess(0x00100000, False, int(line))  # SYNCHRONIZE
        assert child
        agent.kill()
        assert kernel32.WaitForSingleObject(child, 10000) == 0
        kernel32.CloseHandle(child)
    finally:
        agent.kill()
        agent.wait()
        agent.stdout.close()


def test_runner_probe_uses_startup_source_path(monkeypatch, tmp_path):
    source = tmp_path / "src"
    monkeypatch.setattr(runner_module, "_resolve_jiuwenbox_src_dir", lambda: source)
    monkeypatch.setenv("PYTHONPATH", "existing-source")
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "must-not-be-forwarded")
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(subprocess, "run", run)

    assert runner_module.probe_runner_python("runner-python", {"SystemRoot": "C:/Windows"}) is None
    env = run.call_args.kwargs["env"]
    assert env["PYTHONPATH"] == os.pathsep.join((str(source), "existing-source"))
    assert env["SystemRoot"] == "C:/Windows"
    assert "UNRELATED_TEST_SECRET" not in env
    assert os.environ["PYTHONPATH"] == "existing-source"


@pytest.mark.parametrize("result,reason", [
    (SimpleNamespace(returncode=1, stderr="Traceback\nModuleNotFoundError: missing"),
     "ModuleNotFoundError: missing"),
    (SimpleNamespace(returncode=7, stderr=""), "exit 7"),
    (subprocess.TimeoutExpired("runner-python", 15), "timed out"),
    (OSError("cannot start interpreter"), "cannot start interpreter"),
])
def test_runner_probe_reports_failure(monkeypatch, result, reason):
    run = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(subprocess, "run", run)
    assert reason in runner_module.probe_runner_python("runner-python", {})
