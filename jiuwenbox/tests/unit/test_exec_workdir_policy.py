# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from jiuwenbox.models.policy import SecurityPolicy
from jiuwenbox.models.sandbox import BackgroundExecResult, ExecResult, SandboxPhase, SandboxRef
from jiuwenbox.server.limits import SandboxLimits
from jiuwenbox.server.sandbox_manager import (
    CodePolicyDenialError,
    SandboxBackgroundExecRequest,
    SandboxExecRequest,
    SandboxManager,
)


class _Runtime:
    def __init__(self) -> None:
        self.exec_calls = 0

    async def exec(self, sandbox_id, request):
        self.exec_calls += 1
        return ExecResult(exit_code=0)

    async def list_background_jobs(self, sandbox_id):
        return []

    async def exec_background(self, sandbox_id, request):
        raise AssertionError("background exec should be denied before runtime")


def _manager() -> tuple[SandboxManager, _Runtime]:
    manager = SandboxManager.__new__(SandboxManager)
    manager._lock = asyncio.Lock()
    manager._sandboxes = {"sb": SandboxRef(id="sb", phase=SandboxPhase.READY)}
    policy = SecurityPolicy()
    policy.windows.filesystem.deny_read = ["D:/secret"]
    policy.windows.filesystem.deny_write = ["D:/secret"]
    manager._policies = {"sb": policy}
    manager.policy = policy
    runtime = _Runtime()
    manager.runtime = runtime
    manager.audit = type("Audit", (), {"log": staticmethod(lambda *args, **kwargs: None)})()
    manager.limits = SandboxLimits()
    return manager, runtime


def test_exec_workdir_on_denied_path_is_rejected():
    manager, runtime = _manager()

    async def run():
        with pytest.raises(CodePolicyDenialError):
            await manager.exec_in_sandbox(
                "sb",
                SandboxExecRequest(command=["ls"], workdir="D:/secret"),
            )
        with pytest.raises(CodePolicyDenialError):
            await manager.exec_background_in_sandbox(
                "sb",
                SandboxBackgroundExecRequest(command=["ls"], workdir="D:/secret/child"),
            )

    asyncio.run(run())
    assert runtime.exec_calls == 0


def test_exec_without_workdir_skips_path_policy():
    manager, runtime = _manager()

    async def run():
        result = await manager.exec_in_sandbox(
            "sb",
            SandboxExecRequest(command=["ls"]),
        )
        assert result.exit_code == 0

    asyncio.run(run())
    assert runtime.exec_calls == 1


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("explicit_env", [False, True])
def test_windows_exec_inherits_allowlisted_environment(monkeypatch, background, explicit_env):
    from jiuwenbox.server import sandbox_manager

    monkeypatch.setattr(sandbox_manager.sys, "platform", "win32")
    monkeypatch.setenv("CLAW_SKILL_TOKEN", "test-skill-token")
    monkeypatch.setenv("CLAW_PYTHON_EXE", "C:/managed/python.exe")
    monkeypatch.setenv("CUSTOM_TOOL_SETTING", "inherited-setting")
    monkeypatch.setenv("EMPTY_TOOL_SETTING", "")
    monkeypatch.setenv("JIUWENBOX_ENV_ALLOWLIST", json.dumps([
        "CLAW_SKILL_TOKEN", "CLAW_PYTHON_EXE", "CUSTOM_TOOL_SETTING", "EMPTY_TOOL_SETTING",
        "PYTHONPATH", "NODE_OPTIONS",
    ]))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("DATABASE_URL", "must-not-be-forwarded")
    monkeypatch.setenv("PYTHONPATH", "runner-loader-path")
    monkeypatch.setenv("NODE_OPTIONS", "runner-loader-options")
    manager, runtime = _manager()
    request_env = {
        "CLAW_SKILL_TOKEN": "explicit-token", "CUSTOM_TOOL_SETTING": "request-override",
        "PYTHONPATH": "explicit-loader-path",
    } if explicit_env else None
    if background:
        operation = runtime.exec_background = AsyncMock(
            return_value=BackgroundExecResult(started=False, job_id="test-job"),
        )
        request = SandboxBackgroundExecRequest(command=["python"], env=request_env)
        asyncio.run(manager.exec_background_in_sandbox("sb", request))
    else:
        operation = runtime.exec = AsyncMock(return_value=ExecResult(exit_code=0))
        request = SandboxExecRequest(command=["python"], env=request_env)
        asyncio.run(manager.exec_in_sandbox("sb", request))
    env = operation.await_args.args[1].env
    assert env["CLAW_SKILL_TOKEN"] == ("explicit-token" if explicit_env else "test-skill-token")
    assert env["CLAW_PYTHON_EXE"] == "C:/managed/python.exe"
    assert env["CUSTOM_TOOL_SETTING"] == (
        "request-override" if explicit_env else "inherited-setting"
    )
    assert env["EMPTY_TOOL_SETTING"] == ""
    assert sandbox_manager.os.environ["CUSTOM_TOOL_SETTING"] == "inherited-setting"
    assert request.env == request_env
    assert "OPENAI_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert "NODE_OPTIONS" not in env
    if explicit_env:
        assert env["PYTHONPATH"] == "explicit-loader-path"
    else:
        assert "PYTHONPATH" not in env
