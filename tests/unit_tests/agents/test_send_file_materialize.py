# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""send_file_to_user 团队模式「发送即实体化」+ 指纹去重 + source_path 锚定测试。

- 团队模式下源在 project_root 树外 → 实体化到根目录，payload source_path /
  artifact_id / name 一律以实体化路径为准；
- 冲突三态：同名同内容复用 / 异源撞名 (N) 后缀 / 同源更新覆盖写；
- 去重键 = (路径, size, 内容 hash) 指纹：内容变了放行重发，未变才跳过；
- 非团队会话零行为变化（source_path == 源路径，不实体化）。
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from openjiuwen.core.common.exception.codes import StatusCode
from jiuwenswarm.agents.harness.common.tools import send_file_to_user as sfu


def _operation(content_map: dict[str, bytes]):
    """按路径返回不同字节的假 SysOperation。"""

    async def _read(path, mode="bytes"):
        if path in content_map:
            return SimpleNamespace(
                code=StatusCode.SUCCESS.code,
                message="ok",
                data=SimpleNamespace(content=content_map[path]),
            )
        return SimpleNamespace(
            code=StatusCode.SUCCESS.code,
            message=f"File not found: {path}",
            data=None,
        )

    return SimpleNamespace(fs=lambda: SimpleNamespace(read_file=_read))


@pytest.fixture
def env(monkeypatch, tmp_path):
    """公共环境：假 server/落盘、清空两张 session 登记表、给定 project_root。"""
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_history.get_agent_sessions_dir",
        lambda: tmp_path / "sessions",
    )
    sfu._SENT_FILE_PATHS_BY_SESSION.clear()
    sfu._MATERIALIZED_TARGETS_BY_SESSION.clear()
    server = SimpleNamespace(send_push=AsyncMock())
    monkeypatch.setattr(
        "jiuwenswarm.server.agent_ws_server.AgentWebSocketServer.get_instance",
        lambda: server,
    )
    history = Mock()
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_history.append_history_record",
        history,
    )
    monkeypatch.setenv("JIUWENSWARM_FILE_DOWNLOAD_SECRET", "test-only-secret-" * 4)
    root = tmp_path / "project"
    root.mkdir()
    return SimpleNamespace(server=server, history=history, root=root, tmp=tmp_path)


def _team(monkeypatch, env, enabled=True):
    monkeypatch.setattr(sfu, "_is_team_session", lambda session_id: enabled)
    monkeypatch.setattr(sfu, "_resolve_materialize_root", lambda: str(env.root))


async def _send(env, files: dict[str, bytes], paths: list[str]):
    op = _operation(files)
    toolkit = sfu.SendFileToolkit("r", "sess-1", "desktop", operation_provider=lambda: op)
    return await toolkit.get_tools()[0].invoke({"abs_file_path_list": paths})


def _last_files(env):
    return env.server.send_push.await_args.args[0]["payload"]["files"]


def test_is_team_session_reads_locked_mode(monkeypatch):
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_metadata.get_session_metadata",
        lambda sid, enable_writeback=True: {"mode": "team"},
    )
    assert sfu._is_team_session("sess-1") is True
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_metadata.get_session_metadata",
        lambda sid, enable_writeback=True: {"mode": "agent"},
    )
    assert sfu._is_team_session("sess-1") is False


def test_is_team_session_failure_degrades_to_false(monkeypatch):
    def _boom(sid, enable_writeback=True):
        raise RuntimeError("no metadata")

    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_metadata.get_session_metadata",
        _boom,
    )
    assert sfu._is_team_session("sess-1") is False
    assert sfu._is_team_session("") is False


def test_materialize_root_prefers_project_root(monkeypatch, tmp_path):
    root = tmp_path / "root"
    cwd = tmp_path / "cwd"
    root.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(
        "openjiuwen.core.sys_operation.cwd.get_project_root", lambda: str(root)
    )
    monkeypatch.setattr("openjiuwen.core.sys_operation.cwd.get_cwd", lambda: str(cwd))
    assert sfu._resolve_materialize_root() == str(root)
    monkeypatch.setattr("openjiuwen.core.sys_operation.cwd.get_project_root", lambda: "")
    assert sfu._resolve_materialize_root() == str(cwd)


@pytest.mark.asyncio
async def test_team_source_under_root_fast_path(env, monkeypatch):
    """源已在 project_root 树下：不复制，source_path 即源路径。"""
    _team(monkeypatch, env)
    source = env.root / "报告.md"
    content = b"report-v1"
    source.write_bytes(content)
    files = {str(source.resolve()): content}
    result = await _send(env, files, [str(source)])
    assert result == "Sent 1 files"
    assert len(list(env.root.iterdir())) == 1  # 没有多出来的副本
    entry = _last_files(env)[0]
    assert entry["source_path"] == str(source.resolve())
    assert entry["artifact_id"] == sfu.artifact_id_for_path(str(source.resolve()))


@pytest.mark.asyncio
async def test_team_source_outside_root_materialized(env, monkeypatch):
    """源在树外（如 .team/ 挂载）：实体化到 project_root 根，卡片锚定副本。"""
    _team(monkeypatch, env)
    outside = env.tmp / "team-ws" / "方案.md"
    outside.parent.mkdir()
    content = b"plan-v1"
    files = {str(outside.resolve()): content}
    result = await _send(env, files, [str(outside)])
    assert result == "Sent 1 files"
    materialized = env.root / "方案.md"
    assert materialized.read_bytes() == content  # 文件真实存在于 project_dir
    entry = _last_files(env)[0]
    assert entry["source_path"] == str(materialized)
    assert entry["name"] == "方案.md"
    assert entry["artifact_id"] == sfu.artifact_id_for_path(str(materialized))
    # 下载载荷仍指暂存副本，不指源/实体化路径
    assert entry["path"] != entry["source_path"]


@pytest.mark.asyncio
async def test_team_name_conflict_gets_suffix(env, monkeypatch):
    """异源撞名：工作区已有同名不同内容文件 → (1) 后缀。"""
    _team(monkeypatch, env)
    (env.root / "a.md").write_bytes(b"existing-different")
    outside = env.tmp / "o" / "a.md"
    outside.parent.mkdir()
    files = {str(outside.resolve()): b"new-content"}
    await _send(env, files, [str(outside)])
    entry = _last_files(env)[0]
    assert entry["source_path"] == str(env.root / "a (1).md")
    assert (env.root / "a (1).md").read_bytes() == b"new-content"
    assert (env.root / "a.md").read_bytes() == b"existing-different"  # 原文件不动


@pytest.mark.asyncio
async def test_team_same_name_same_content_reused(env, monkeypatch):
    """异源撞名但内容一致：复用已有文件，不产副本。"""
    _team(monkeypatch, env)
    (env.root / "a.md").write_bytes(b"same")
    outside = env.tmp / "o" / "a.md"
    outside.parent.mkdir()
    files = {str(outside.resolve()): b"same"}
    await _send(env, files, [str(outside)])
    entry = _last_files(env)[0]
    assert entry["source_path"] == str(env.root / "a.md")
    assert not (env.root / "a (1).md").exists()


@pytest.mark.asyncio
async def test_unchanged_resend_skipped(env, monkeypatch):
    """内容未变的重发被跳过（防压缩丢历史后的重复推送）。"""
    _team(monkeypatch, env)
    source = env.root / "报告.md"
    files = {str(source.resolve()): b"v1"}
    assert await _send(env, files, [str(source)]) == "Sent 1 files"
    result = await _send(env, files, [str(source)])
    assert "already sent" in result
    assert env.server.send_push.await_count == 1


@pytest.mark.asyncio
async def test_changed_resend_allowed_and_overwrites(env, monkeypatch):
    """内容变了放行重发；同源更新覆盖写同一目标，不产 (N) 副本。"""
    _team(monkeypatch, env)
    outside = env.tmp / "team-ws" / "方案.md"
    outside.parent.mkdir()
    key = str(outside.resolve())
    assert await _send(env, {key: b"v1"}, [str(outside)]) == "Sent 1 files"
    result = await _send(env, {key: b"v2-changed"}, [str(outside)])
    assert result == "Sent 1 files"
    assert env.server.send_push.await_count == 2
    materialized = env.root / "方案.md"
    assert materialized.read_bytes() == b"v2-changed"  # 覆盖写
    assert not (env.root / "方案 (1).md").exists()  # 不产版本坟场
    # 两次推送 artifact_id 相同（同键合并为同一产物的新版本）
    first = env.server.send_push.await_args_list[0].args[0]["payload"]["files"][0]
    second = _last_files(env)[0]
    assert first["artifact_id"] == second["artifact_id"]


@pytest.mark.asyncio
async def test_non_team_no_materialization(env, monkeypatch):
    _team(monkeypatch, env, enabled=False)
    outside = env.tmp / "o" / "x.md"
    outside.parent.mkdir()
    key = str(outside.resolve())
    result = await _send(env, {key: b"data"}, [str(outside)])
    assert result == "Sent 1 files"
    assert list(env.root.iterdir()) == []  # 不实体化
    entry = _last_files(env)[0]
    assert entry["source_path"] == key
    assert entry["artifact_id"] == sfu.artifact_id_for_path(key)


@pytest.mark.asyncio
async def test_team_gate_off_without_project_root(env, monkeypatch):
    """团队但 project_root 不可解析：退化为不实体化。"""
    monkeypatch.setattr(sfu, "_is_team_session", lambda session_id: True)
    monkeypatch.setattr(sfu, "_resolve_materialize_root", lambda: None)
    outside = env.tmp / "o" / "x.md"
    outside.parent.mkdir()
    key = str(outside.resolve())
    result = await _send(env, {key: b"data"}, [str(outside)])
    assert result == "Sent 1 files"
    assert list(env.root.iterdir()) == []


@pytest.mark.asyncio
async def test_clear_session_drops_both_registries(env, monkeypatch):
    _team(monkeypatch, env)
    source = env.root / "a.md"
    key = str(source.resolve())
    await _send(env, {key: b"v"}, [str(source)])
    assert sfu._SENT_FILE_PATHS_BY_SESSION
    sfu._MATERIALIZED_TARGETS_BY_SESSION["sess-1"] = {"k": "v"}
    sfu.clear_sent_files_for_session("sess-1")
    assert "sess-1" not in sfu._SENT_FILE_PATHS_BY_SESSION
    assert "sess-1" not in sfu._MATERIALIZED_TARGETS_BY_SESSION
