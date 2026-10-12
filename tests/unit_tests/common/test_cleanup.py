# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import os
import time

import pytest

from jiuwenswarm.common.cleanup import (
    cleanup_old_sessions,
    cleanup_orphan_file_ops,
)


class TestCleanup:

    @staticmethod
    @pytest.mark.parametrize(
        ("desktop", "retention"),
        [
            ("1", None), ("1", "invalid"), ("1", "0"), ("1", "-1"),
            (None, None), (None, ""), (None, "invalid"), (None, "0"), (None, "-1"),
        ],
    )
    def test_preserves_old_sessions_when_cleanup_disabled(tmp_path, monkeypatch, desktop, retention):
        """所有启动方式默认永久保留会话，非正保留天数也禁用过期清理。"""
        sessions_dir = tmp_path / "sessions"
        stale = sessions_dir / "session-old"
        stale.mkdir(parents=True)
        history = stale / "history.json"
        history.write_text('{"message": "keep me"}', encoding="utf-8")
        old_time = time.time() - 999 * 86400
        os.utime(stale, (old_time, old_time))
        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_agent_sessions_dir", lambda: sessions_dir)
        for key, value in (("JIUWENSWARM_DESKTOP", desktop), ("CLEANUP_PERIOD_DAYS", retention)):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        result = cleanup_old_sessions()

        assert result == {"removed": 0, "errors": 0}
        assert history.read_text(encoding="utf-8") == '{"message": "keep me"}'

    @staticmethod
    @pytest.mark.parametrize(("desktop", "retention"), [(None, "7"), (None, "30"), ("1", "30")])
    def test_positive_retention_removes_only_old_sessions(tmp_path, monkeypatch, desktop, retention):
        """显式设置正数保留天数时，所有启动方式均清理过期会话。"""
        sessions_dir = tmp_path / "sessions"
        stale = sessions_dir / "session-old"
        recent = sessions_dir / "session-recent"
        stale.mkdir(parents=True)
        recent.mkdir()
        old_time = time.time() - 40 * 86400
        os.utime(stale, (old_time, old_time))
        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_agent_sessions_dir", lambda: sessions_dir)
        for key, value in (("JIUWENSWARM_DESKTOP", desktop), ("CLEANUP_PERIOD_DAYS", retention)):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        result = cleanup_old_sessions()

        assert result == {"removed": 1, "errors": 0}
        assert not stale.exists()
        assert recent.is_dir()

    @staticmethod
    def test_removes_old_sessions_by_mtime(tmp_path, monkeypatch):
        """会话目录 mtime 超过保留期时被删除（与 cc 一致）。"""
        sessions_dir = tmp_path / "sessions"
        stale = sessions_dir / "session-old"
        stale.mkdir(parents=True)

        # 直接修改目录 mtime 到很久以前
        old_time = time.time() - 999 * 86400
        os.utime(stale, (old_time, old_time))

        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_agent_sessions_dir", lambda: sessions_dir)
        monkeypatch.setattr("jiuwenswarm.common.cleanup._get_cleanup_period_days", lambda: 30)

        result = cleanup_old_sessions()
        assert result["removed"] >= 1
        assert not stale.exists()

    @staticmethod
    def test_removes_file_ops_without_session(tmp_path, monkeypatch):
        """file_ops 对应的 session 目录已不存在时被删除。"""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True)

        hist_dir = tmp_path / "agent-ws" / ".agent_history"
        hist_dir.mkdir(parents=True)
        fop = hist_dir / "file_ops_jiuwenswarm_dead_session.json"
        fop.write_text("{}", encoding="utf-8")

        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_agent_workspace_dir", lambda: tmp_path / "agent-ws")
        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_user_workspace_dir", lambda: tmp_path / "user-ws")
        monkeypatch.setattr("jiuwenswarm.common.cleanup.get_agent_sessions_dir", lambda: sessions_dir)

        result = cleanup_orphan_file_ops()
        assert result["removed"] >= 1
        assert not fop.exists()
