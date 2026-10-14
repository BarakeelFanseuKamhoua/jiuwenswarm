# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

POLL_TABLES: tuple[str, ...] = (
    "channel_config",
    "logging_config",
    "log_masking_rule",
    "audit_log_config",
)
_POLL_TABLES = frozenset(POLL_TABLES)
# 本进程 DB handler 未注册的表：只告警一次，避免每轮刷屏
_skipped_uninitialized: set[str] = set()
_warned_no_store = False


class ConfigPollReadError(Exception):
    """读库失败：本轮不得当作空表 apply，也不得更新 snapshot。"""


def _normalize_updated_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def row_key(table: str, row: dict[str, Any]) -> str:
    if table == "channel_config":
        return str(row.get("channel_id") or "").strip()
    if table == "log_masking_rule":
        return str(row.get("rule_id") or "").strip()
    # logging_config / audit_log_config：单文档表
    return str(row.get("id") or "default").strip()


def row_stamp(row: dict[str, Any]) -> str:
    ts = _normalize_updated_at(row.get("updated_at"))
    return ts.isoformat() if ts is not None else ""


def row_snapshot(table: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for row in rows:
        key = row_key(table, row)
        if not key:
            continue
        snapshot[key] = row_stamp(row)
    return snapshot


async def list_table_records(table: str) -> list[dict[str, Any]]:
    """拉取 poll 表全行：仅经 ``PersistentStore``。

    成功时返回行列表（可为空，表示表内确实无行）。
    读失败（无 store / 表未注册）抛 ``ConfigPollReadError``，
    避免 syncer 把失败当成空表而清空已生效配置。
    """
    global _warned_no_store

    if table not in _POLL_TABLES:
        raise ConfigPollReadError(f"unsupported poll table: {table}")

    from jiuwenswarm.gateway.storage.access import get_persistent_store

    store = get_persistent_store()
    if store is None:
        if not _warned_no_store:
            _warned_no_store = True
            logger.warning(
                "[ConfigPoll] PersistentStore not wired; skip poll until store is set"
            )
        raise ConfigPollReadError("PersistentStore not wired")

    await store.ensure_ready()
    try:
        rows = await store.list(table)
    except ValueError as exc:
        # 例如 ``Table xxx not initialized``：当前 Gateway
        # 进程未注册该表模型，跳过即可（其它 poll 表仍正常）。
        if "not initialized" in str(exc):
            if table not in _skipped_uninitialized:
                _skipped_uninitialized.add(table)
                logger.warning(
                    "[ConfigPoll] skip table=%s (%s); other poll tables unaffected",
                    table,
                    exc,
                )
            raise ConfigPollReadError(f"table not initialized: {table}") from exc
        raise
    return [dict(row) for row in rows or []]
