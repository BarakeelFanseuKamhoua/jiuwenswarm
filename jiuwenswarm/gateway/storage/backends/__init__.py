# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Backends package — heavy backends (file/db) import lazily."""

from __future__ import annotations

from typing import Any

from jiuwenswarm.gateway.storage.backends.memory_ephemeral import MemoryEphemeralBackend
from jiuwenswarm.gateway.storage.backends.redis_ephemeral import RedisEphemeralBackend

__all__ = [
    "FilePersistentBackend",
    "DbPersistentBackend",
    "PersistentDbConnection",
    "MemoryEphemeralBackend",
    "RedisEphemeralBackend",
]


def __getattr__(name: str) -> Any:
    if name == "FilePersistentBackend":
        from jiuwenswarm.gateway.storage.backends.file_persistent import (
            FilePersistentBackend,
        )

        return FilePersistentBackend
    if name in ("DbPersistentBackend", "PersistentDbConnection"):
        from jiuwenswarm.gateway.storage.backends.db import (
            DbPersistentBackend,
            PersistentDbConnection,
        )

        return {
            "DbPersistentBackend": DbPersistentBackend,
            "PersistentDbConnection": PersistentDbConnection,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
