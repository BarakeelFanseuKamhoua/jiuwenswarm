# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Read outbound sources through the owning agent's filesystem backend."""

from collections.abc import Callable

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.sys_operation import SysOperation
from openjiuwen.harness.security.permission_engine.fileguard.outbound_paths import resolve_outbound_path

OperationProvider = Callable[[], SysOperation | None]


async def read_outbound_file(path: str, operation: SysOperation | None) -> bytes:
    if operation is None:
        raise RuntimeError("Outbound file access requires the agent's SysOperation binding")
    resolved_path = str(resolve_outbound_path(path))
    result = await operation.fs().read_file(resolved_path, mode="bytes")
    if result.code != StatusCode.SUCCESS.code or result.data is None:
        if f"File not found: {resolved_path}" in (result.message or ""):
            raise FileNotFoundError(resolved_path)
        raise RuntimeError(f"Unable to read outbound file {path}: {result.message}")
    content = result.data.content
    if not isinstance(content, bytes):
        raise RuntimeError("Outbound file backend did not return binary content")
    return content
