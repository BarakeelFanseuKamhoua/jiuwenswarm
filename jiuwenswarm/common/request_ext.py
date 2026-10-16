"""请求级扩展字段透传（HTTP header / query → Message.metadata['ext']）。

要透传哪些字段由环境变量声明（逗号分隔），两套名字等价，先读 Swarm 名：

- ``JIUWENSWARM_REQUEST_EXT_FORWARD_HEADERS``
- ``JIUWENCLAW_REQUEST_EXT_FORWARD_HEADERS``（企业 kub 兼容）

名单为空则不透传。载体内的键名与值含义由集成方约定。
"""
from __future__ import annotations

import os
from typing import Any, Mapping

ENV_FORWARD_HEADERS = "JIUWENSWARM_REQUEST_EXT_FORWARD_HEADERS"
ENV_FORWARD_HEADERS_LEGACY = "JIUWENCLAW_REQUEST_EXT_FORWARD_HEADERS"
METADATA_KEY = "ext"

# Gateway -> AgentServer 的内部载体名。浏览器侧仍使用白名单中的原始
# header/query；同名头不能被当作扩展字段再次封装。
INTERNAL_HEADER_NAME = "X-Jiuwenswarm-Request-Ext"


def _env_forward_headers() -> list[str]:
    raw = os.environ.get(ENV_FORWARD_HEADERS, "") or os.environ.get(ENV_FORWARD_HEADERS_LEGACY, "")
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def build_ext_from_source(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """从入站 mapping（HTTP header 或 query 字典）按环境变量白名单抽取扩展字段。

    支持值为字符串或字符串列表（``parse_qs`` 输出）；列表取首元素。
    键名匹配大小写不敏感。
    """
    names = _env_forward_headers()
    if not names or not source:
        return None
    lower_to_actual = {k.lower(): k for k in source.keys() if isinstance(k, str)}
    out: dict[str, Any] = {}
    for name in names:
        if name.lower() == INTERNAL_HEADER_NAME.lower():
            continue
        actual = lower_to_actual.get(name.lower())
        if actual is None:
            continue
        val = source[actual]
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        if val is None or val == "":
            continue
        out[name] = val
    return out or None


def attach_to_metadata(
    metadata: dict[str, Any] | None,
    ext: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """构造 Message 时使用：将 ext 写入 metadata。无 ext 时返回原 metadata。"""
    if not ext:
        return metadata
    out = dict(metadata or {})
    out[METADATA_KEY] = dict(ext)
    return out
