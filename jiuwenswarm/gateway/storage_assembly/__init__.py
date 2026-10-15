# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""存储装配层：按 edition 注入 backend，不属于 ``gateway/storage/`` 基础设施。

序 3：启用 2-C 接线导出（``setup_gateway_storage_repositories`` / ``create_*``）。
仍不改 ``app_gateway``（3-F / 2-D 另批）。
"""

from jiuwenswarm.gateway.storage_assembly.db_connection import (
    GatewayDbConnection,
    assert_replicas_db_compat,
)
from jiuwenswarm.gateway.storage_assembly.layouts import build_gateway_store_registry
from jiuwenswarm.gateway.storage_assembly.setup import (
    create_channel_config_repository,
    create_enterprise_record_repository,
    create_enterprise_record_repositories,
    create_gateway_storage_context,
    create_logging_config_repository,
    create_memory_config_repository,
    create_permissions_config_repository,
    create_session_map_repository,
    ensure_enterprise_storage_context,
    ensure_gateway_storage_context_for_ephemeral,
    is_ephemeral_state_enabled,
    is_session_map_repository_enabled,
    is_storage_repositories_enabled,
    resolve_storage_instance_id,
    setup_gateway_storage_repositories,
    setup_session_map_repository,
    teardown_gateway_storage_repositories,
    teardown_session_map_repository,
    wire_enterprise_persistent_repositories,
    wire_enterprise_persistent_repositories_async,
)

__all__ = [
    "GatewayDbConnection",
    "assert_replicas_db_compat",
    "build_gateway_store_registry",
    "create_channel_config_repository",
    "create_enterprise_record_repository",
    "create_enterprise_record_repositories",
    "create_gateway_storage_context",
    "create_logging_config_repository",
    "create_memory_config_repository",
    "create_permissions_config_repository",
    "create_session_map_repository",
    "ensure_enterprise_storage_context",
    "ensure_gateway_storage_context_for_ephemeral",
    "is_ephemeral_state_enabled",
    "is_session_map_repository_enabled",
    "is_storage_repositories_enabled",
    "resolve_storage_instance_id",
    "setup_gateway_storage_repositories",
    "setup_session_map_repository",
    "teardown_gateway_storage_repositories",
    "teardown_session_map_repository",
    "wire_enterprise_persistent_repositories",
    "wire_enterprise_persistent_repositories_async",
]
