# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Send File Toolkit

提供发送文件到用户的工具。支持发送一个或多个文件。

使用方式：
1. 创建 SendFileToolkit 实例
2. 调用 get_tools() 获取工具列表
3. 工具会自动注册到 Runner 中
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import logging
from typing import Any, List, Union

from openjiuwen.core.foundation.tool import LocalFunction, Tool, ToolCard
from openjiuwen.harness.security.permission_engine.fileguard.outbound_paths import resolve_outbound_path

from .outbound_file_access import OperationProvider, read_outbound_file

from jiuwenswarm.agents.harness.common.tools.turn_request_identity import (
    resolve_toolkit_delivery,
)

logger = logging.getLogger(__name__)

# Session-level dedup for send_file_to_user. Compression may drop prior tool
# results, so the agent can re-call the same path; IM request-level dedup alone
# cannot stop cross-turn duplicates.
# 去重键为 (归一化路径, size, 内容hash) 指纹而非纯路径：同一文件编辑后重发
# 必须放行（指纹变了），未变重发才跳过（防压缩丢历史后的重复推送）。
_SENT_FILE_PATHS_BY_SESSION: dict[str, set[str]] = {}

# 团队模式「发送即实体化」登记：session → {归一化源路径: 实体化目标路径}。
# 同源更新覆盖写依赖此表找回上次的目标路径；异源撞名才加 (N) 后缀。
_MATERIALIZED_TARGETS_BY_SESSION: dict[str, dict[str, str]] = {}


def _normalize_sent_file_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/").lower()


def _normalize_abs_file_path_list(value: Any) -> list[str]:
    """把 send_file_to_user 的入参归一化成路径列表。

    模型实际会给出四种形态，全部接受：
      1. 数组（``["a.pdf", "b.xlsx"]``）—— 模型最自然的写法，优先引导；
      2. 双引号 JSON 数组字符串（``'["a.pdf", "b.xlsx"]'``）；
      3. 单引号 Python 字面量字符串（``"['a.pdf', 'b.xlsx']"``）——
         兼容历史工具描述里的示例写法：``json.loads`` 解析不了，用 literal_eval 兜住，
         否则整串会被当成一个路径、最后报「文件不存在」；
      4. 单个绝对路径字符串。

    非字符串/列表的值（None、数字等）按字符串处理，行为与原实现一致。
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[(" and text[-1] in ")]":
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (TypeError, ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
                if isinstance(parsed, str):
                    return [parsed.strip()] if parsed.strip() else []
                break
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def artifact_id_for_path(path: str) -> str:
    """产物稳定标识：供桌面端「对话卡片」与「右侧工作台产物」共用同一个键。

    用「归一化绝对路径」而不是文件内容/时间：同一路径重复产出（重写、重新导出、
    下载 URL 重新签名）都视为同一产物，跨轮、跨历史重放都稳定。WSL 的
    ``/mnt/c/...`` 先还原成 ``C:/...``，避免沙箱路径与 Windows 路径被当成两个产物。
    """
    # 先还原 WSL 盘符再 abspath：Windows 上 os.path.abspath("/mnt/c/x") 会变成
    # "C:/mnt/c/x"（相对当前盘符），必须在绝对化之前处理。
    raw = path.replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if match:
        raw = f"{match.group(1).upper()}:/{match.group(2)}"
    normalized = os.path.abspath(raw).replace("\\", "/")
    digest = hashlib.sha1(normalized.lower().encode("utf-8")).hexdigest()
    return f"art-{digest[:16]}"


def _sent_fingerprint(path: str, content: bytes) -> str:
    """发送去重指纹：归一化路径 + 内容指纹（size + sha1）。

    字节在调用点已经在内存里，sha1 零额外 IO；比 mtime 可靠（mtime 变而
    内容不变不应触发重发）。
    """
    digest = hashlib.sha1(content).hexdigest()[:16]
    return f"{_normalize_sent_file_path(path)}|{len(content)}|{digest}"


def _partition_sent_files(
        session_id: str,
        items: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Split ``(path, fingerprint)`` items into (new_to_send, already_sent).

    Does not mutate the registry.
    """
    sid = (session_id or "").strip() or "default"
    sent = _SENT_FILE_PATHS_BY_SESSION.get(sid) or set()
    new_paths: list[str] = []
    skipped: list[str] = []
    for path, fingerprint in items:
        if fingerprint in sent:
            skipped.append(path)
        else:
            new_paths.append(path)
    return new_paths, skipped


def _mark_files_sent(session_id: str, items: list[tuple[str, str]]) -> None:
    sid = (session_id or "").strip() or "default"
    sent = _SENT_FILE_PATHS_BY_SESSION.setdefault(sid, set())
    for _path, fingerprint in items:
        sent.add(fingerprint)


def clear_sent_files_for_session(session_id: str | None) -> None:
    """Drop session dedup state when the session adapter is cleaned up."""
    sid = (session_id or "").strip() or "default"
    _SENT_FILE_PATHS_BY_SESSION.pop(sid, None)
    _MATERIALIZED_TARGETS_BY_SESSION.pop(sid, None)


def _is_team_session(session_id: str) -> bool:
    """按会话元数据的锁定 mode 判定团队会话（leader/成员全覆盖）。

    不靠 ``get_member_id()``：leader 协程读不到 member_id。metadata.json 有
    进程内缓存，单次发送一次读，成本可忽略；任何读取失败都退化为非团队
    （不实体化，保持旧行为），绝不阻断发送。
    """
    sid = (session_id or "").strip()
    if not sid:
        return False
    try:
        from jiuwenswarm.common.mode_matrix import is_team_mode
        from jiuwenswarm.server.runtime.session.session_metadata import (
            get_session_metadata,
        )

        meta = get_session_metadata(sid, enable_writeback=False)
        return is_team_mode(meta.get("mode"))
    except Exception as exc:
        logger.debug("[SendFileToolkit] team 判定失败，按非团队处理: %s", exc)
        return False


def _resolve_materialize_root() -> str | None:
    """实体化目标根：project_root（身份锚点，装配期=project_dir）优先。

    不用 get_cwd() 作主锚——cwd 随成员 shell ``cd`` 高频漂移，会把文件落进
    子目录。两者皆空或目录不存在时返回 None（调用方跳过实体化）。
    """
    try:
        from openjiuwen.core.sys_operation.cwd import get_cwd, get_project_root

        root = (get_project_root() or "").strip() or (get_cwd() or "").strip()
    except Exception:
        return None
    if root and os.path.isdir(root):
        return root
    return None


def _paths_equal(a: str, b: str) -> bool:
    return _normalize_sent_file_path(a) == _normalize_sent_file_path(b)


def _path_under(root: str, path: str) -> bool:
    try:
        from pathlib import Path

        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _write_bytes_if_changed(target: str, content: bytes) -> None:
    """写入 target；已存在且内容一致则跳过（同名同内容复用）。"""
    if os.path.isfile(target):
        try:
            with open(target, "rb") as f:
                if f.read() == content:
                    return
        except OSError:
            pass
    with open(target, "wb") as f:
        f.write(content)


def _materialize_into_root(
        source: str,
        content: bytes,
        session_id: str,
        root: str,
) -> str:
    """发送即实体化：确保文件在 project_dir 树根下存在一份，返回目标路径。

    - 源已在根树下 → 快路径，原样返回（不复制）；
    - 同源重发 → 覆盖写上次的实体化目标（用户要最新版，不是版本坟场）；
    - 异源撞名 → 内容一致复用，否则 ``name (N).ext`` 后缀（与前端
      stripCollisionSuffix 归一规则兼容）；
    - 任何写入失败 → 返回原源路径，退化为不实体化（不阻断发送）。
    """
    if _path_under(root, source):
        return source
    sid = (session_id or "").strip() or "default"
    registry = _MATERIALIZED_TARGETS_BY_SESSION.setdefault(sid, {})
    norm_source = _normalize_sent_file_path(source)
    previous = registry.get(norm_source)
    try:
        if previous:
            _write_bytes_if_changed(previous, content)
            return previous
        base = os.path.basename(source) or "artifact"
        stem, ext = os.path.splitext(base)
        candidate = os.path.join(root, base)
        if _paths_equal(candidate, source):
            return source
        suffix = 0
        while os.path.isfile(candidate):
            # 撞名：内容一致直接复用；否则 (N) 后缀找下一个空位
            try:
                with open(candidate, "rb") as f:
                    if f.read() == content:
                        registry[norm_source] = candidate
                        return candidate
            except OSError:
                pass
            suffix += 1
            candidate = os.path.join(root, f"{stem} ({suffix}){ext}")
        with open(candidate, "wb") as f:
            f.write(content)
        registry[norm_source] = candidate
        logger.info(
            "[SendFileToolkit] 实体化发送产物 session_id=%s source=%s -> %s",
            session_id,
            source,
            candidate,
        )
        return candidate
    except OSError as exc:
        logger.warning(
            "[SendFileToolkit] 实体化失败，按原路径发送: source=%s error=%s",
            source,
            exc,
        )
        return source


def _tool_fail(message: str) -> str:
    """失败信封，供 stream_event_rail / 桌面步骤卡认 is_error。

    不要带 data=None：桌面把 success=False + data=None + 空 error 当成 HITL 占位
    （步骤卡停在「运行中」）。多行 error 还会让空 error 正则误判。
    """
    escaped = message.replace("\\", "\\\\").replace("'", "\\'")
    return f"success=False error='{escaped}'"


class SendFileToolkit:
    """Toolkit for sending files to users."""

    def __init__(
        self,
        request_id: str,
        session_id: str,
        channel_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        operation_provider: OperationProvider | None = None,
    ) -> None:
        """Initialize SendFileToolkit.

        Args:
            request_id: Request identifier for message routing.
            session_id: Session identifier for message routing.
            channel_id: Channel identifier for message routing.
            metadata: 与 AgentRequest.metadata 一致（E2A channel_context 映射结果），用于 send_push。
        """
        self._operation_provider = operation_provider
        self.request_id = request_id
        self.session_id = session_id
        self.channel_id = channel_id
        self._request_metadata = dict(metadata) if metadata else None
        logger.debug(
            "[SendFileToolkit] 初始化 request_id=%s session_id=%s channel_id=%s has_metadata=%s",
            request_id,
            session_id,
            channel_id,
            bool(self._request_metadata),
        )

    def update_runtime_context(
        self,
        *,
        request_id: str,
        session_id: str,
        channel_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Cached for send-path fallback when InvocationContext is unbound.
        self.request_id = request_id
        self.session_id = session_id
        self.channel_id = channel_id
        self._request_metadata = dict(metadata) if metadata else None
        logger.debug(
            "[SendFileToolkit] update_runtime_context session_id=%s channel_id=%s has_metadata=%s",
            session_id,
            channel_id,
            bool(self._request_metadata),
        )

    @staticmethod
    def _normalize_target_channels(target_channels: Any) -> list[str]:
        """Normalize target_channels into a list of non-empty strings.

        Accepts a single string, a JSON array string, or a native list.
        Returns [] when absent/empty.
        """
        if target_channels is None:
            return []
        if isinstance(target_channels, str):
            stripped = target_channels.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
                if isinstance(parsed, str):
                    return [parsed.strip()] if parsed.strip() else []
                return [stripped]
            except (TypeError, ValueError):
                return [stripped]
        if isinstance(target_channels, (list, tuple)):
            return [str(x).strip() for x in target_channels if str(x).strip()]
        return [str(target_channels).strip()]

    async def send_file(
        self,
        abs_file_path_list: Union[List[str], str],
        target_channels: Union[List[str], str, None] = None,
        **_ignored: Any,
    ) -> str:
        """Send files to user.

        Args:
            abs_file_path_list: List of absolute file paths to send.
            target_channels: Optional explicit delivery targets. Each item is
                a channel id (e.g. "feishu", "web") or a team human-agent
                seat name (the member_name used in /join). When omitted the
                Gateway auto-routes the file to all channels joined to the
                session (team mode). When provided, the file is delivered
                only to the specified targets.

        Returns:
            Success message or error description.
        """
        target_channel_list = SendFileToolkit._normalize_target_channels(target_channels)
        if target_channel_list:
            logger.info(
                "[SendFileToolkit] send_file target_channels=%s session_id=%s",
                target_channel_list, self.session_id,
            )
        # 数组 / JSON 数组串 / 单引号字面量串 / 单路径字符串 → 统一成 list[str]
        abs_file_path_list = _normalize_abs_file_path_list(abs_file_path_list)

        valid_files = []
        missing_files = []
        contents: dict[str, bytes] = {}
        try:
            operation = self._operation_provider() if self._operation_provider else None
            for fp in abs_file_path_list:
                fp = str(resolve_outbound_path(fp))
                try:
                    contents[fp] = await read_outbound_file(fp, operation)
                except FileNotFoundError:
                    missing_files.append(fp)
                    logger.warning("[SendFileToolkit] 文件不存在: %s", fp)
                    continue
                valid_files.append(fp)
        except Exception as exc:
            return _tool_fail(f"Failed to send files: {exc}")

        if not valid_files:
            msg_parts = ["Failed to send files: none of the files exist"]
            for mf in missing_files:
                msg_parts.append(f"  - {mf}")
            return _tool_fail("\n".join(msg_parts))

        delivery = resolve_toolkit_delivery(self)

        # 团队模式「发送即实体化」：凡发送的产物（含中间产物）必须在任务工作区
        # （project_root）存在一份，展示/身份/去重一律以实体化路径为准。
        # 非团队会话、无 project_root、写入失败均退化为源路径（旧行为）。
        source_paths: dict[str, str] = {fp: fp for fp in valid_files}
        if _is_team_session(delivery.session_id):
            root = _resolve_materialize_root()
            if root:
                for fp in valid_files:
                    source_paths[fp] = await asyncio.to_thread(
                        _materialize_into_root, fp, contents[fp], delivery.session_id, root,
                    )

        # 指纹去重：键 = (实体化路径, size, 内容 hash)。内容变 → 放行重发；
        # 内容未变 → 跳过（防压缩丢历史后的重复推送）。
        fingerprint_items = [
            (source_paths[fp], _sent_fingerprint(source_paths[fp], contents[fp]))
            for fp in valid_files
        ]
        sendable, skipped_files = _partition_sent_files(delivery.session_id, fingerprint_items)
        valid_files = [fp for fp in valid_files if source_paths[fp] in set(sendable)]
        if not valid_files:
            logger.info(
                "[SendFileToolkit] skip duplicate send session_id=%s skipped=%s missing=%s",
                delivery.session_id,
                skipped_files,
                missing_files,
            )
            msg_parts: list[str] = []
            if skipped_files:
                msg_parts.append("Files already sent in this session; skipping duplicate delivery:")
                for sf in skipped_files:
                    msg_parts.append(f"  - {sf}")
            if missing_files:
                msg_parts.append("The following files do not exist and were not sent:")
                for mf in missing_files:
                    msg_parts.append(f"  - {mf}")
            if not msg_parts:
                msg_parts.append("No files available to send")
            return "\n".join(msg_parts)

        logger.info(
            "[SendFileToolkit] send_file 开始 session_id=%s turn_request_id=%s 有效文件=%d 缺失=%d 跳过重复=%d",
            delivery.session_id,
            delivery.turn_request_id,
            len(valid_files),
            len(missing_files),
            len(skipped_files),
        )

        try:
            # Access has been checked through the owning agent's SysOperation.
            # Deliver the live source path; no outbound snapshot or version cache.
            from jiuwenswarm.server.agent_ws_server import AgentWebSocketServer

            server = AgentWebSocketServer.get_instance()

            files_payload = []
            try:
                from jiuwenswarm.agents.harness.common.tools.web_file_download import (
                    build_file_download_info,
                )

                for file_path in valid_files:
                    source_path = source_paths[file_path]
                    base_name = os.path.basename(source_path)
                    download_info = build_file_download_info(
                        source_path, base_name, delivery.session_id, approved_outbound=True
                    )
                    files_payload.append({
                        "path": source_path,
                        # 展示与下载共用源路径（团队为实体化后的工作区路径）。
                        "source_path": source_path,
                        "name": base_name,
                        # 产物稳定标识：桌面端「对话卡片」与「右侧工作台产物」共用同一个键去重
                        "artifact_id": artifact_id_for_path(source_path),
                        "size": download_info["size"],
                        "mime_type": download_info["mime_type"],
                        "download_url": download_info["download_url"],
                        "download_token": download_info["download_token"],
                    })
            except Exception as download_err:
                logger.warning(
                    "[SendFileToolkit] 生成下载信息失败，回退到基础模式: %s",
                    download_err,
                )
                files_payload = [
                    {
                        "path": source_paths[file_path],
                        "source_path": source_paths[file_path],
                        "name": os.path.basename(source_paths[file_path]),
                        "artifact_id": artifact_id_for_path(source_paths[file_path]),
                    }
                    for file_path in valid_files
                ]

            import time
            from jiuwenswarm.server.runtime.session.session_history import (
                append_history_record,
            )
            # 团队模式下成员经 send_file 直推（不经 team 流，吃不到 team_helpers 的
            # 成员归因）——从成员执行上下文读 member_name（agent_teams 在成员协程内
            # set_member_id），让推送帧与落盘记录都能归到成员子聊天框。
            # leader / 非 team 会话读不到（""），行为与之前完全一致。
            member_name = ""
            try:
                from openjiuwen.core.common.logging import get_member_id

                member_name = str(get_member_id() or "").strip()
            except Exception as exe:
                member_name = ""
                logger.warning(
                    "[SendFileToolkit] 获取 member_name 失败，不阻断发文件: %s",
                    exe,
                )
            history_extra: dict[str, Any] = {"files": files_payload}
            if member_name:
                history_extra["member_name"] = member_name
            append_history_record(
                session_id=delivery.session_id,
                request_id=delivery.turn_request_id,
                channel_id=delivery.channel_id,
                role="assistant",
                event_type="chat.file",
                content="",
                timestamp=time.time(),
                extra=history_extra,
                mode="team" if member_name else None,
            )

            msg = {
                "request_id": delivery.turn_request_id,
                "channel_id": delivery.channel_id,
                "session_id": delivery.session_id,
                "payload": {
                    "event_type": "chat.file",
                    "files": files_payload,
                    **({"member_name": member_name} if member_name else {}),
                },
                "is_complete": False,
            }
            # 合并 metadata：原始 request metadata + 文件投递目标提示。
            # send_file_targets 由 Gateway 的 dispatch 层解析为 fan_out_targets，
            # 使文件可跨 channel 投递到 team 会话已接入的 channel（如飞书）。
            # xiaoyi_task_id 只留在 metadata，给手机 A2A；history / Message.id 用本轮 turn id。
            merged_meta: dict[str, Any] = dict(delivery.metadata)
            if target_channel_list:
                merged_meta["send_file_targets"] = list(target_channel_list)
            if merged_meta:
                msg["metadata"] = merged_meta
            await server.send_push(msg)
            sent_items = [
                (source_paths[fp], _sent_fingerprint(source_paths[fp], contents[fp]))
                for fp in valid_files
            ]
            _mark_files_sent(delivery.session_id, sent_items)
            result_parts = [f"Sent {len(valid_files)} files"]
            if skipped_files:
                result_parts.append("The following files were already sent in this session and were skipped:")
                for sf in skipped_files:
                    result_parts.append(f"  - {sf}")
            if missing_files:
                result_parts.append("The following files do not exist and were not sent:")
                for mf in missing_files:
                    result_parts.append(f"  - {mf}")
            return "\n".join(result_parts)
        except Exception as e:
            logger.exception(
                "[SendFileToolkit] send_file 失败 session_id=%s error=%s",
                self.session_id,
                str(e),
            )
            return _tool_fail(f"Failed to submit files: {str(e)}")

    def get_tools(self) -> List[Tool]:
        """Return tools for registration in Runner.

        Returns:
            List of tools for sending files.
        """

        def make_tool(
            name: str,
            description: str,
            input_params: dict,
            func,
        ) -> Tool:
            card = ToolCard(
                name=name,
                description=description,
                input_params=input_params,
            )
            return LocalFunction(card=card, func=func)

        return [
            make_tool(
                name="send_file_to_user",
                description=(
                    "[MANDATORY] Every time a deliverable (file, document, image, video, "
                    "music, etc.) is created, modified, downloaded, or renamed during the "
                    "task, you MUST call this tool to send it to the user; do not skip it. "
                    "[Deliverable sending tool] Use this tool whenever generated, downloaded, "
                    "exported, created, modified, or renamed documents, images, videos, music, "
                    "or other deliverables need to be sent to the user. Typical scenarios: the "
                    "user asks to export/download files, deliverables must be handed over after "
                    "the task, or a report/document has been generated and should be sent. "
                    "Parameter format: put the file(s) in abs_file_path_list as a JSON array of "
                    "absolute paths, e.g. [\"/tmp/report.pdf\"] or "
                    "[\"/tmp/file1.csv\", \"/tmp/file2.xlsx\"]; a single absolute path string "
                    "like \"/tmp/report.pdf\" is also accepted when there is exactly one file. "
                    "Paths must be absolute, and quotes inside the JSON array must be double "
                    "quotes (do not use single quotes or Python-style list syntax). "
                    "When several deliverables are ready at the same time, send them in ONE call "
                    "instead of calling this tool once per file. "
                    "target_channels is optional: delivery targets for the files; each item "
                    "can be a channel id (e.g. 'web') or a team human seat name "
                    "(e.g. 'human-player-1'); a single channel id string is also accepted. "
                    "When omitted, files are delivered to the human member who issued the most "
                    "recent request (the initiator recorded on the session); if the request came "
                    "from web or no human member exists, deliver to web. "
                    "To target a specific feishu user in multi-app scenarios, pass that user's "
                    "member_name (avoids mis-delivery to other apps); "
                    "for cross-device delivery (e.g. sending files to a feishu user or to web), "
                    "pass the corresponding member_name or 'web'."
                ),
                input_params={
                    "type": "object",
                    "properties": {
                        "abs_file_path_list": {
                            # 两种形态都收：模型常直接给数组，声明成 string 会在参数校验层
                            # 直接以 [189001] 失败（tools 实现本来就有 list 分支）。
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ],
                            "description": (
                                "Absolute path(s) of the files to send. "
                                "Prefer a JSON array of absolute paths, e.g. "
                                "[\"/path/file1.csv\", \"/path/file2.xlsx\"] "
                                "(double quotes only, no single quotes / Python list syntax); "
                                "a single absolute path string such as "
                                "\"/path/to/file.pdf\" is also accepted. "
                                "Any file type is supported (pdf, xlsx, docx, png, zip, etc.)."
                            ),
                        },
                        "target_channels": {
                            # 实现侧 _normalize_target_channels 字符串/数组都吃，声明保持一致
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            ],
                            "description": (
                                "Optional: delivery target list. Each item can be a channel id "
                                "(e.g. 'web') or a team human seat name (e.g. 'human-player-1'). "
                                "A single channel id string (e.g. 'web') is also accepted. "
                                "When omitted, delivered to the human member who issued the most "
                                "recent request; if the request came from web or no human member "
                                "exists, deliver to web. "
                                "Pass a feishu user's member_name to target that user; for "
                                "cross-device delivery pass the corresponding member_name or 'web'."
                            ),
                        },
                    },
                    "required": ["abs_file_path_list"],
                },
                func=self.send_file,
            ),
        ]
