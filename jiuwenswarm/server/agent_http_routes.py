# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""HTTP 路由。每条路径直接调用一个处理函数，不查 WebSocket 的分发表。

第一期只开放列 Agent 和聊天。别的命令还走 WebSocket。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from jiuwenswarm.common.log_context import (
    bind_log_session,
    current_log_session_id,
    unbind_log_session,
)
from jiuwenswarm.common.schema.agent import AgentRequest
from jiuwenswarm.common.schema.message import ReqMethod
from jiuwenswarm.common.ws_limits import AGENT_WS_MAX_MESSAGE_BYTES
from jiuwenswarm.server import http_chat
from jiuwenswarm.server.app_services import AppServices
from jiuwenswarm.server.context import AgentServerServices, RequestContext
from jiuwenswarm.server.handlers import agents as agent_handlers
from jiuwenswarm.server.http_invoke import invoke_stream, invoke_unary
from jiuwenswarm.server.transports.http_sink import SSESink, UnaryHTTPSink

logger = logging.getLogger(__name__)

HTTP_CONNECTION_ID = "http"
API_PREFIX = "/api/v1"


class _RequestBodyTooLarge(Exception):
    """请求体超过和 WebSocket 相同的单条上限。"""


class RequestBodyLimitMiddleware:
    """请求体超过 WebSocket 单条上限时直接拒绝，避免整段读进内存。"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > AGENT_WS_MAX_MESSAGE_BYTES:
            await _payload_too_large()(scope, receive, send)
            return
        try:
            body = await _read_bounded_body(receive)
        except _RequestBodyTooLarge:
            await _payload_too_large()(scope, receive, send)
            return
        if body is None:
            return
        await self.app(scope, _replay_body(body), send)


class SessionLogMiddleware(BaseHTTPMiddleware):
    """每条网页请求进来时，把日志绑到这次的会话上，离开时解开。"""

    async def dispatch(self, request: Request, call_next):
        session_id = await _session_id_from_request(request)
        token, previous_trace = bind_log_session(session_id)
        logger.info(
            "[AgentHTTPServer] request arrived: method=%s path=%s session_id=%s",
            request.method,
            request.url.path,
            current_log_session_id(),
        )
        try:
            return await call_next(request)
        finally:
            unbind_log_session(token, previous_trace)


def build_http_app(app_services: AppServices) -> FastAPI:
    """造出 HTTP 应用。业务函数通过工具箱拿服务，不接触 WebSocket 服务器。"""
    app = FastAPI(title="JiuwenSwarm HTTP", docs_url=None, redoc_url=None)
    app.add_middleware(SessionLogMiddleware)
    # 后加的包在最外层，先拦超大请求体，再记日志。
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.post(f"{API_PREFIX}/agents/list")
    async def agents_list(request: Request) -> JSONResponse:
        ctx = await _context_from_body(
            app_services,
            request,
            req_method=ReqMethod.AGENTS_LIST,
            sink=UnaryHTTPSink(),
        )
        await invoke_unary(ctx, agent_handlers.handle_agents_list)
        return JSONResponse(ctx.sink.body or {}, status_code=ctx.sink.status_code)

    @app.post(f"{API_PREFIX}/chat")
    async def chat(request: Request):
        body = await _read_json(request)
        is_stream = bool(body.get("is_stream"))
        if is_stream:
            sink = SSESink()
            ctx = _context_from_mapping(
                app_services,
                body,
                req_method=_chat_method(body),
                sink=sink,
                is_stream=True,
            )

            async def _frames():
                task = asyncio.create_task(invoke_stream(ctx, http_chat.handle_http_chat))
                try:
                    async for frame in sink.frames():
                        yield frame
                finally:
                    await _stop_background_chat(task)

            return StreamingResponse(_frames(), media_type="text/event-stream")

        sink = UnaryHTTPSink()
        ctx = _context_from_mapping(
            app_services,
            body,
            req_method=_chat_method(body),
            sink=sink,
            is_stream=False,
        )
        await invoke_unary(ctx, http_chat.handle_http_chat)
        return JSONResponse(ctx.sink.body or {}, status_code=ctx.sink.status_code)

    return app


async def _stop_background_chat(task: asyncio.Task[None]) -> None:
    """流结束时等后台聊天收尾；客户端断开时先把它停掉。

    Python 3.11 起，当前任务已被取消时，``finally`` 里的 ``await`` 会马上再抛
    「已取消」，后台任务就没人取消、也没人等待。先把这次取消暂存起来，等后台
    停稳，再把取消还回去，让这次网页请求仍然按断开结束。

    正常读完时不要取消。取消只发生在请求本身已被取消，或生成器被关掉
    （客户端中途离开，生成器收到 GeneratorExit）的时候。
    """
    current = asyncio.current_task()
    abandoned = sys.exception() is not None or (
        current is not None and current.cancelling() > 0
    )
    if abandoned and not task.done():
        task.cancel()
    parked = 0
    if abandoned and current is not None:
        while current.cancelling():
            current.uncancel()
            parked += 1
    try:
        await task
    except asyncio.CancelledError:
        if not abandoned:
            raise
    finally:
        if current is None:
            return
        for _ in range(parked):
            current.cancel()


def _chat_method(body: dict[str, Any]) -> ReqMethod:
    raw = str(body.get("req_method") or ReqMethod.CHAT_SEND.value)
    try:
        method = ReqMethod(raw)
    except ValueError:
        return ReqMethod.CHAT_SEND
    if method in {ReqMethod.CHAT_SEND, ReqMethod.CHAT_RESUME, ReqMethod.CHAT_ANSWER}:
        return method
    return ReqMethod.CHAT_SEND


async def _context_from_body(
    app_services: AppServices,
    request: Request,
    *,
    req_method: ReqMethod,
    sink: UnaryHTTPSink,
) -> RequestContext:
    return _context_from_mapping(
        app_services,
        await _read_json(request),
        req_method=req_method,
        sink=sink,
        is_stream=False,
    )


def _context_from_mapping(
    app_services: AppServices,
    body: dict[str, Any],
    *,
    req_method: ReqMethod,
    sink: Any,
    is_stream: bool,
) -> RequestContext:
    params = body.get("params")
    agent_request = AgentRequest(
        request_id=str(body.get("request_id") or "http"),
        channel_id=str(body.get("channel_id") or "http"),
        session_id=str(body.get("session_id") or "") or None,
        req_method=req_method,
        params=params if isinstance(params, dict) else {},
        is_stream=is_stream,
        metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
        user_id=str(body.get("user_id") or "").strip(),
    )
    return RequestContext(
        request=agent_request,
        sink=sink,
        connection_id=HTTP_CONNECTION_ID,
        services=AgentServerServices(app_services),
    )


def _content_length(scope: Scope) -> int | None:
    raw = Headers(scope=scope).get("content-length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except ValueError:
        return None
    if size < 0:
        return None
    return size


def _payload_too_large() -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "payload": {
                "code": "PAYLOAD_TOO_LARGE",
                "error": "request body exceeds the size limit",
            },
        },
        status_code=413,
    )


async def _read_bounded_body(receive: Receive) -> bytes | None:
    """按块读取。超过上限就停，不再把剩下的收进来。客户端先断开时返回 None。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        if message["type"] != "http.request":
            continue
        piece = message.get("body", b"")
        total += len(piece)
        if total > AGENT_WS_MAX_MESSAGE_BYTES:
            raise _RequestBodyTooLarge
        if piece:
            chunks.append(piece)
        if not message.get("more_body", False):
            return b"".join(chunks)


def _replay_body(body: bytes) -> Receive:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def _session_id_from_request(request: Request) -> str:
    header = str(request.headers.get("x-session-id") or "").strip()
    if header:
        return header
    query = str(request.query_params.get("session_id") or "").strip()
    if query:
        return query
    body = await _read_json(request)
    return str(body.get("session_id") or "").strip()
