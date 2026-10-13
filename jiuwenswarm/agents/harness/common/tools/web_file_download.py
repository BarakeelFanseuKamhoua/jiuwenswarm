# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Web File Download Token Manager

提供基于 HMAC 签名的文件下载令牌生成与验证，支持跨进程（AgentServer / Gateway / app_web.py）
无需共享内存即可安全校验。

协议：
- 令牌格式: Base64URL(payload_json) + "." + Hex(HMAC-SHA256)
- payload 包含: path, exp, session_id
- 密钥来源: 环境变量 JIUWENSWARM_FILE_DOWNLOAD_SECRET 或自动生成并写入共享文件
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

_DEFAULT_EXPIRES_SECONDS = 600
_SECRET_ENV_KEY = "JIUWENSWARM_FILE_DOWNLOAD_SECRET"
_SECRET_FILE_NAME = ".file_download_secret"
OUTBOUND_DOWNLOAD_PURPOSE = "approved-outbound-file"
_SESSION_DOWNLOAD_KEY_FILE = ".outbound_download_key"


def _session_download_generation(session_id: str, *, create: bool = False) -> str:
    """Bind downloads to one session lifetime, without a revocation database.

    Never create a session here. An atomic hard link publishes a complete random
    key so concurrent issuers share one generation, even across processes. Only
    its digest is included in tokens; deleting the session removes the key.
    """
    from jiuwenswarm.common.utils import is_persistable_session_id
    from jiuwenswarm.server.runtime.session.session_history import resolve_session_dir

    if not isinstance(session_id, str) or not is_persistable_session_id(session_id):
        raise ValueError("outbound download requires an existing session")
    session_dir, _ = resolve_session_dir(session_id)
    if session_dir is None or not session_dir.is_dir():
        raise ValueError("outbound download requires an existing session")
    key_path = session_dir / _SESSION_DOWNLOAD_KEY_FILE

    def read_generation() -> str:
        with open_outbound_download(str(key_path)) as stream:
            key = stream.read(33)
        if len(key) != 32:
            raise ValueError("invalid session download key")
        return hashlib.sha256(key).hexdigest()

    try:
        return read_generation()
    except FileNotFoundError:
        if not create:
            raise
    # Do not replace an existing key: that would revoke concurrent deliveries.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=session_dir, prefix=".download-key-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(secrets.token_bytes(32))
        try:
            os.link(temporary, key_path)
        except FileExistsError:
            pass
        return read_generation()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _get_secret_file_path() -> Path:
    workspace = os.getenv("JIUWENSWARM_WORKSPACE")
    if workspace:
        return Path(workspace) / "config" / _SECRET_FILE_NAME
    return Path.home() / ".jiuwenswarm" / "config" / _SECRET_FILE_NAME


def _load_or_create_secret() -> str:
    secret = os.getenv(_SECRET_ENV_KEY)
    if secret and len(secret) >= 32:
        return secret

    secret_file = _get_secret_file_path()
    try:
        if secret_file.exists():
            existing = secret_file.read_text(encoding="utf-8").strip()
            if existing and len(existing) >= 32:
                return existing
    except Exception:
        logger.debug("[WebFileDownload] 读取密钥文件失败，将重新生成")

    new_secret = secrets.token_hex(32)
    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        secret_file.write_text(new_secret, encoding="utf-8")
        os.chmod(secret_file, 0o600)
    except Exception:
        logger.warning("[WebFileDownload] 写入密钥文件失败，使用内存密钥（重启后失效）")

    return new_secret


class WebFileDownloadManager:
    """管理 Web 端文件下载令牌的生成与验证。

    使用 HMAC-SHA256 签名保证令牌不可伪造，
    密钥通过环境变量或共享文件在 AgentServer / Gateway / app_web.py 间共享。
    """

    _instance: WebFileDownloadManager | None = None

    def __init__(self, secret: str | None = None) -> None:
        self._secret = secret or _load_or_create_secret()

    @classmethod
    def get_instance(cls) -> WebFileDownloadManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def generate_token(
        self,
        file_path: str,
        session_id: str = "",
        expires_in: int = _DEFAULT_EXPIRES_SECONDS,
        *,
        approved_outbound: bool = False,
    ) -> str:
        payload = {
            "path": file_path,
            "exp": int(time.time()) + expires_in,
            "sid": session_id,
        }
        if approved_outbound:
            # The caller must have checked this exact path through SysOperation.
            # Do not resolve again here: a replaced link must not gain authority.
            if not Path(file_path).is_absolute():
                raise ValueError("outbound download requires an absolute path")
            payload["purpose"] = OUTBOUND_DOWNLOAD_PURPOSE
            payload["session_generation"] = _session_download_generation(session_id, create=True)
        payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        payload_b64 = base64.urlsafe_b64encode(
            payload_json.encode("utf-8")
        ).decode("ascii")
        signature = hmac.new(
            self._secret.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload_b64}.{signature}"

    def validate_token(self, token: str) -> dict[str, Any] | None:
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            payload_b64, signature = parts
            expected_sig = hmac.new(
                self._secret.encode("utf-8"),
                payload_b64.encode("ascii"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                logger.warning("[WebFileDownload] 令牌签名校验失败")
                return None
            payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):
                return None
            if payload.get("purpose") == OUTBOUND_DOWNLOAD_PURPOSE:
                generation = payload.get("session_generation")
                if not isinstance(generation, str) or not hmac.compare_digest(
                    generation, _session_download_generation(payload.get("sid")),
                ):
                    return None
            return payload
        except Exception:
            logger.debug("[WebFileDownload] 令牌解析异常", exc_info=True)
            return None

    @staticmethod
    def generate_download_url(token: str) -> str:
        return f"/file-api/download?token={token}"


def generate_file_download_token(
    file_path: str,
    session_id: str = "",
    expires_in: int = _DEFAULT_EXPIRES_SECONDS,
    *,
    approved_outbound: bool = False,
) -> str:
    return WebFileDownloadManager.get_instance().generate_token(
        file_path, session_id, expires_in, approved_outbound=approved_outbound
    )


def validate_file_download_token(token: str) -> dict[str, Any] | None:
    return WebFileDownloadManager.get_instance().validate_token(token)


def build_file_download_info(
    file_path: str,
    file_name: str,
    session_id: str = "",
    expires_in: int = _DEFAULT_EXPIRES_SECONDS,
    *,
    approved_outbound: bool = False,
) -> dict[str, Any]:
    token = generate_file_download_token(
        file_path, session_id, expires_in, approved_outbound=approved_outbound
    )
    download_url = WebFileDownloadManager.get_instance().generate_download_url(token)

    file_size = 0
    mime_type = "application/octet-stream"
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        pass

    import mimetypes

    guessed_type, _ = mimetypes.guess_type(file_name)
    if guessed_type:
        mime_type = guessed_type

    return {
        "name": file_name,
        "size": file_size,
        "mime_type": mime_type,
        "download_url": download_url,
        "download_token": token,
    }


def open_outbound_download(file_path: str) -> BinaryIO:
    """Open the signed live path without allowing link redirection.

    Validate the Windows handle before reading; on POSIX walk directory handles
    with O_NOFOLLOW. The caller must stream this same handle, never reopen by path.
    """
    path = Path(file_path)
    if not path.is_absolute() or ".." in path.parts:
        raise PermissionError("forbidden_path")
    fd = None
    try:
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            fd = os.open(path, os.O_RDONLY | os.O_BINARY)
            get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
            get_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
            get_path.restype = wintypes.DWORD
            buffer = ctypes.create_unicode_buffer(32768)
            length = get_path(msvcrt.get_osfhandle(fd), buffer, len(buffer), 0)
            if not length or length >= len(buffer):
                raise PermissionError("forbidden_path")

            def normalized(value: str) -> str:
                if value.startswith("\\\\?\\UNC\\"):
                    value = "\\\\" + value[8:]
                elif value.startswith("\\\\?\\"):
                    value = value[4:]
                return os.path.normcase(os.path.normpath(value))

            if normalized(buffer.value) != normalized(file_path):
                raise PermissionError("forbidden_path")
        else:
            directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
            try:
                for part in path.parts[1:-1]:
                    next_directory = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory,
                    )
                    os.close(directory)
                    directory = next_directory
                fd = os.open(
                    path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory,
                )
            finally:
                os.close(directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PermissionError("forbidden_path")
        stream = os.fdopen(fd, "rb")
        fd = None
        return stream
    finally:
        if fd is not None:
            os.close(fd)
