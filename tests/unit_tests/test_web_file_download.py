from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from jiuwenswarm.agents.harness.common.tools import web_file_download
from jiuwenswarm.agents.harness.common.tools.web_file_download import (
    WebFileDownloadManager,
)
from jiuwenswarm.channels.web.app_web import _SpaStaticHandler


class _DownloadHandlerStub:
    def __init__(
        self,
        *,
        command: str = "HEAD",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.headers = headers or {}
        self.wfile = io.BytesIO()
        self.connection = Mock(spec=["settimeout"])
        self.status: int | None = None
        self.response_headers: dict[str, str] = {}
        self.headers_ended = False

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        self.response_headers[name] = value

    def end_headers(self) -> None:
        self.headers_ended = True

    def _write_json(self, status: int, payload: dict) -> None:
        raise AssertionError(f"unexpected JSON response: {status} {payload}")

    def log_error(self, message: str, *args: object) -> None:
        raise AssertionError(message % args)

    def _is_path_under_allowed_root(self, target: Path) -> bool:
        return True


def _serve_file(
    monkeypatch: pytest.MonkeyPatch,
    file_path: Path,
    query: dict[str, str],
    *,
    command: str = "HEAD",
    headers: dict[str, str] | None = None,
) -> _DownloadHandlerStub:
    monkeypatch.setattr(
        web_file_download,
        "validate_file_download_token",
        lambda _token: {"path": str(file_path)},
    )
    handler = _DownloadHandlerStub(command=command, headers=headers)
    _SpaStaticHandler._handle_file_download(handler, query)
    return handler


def _signed_token(secret: str, payload: object) -> str:
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    signature = hmac.new(
        secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{encoded}.{signature}"


def test_valid_token_is_accepted() -> None:
    manager = WebFileDownloadManager(secret="s" * 32)
    token = manager.generate_token("/tmp/report.xlsx", "session-1", expires_in=60)

    payload = manager.validate_token(token)
    assert payload is not None
    assert payload["path"] == "/tmp/report.xlsx"
    assert payload["exp"] == pytest.approx(int(time.time()) + 60, abs=1)
    assert payload["sid"] == "session-1"


def test_expiration_is_not_required_for_download() -> None:
    secret = "s" * 32
    manager = WebFileDownloadManager(secret=secret)
    token = _signed_token(
        secret,
        {"path": "/tmp/report.xlsx", "sid": "session-1"},
    )

    assert manager.validate_token(token) == {
        "path": "/tmp/report.xlsx",
        "sid": "session-1",
    }


def test_tampered_token_is_rejected() -> None:
    manager = WebFileDownloadManager(secret="s" * 32)
    token = manager.generate_token("/tmp/report.xlsx", expires_in=60)
    encoded, signature = token.split(".")

    assert manager.validate_token(f"{encoded}x.{signature}") is None


@pytest.mark.parametrize("inline_value", ["1", "true", "TRUE"])
def test_download_handler_uses_inline_disposition_for_preview(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    inline_value: str,
) -> None:
    file_path = tmp_path / "preview sample.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    handler = _serve_file(
        monkeypatch,
        file_path,
        {"token": "signed-token", "inline": inline_value},
    )

    assert handler.status == 200
    assert handler.headers_ended is True
    assert handler.response_headers["Content-Type"] == "application/pdf"
    assert handler.response_headers["Accept-Ranges"] == "bytes"
    assert handler.response_headers["Content-Disposition"] == (
        "inline; filename*=UTF-8''preview%20sample.pdf"
    )


@pytest.mark.parametrize(
    "query", [{"token": "signed-token"}, {"token": "signed-token", "inline": "0"}]
)
def test_download_handler_keeps_attachment_disposition_for_download(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    query: dict[str, str],
) -> None:
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    handler = _serve_file(monkeypatch, file_path, query)

    assert handler.status == 200
    assert handler.response_headers["Content-Disposition"] == (
        "attachment; filename*=UTF-8''report.pdf"
    )


@pytest.mark.parametrize(
    ("range_header", "expected_body", "expected_content_range"),
    [
        ("bytes=2-5", b"2345", "bytes 2-5/10"),
        ("bytes=7-", b"789", "bytes 7-9/10"),
        ("bytes=-3", b"789", "bytes 7-9/10"),
        ("bytes=7-20", b"789", "bytes 7-9/10"),
    ],
)
def test_download_handler_serves_single_byte_range(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    range_header: str,
    expected_body: bytes,
    expected_content_range: str,
) -> None:
    file_path = tmp_path / "media.bin"
    file_path.write_bytes(b"0123456789")
    handler = _serve_file(
        monkeypatch,
        file_path,
        {"token": "signed-token", "inline": "1"},
        command="GET",
        headers={"Range": range_header},
    )

    assert handler.status == 206
    assert handler.response_headers["Content-Length"] == str(len(expected_body))
    assert handler.response_headers["Accept-Ranges"] == "bytes"
    assert handler.response_headers["Content-Range"] == expected_content_range
    assert handler.wfile.getvalue() == expected_body


@pytest.mark.parametrize(
    "range_header",
    [
        "items=0-1",
        "bytes=10-12",
        "bytes=3-2",
        "bytes=-0",
        "bytes=0-1,3-4",
    ],
)
def test_download_handler_rejects_invalid_byte_range(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    range_header: str,
) -> None:
    file_path = tmp_path / "media.bin"
    file_path.write_bytes(b"0123456789")
    handler = _serve_file(
        monkeypatch,
        file_path,
        {"token": "signed-token", "inline": "1"},
        command="GET",
        headers={"Range": range_header},
    )

    assert handler.status == 416
    assert handler.response_headers["Content-Range"] == "bytes */10"
    assert handler.wfile.getvalue() == b""


def test_download_handler_rejects_path_outside_allowed_root(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "secret.txt"
    file_path.write_text("secret", encoding="utf-8")

    class _RejectingStub(_DownloadHandlerStub):
        def _is_path_under_allowed_root(self, target: Path) -> bool:
            return False

        def _write_json(self, status: int, payload: dict) -> None:
            self.status = status
            self.json_payload = payload

    monkeypatch.setattr(
        web_file_download,
        "validate_file_download_token",
        lambda _token: {"path": str(file_path)},
    )
    handler = _RejectingStub()
    _SpaStaticHandler._handle_file_download(handler, {"token": "signed-token"})

    assert handler.status == 403
    assert handler.json_payload == {"error": "forbidden_path"}
    assert str(file_path) not in str(handler.json_payload)


@pytest.fixture
def signed_download(tmp_path, monkeypatch):
    """Exercise real signatures and the production root predicate together."""
    from jiuwenswarm.channels.web import app_web

    sessions = tmp_path / "sessions"
    (sessions / "session-1").mkdir(parents=True)
    monkeypatch.setattr(
        "jiuwenswarm.server.runtime.session.session_history.get_agent_sessions_dir", lambda: sessions,
    )
    manager = WebFileDownloadManager(secret="download-test-" * 4)
    monkeypatch.setattr(WebFileDownloadManager, "_instance", manager)
    monkeypatch.setattr(app_web, "_web_file_limit_bytes", lambda: 1024)
    source = tmp_path / "user-project" / "report.txt"
    source.parent.mkdir()
    source.write_bytes(b"0123456789")

    class Handler(_DownloadHandlerStub):
        workspace_root = tmp_path / "workspace"
        agent_teams_root = tmp_path / "teams"
        logs_root = tmp_path / "logs"
        auto_harness_root = tmp_path / "harness"
        _is_path_under_allowed_root = classmethod(
            _SpaStaticHandler._is_path_under_allowed_root.__func__
        )

        def _write_json(self, status, payload):
            self.status = status
            self.json_payload = payload

    def serve(token, *, command="GET", headers=None, **query):
        handler = Handler(command=command, headers=headers)
        _SpaStaticHandler._handle_file_download(handler, {"token": token, **query})
        return handler

    return manager, source, serve


@pytest.mark.parametrize("approved", [False, True])
def test_signed_external_download_requires_outbound_purpose(signed_download, approved):
    manager, source, serve = signed_download
    info = web_file_download.build_file_download_info(
        str(source), source.name, "session-1", approved_outbound=approved,
    )
    assert info["download_url"].endswith(info["download_token"])
    payload = manager.validate_token(info["download_token"])
    assert ("purpose" in payload) is approved
    # Query parameters cannot grant the exception.
    handler = serve(info["download_token"], purpose=web_file_download.OUTBOUND_DOWNLOAD_PURPOSE)
    assert handler.status == (200 if approved else 403)
    assert handler.wfile.getvalue() == (b"0123456789" if approved else b"")


def test_ordinary_signed_download_still_serves_allowed_root(signed_download, tmp_path):
    manager, _, serve = signed_download
    source = tmp_path / "workspace" / "normal.txt"
    source.parent.mkdir()
    source.write_bytes(b"normal")
    handler = serve(manager.generate_token(str(source)))
    assert handler.status == 200
    assert handler.wfile.getvalue() == b"normal"


@pytest.mark.parametrize("field", ["path", "purpose"])
def test_outbound_path_and_purpose_cannot_be_forged(signed_download, field):
    manager, source, serve = signed_download
    token = manager.generate_token(str(source), "session-1", approved_outbound=(field == "path"))
    encoded, signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(encoded))
    payload[field] = str(source.parent / "secret.txt") if field == "path" else web_file_download.OUTBOUND_DOWNLOAD_PURPOSE
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    handler = serve(f"{encoded}.{signature}")
    assert handler.status == 403
    assert handler.wfile.getvalue() == b""


@pytest.mark.parametrize("command,headers,status,body", [
    ("GET", {}, 413, b""),
    ("HEAD", {}, 413, b""),
    ("GET", {"Range": "bytes=2-4"}, 206, b"234"),
    ("HEAD", {"Range": "bytes=2-4"}, 206, b""),
    ("GET", {"Range": "bytes=30-40"}, 416, b""),
])
def test_outbound_download_keeps_limits_and_ranges(
    signed_download, monkeypatch, command, headers, status, body,
):
    from jiuwenswarm.channels.web import app_web

    manager, source, serve = signed_download
    monkeypatch.setattr(app_web, "_web_file_limit_bytes", lambda: 4)
    token = manager.generate_token(str(source), "session-1", approved_outbound=True)
    handler = serve(token, command=command, headers=headers)
    assert handler.status == status
    assert handler.wfile.getvalue() == body


def test_outbound_download_missing_source_returns_404(signed_download):
    manager, source, serve = signed_download
    token = manager.generate_token(str(source), "session-1", approved_outbound=True)
    source.unlink()
    assert serve(token).status == 404


@pytest.mark.parametrize("redirect_parent", [False, True])
def test_outbound_download_rejects_replaced_symlink(signed_download, tmp_path, redirect_parent):
    manager, source, serve = signed_download
    token = manager.generate_token(str(source), "session-1", approved_outbound=True)
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret = secret_dir / source.name
    secret.write_bytes(b"secret")
    source.unlink()
    try:
        if redirect_parent:
            source.parent.rmdir()
            source.parent.symlink_to(secret_dir, target_is_directory=True)
        else:
            source.symlink_to(secret)
    except OSError:
        pytest.skip("symlink is not permitted in this environment")
    handler = serve(token)
    assert handler.status == 403
    assert handler.wfile.getvalue() == b""


@pytest.mark.skipif(os.name != "nt", reason="Windows junction")
def test_outbound_download_rejects_replaced_junction(signed_download, tmp_path):
    import _winapi

    manager, source, serve = signed_download
    token = manager.generate_token(str(source), "session-1", approved_outbound=True)
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / source.name).write_bytes(b"secret")
    source.unlink()
    source.parent.rmdir()
    _winapi.CreateJunction(str(secret_dir), str(source.parent))
    try:
        handler = serve(token)
        assert handler.status == 403
        assert handler.wfile.getvalue() == b""
    finally:
        # Remove the junction itself, leaving its target intact.
        os.rmdir(source.parent)


def test_session_deletion_revokes_only_its_downloads(signed_download, tmp_path):
    manager, source, serve = signed_download
    session = tmp_path / "sessions" / "session-1"
    other = session.parent / "session-2"
    other.mkdir()
    token = manager.generate_token(str(source), session.name, approved_outbound=True)
    other_token = manager.generate_token(str(source), other.name, approved_outbound=True)
    assert serve(token).status == 200
    # The session contains only the 32-byte key, not a source-file copy.
    files = list(session.iterdir())
    assert len(files) == 1 and files[0].stat().st_size == 32
    files[0].unlink()
    session.rmdir()
    assert serve(token).status == 403
    assert source.read_bytes() == b"0123456789"
    assert serve(other_token).status == 200
    assert not session.exists()  # Validation must never recreate deleted sessions.

    session.mkdir()
    assert serve(token).status == 403
    replacement = manager.generate_token(str(source), session.name, approved_outbound=True)
    assert serve(replacement).status == 200
    assert serve(token).status == 403  # Same sid, different lifetime.


def test_session_generation_is_shared_across_issuers(signed_download, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    manager, source, _ = signed_download
    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(pool.map(
            lambda _: WebFileDownloadManager(secret="download-test-" * 4).generate_token(
                str(source), "session-1", approved_outbound=True,
            ), range(32),
        ))
    generations = {manager.validate_token(token)["session_generation"] for token in tokens}
    assert len(generations) == 1
    assert len(list((tmp_path / "sessions" / "session-1").iterdir())) == 1


@pytest.mark.parametrize("sid", ["", "default", "missing-session", "../escape", "session-1/child", None, []])
def test_outbound_token_requires_existing_valid_session(signed_download, tmp_path, sid):
    manager, source, _ = signed_download
    with pytest.raises(ValueError):
        manager.generate_token(str(source), sid, approved_outbound=True)
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == ["session-1"]
    assert not list((tmp_path / "sessions" / "session-1").iterdir())


def test_outbound_token_without_generation_is_rejected(signed_download):
    _, source, serve = signed_download
    token = _signed_token("download-test-" * 4, {
        "path": str(source), "sid": "session-1",
        "purpose": web_file_download.OUTBOUND_DOWNLOAD_PURPOSE,
    })
    assert serve(token).status == 403


def test_corrupt_session_key_fails_closed_without_replacing_it(signed_download, tmp_path):
    manager, source, serve = signed_download
    token = manager.generate_token(str(source), "session-1", approved_outbound=True)
    key = next((tmp_path / "sessions" / "session-1").iterdir())
    key.write_bytes(b"corrupt")
    assert serve(token).status == 403
    with pytest.raises(ValueError, match="invalid session download key"):
        manager.generate_token(str(source), "session-1", approved_outbound=True)
    assert key.read_bytes() == b"corrupt"
