# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""How shots that share a setting stay continuous.

Each shot is generated on its own, so a cut between two shots is where seating,
poses and cast drift. The cheapest continuity is no cut at all: a setting made of
short beats is generated as one take while the video model's window covers it.
When the model cannot hold the setting, a configured vision backend lets the
Director read a *set* of sampled frames from the previous shot — one frame can be
a blur, a smile or an empty doorway — and write the next beat from what it sees.
Without either, the remaining shots keep one locked wide camera and nobody enters
or leaves mid-shot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

STRATEGY_SINGLE_CLIP = "single_clip"
STRATEGY_DIRECTOR_FRAME = "director_frame_handoff"
STRATEGY_STATIC_WIDE = "static_wide"

logger = logging.getLogger(__name__)

_DEFAULT_SHOT_SECONDS = 5.0
_DEFAULT_FRAME_SAMPLES = 4

_STATIC_WIDE_CAMERA = {
    "zh": "固定全景，机位锁定，不推近、不横移；全员同框，镜头内无人进出画面。",
    "en": (
        "Static wide framing on a locked camera, no push-in and no pan; the whole "
        "cast stays in frame and nobody enters or leaves during the shot."
    ),
}


_CHAT_VISION_VERDICT: bool | None = None


def chat_model_config() -> dict[str, Any]:
    """The Director's own chat model: base url, key, model id, declared image support."""
    try:
        from jiuwenswarm.common.config import get_config, resolve_env_vars
    except Exception:  # noqa: BLE001
        return {}
    models = (get_config() or {}).get("models") or {}
    for entry in models.get("defaults") or []:
        if not isinstance(entry, dict):
            continue
        mc = entry.get("model_client_config")
        if not isinstance(mc, dict):
            continue
        base = resolve_env_vars(str(mc.get("api_base") or "")).strip()
        key = resolve_env_vars(str(mc.get("api_key") or "")).strip()
        name = resolve_env_vars(str(mc.get("model_name") or "")).strip()
        if not (base and key and name):
            continue
        declared = mc.get("supports_vision")
        return {
            "api_base": base,
            "api_key": key,
            "model": name,
            "declared_vision": declared if isinstance(declared, bool) else None,
        }
    return {}


def _chat_completion(conf: dict[str, Any], content: list[dict[str, Any]], *, timeout: int) -> str:
    """One chat round-trip with image parts; empty string on any failure."""
    payload = {
        "model": conf["model"],
        "messages": [{"role": "user", "content": content}],
        # Reasoning models spend tokens on reasoning first; a small cap returns
        # an empty answer even when the image was read correctly.
        "max_tokens": 1200,
        "stream": False,
    }
    req = urllib.request.Request(
        conf["api_base"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {conf['api_key']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except Exception:
        logger.debug("chat completion failed", exc_info=True)
        return ""
    choices = data.get("choices") or []
    message = (choices[0] or {}).get("message") if choices else {}
    message = message or {}
    # Only the answer: a reasoning model's private reasoning is off-task text.
    return str(message.get("content") or "").strip()


def _image_part(image_path: Path, mime: str = "image/jpeg") -> dict[str, Any]:
    import base64

    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def _solid_png(size: int = 48, rgb: tuple[int, int, int] = (0, 0, 255)) -> bytes:
    import base64
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = struct.pack(">I", len(data)) + tag + data
        return body + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png)


def _probe_chat_image_support(conf: dict[str, Any]) -> bool:
    """One tiny colour question decides whether the chat model accepts images."""
    content = [
        {"type": "text", "text": "What colour is this square? One word."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + _solid_png().decode()},
        },
    ]
    answer = _chat_completion(conf, content, timeout=60).casefold()
    return "blue" in answer


def chat_model_reads_images() -> bool:
    """Cached verdict: does the Director's chat model accept image input?"""
    global _CHAT_VISION_VERDICT
    if _CHAT_VISION_VERDICT is not None:
        return _CHAT_VISION_VERDICT
    conf = chat_model_config()
    if not conf:
        _CHAT_VISION_VERDICT = False
        return False
    declared = conf.get("declared_vision")
    _CHAT_VISION_VERDICT = (
        bool(declared) if declared is not None else _probe_chat_image_support(conf)
    )
    return _CHAT_VISION_VERDICT


CONTINUITY_ENV = "DESIGNER_SHOT_CONTINUITY"


def shots_run_in_order() -> bool:
    """Whether shots must run in order: reading a shot is opt-in and gated.

    Sequencing a setting costs its parallelism (shots render one after another),
    so it stays off unless the build is explicitly asked for observed continuity.
    """
    if str(os.getenv(CONTINUITY_ENV) or "").strip().casefold() not in {"observe", "on", "1", "true"}:
        return False
    return _vision_available()


def director_reads_shots() -> bool:
    """Whether the Director can read a finished shot (a vision slot or a multimodal chat model)."""
    return shots_run_in_order()


def _vision_available() -> bool:
    """True when the Director can read a frame: a vision slot, or a multimodal chat model."""
    try:
        from jiuwenswarm.server.runtime.designer.capabilities import (
            _vision_backend_configured,
        )

        if _vision_backend_configured().get("available"):
            return True
    except Exception:
        logger.debug("vision slot lookup failed", exc_info=True)
    return chat_model_reads_images()


def video_max_seconds() -> float:
    """Longest single generation the configured video model allows."""
    try:
        from jiuwenswarm.server.runtime.designer.pipeline.model_capacity import (
            active_video_capacity,
        )

        return float(active_video_capacity().max_sec)
    except Exception:  # noqa: BLE001
        return _DEFAULT_SHOT_SECONDS


def video_beat_seconds() -> float:
    """The model's natural beat length — what a storyboard shot is planned at."""
    try:
        from jiuwenswarm.server.runtime.designer.pipeline.model_capacity import (
            active_video_capacity,
        )

        return float(active_video_capacity().default_sec)
    except Exception:  # noqa: BLE001
        return _DEFAULT_SHOT_SECONDS


def per_shot_seconds(
    shots: list[dict[str, Any]] | None,
    analysis: dict[str, Any] | None = None,
) -> float:
    """Beat length the storyboard planned per shot."""
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        for key in ("duration_sec", "duration"):
            try:
                seconds = float(shot.get(key))
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                return seconds
    try:
        total = float((analysis or {}).get("target_duration_sec"))
    except (TypeError, ValueError):
        total = 0.0
    count = len([s for s in (shots or []) if isinstance(s, dict)])
    if total > 0 and count:
        return round(total / count, 3)
    return _DEFAULT_SHOT_SECONDS


def shots_by_setting(shots: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    """Setting id -> its shots, in storyboard order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        setting = str(shot.get("setting_id") or "set_1").strip() or "set_1"
        grouped.setdefault(setting, []).append(shot)
    return grouped


def same_cast(group: list[dict[str, Any]]) -> bool:
    """True when every shot of a setting shows the same cast."""
    casts: list[frozenset[str]] = []
    for shot in group:
        cast = shot.get("on_screen") or shot.get("character_ids") or []
        casts.append(frozenset(str(c) for c in cast if str(c)))
    return bool(casts) and all(cast == casts[0] for cast in casts)


def continuity_strategy(
    shots: list[dict[str, Any]] | None,
    *,
    analysis: dict[str, Any] | None = None,
    max_seconds: float | None = None,
    beat_seconds: float | None = None,
) -> str:
    """Which continuity the shots need, given the model and the Director's eyes.

    The storyboard's shot count is the plan, so shots are never merged. Continuity
    inside a setting comes from the Director reading the shot before it when a
    vision backend is configured; without one, the only lever the text has is a
    locked wide camera on the shots that follow.
    """
    if str(os.getenv(CONTINUITY_ENV) or "").strip().casefold() not in {"observe", "on", "1", "true"}:
        # Default: plain parallel rendering, no sequencing, no camera rewrite.
        return STRATEGY_SINGLE_CLIP
    grouped = shots_by_setting(shots)
    if not any(len(group) > 1 and same_cast(group) for group in grouped.values()):
        return STRATEGY_SINGLE_CLIP
    if _vision_available():
        return STRATEGY_DIRECTOR_FRAME
    return STRATEGY_STATIC_WIDE


def _static_wide(shot: dict[str, Any], lang: str = "en") -> dict[str, Any]:
    """Freeze framing and cast for a shot that follows another in the setting."""
    out = dict(shot)
    out["camera"] = _STATIC_WIDE_CAMERA.get(lang, _STATIC_WIDE_CAMERA["en"])
    out["exiting_character_ids"] = []
    out["exited_ids"] = []
    return out


def apply_continuity_plan(
    shots: list[dict[str, Any]] | None,
    *,
    analysis: dict[str, Any] | None = None,
    max_seconds: float | None = None,
    beat_seconds: float | None = None,
    lang: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return the shot list the graph should build, plus the strategy used.

    The storyboard owns the shot count, so shots are never merged here: continuity
    comes from the Director reading the previous shot when a vision backend is
    configured, and from a locked wide camera when there is none.
    """
    ordered = [s for s in (shots or []) if isinstance(s, dict)]
    if not ordered:
        return [], STRATEGY_SINGLE_CLIP
    strategy = continuity_strategy(
        ordered, analysis=analysis, max_seconds=max_seconds, beat_seconds=beat_seconds
    )
    if lang is None:
        try:
            from jiuwenswarm.server.runtime.designer.pipeline.content_language import (
                current_content_language,
            )

            lang = current_content_language()
        except Exception:  # noqa: BLE001
            lang = "en"
    if strategy != STRATEGY_STATIC_WIDE:
        return ordered, strategy
    out: list[dict[str, Any]] = []
    for group in shots_by_setting(ordered).values():
        if len(group) < 2 or not same_cast(group):
            out.extend(group)
            continue
        out.append(group[0])
        out.extend(_static_wide(shot, lang) for shot in group[1:])
    return out, strategy


def _ffmpeg_exe() -> str:
    """ffmpeg for frame sampling: env override, PATH, then the bundled wheel."""
    env = str(os.environ.get("FFMPEG_BINARY") or os.environ.get("FFMPEG_PATH") or "").strip()
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        exe = str(imageio_ffmpeg.get_ffmpeg_exe() or "").strip()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        logger.debug("imageio_ffmpeg is not available", exc_info=True)
    raise RuntimeError("ffmpeg is required to sample clip frames")


def _clip_seconds(ffmpeg: str, clip_path: Path) -> float:
    """Length of a finished clip, from the bundled reader or ffmpeg's own report."""
    try:
        import imageio_ffmpeg

        _frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(clip_path))
        if seconds and float(seconds) > 0:
            return float(seconds)
    except Exception:
        logger.debug("clip length unavailable from imageio_ffmpeg", exc_info=True)
    probed = subprocess.run(
        [ffmpeg, "-i", str(clip_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    for line in f"{probed.stderr or ''}\n{probed.stdout or ''}".splitlines():
        text = line.strip()
        marker = text.find("Duration:")
        if marker < 0:
            continue
        stamp = text[marker + len("Duration:") :].split(",")[0].strip()
        parts = stamp.split(":")
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            continue
        if len(numbers) == 3:
            return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return 0.0


def sample_clip_frames(
    clip_path: str | Path,
    *,
    count: int = _DEFAULT_FRAME_SAMPLES,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Evenly spaced frames of a finished clip.

    One frame is a lottery — it can be a blur, a smile that hides the table, or an
    empty doorway. The samples start after the first frame and stop before the
    last, so the Director reads the staging the shot actually held.
    """
    path = Path(clip_path)
    if not path.is_file():
        return []
    wanted = max(2, min(6, int(count or _DEFAULT_FRAME_SAMPLES)))
    ffmpeg = _ffmpeg_exe()
    seconds = _clip_seconds(ffmpeg, path)
    if seconds <= 0:
        seconds = float(wanted)
    target = Path(out_dir) if out_dir else path.parent / f"{path.stem}_frames"
    target.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index in range(wanted):
        at = seconds * (index + 0.5) / wanted
        frame = target / f"{path.stem}_f{index + 1}.jpg"
        done = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{at:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if done.returncode == 0 and frame.is_file() and frame.stat().st_size > 0:
            frames.append(frame)
    return frames


def previous_clip_frame_sets(
    clip_paths: list[str | Path] | None,
    *,
    count: int = _DEFAULT_FRAME_SAMPLES,
) -> list[list[Path]]:
    """Sampled frames per earlier clip, oldest first — what the Director reads."""
    sets: list[list[Path]] = []
    for clip in clip_paths or []:
        frames = sample_clip_frames(clip, count=count)
        if frames:
            sets.append(frames)
    return sets

_OBSERVE_QUESTION = {
    "zh": (
        "???????????????????????????????????"
        "?????/?/??????????????????????????????????"
    ),
    "en": (
        "These are sampled frames from a finished shot. Report only what you see: who is "
        "in frame, where each person sits or stands relative to the table, door or window, "
        "and what each is holding. No judgement, no advice."
    ),
}


async def observe_previous_clip(
    clip_paths: list[str | Path] | None,
    *,
    count: int = _DEFAULT_FRAME_SAMPLES,
    lang: str = "en",
    max_frames: int = 6,
) -> str:
    """What the earlier shot actually looked like, read from a set of its frames.

    Empty when no vision backend is configured or nothing could be read.
    """
    if not _vision_available():
        return ""
    frames: list[Path] = []
    for clip in clip_paths or []:
        frames.extend(sample_clip_frames(clip, count=count))
    if not frames:
        return ""
    try:
        from jiuwenswarm.server.runtime.designer.capabilities import (
            inspect_image_for_rating,
        )
    except Exception:  # noqa: BLE001
        return ""
    question = _OBSERVE_QUESTION.get(lang, _OBSERVE_QUESTION["en"])
    conf = chat_model_config()
    seen: list[str] = []
    for frame in frames[:max_frames]:
        text = ""
        if conf:
            content = [{"type": "text", "text": question}, _image_part(frame)]
            text = await asyncio.to_thread(_chat_completion, conf, content, timeout=90)
        if not text:
            try:
                text = await inspect_image_for_rating(str(frame), question)
            except Exception:
                logger.debug("frame observation failed: %s", frame, exc_info=True)
                continue
        text = str(text or "").strip()
        if text and text not in seen:
            seen.append(text)
    return " / ".join(seen)[:1200]


def previous_clip_paths(ctx: Any, node: dict[str, Any]) -> list[Path]:
    """Files of the shot just before this one in the same setting."""
    cfg = node.get("config") if isinstance(node.get("config"), dict) else {}
    try:
        index = int(cfg.get("clip_index") or 1)
    except (TypeError, ValueError):
        index = 1
    setting = str(cfg.get("setting_id") or "").strip()
    graph = getattr(ctx, "graph", None) or {}
    states = ((getattr(ctx, "run", None) or {}).get("node_states") or {})
    best_index = -1
    best_node = ""
    for other in graph.get("nodes") or []:
        if not isinstance(other, dict):
            continue
        ocfg = other.get("config") if isinstance(other.get("config"), dict) else {}
        if str(ocfg.get("pipeline") or "") != "clip":
            continue
        if str(ocfg.get("setting_id") or "").strip() != setting:
            continue
        try:
            other_index = int(ocfg.get("clip_index") or 0)
        except (TypeError, ValueError):
            continue
        if other_index <= 0 or other_index >= index or other_index <= best_index:
            continue
        best_index = other_index
        best_node = str(other.get("id") or "")
    if not best_node:
        return []
    paths: list[Path] = []
    for ref in (states.get(best_node) or {}).get("output_refs") or []:
        uri = str((ref or {}).get("uri") or "")
        if not uri.startswith("file:///"):
            continue
        path = Path(uri[8:])
        if path.is_file():
            paths.append(path)
    return paths
