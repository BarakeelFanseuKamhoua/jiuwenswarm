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

import logging
import os
import shutil
import subprocess
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


def _vision_available() -> bool:
    """True when a vision backend is configured, so the Director can read a frame."""
    try:
        from jiuwenswarm.server.runtime.designer.capabilities import (
            _vision_backend_configured,
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(_vision_backend_configured().get("available"))


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
    """Which continuity the shots need, given the model and the Director's eyes."""
    grouped = shots_by_setting(shots)
    if not any(len(group) > 1 for group in grouped.values()):
        return STRATEGY_SINGLE_CLIP
    if _vision_available():
        return STRATEGY_DIRECTOR_FRAME
    window = float(max_seconds) if max_seconds is not None else video_max_seconds()
    beat = float(beat_seconds) if beat_seconds is not None else video_beat_seconds()
    per_shot = per_shot_seconds(shots, analysis)
    for group in grouped.values():
        if len(group) < 2 or not same_cast(group):
            # A cast that changes between shots is the storyboard's own cut.
            continue
        if per_shot > beat or per_shot * len(group) > window:
            return STRATEGY_STATIC_WIDE
    return STRATEGY_SINGLE_CLIP


def _merge_list_field(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    merged: list[Any] = []
    for part in (target.get(key), source.get(key)):
        if not isinstance(part, list):
            continue
        for item in part:
            if item not in merged:
                merged.append(item)
    if merged:
        target[key] = merged


def _merge_map_field(target: dict[str, Any], source: dict[str, Any], key: str) -> None:
    merged: dict[str, Any] = {}
    for part in (target.get(key), source.get(key)):
        if isinstance(part, dict):
            merged.update(part)
    if merged:
        target[key] = merged


_LIST_FIELDS = (
    "on_screen",
    "offscreen",
    "character_ids",
    "featured_cast_ids",
    "ensemble_cast_ids",
    "already_done",
    "exited_ids",
)
_MAP_FIELDS = (
    "cast_actions",
    "speech_by_character",
    "wardrobe_by_character",
    "cast_states",
    "screen_positions",
)


def merge_shots(shots: list[dict[str, Any]], *, seconds: float) -> dict[str, Any]:
    """One take: the beats stay in order, the cast and its wardrobe stay put."""
    merged = dict(shots[0])
    actions = [str(s.get("action") or "").strip() for s in shots if str(s.get("action") or "").strip()]
    speech = [str(s.get("speech_line") or "").strip() for s in shots if str(s.get("speech_line") or "").strip()]
    for shot in shots[1:]:
        for key in _LIST_FIELDS:
            _merge_list_field(merged, shot, key)
        for key in _MAP_FIELDS:
            _merge_map_field(merged, shot, key)
        for key in ("crowd_spec", "start_state", "continuity_lock", "occupancy", "time_of_day_lock"):
            if not merged.get(key) and shot.get(key):
                merged[key] = shot[key]
        if shot.get("end_state"):
            merged["end_state"] = shot["end_state"]
        if shot.get("timeline"):
            merged["timeline"] = shot["timeline"]
    if actions:
        merged["action"] = "；".join(dict.fromkeys(actions))
    if speech:
        merged["speech_line"] = " ".join(dict.fromkeys(speech))
    merged["duration_sec"] = round(float(seconds), 3)
    merged["merged_from"] = [
        int(s.get("clip_index") or 0) for s in shots if s.get("clip_index") is not None
    ]
    merged["first_of_setting"] = True
    return merged


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
    """Return the shot list the graph should build, plus the strategy used."""
    ordered = [s for s in (shots or []) if isinstance(s, dict)]
    if not ordered:
        return [], STRATEGY_SINGLE_CLIP
    strategy = continuity_strategy(
        ordered, analysis=analysis, max_seconds=max_seconds, beat_seconds=beat_seconds
    )
    window = float(max_seconds) if max_seconds is not None else video_max_seconds()
    per_shot = per_shot_seconds(ordered, analysis)
    if lang is None:
        try:
            from jiuwenswarm.server.runtime.designer.pipeline.content_language import (
                current_content_language,
            )

            lang = current_content_language()
        except Exception:  # noqa: BLE001
            lang = "en"
    out: list[dict[str, Any]] = []
    for setting, group in shots_by_setting(ordered).items():
        if len(group) < 2:
            out.extend(group)
            continue
        stable_cast = same_cast(group)
        fits = len(group) * per_shot <= window
        if strategy == STRATEGY_SINGLE_CLIP and stable_cast and fits:
            merged = merge_shots(group, seconds=len(group) * per_shot)
            merged["setting_id"] = setting
            out.append(merged)
            continue
        if strategy == STRATEGY_STATIC_WIDE and stable_cast and (not fits or per_shot > video_beat_seconds()):
            out.append(group[0])
            out.extend(_static_wide(shot, lang) for shot in group[1:])
            continue
        out.extend(group)
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
