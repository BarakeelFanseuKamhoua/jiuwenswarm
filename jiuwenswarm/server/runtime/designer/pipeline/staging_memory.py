# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""What each finished shot actually looked like, kept for the rest of the film.

A shot is written from text, so the layout the model renders is only knowable by
looking at the render. Each finished shot is sampled and read, and the reading is
stored per setting. Later shots of that setting — including shots that do not
touch it, e.g. a bird's-eye in shot 1 and a zoom in shot 5 — receive every earlier
reading, so the seating, the sides of the armies and the positions of the leaders
stay one arrangement instead of a fresh invention per cut.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_KEY = "staging_memory"

_OBSERVE_QUESTION = {
    "en": (
        "These are frames sampled from one finished shot. Report only what is visible: "
        "who is in frame, on which side of the frame each person or group stands, who is "
        "seated or standing, what each holds, and where the fixed landmarks are. No "
        "judgement, no advice."
    ),
    "zh": (
        "这是一段已生成镜头的采样帧。只描述看得见的内容：画面里有谁；每个人或每支队伍"
        "位于画面哪一侧；谁坐着谁站着；各自拿着什么；固定地标在哪里。不要评价，不要建议。"
    ),
}


def _memory(graph: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    graph = graph if isinstance(graph, dict) else {}
    meta = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    store = meta.get(MEMORY_KEY)
    if not isinstance(store, dict):
        store = {}
        meta[MEMORY_KEY] = store
        graph["metadata"] = meta
    return store


def record(graph: dict[str, Any] | None, *, setting_id: str, clip_index: int, camera: str, observation: str) -> None:
    """Store one shot's reading, replacing any earlier reading of the same shot."""
    text = str(observation or "").strip()
    if not text:
        return
    store = _memory(graph)
    setting = str(setting_id or "set_1").strip() or "set_1"
    entries = [e for e in (store.get(setting) or []) if isinstance(e, dict)]
    entries = [e for e in entries if int(e.get("clip_index") or 0) != int(clip_index or 0)]
    entries.append(
        {
            "clip_index": int(clip_index or 0),
            "camera": str(camera or "").strip()[:120],
            "observation": text[:1200],
        }
    )
    entries.sort(key=lambda e: int(e.get("clip_index") or 0))
    store[setting] = entries


def digest(graph: dict[str, Any] | None, *, setting_id: str, upto_index: int, limit: int = 1400) -> str:
    """Every earlier shot of this setting, as the stage the next shot must keep."""
    store = _memory(graph)
    setting = str(setting_id or "set_1").strip() or "set_1"
    parts: list[str] = []
    for entry in store.get(setting) or []:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("clip_index") or 0) >= int(upto_index or 0):
            continue
        camera = str(entry.get("camera") or "").strip()
        head = f"shot {entry.get('clip_index')}" + (f" ({camera})" if camera else "")
        parts.append(f"{head}: {entry.get('observation')}")
    text = " | ".join(parts)
    return text[:limit]


def observed_clip_indexes(graph: dict[str, Any] | None, setting_id: str) -> set[int]:
    store = _memory(graph)
    setting = str(setting_id or "set_1").strip() or "set_1"
    return {
        int(entry.get("clip_index") or 0)
        for entry in store.get(setting) or []
        if isinstance(entry, dict)
    }


def completed_clip_paths(ctx: Any, *, setting_id: str, upto_index: int) -> dict[int, Any]:
    """Earlier clips of this setting that already produced a video file."""
    from pathlib import Path

    graph = getattr(ctx, "graph", None) or {}
    run = getattr(ctx, "run", None) or {}
    states = run.get("node_states") if isinstance(run, dict) else None
    states = states if isinstance(states, dict) else {}
    out: dict[int, Path] = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        cfg = node.get("config") if isinstance(node.get("config"), dict) else {}
        if str(cfg.get("pipeline") or "") != "clip":
            continue
        if str(cfg.get("setting_id") or "").strip() != str(setting_id or "").strip():
            continue
        try:
            index = int(cfg.get("clip_index") or 0)
        except (TypeError, ValueError):
            continue
        if index <= 0 or index >= int(upto_index or 0):
            continue
        state = states.get(str(node.get("id") or "")) or {}
        if not isinstance(state, dict) or state.get("status") != "completed":
            continue
        for ref in state.get("output_refs") or []:
            uri = str((ref or {}).get("uri") or "")
            if not uri.startswith("file:///"):
                continue
            path = Path(uri[8:])
            if path.is_file():
                out[index] = path
                break
    return out


async def observe_new_shots(ctx: Any, node: dict[str, Any], *, lang: str | None = None) -> str:
    """Read any earlier clip of this setting that has not been read yet."""
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    cfg = node.get("config") if isinstance(node.get("config"), dict) else {}
    setting = str(cfg.get("setting_id") or "set_1").strip() or "set_1"
    try:
        index = int(cfg.get("clip_index") or 1)
    except (TypeError, ValueError):
        index = 1
    graph = getattr(ctx, "graph", None) or {}
    seen = observed_clip_indexes(graph, setting)
    for earlier_index, path in sorted(completed_clip_paths(ctx, setting_id=setting, upto_index=index).items()):
        if earlier_index in seen:
            continue
        try:
            observation = await cs.observe_previous_clip([path], lang=lang or "en")
        except Exception:  # noqa: BLE001
            logger.debug("staging observation failed for shot %s", earlier_index, exc_info=True)
            continue
        if not observation:
            continue
        earlier_camera = ""
        for other in graph.get("nodes") or []:
            ocfg = (other or {}).get("config") if isinstance(other, dict) else None
            if not isinstance(ocfg, dict):
                continue
            if (
                str(ocfg.get("pipeline") or "") == "clip"
                and str(ocfg.get("setting_id") or "").strip() == setting
                and int(ocfg.get("clip_index") or 0) == earlier_index
            ):
                earlier_camera = str(ocfg.get("camera") or "")
                break
        record(
            graph,
            setting_id=setting,
            clip_index=earlier_index,
            camera=earlier_camera,
            observation=observation,
        )
    return digest(graph, setting_id=setting, upto_index=index)
