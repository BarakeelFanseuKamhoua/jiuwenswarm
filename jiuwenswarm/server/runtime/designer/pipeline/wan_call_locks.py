# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Still-image locks, and the short prompt sent to the video model.

Keyframe stills still receive style and wardrobe locks. Clip calls are
rewritten into a concise story-form prompt (wardrobe/seats/visibility
folded into narrative); those locks stay on the node for the agent.
"""

from __future__ import annotations

from typing import Any


def _style_from(cfg: dict[str, Any], graph: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    for src in (
        cfg.get("style_lock"),
        (graph.get("metadata") or {}).get("style_lock") if isinstance(graph.get("metadata"), dict) else None,
        analysis.get("style_lock"),
    ):
        if isinstance(src, dict) and src:
            return src
    return {}


def _aspect_from(cfg: dict[str, Any], graph: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    for src in (
        cfg.get("aspect_lock"),
        (graph.get("metadata") or {}).get("aspect_lock") if isinstance(graph.get("metadata"), dict) else None,
        analysis.get("aspect_lock"),
    ):
        if isinstance(src, dict) and src:
            return src
    return {}


def set_orientation_lock_clause(*, scene_card: bool = False) -> str:
    """Keep first-frame room orientation stable (fixes spin/flip drift)."""
    subject = "Scene card / Image 1" if scene_card else "Image 1"
    return (
        f"SET/ORIENTATION LOCK: keep {subject}'s exact room geometry, wall/window sides, "
        "furniture layout, and camera roll/horizon — FORBIDDEN: spin or rotate the set, "
        "mirror/flip architecture left-right, tilt the world, or rebuild a different room. "
        "Camera may push-in/pan slightly only if it does not reorient the space."
    )


def apply_keyframe_call_locks(
    prompt: str,
    *,
    cfg: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
) -> str:
    """Ensure the still API body is a positive practice prompt (no LOCK essays)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    graph = graph if isinstance(graph, dict) else {}
    role = str(cfg.get("role") or "").lower()
    try:
        from jiuwenswarm.server.runtime.designer.pipeline.image_prompt_practice import (
            ensure_still_tool_prompt,
        )

        role_eff = role or "character"
        if "scene" in role_eff:
            role_eff = "scene"
        elif "character" in role_eff:
            role_eff = "character"
        text, _notes = ensure_still_tool_prompt(
            str(prompt or ""),
            role=role_eff,
            cfg=cfg,
            graph=graph,
        )
        return text[:6000]
    except Exception:  # noqa: BLE001
        return str(prompt or "").strip()[:6000]


def _stamp_sent_prompt(cfg: dict[str, Any], approved: str, *, user_origin: bool) -> None:
    """Keep saved surfaces honest with the text actually sent to the video API."""
    text = str(approved or "").strip()
    if not text:
        return
    cfg["prompt"] = text
    generate = dict(cfg.get("generate") or {}) if isinstance(cfg.get("generate"), dict) else {}
    generate["prompt"] = text
    if user_origin:
        generate["prompt_origin"] = "user"
    cfg["generate"] = generate
    cfg["last_wan_prompt"] = text[:4000]
    cfg["last_approved_prompt"] = text[:4000]
    packet = cfg.get("regenerate_packet")
    if isinstance(packet, dict):
        cfg["regenerate_packet"] = {**packet, "prompt": text[:4000]}


def apply_wan_call_locks(
    prompt: str,
    *,
    cfg: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    shot_index: int = 0,
) -> str:
    """Rewrite the video call into a short image-binding prompt.

    Style, wardrobe, and storyboard locks stay on the node. The video model
    receives who each image is, where they are, what they do, and what they say.

    When generate.prompt_origin == user with a non-empty user prompt, that text
    seeds action/camera instead of stale shot_action/camera beats.
    """
    # Stamp/notes must mutate the caller's cfg. ensure_prior_clip_story_on_cfg
    # returns a shallow copy — use that copy only as the director working view.
    cfg_map = cfg if isinstance(cfg, dict) else {}
    director_cfg = cfg_map
    if isinstance(graph, dict):
        try:
            from jiuwenswarm.server.runtime.designer.pipeline.clip_story_state import (
                ensure_prior_clip_story_on_cfg,
            )

            director_cfg = ensure_prior_clip_story_on_cfg(cfg_map, graph)
        except Exception:  # noqa: BLE001
            director_cfg = cfg_map
    from jiuwenswarm.server.runtime.designer.pipeline.video_prompt_practice import (
        _pull_labeled,
        director_approve_video_prompt,
        narrative_seed_from_user_prompt,
        resolve_user_origin_prompt,
    )

    user_text = resolve_user_origin_prompt(cfg_map, prompt)
    if user_text:
        candidate = user_text
        action = narrative_seed_from_user_prompt(user_text) or user_text
        camera = _pull_labeled(user_text, ("camera move", "camera for shot", "camera"))
    else:
        candidate = prompt
        action = str(director_cfg.get("shot_action") or "")
        camera = str(director_cfg.get("camera") or "")

    approved, notes = director_approve_video_prompt(
        candidate,
        cfg=director_cfg,
        graph=graph if isinstance(graph, dict) else {},
        shot_index=shot_index,
        action=action,
        camera=camera,
    )
    cfg_map["director_video_prompt_notes"] = list(notes)
    cfg_map["director_video_prompt_approved"] = True
    _stamp_sent_prompt(cfg_map, approved, user_origin=bool(user_text))
    return approved
