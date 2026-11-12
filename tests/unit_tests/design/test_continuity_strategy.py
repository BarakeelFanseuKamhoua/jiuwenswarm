def _shot(idx: int, *, setting: str = "set_1", action: str = "beat", cast: list[str] | None = None) -> dict:
    return {
        "clip_index": idx,
        "setting_id": setting,
        "action": action,
        "camera": "medium / eye-level",
        "timeline": f"{idx - 1}-{idx}s",
        "on_screen": cast or ["char_1"],
        "cast_actions": {"char_1": action},
        "speech_line": f"line {idx}",
        "duration_sec": 5.0,
    }


def test_the_storyboard_shot_count_is_never_merged_away(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: False)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1", "char_2"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5)

    assert len(out) == 2
    assert strategy != cs.STRATEGY_SINGLE_CLIP


def test_without_vision_the_following_shot_keeps_a_locked_wide_camera(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: False)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1", "char_2"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5, lang="en")

    assert strategy == cs.STRATEGY_STATIC_WIDE
    assert out[1]["camera"] == cs._STATIC_WIDE_CAMERA["en"]
    assert out[1]["exiting_character_ids"] == []


def test_a_cast_that_changes_between_shots_is_left_alone(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: False)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5, lang="en")

    assert strategy != cs.STRATEGY_STATIC_WIDE
    assert out[1]["camera"] == "medium / eye-level"


def test_with_vision_the_director_reads_the_previous_shot_instead(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: True)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1", "char_2"])]

    assert cs.continuity_strategy(shots, max_seconds=5) == cs.STRATEGY_DIRECTOR_FRAME
    out, strategy = cs.apply_continuity_plan(shots, max_seconds=5)
    assert strategy == cs.STRATEGY_DIRECTOR_FRAME
    assert len(out) == 2
    assert out[1]["camera"] == "medium / eye-level"


def test_the_observation_is_empty_without_a_vision_backend(monkeypatch) -> None:
    import asyncio

    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: False)

    assert asyncio.run(cs.observe_previous_clip(["does-not-exist.mp4"])) == ""


def test_the_director_reads_a_set_of_sampled_frames_not_one_frame(tmp_path) -> None:
    import pytest

    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    try:
        ffmpeg = cs._ffmpeg_exe()
    except RuntimeError:
        pytest.skip("ffmpeg is not available")
    clip = tmp_path / "clip.mp4"
    made = cs.subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=12:duration=2",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if made.returncode != 0 or not clip.is_file():
        pytest.skip("ffmpeg could not build a test clip")

    frames = cs.sample_clip_frames(clip, count=4)

    assert len(frames) == 4
    assert all(frame.is_file() and frame.stat().st_size > 0 for frame in frames)
    assert len({frame.read_bytes() for frame in frames}) == 4


def test_the_composed_prompt_states_what_the_previous_shot_looked_like() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline.video_prompt_practice import (
        compose_practice_prompt,
    )

    cfg = {
        "pipeline": "clip",
        "on_screen": ["char_1"],
        "clip_action": "Ruth waits.",
        "observed_previous_state": "Ruth is seated at the left of the table, holding a cup",
    }

    prompt = compose_practice_prompt(cfg=cfg, graph=None, action="Ruth waits.")

    assert "Ruth is seated at the left of the table, holding a cup" in prompt
    assert "continues from that state" in prompt
