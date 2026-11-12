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


def test_two_shots_in_a_setting_become_one_take_when_the_model_can_hold_them() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline.continuity_strategy import (
        STRATEGY_SINGLE_CLIP,
        apply_continuity_plan,
    )

    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_2", "char_1"])]
    out, strategy = apply_continuity_plan(shots, max_seconds=15)

    assert strategy == STRATEGY_SINGLE_CLIP
    assert len(out) == 1
    merged = out[0]
    assert merged["duration_sec"] == 10.0
    assert merged["merged_from"] == [1, 2]
    assert merged["on_screen"] == ["char_1", "char_2"]
    assert "beat" in merged["action"]
    assert merged["speech_line"] == "line 1 line 2"


def test_a_cast_that_changes_between_shots_keeps_its_cut() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline.continuity_strategy import (
        apply_continuity_plan,
    )

    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1"])]
    out, _ = apply_continuity_plan(shots, max_seconds=15)

    assert len(out) == 2
    assert out[1]["camera"] == "medium / eye-level"


def test_each_setting_is_planned_independently() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline.continuity_strategy import (
        apply_continuity_plan,
    )

    shots = [
        _shot(1, setting="set_1", cast=["char_1", "char_2"]),
        _shot(2, setting="set_1", cast=["char_1", "char_2"]),
        _shot(3, setting="set_2", cast=["char_1", "char_3"]),
        _shot(4, setting="set_2", cast=["char_1", "char_3"]),
    ]
    out, _ = apply_continuity_plan(shots, max_seconds=15)

    assert [s["setting_id"] for s in out] == ["set_1", "set_2"]
    assert [s["merged_from"] for s in out] == [[1, 2], [3, 4]]


def test_a_setting_longer_than_the_window_keeps_its_cut_and_locks_the_camera() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline.continuity_strategy import (
        STRATEGY_STATIC_WIDE,
        _STATIC_WIDE_CAMERA,
        apply_continuity_plan,
    )

    shots = [_shot(1), _shot(2)]
    out, strategy = apply_continuity_plan(shots, max_seconds=6, lang="en")

    assert strategy == STRATEGY_STATIC_WIDE
    assert len(out) == 2
    assert out[1]["camera"] == _STATIC_WIDE_CAMERA["en"]
    assert out[1]["exiting_character_ids"] == []


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
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=12:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(clip),
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


def test_the_director_reads_the_previous_beats_when_a_vision_backend_exists(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setattr(cs, "_vision_available", lambda: True)
    shots = [_shot(1), _shot(2)]

    assert cs.continuity_strategy(shots, max_seconds=5) == cs.STRATEGY_DIRECTOR_FRAME
    out, strategy = cs.apply_continuity_plan(shots, max_seconds=5)
    assert strategy == cs.STRATEGY_DIRECTOR_FRAME
    assert len(out) == 2
