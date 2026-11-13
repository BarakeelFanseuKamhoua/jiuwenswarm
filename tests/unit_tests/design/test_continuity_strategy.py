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


def test_by_default_shots_render_in_parallel(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.delenv(cs.CONTINUITY_ENV, raising=False)
    monkeypatch.setattr(cs, "_vision_available", lambda: True)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1", "char_2"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5)

    assert strategy == cs.STRATEGY_SINGLE_CLIP
    assert len(out) == 2
    assert out[1]["camera"] == "medium / eye-level"


def test_without_vision_the_following_shot_keeps_a_locked_wide_camera(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setenv(cs.CONTINUITY_ENV, "observe")
    monkeypatch.setattr(cs, "_vision_available", lambda: False)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1", "char_2"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5, lang="en")

    assert strategy == cs.STRATEGY_STATIC_WIDE
    assert out[1]["camera"] == cs._STATIC_WIDE_CAMERA["en"]
    assert out[1]["exiting_character_ids"] == []


def test_a_cast_that_changes_between_shots_is_left_alone(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setenv(cs.CONTINUITY_ENV, "observe")
    monkeypatch.setattr(cs, "_vision_available", lambda: False)
    shots = [_shot(1, cast=["char_1", "char_2"]), _shot(2, cast=["char_1"])]

    out, strategy = cs.apply_continuity_plan(shots, max_seconds=15, beat_seconds=5, lang="en")

    assert strategy != cs.STRATEGY_STATIC_WIDE
    assert out[1]["camera"] == "medium / eye-level"


def test_with_vision_the_director_reads_the_previous_shot_instead(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setenv(cs.CONTINUITY_ENV, "observe")
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
    assert "Keep those positions" in prompt

def test_the_chat_model_can_be_the_directors_eyes(monkeypatch, tmp_path) -> None:
    import asyncio

    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    monkeypatch.setattr(cs, "_vision_available", lambda: True)
    monkeypatch.setattr(
        cs, "chat_model_config", lambda: {"api_base": "https://x", "api_key": "k", "model": "m"}
    )
    monkeypatch.setattr(cs, "sample_clip_frames", lambda *a, **k: [frame])
    monkeypatch.setattr(
        cs, "_chat_completion", lambda conf, content, timeout=0: "Five people; mother sits left"
    )

    observed = asyncio.run(cs.observe_previous_clip(["clip.mp4"], count=2))

    assert observed == "Five people; mother sits left"


def test_the_image_probe_accepts_a_colour_answer(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    conf = {"api_base": "https://x", "api_key": "k", "model": "m"}
    monkeypatch.setattr(cs, "_chat_completion", lambda *a, **k: "Blue")
    assert cs._probe_chat_image_support(conf) is True
    monkeypatch.setattr(cs, "_chat_completion", lambda *a, **k: "I cannot see images.")
    assert cs._probe_chat_image_support(conf) is False

def _two_shot_graph(vision: bool, monkeypatch):
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs
    from jiuwenswarm.server.runtime.designer.smart_graph import build_smart_video_graph

    monkeypatch.setattr(cs, "_vision_available", lambda: vision)
    analysis = {
        "characters": [{"id": "char_1", "name": "Ann", "description": "green blouse"}],
        "scenes": [{"id": "set_1", "name": "kitchen", "description": "table"}],
        "clips": [
            {
                "clip_index": 1,
                "setting_id": "set_1",
                "action": "Ann sits.",
                "on_screen": ["char_1"],
                "camera": "wide",
                "timeline": "0-5s",
            },
            {
                "clip_index": 2,
                "setting_id": "set_1",
                "action": "Ann eats.",
                "on_screen": ["char_1"],
                "camera": "medium",
                "timeline": "5-10s",
            },
        ],
        "target_duration_sec": 10,
    }
    return build_smart_video_graph(project_id="p_seq", prompt="two shots", analysis=analysis)


def test_shots_run_in_order_only_when_something_reads_them(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs

    monkeypatch.setenv(cs.CONTINUITY_ENV, "observe")
    read_graph = _two_shot_graph(True, monkeypatch)
    assert [e for e in read_graph["edges"] if e.get("label") == "sequence"]

    blind_graph = _two_shot_graph(False, monkeypatch)
    assert not [e for e in blind_graph["edges"] if e.get("label") == "sequence"]
    clip2 = next(n for n in blind_graph["nodes"] if n["id"] == "n_clip_2")
    assert not (clip2.get("config") or {}).get("sequence_from")

def test_the_staging_memory_remembers_earlier_shots_of_the_same_setting() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import staging_memory as sm

    graph = {"metadata": {}}
    sm.record(graph, setting_id="set_1", clip_index=1, camera="bird's eye", observation="Saul on the ridge north")
    sm.record(graph, setting_id="set_1", clip_index=2, camera="low angle", observation="Goliath in the valley")
    sm.record(graph, setting_id="set_2", clip_index=5, camera="wide", observation="road")

    digest = sm.digest(graph, setting_id="set_1", upto_index=5)

    assert "Saul on the ridge north" in digest
    assert "Goliath in the valley" in digest
    assert "road" not in digest
    assert digest.index("Saul") < digest.index("Goliath")


def test_a_later_shot_only_sees_the_shots_before_it() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import staging_memory as sm

    graph = {"metadata": {}}
    sm.record(graph, setting_id="set_1", clip_index=1, camera="", observation="shot one")
    sm.record(graph, setting_id="set_1", clip_index=6, camera="", observation="shot six")

    digest = sm.digest(graph, setting_id="set_1", upto_index=5)

    assert "shot one" in digest
    assert "shot six" not in digest


def test_recording_the_same_shot_twice_keeps_one_entry() -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import staging_memory as sm

    graph = {"metadata": {}}
    sm.record(graph, setting_id="set_1", clip_index=3, camera="a", observation="first read")
    sm.record(graph, setting_id="set_1", clip_index=3, camera="a", observation="second read")

    digest = sm.digest(graph, setting_id="set_1", upto_index=4)

    assert "second read" in digest
    assert "first read" not in digest


def test_every_setting_gets_its_shot_order(monkeypatch) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import continuity_strategy as cs
    from jiuwenswarm.server.runtime.designer.smart_graph import build_smart_video_graph

    monkeypatch.setenv(cs.CONTINUITY_ENV, "observe")
    monkeypatch.setattr(cs, "_vision_available", lambda: True)
    analysis = {
        "characters": [
            {"id": "char_1", "name": "David", "description": "tunic"},
            {"id": "char_2", "name": "Saul", "description": "armour"},
            {"id": "char_3", "name": "Goliath", "description": "bronze"},
        ],
        "scenes": [
            {"id": "set_1", "name": "valley", "description": "two armies"},
            {"id": "set_2", "name": "camp", "description": "tents"},
        ],
        "clips": [
            {"clip_index": i, "setting_id": "set_1", "action": f"beat {i}", "on_screen": ["char_1"], "camera": "wide", "timeline": f"{(i-1)*5}-{i*5}s"}
            for i in range(1, 3)
        ] + [
            {"clip_index": i, "setting_id": "set_2", "action": f"beat {i}", "on_screen": ["char_2"], "camera": "wide", "timeline": f"{(i-1)*5}-{i*5}s"}
            for i in range(3, 5)
        ],
        "target_duration_sec": 20,
    }
    graph = build_smart_video_graph(project_id="p_two", prompt="four shots", analysis=analysis)

    seq = {(e["source"], e["target"]) for e in graph["edges"] if e.get("label") == "sequence"}
    assert ("n_clip_1", "n_clip_2") in seq
    assert ("n_clip_3", "n_clip_4") in seq, "the second setting lost its order"
    assert len(seq) == 2

def test_a_finished_shot_is_remembered_by_its_file(tmp_path) -> None:
    from jiuwenswarm.server.runtime.designer.pipeline import staging_memory as sm

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    graph = {"metadata": {}}

    sm.register_output(graph, setting_id="set_1", clip_index=1, path=clip)

    recorded = sm.recorded_outputs(graph, "set_1")
    assert recorded[1] == clip.resolve()
    assert sm.recorded_outputs(graph, "set_2") == {}

def test_the_video_handler_sends_and_records_the_staging_memory(tmp_path, monkeypatch) -> None:
    import asyncio

    from jiuwenswarm.server.runtime.designer.handlers import media_nodes
    from jiuwenswarm.server.runtime.designer.pipeline import staging_memory as sm

    prev = tmp_path / "prev.mp4"
    prev.write_bytes(b"\x00")
    graph = {
        "metadata": {},
        "edges": [],
        "nodes": [
            {"id": "n_clip_1", "type": "video", "config": {"pipeline": "clip", "setting_id": "set_1", "clip_index": 1}},
            {
                "id": "n_clip_2",
                "type": "video",
                "config": {
                    "pipeline": "clip",
                    "setting_id": "set_1",
                    "clip_index": 2,
                    "prompt": "seed prompt for shot two",
                },
            },
        ],
    }
    sm.record(graph, setting_id="set_1", clip_index=1, camera="bird's eye", observation="Saul north, Philistines south")
    sm.register_output(graph, setting_id="set_1", clip_index=1, path=prev)

    captured: dict[str, str] = {}

    async def fake_generate(prompt, **_kwargs):
        captured["prompt"] = prompt
        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00")
        return {"video_path": str(out)}

    monkeypatch.setattr(media_nodes, "generate_clip_video", fake_generate)
    monkeypatch.setattr(
        media_nodes,
        "video_settings",
        lambda _g: type("V", (), {"aspect_ratio": "16:9", "resolution": "720P"})(),
    )
    from jiuwenswarm.server.runtime.designer.handlers import clip as clip_handler

    monkeypatch.setattr(clip_handler, "video_duration_seconds", lambda _c, _n: 5)
    monkeypatch.setattr(clip_handler, "note_video_duration", lambda *_a, **_k: None)

    ctx = type("C", (), {"graph": graph, "run": {"node_states": {}}, "run_id": "r", "node_id": "n_clip_2", "media_dir": tmp_path})()
    asyncio.run(media_nodes.VideoNodeHandler().execute(graph["nodes"][1], ctx))

    assert "Saul north, Philistines south" in captured["prompt"]
    assert captured["prompt"].startswith("seed prompt for shot two")
    assert sm.recorded_outputs(graph, "set_1").get(2) is not None
