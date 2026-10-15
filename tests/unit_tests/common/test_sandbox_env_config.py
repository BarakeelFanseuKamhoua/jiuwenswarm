"""Sandbox parent-environment allowlist configuration."""

import pytest
import yaml
from pathlib import Path

from jiuwenswarm.common import config


def test_env_allowlist_reads_yaml_names(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        "sandbox:\n  env_allowlist:\n    - CUSTOM_TOOL_SETTING\n    - EMPTY_TOOL_SETTING\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "get_config", lambda: yaml.safe_load(path.read_text(encoding="utf-8")))
    assert config.get_sandbox_env_allowlist() == ["CUSTOM_TOOL_SETTING", "EMPTY_TOOL_SETTING"]


def test_env_allowlist_default_and_explicit_empty():
    defaults = config.get_sandbox_env_allowlist({})
    assert "CLAW_SKILL_TOKEN" in defaults
    assert "OPENAI_API_KEY" not in defaults
    assert "DATABASE_URL" not in defaults
    defaults.clear()
    assert config.get_sandbox_env_allowlist({})
    assert config.get_sandbox_env_allowlist({"sandbox": {"env_allowlist": []}}) == []


@pytest.mark.parametrize("value", [None, "PATH", ["*"], ["CLAW_*"], [1], ["A=B"], [""]])
def test_env_allowlist_rejects_invalid_names(value):
    with pytest.raises(ValueError, match="sandbox.env_allowlist"):
        config.get_sandbox_env_allowlist({"sandbox": {"env_allowlist": value}})


def test_env_allowlist_deduplicates_names():
    assert config.get_sandbox_env_allowlist({"sandbox": {"env_allowlist": ["PATH", "PATH"]}}) == ["PATH"]


def test_template_matches_default_allowlist():
    path = Path(__file__).resolve().parents[3] / "jiuwenswarm" / "resources" / "config.yaml"
    template = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config.get_sandbox_env_allowlist(template) == config.get_sandbox_env_allowlist({})
