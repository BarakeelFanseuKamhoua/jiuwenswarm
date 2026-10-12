# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for team config loading."""

from pathlib import Path

import pytest
import yaml

from jiuwenswarm.common.config import resolve_env_vars
from jiuwenswarm.common.config import _normalize_config
from jiuwenswarm.agents.harness.team.config_loader import (
    TeamTemplateNotFoundError,
    get_team_template_snapshot,
    list_team_template_summaries,
    load_team_spec_dict,
    resolve_team_sqlite_db_path,
)


def _wrap_modes_team(team_mapping: dict[str, dict]) -> dict:
    return {"modes": {"team": team_mapping}}


def test_load_team_spec_dict_tolerates_empty_custom_headers():
    """CUSTOM_HEADERS 为空串时（resolve_env_vars 展开为 ''），team 模型自定义头
    必须被兜底剔除，不能让 '' 流到 TeamAgentSpec 校验炸 dict_type。"""
    from openjiuwen.agent_teams.schema.blueprint import TeamAgentSpec

    config = {
        "models": {
            "defaults": [
                {
                    "model_client_config": {
                        "api_base": "https://example.test/v1",
                        "api_key": "sk-test",
                        "model_name": "gpt-test",
                        "client_provider": "OpenAI",
                        "custom_headers": "",
                    },
                    "model_config_obj": {},
                },
            ]
        },
        **_wrap_modes_team({"demo_team": {"team_name": "demo_team", "agents": {}}}),
    }

    spec_dict = load_team_spec_dict(config_base=config)

    for role in ("leader", "teammate"):
        mcc = spec_dict["agents"][role]["model"]["model_client_config"]
        assert mcc.get("custom_headers") is None
    # 关键回归：不再抛 dict_type 校验错误
    TeamAgentSpec.model_validate(spec_dict)


def test_normalize_config_walks_models_defaults_list():
    """_normalize_config 必须覆盖 models.defaults 列表条目（此前只处理 dict 值，
    列表里的 custom_headers 空串漏网）。"""
    config = {
        "models": {
            "defaults": [
                {"model_client_config": {"custom_headers": ""}},
                {"model_client_config": {"custom_headers": '{"x-a": "1"}'}},
            ]
        }
    }

    _normalize_config(config)

    defaults = config["models"]["defaults"]
    assert defaults[0]["model_client_config"]["custom_headers"] is None
    assert defaults[1]["model_client_config"]["custom_headers"] == {"x-a": "1"}


def test_load_team_spec_dict_reads_models_defaults_from_repository_config(monkeypatch):
    """Repository config template should provide the default team model from models.defaults."""
    repo_config = Path(__file__).resolve().parents[3] / "jiuwenswarm" / "resources" / "config.yaml"
    monkeypatch.setenv("API_BASE", "https://example.test/v1")
    monkeypatch.setenv("API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_NAME", "gpt-template")
    monkeypatch.setenv("MODEL_PROVIDER", "OpenAI")

    config = resolve_env_vars(yaml.safe_load(repo_config.read_text(encoding="utf-8")) or {})

    spec = load_team_spec_dict(config_base=config)

    model = spec["agents"]["leader"]["model"]
    assert model["model_client_config"]["api_base"] == "https://example.test/v1"
    assert model["model_client_config"]["api_key"] == "sk-test"
    assert model["model_client_config"]["model_name"] == "gpt-template"
    assert model["model_client_config"]["client_provider"] == "OpenAI"
    assert model["model_request_config"]["model"] == "gpt-template"


def _two_entry_models_config() -> dict:
    return {
        "models": {
            "defaults": [
                {
                    "model_client_config": {
                        "api_base": "https://first.example.test/v1",
                        "api_key": "sk-first",
                        "model_name": "first-model",
                        "client_provider": "OpenAI",
                    },
                    "model_config_obj": {"temperature": 0.1},
                    "is_default": True,
                },
                {
                    "model_client_config": {
                        "api_base": "https://second.example.test/v1",
                        "api_key": "sk-second",
                        "model_name": "second-model",
                        "client_provider": "OpenAI",
                    },
                    "model_config_obj": {"temperature": 0.9},
                },
            ]
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {"leader": {}, "teammate": {}},
                }
            }
        ),
    }


def test_load_team_spec_dict_requested_model_hit_selects_matching_entry():
    """页面选择命中 models.defaults 条目时，整团（leader+teammate）用匹配条目。"""
    spec = load_team_spec_dict(
        config_base=_two_entry_models_config(),
        requested_model_name="second-model",
    )

    for role in ("leader", "teammate"):
        model = spec["agents"][role]["model"]
        assert model["model_client_config"]["model_name"] == "second-model"
        assert model["model_client_config"]["api_key"] == "sk-second"
        assert model["model_request_config"]["model"] == "second-model"


def test_load_team_spec_dict_requested_model_miss_overrides_first_entry(monkeypatch):
    """页面选择未命中 models.defaults 时，不再静默回退首条目
    模型名——以首条目为底覆盖 model_name（凭证复用），对齐单 agent
    model_name_override 语义。"""
    from unittest.mock import MagicMock

    # 可观测性断言用 spy 替换模块 logger（caplog 在本仓日志体系下收不到记录）
    spy = MagicMock()
    monkeypatch.setattr("jiuwenswarm.agents.harness.team.config_loader.logger", spy)
    spec = load_team_spec_dict(
        config_base=_two_entry_models_config(),
        requested_model_name="page-selected-model",
    )

    for role in ("leader", "teammate"):
        model = spec["agents"][role]["model"]
        assert model["model_client_config"]["model_name"] == "page-selected-model"
        # 凭证复用首条目
        assert model["model_client_config"]["api_base"] == "https://first.example.test/v1"
        assert model["model_client_config"]["api_key"] == "sk-first"
        assert model["model_client_config"]["client_provider"] == "OpenAI"
        assert model["model_request_config"]["model"] == "page-selected-model"
        assert model["model_request_config"]["temperature"] == 0.1
    # 覆盖动作必须可观测（info 日志含 requested 与命中失败说明）
    logged = [c.args[0] % c.args[1:] if len(c.args) > 1 else c.args[0] for c in spy.info.call_args_list if c.args]
    assert any("page-selected-model" in m and "not in models.defaults" in m for m in logged)


def test_load_team_spec_dict_requested_model_miss_overrides_legacy_default():
    """models.default 旧格式同样适用覆盖语义。"""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "api_base": "https://legacy.example.test/v1",
                    "api_key": "sk-legacy",
                    "model_name": "legacy-model",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team({"demo_team": {"team_name": "demo_team", "agents": {"leader": {}}}}),
    }

    spec = load_team_spec_dict(config_base=config, requested_model_name="page-model")

    model = spec["agents"]["leader"]["model"]
    assert model["model_client_config"]["model_name"] == "page-model"
    assert model["model_client_config"]["api_key"] == "sk-legacy"
    assert model["model_request_config"]["model"] == "page-model"


def test_load_team_spec_dict_requested_model_equal_to_default_is_noop():
    """页面选择恰等于默认条目时走命中/恒等路径，不产生覆盖。"""
    spec = load_team_spec_dict(
        config_base=_two_entry_models_config(),
        requested_model_name="first-model",
    )

    model = spec["agents"]["leader"]["model"]
    assert model["model_client_config"]["model_name"] == "first-model"
    assert model["model_client_config"]["api_key"] == "sk-first"


def test_load_team_spec_dict_requested_model_blank_keeps_first_entry():
    """空白 requested_model_name 不触发覆盖，保持首条目回退（现状行为）。"""
    spec = load_team_spec_dict(
        config_base=_two_entry_models_config(),
        requested_model_name="   ",
    )

    model = spec["agents"]["leader"]["model"]
    assert model["model_client_config"]["model_name"] == "first-model"


def test_load_team_spec_dict_requested_model_does_not_override_explicit_member_model():
    """成员 manifest 显式配置的 model 不受页面选择影响（setdefault 不覆盖）。"""
    config = _two_entry_models_config()
    config["modes"]["team"]["demo_team"]["agents"]["teammate"]["model"] = {
        "model_client_config": {
            "api_base": "https://member.example.test/v1",
            "api_key": "sk-member",
            "model_name": "member-pinned-model",
            "client_provider": "OpenAI",
        },
        "model_config_obj": {},
    }

    spec = load_team_spec_dict(config_base=config, requested_model_name="page-model")

    assert spec["agents"]["teammate"]["model"]["model_client_config"]["model_name"] == "member-pinned-model"
    assert spec["agents"]["leader"]["model"]["model_client_config"]["model_name"] == "page-model"


def test_load_team_spec_dict_uses_first_models_defaults_entry_for_team(monkeypatch):
    """Team config loading should use the first models.defaults entry."""
    config = {
        "models": {
            "defaults": [
                {
                    "model_client_config": {
                        "api_base": "https://first.example.test/v1",
                        "api_key": "sk-first",
                        "model_name": "first-model",
                        "client_provider": "OpenAI",
                    },
                    "model_config_obj": {"temperature": 0.1},
                    "is_default": False,
                },
                {
                    "model_client_config": {
                        "api_base": "https://second.example.test/v1",
                        "api_key": "sk-second",
                        "model_name": "second-model",
                        "client_provider": "OpenAI",
                    },
                    "model_config_obj": {"temperature": 0.9},
                    "is_default": True,
                },
            ]
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                        "teammate": {},
                    },
                }
            }
        ),
    }

    spec = load_team_spec_dict(config_base=config)

    model = spec["agents"]["leader"]["model"]
    assert model["model_client_config"]["api_base"] == "https://first.example.test/v1"
    assert model["model_client_config"]["api_key"] == "sk-first"
    assert model["model_client_config"]["model_name"] == "first-model"
    assert model["model_request_config"]["model"] == "first-model"
    assert model["model_request_config"]["temperature"] == 0.1


def test_load_team_spec_dict_supports_member_specific_agents(monkeypatch, tmp_path):
    """Predefined members should resolve to member_name-keyed DeepAgentSpec entries."""
    fake_agent_teams_home = tmp_path / ".agent_teams"
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-test",
                    "client_provider": "openai",
                },
                "model_config_obj": {"temperature": 0.2},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "leader": {
                        "member_name": "team_leader",
                        "display_name": "TeamLeader",
                        "persona": "Lead the team",
                    },
                    "workspace": {
                        "enabled": True,
                        "artifact_dirs": ["artifacts/reports"],
                    },
                    "agents": {
                        "leader": {},
                        "teammate": {},
                        "analyst": {
                            "name": "Analyst",
                            "skills": ["skill-a", "skill-b"],
                        },
                    },
                    "predefined_members": [
                        {
                            "member_name": "analyst",
                            "display_name": "Data Analyst",
                            "persona": "Analyze data",
                            "prompt_hint": "Focus on trends",
                            "toolkits": ["sql", "python"],
                        }
                    ],
                    "storage": {
                        "type": "sqlite",
                        "params": {
                            "connection_string": "team.db",
                        },
                    },
                    "planning": {
                        "enabled": True,
                        "max_parallel_tasks": 3,
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: fake_agent_teams_home,
    )

    spec = load_team_spec_dict()

    assert spec["team_name"] == "demo_team"
    assert spec["leader"]["member_name"] == "team_leader"
    assert spec["leader"]["display_name"] == "TeamLeader"
    assert spec["leader"]["persona"] == "Lead the team"
    assert spec["predefined_members"][0]["member_name"] == "analyst"
    assert spec["predefined_members"][0]["display_name"] == "Data Analyst"
    assert spec["predefined_members"][0]["prompt_hint"] == "Focus on trends"
    assert spec["predefined_members"][0]["toolkits"] == ["sql", "python"]
    assert spec["workspace"]["enabled"] is True
    assert spec["workspace"]["artifact_dirs"] == ["artifacts/reports"]
    assert spec["planning"] == {
        "enabled": True,
        "max_parallel_tasks": 3,
    }
    assert spec["agents"]["analyst"]["skills"] == ["skill-a", "skill-b"]
    assert spec["agents"]["analyst"]["model"]["model_request_config"]["model"] == "gpt-test"
    assert spec["agents"]["analyst"]["workspace"] == {"stable_base": True}
    assert spec["storage"]["params"]["connection_string"] == str(
        fake_agent_teams_home / "team.db"
    )


def test_load_team_spec_dict_uses_first_team_from_modes_team(monkeypatch, tmp_path):
    """The current runtime should default to the first team entry in modes.team."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-first",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "alpha_team": {
                    "team_name": "alpha_team",
                    "leader": {
                        "member_name": "alpha_leader",
                        "display_name": "Alpha Leader",
                        "persona": "Lead alpha",
                    },
                    "agents": {"leader": {"skills": ["alpha-skill"]}},
                },
                "beta_team": {
                    "team_name": "beta_team",
                    "leader": {
                        "member_name": "beta_leader",
                        "display_name": "Beta Leader",
                        "persona": "Lead beta",
                    },
                    "agents": {"leader": {"skills": ["beta-skill"]}},
                },
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["team_name"] == "alpha_team"
    assert spec["leader"]["member_name"] == "alpha_leader"
    assert spec["agents"]["leader"]["skills"] == ["alpha-skill"]


def test_load_team_spec_dict_selects_requested_template_id(monkeypatch, tmp_path):
    """Runtime binding should be able to pick a specific configured team template."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-template",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "alpha": {
                    "team_name": "alpha_default_name",
                    "agents": {"leader": {"skills": ["alpha-skill"]}},
                },
                "beta": {
                    "team_name": "beta_default_name",
                    "leader": {"member_name": "beta_leader"},
                    "agents": {"leader": {"skills": ["beta-skill"]}},
                },
            }
        ),
    }
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict(config_base=config, template_id="beta")

    assert spec["team_name"] == "beta_default_name"
    assert spec["leader"]["member_name"] == "beta_leader"
    assert spec["agents"]["leader"]["skills"] == ["beta-skill"]


def test_team_template_snapshot_survives_deleted_template(monkeypatch, tmp_path):
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-template",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "alpha": {
                    "team_name": "alpha_default_name",
                    "agents": {"leader": {"skills": ["alpha-skill"]}},
                },
                "beta": {
                    "team_name": "beta_default_name",
                    "leader": {"member_name": "beta_leader"},
                    "agents": {"leader": {"skills": ["beta-skill"]}},
                },
            }
        ),
    }
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    snapshot = get_team_template_snapshot(config_base=config, template_id="beta")
    config["modes"]["team"].pop("beta")
    spec = load_team_spec_dict(
        config_base=config,
        template_id="beta",
        template_snapshot=snapshot,
        strict_template=True,
    )

    assert spec["team_name"] == "beta_default_name"
    assert spec["leader"]["member_name"] == "beta_leader"
    assert spec["agents"]["leader"]["skills"] == ["beta-skill"]


def test_load_team_spec_dict_strict_template_rejects_missing_id(monkeypatch, tmp_path):
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-template",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "alpha": {
                    "team_name": "alpha_default_name",
                    "agents": {"leader": {"skills": ["alpha-skill"]}},
                },
            }
        ),
    }
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    with pytest.raises(TeamTemplateNotFoundError, match="missing-template"):
        load_team_spec_dict(
            config_base=config,
            template_id="missing-template",
            strict_template=True,
        )


def test_list_team_template_summaries_reads_modes_team_templates() -> None:
    config = _wrap_modes_team(
        {
            "default": {
                "team_name": "configured_team",
                "display_name": "Default Team",
                "agents": {"leader": {}},
            },
            "research": {
                "name": "Research Template",
                "agents": {"leader": {}},
            },
        }
    )

    summaries = list_team_template_summaries(config)

    assert summaries == [
        {
            "template_id": "default",
            "display_name": "Default Team",
            "available": True,
            "source": "modes.team.default",
            "team_name": "configured_team",
        },
        {
            "template_id": "research",
            "display_name": "Research Template",
            "available": True,
            "source": "modes.team.research",
            "team_name": "",
        },
    ]


def test_list_team_template_summaries_falls_back_to_legacy_team_config() -> None:
    config = {
        "team": {
            "team_name": "legacy_team",
            "leader": {"member_name": "legacy_leader"},
            "agents": {"leader": {}},
        },
    }

    summaries = list_team_template_summaries(config)

    assert summaries == [
        {
            "template_id": "legacy_team",
            "display_name": "legacy_team",
            "available": True,
            "source": "team",
            "team_name": "legacy_team",
        },
    ]


def test_load_team_spec_dict_selects_legacy_template_id(monkeypatch, tmp_path):
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-legacy",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        "team": {
            "team_name": "legacy_team",
            "leader": {"member_name": "legacy_leader"},
            "agents": {"leader": {"skills": ["legacy-skill"]}},
        },
    }
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict(config_base=config, template_id="legacy_team")

    assert spec["team_name"] == "legacy_team"
    assert spec["leader"]["member_name"] == "legacy_leader"
    assert spec["agents"]["leader"]["skills"] == ["legacy-skill"]


def test_load_team_spec_dict_fills_default_transport_and_workspace(monkeypatch, tmp_path):
    """Missing team transport/workspace should fall back to local inprocess defaults."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-defaults",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                        "reviewer": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["transport"] == {"type": "inprocess"}
    assert spec["workspace"] == {
        "enabled": True,
        "version_control": False,
    }
    assert spec["agents"]["leader"]["workspace"] == {"stable_base": True}
    assert spec["agents"]["reviewer"]["workspace"] == {"stable_base": True}


def test_load_team_spec_dict_defaults_enable_hitt_to_true(monkeypatch, tmp_path):
    """Missing enable_hitt should default to enabled for team mode."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-hitt-default",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["enable_hitt"] is True


def test_load_team_spec_dict_preserves_explicit_enable_hitt_false(monkeypatch, tmp_path):
    """Explicit enable_hitt false should not be overwritten by defaults."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-hitt-disabled",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "enable_hitt": False,
                    "agents": {
                        "leader": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["enable_hitt"] is False


@pytest.mark.skip(reason="enable_swarmflow default injection has been removed")
def test_load_team_spec_dict_defaults_enable_swarmflow_to_true(monkeypatch, tmp_path):
    """Missing enable_swarmflow should default to enabled for team mode."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-swarmflow-default",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["enable_swarmflow"] is True


@pytest.mark.skip(reason="enable_swarmflow loader behavior is no longer validated")
def test_load_team_spec_dict_preserves_explicit_enable_swarmflow_false(monkeypatch, tmp_path):
    """Explicit enable_swarmflow false should not be overwritten by defaults."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-swarmflow-disabled",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "enable_swarmflow": False,
                    "agents": {
                        "leader": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["enable_swarmflow"] is False


def test_load_team_spec_dict_adds_default_teammate_when_only_leader_configured(monkeypatch, tmp_path):
    """A leader-only team config still needs a teammate template for dynamic spawns."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-role-default",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {
                            "skills": ["team-management"],
                        },
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert set(spec["agents"]) == {"leader", "teammate"}
    assert spec["agents"]["leader"]["skills"] == ["team-management"]
    assert "skills" not in spec["agents"]["teammate"]


def test_load_team_spec_dict_keeps_role_defaults_when_member_alias_is_added(monkeypatch, tmp_path):
    """Role keys should remain usable after member_name aliases are injected."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-role",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                        "teammate": {
                            "skills": ["shared-skill"],
                        },
                        "default_teammate": {
                            "skills": ["member-skill"],
                        },
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert "leader" in spec["agents"]
    assert "teammate" in spec["agents"]
    assert "default_teammate" in spec["agents"]
    assert spec["agents"]["default_teammate"]["skills"] == ["member-skill"]
    assert spec["agents"]["teammate"]["skills"] == ["shared-skill"]


def test_load_team_spec_dict_preserves_explicit_empty_skills(monkeypatch, tmp_path):
    """Explicit empty skill lists should not be treated as missing config."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-empty",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                        "reviewer": {
                            "skills": [],
                        },
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert "reviewer" in spec["agents"]
    assert spec["agents"]["reviewer"]["skills"] == []


def test_load_team_spec_dict_no_auto_fill_skills_when_missing(monkeypatch, tmp_path):
    """Missing skills config should not auto-fill with global skills (new behavior)."""
    global_skills_dir = tmp_path / "skills"
    (global_skills_dir / "skill-a").mkdir(parents=True)
    (global_skills_dir / "skill-a" / "SKILL.md").write_text("# skill-a", encoding="utf-8")
    (global_skills_dir / "skill-b").mkdir(parents=True)
    (global_skills_dir / "skill-b" / "SKILL.md").write_text("# skill-b", encoding="utf-8")
    (global_skills_dir / "_internal").mkdir(parents=True)

    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-all",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                        "writer": {},
                    },
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    # skills should not be auto-filled when not configured
    assert "skills" not in spec["agents"]["leader"]
    assert "skills" not in spec["agents"]["writer"]


def test_resolve_team_sqlite_db_path_defaults_to_agent_teams_home(monkeypatch, tmp_path):
    """Missing connection_string should fall back to openjiuwen agent-teams team.db."""
    config = _wrap_modes_team(
        {
            "demo_team": {
                "team_name": "demo_team",
                "storage": {
                    "type": "sqlite",
                    "params": {},
                },
            }
        }
    )

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    db_path = resolve_team_sqlite_db_path()

    assert db_path == Path(tmp_path / ".agent_teams" / "team.db")


def test_load_team_spec_dict_preserves_arbitrary_team_top_level_fields(monkeypatch, tmp_path):
    """Unknown team-level fields should be preserved in the final spec dict."""
    config = {
        "models": {
            "default": {
                "model_client_config": {
                    "model_name": "gpt-custom",
                    "client_provider": "openai",
                },
                "model_config_obj": {},
            }
        },
        **_wrap_modes_team(
            {
                "demo_team": {
                    "team_name": "demo_team",
                    "agents": {
                        "leader": {},
                    },
                    "runtime_flags": {
                        "enable_observer": True,
                        "retry_limit": 5,
                    },
                    "custom_labels": ["a", "b"],
                }
            }
        ),
    }

    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "jiuwenswarm.agents.harness.team.config_loader.get_agent_teams_home",
        lambda: tmp_path / ".agent_teams",
    )

    spec = load_team_spec_dict()

    assert spec["runtime_flags"] == {
        "enable_observer": True,
        "retry_limit": 5,
    }
    assert spec["custom_labels"] == ["a", "b"]
