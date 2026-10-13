# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""档位关键字（fast/balanced/extreme/auto）必须在构建 Model 的唯一入口翻成具体模型。

背景：档位关键字有两条通道。单 agent 主链路走 request param，由 ModelRoutingRail
翻译；另一条经有效模型 env → MODEL_NAME → config.yaml models.defaults 直接流进
``_build_model_from_entry`` 构建 Model。没有 rail 的消费方——子代理（deep_agent
直接继承 ``_deep_config.model``，且 ``subagent_rails=None`` 不继承父 rail）、code
模式、DeepResearch 隔离子进程——会把裸档位当模型名直送上游网关，触发模型无效类错误。
"""

from __future__ import annotations

import pytest

from jiuwenswarm.server.runtime.agent_adapter.interface_deep import JiuWenSwarmDeepAdapter


def _entry(model_name: str) -> dict:
    return {
        "is_default": True,
        "model_client_config": {
            "model_name": model_name,
            "client_provider": "OpenAI",
            "api_base": "https://llm.example/v2/",
            "api_key": "k",
        },
        "model_config_obj": {},
    }


def _wire_model_name(model) -> str:
    """实际发往上游网关的模型名（ModelRequestConfig.model_name，alias="model"）。"""
    return str(getattr(model.model_config, "model_name", "") or "")


@pytest.mark.parametrize(
    ("gear", "concrete"),
    [
        ("extreme", "glm-5.2"),
        ("fast", "deepseek-v4.1-flash"),
        ("balanced", "deepseek-v4.1-flash"),
        ("auto", "deepseek-v4.1-flash"),
    ],
)
def test_build_model_from_entry_translates_gear_for_wire_name(gear: str, concrete: str) -> None:
    mcc = _entry(gear)["model_client_config"]
    model = JiuWenSwarmDeepAdapter._build_model_from_entry(mcc, {})

    assert _wire_model_name(model) == concrete
    # mcc 不被改写：cache key / identity 仍按档位关键字登记，relay 按档位选择不受影响。
    assert mcc["model_name"] == gear


@pytest.mark.parametrize("concrete", ["glm-5.2", "deepseek-v4.1-flash", "gpt-4"])
def test_build_model_from_entry_passes_through_concrete_model(concrete: str) -> None:
    model = JiuWenSwarmDeepAdapter._build_model_from_entry(_entry(concrete)["model_client_config"], {})

    assert _wire_model_name(model) == concrete


def test_subagent_resolving_by_gear_gets_concrete_wire_model() -> None:
    """子代理按档位关键字解析时，拿到的 Model 必须发具体模型 id。"""
    adapter = object.__new__(JiuWenSwarmDeepAdapter)
    adapter._model_cache = {}
    adapter._model_name_to_keys = {}
    adapter._model_identity_to_keys = {}
    adapter._model_identity_owners = {}
    adapter._tier_model_cache = {}
    adapter._model = None
    adapter._instance = None

    adapter._register_model_cache_entry(_entry("extreme"), {})

    # 档位关键字仍是合法查找键（relay 按档位下发 model_name / model_ref 时命中）。
    assert "extreme" in adapter._model_cache
    # 真实链路里 _create_model 一定会给出默认 Model；子代理解析入口要求它非空。
    adapter._model = adapter._model_cache["extreme"]

    model, _fallback = adapter._resolve_model_for_subagent(model_name="extreme")

    assert _wire_model_name(model) == "glm-5.2"


def test_mode_model_map_covers_four_gears() -> None:
    """四档关键字必须全部出现在 MODE_MODEL_MAP，避免漏档导致上游模型无效。

    宿主产品若另有档位映射表，须与本表同表同源；新增档位时两边同步改。
    本测试锁当前四档集合与市场 wire id。
    """
    from jiuwenswarm.agents.harness.common.model_gears import MODE_MODEL_MAP, MODEL_GEAR_NAMES

    expected_gears = {
        "fast": "deepseek-v4.1-flash",
        "balanced": "deepseek-v4.1-flash",
        "extreme": "glm-5.2",
        "auto": "deepseek-v4.1-flash",
    }
    assert MODEL_GEAR_NAMES == frozenset(expected_gears)
    assert MODE_MODEL_MAP == expected_gears


def test_sync_agent_compressor_models_rebinds_live_processors() -> None:
    """主对话换模后，已缓存的压缩器 _model 必须跟着换。"""
    from types import SimpleNamespace

    from jiuwenswarm.agents.harness.common.model_gears import sync_agent_compressor_models

    class _Proc:
        def __init__(self) -> None:
            self._model = SimpleNamespace(model_config=SimpleNamespace(model_name="extreme"))
            self._compression_executor = object()

    old_model = SimpleNamespace(
        model_config=SimpleNamespace(model_name="extreme"),
        model_client_config={"api_base": "https://old"},
    )
    new_model = SimpleNamespace(
        model_config=SimpleNamespace(model_name="glm-5.2"),
        model_client_config={"api_base": "https://new"},
    )
    proc = _Proc()
    pcfg = SimpleNamespace(model=old_model.model_config, model_client=old_model.model_client_config)
    react = SimpleNamespace(
        _config=SimpleNamespace(context_processors=[("DialogueCompressor", pcfg)]),
        context_engine=SimpleNamespace(_context_pool={"sid_ctx": SimpleNamespace(_processors=[proc])}),
    )

    updated = sync_agent_compressor_models(react, new_model)

    assert updated == 1
    assert proc._model is new_model
    assert pcfg.model is new_model.model_config
    assert pcfg.model_client is new_model.model_client_config
    # 依赖在位时 CompressionExecutor 必须重建成功，且不再是换模前的 sentinel。
    assert proc._compression_executor is not None


def test_sync_agent_compressor_models_skips_failed_executor_rebuild(monkeypatch) -> None:
    """执行器重建失败时置空，且不把该 processor 算进成功数。"""
    from types import SimpleNamespace

    from jiuwenswarm.agents.harness.common import model_gears

    class _Proc:
        def __init__(self) -> None:
            self._model = None
            self._compression_executor = object()

    def _boom(_model: object) -> object:
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(model_gears, "_new_compression_executor", _boom)
    new_model = SimpleNamespace(
        model_config=SimpleNamespace(model_name="glm-5.2"),
        model_client_config={"api_base": "https://new"},
    )
    proc = _Proc()
    react = SimpleNamespace(
        context_engine=SimpleNamespace(_context_pool={"sid": SimpleNamespace(_processors=[proc])}),
    )

    updated = model_gears.sync_agent_compressor_models(react, new_model)

    assert updated == 0
    assert proc._model is new_model
    assert proc._compression_executor is None


def test_sync_rebinds_inherited_session_memory_and_keeps_explicit_model() -> None:
    """只重绑从主模型继承来的会话记忆，显式 memory.model 不动。"""
    from types import SimpleNamespace

    from jiuwenswarm.agents.harness.common.model_gears import sync_agent_compressor_models

    old_model = SimpleNamespace(
        model_config=SimpleNamespace(model_name="extreme"),
        model_client_config={"api_base": "https://old"},
    )
    new_model = SimpleNamespace(
        model_config=SimpleNamespace(model_name="glm-5.2"),
        model_client_config={"api_base": "https://new"},
    )
    explicit_model = SimpleNamespace(model_name="custom-memory")
    explicit_client = {"api_base": "https://explicit"}
    inherited = SimpleNamespace(
        config=SimpleNamespace(
            model=old_model.model_config,
            model_client=old_model.model_client_config,
        ),
        _update_agent=SimpleNamespace(_agent=object(), _direct_model=object(), _workspace_root="/old"),
    )
    explicit = SimpleNamespace(
        config=SimpleNamespace(model=explicit_model, model_client=explicit_client),
        _update_agent=SimpleNamespace(_agent=object(), _direct_model=object(), _workspace_root="/keep"),
    )
    host = SimpleNamespace(
        _registered_rails=[
            SimpleNamespace(_session_memory_mgr=inherited),
            SimpleNamespace(_session_memory_mgr=explicit),
        ],
    )
    react = SimpleNamespace()

    updated = sync_agent_compressor_models(
        react,
        new_model,
        host=host,
        previous_model=old_model,
    )

    assert updated == 1
    assert inherited.config.model is new_model.model_config
    assert inherited.config.model_client is new_model.model_client_config
    assert inherited._update_agent._agent is None
    assert inherited._update_agent._direct_model is None
    assert inherited._update_agent._workspace_root is None
    assert explicit.config.model is explicit_model
    assert explicit.config.model_client is explicit_client
    assert explicit._update_agent._agent is not None
