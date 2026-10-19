"""step77：DSpark 的**边界与明确拒绝**（需求 077 §3.4/§3.5 与 §4 的最后两条）。

每一条都是"上游会报错、或者本仓库不支持时必须当场报错"的情形；**不许**出现"配置写了、
实际静默按别的语义跑"：

    配置期  PP>1 / 非法 top-k / 非 Qwen3 架构 / K 与 n_predict 的整除 / 79 关的自适应验证
    加载期  checkpoint 没有 confidence 头却要求自适应验证
    共享    checkpoint 没有 embed_tokens / lm_head 时挂 target 的那一份
    V1      DSpark 只有 V2 通路（上游 `use_v2_model_runner` 为 dspark 强制 V2）
"""

import json

import pytest
import torch

from minivllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from minivllm.worker.gpu.spec_decode.dspark.utils import load_dspark_model

from dspark_helpers import (fake_pp_config, make_spec_config,  # noqa: E402
                            make_speculator, model_configs, tiny_pair)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSpark 的提议者要加载模型并绑 KV 缓存（只在 CUDA 上跑）")


# ---------------- 配置期 ----------------

def test_pipeline_parallelism_is_rejected():
    """PP>1：上游 `load_dspark_model()` 里那条 `NotImplementedError` 原样保留。"""
    spec = make_spec_config(k=3)
    fake = fake_pp_config(spec, pp_size=2)
    with pytest.raises(NotImplementedError, match="pipeline parallel"):
        load_dspark_model(target_model=None, vllm_config=fake)


def test_illegal_topk_is_rejected():
    """top-k 必须落在 [1, draft 词表]：0 与超过词表都报错（不是截到边界）。"""
    for bad in (0, 12):
        with pytest.raises(ValueError, match="dspark_draft_topk"):
            make_spec_config(k=3, draft_topk=bad)


def test_wrong_architecture_is_rejected():
    """非 Qwen3 的 DSpark 架构（Gemma4 / K3 / DeepSeek-V4）明确拒绝。"""
    _, draft_config = model_configs()
    hf = dict(draft_config.hf_config)
    hf["architectures"] = ["Gemma4DSparkModel"]
    draft = ModelConfig(model=draft_config.model, dtype="float32", max_model_len=64,
                        hf_config=hf)
    with pytest.raises(NotImplementedError, match="Gemma4DSparkModel"):
        SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                          draft_model_config=draft)


def test_k_must_be_divisible_by_n_predict():
    """超过 `n_predict` 时 K 必须能整除它（上游的模块复用约束）。"""
    _, draft_config = model_configs()
    hf = dict(draft_config.hf_config)
    hf["n_predict"] = 2
    draft = ModelConfig(model=draft_config.model, dtype="float32", max_model_len=64,
                        hf_config=hf)
    # 3 > 2 且 3 % 2 != 0 → 上游同款报错
    with pytest.raises(ValueError, match="n_predict"):
        SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                          draft_model_config=draft)


def test_k_can_default_from_n_predict():
    """K 没给（0）时取 draft 配置的 `n_predict`（上游同一条缺省）。"""
    _, draft_config = model_configs()
    hf = dict(draft_config.hf_config)
    hf["n_predict"] = 4
    draft = ModelConfig(model=draft_config.model, dtype="float32", max_model_len=64,
                        hf_config=hf)
    spec = SpeculativeConfig(method="dspark", num_speculative_tokens=0,
                             draft_model_config=draft)
    assert spec.num_speculative_tokens == 4


def test_adaptive_verification_belongs_to_step_79():
    """`enable_adaptive_verification=True` 在 77 关明确拒绝（预算决策属 79 关）。"""
    _, draft_config = model_configs(confidence_head=True)
    with pytest.raises(NotImplementedError, match="79"):
        SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                          draft_model_config=draft_config,
                          enable_adaptive_verification=True)


def test_adaptive_verification_requires_dspark():
    """非 DSpark 方法打开它 → 上游原话的那条 ValueError。"""
    _, draft_config = model_configs()
    with pytest.raises(ValueError, match="only supported with DSpark"):
        SpeculativeConfig(method="draft_model", num_speculative_tokens=2,
                          draft_model_config=draft_config,
                          enable_adaptive_verification=True)


def test_v1_runner_is_rejected():
    """DSpark 只有 V2：不设 `VLLM_USE_V2_MODEL_RUNNER` 时配置期就报错。"""
    spec = make_spec_config(k=3)
    target_config, _ = model_configs()
    import os

    old = os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    try:
        with pytest.raises(NotImplementedError, match="VLLM_USE_V2_MODEL_RUNNER"):
            VllmConfig(model_config=target_config, speculative_config=spec)
    finally:
        if old is not None:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old


# ---------------- 注意力配置（上游 `dflash_has_any_non_causal` 的等价物） ----------------

@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_draft_attention_causality_comes_from_the_model(causal):
    """草稿的因果性由**模型逐层配置**决定（上游 `dflash_has_any_non_causal` 同一件事）。

    本仓库没有 `attention_config` 这一轴可切换后端，所以 `load_dspark_model()` 不需要改写
    它——但"预设器读到的因果性"必须与配置一致，否则注意力掩码会按另一套语义算。
    """
    speculator, _ = make_speculator(k=3, causal=causal)
    assert speculator.dflash_causal is causal
    assert speculator.model.get_draft_attn_causal() == [causal] * 2


# ---------------- 加载期 ----------------

@requires_cuda
def test_missing_confidence_head_is_rejected_when_adaptive_is_on():
    """打开自适应验证但 checkpoint 没有 confidence 头 → 加载期报错（上游同款）。"""
    speculator, target_model = make_speculator(k=3, confidence_head=False)
    assert speculator.model.model.confidence_head is None
    speculator.enable_adaptive_verification = True
    with pytest.raises(ValueError, match="confidence"):
        speculator.load_draft_model(target_model)


@requires_cuda
def test_missing_embed_tokens_and_lm_head_are_shared_from_target():
    """checkpoint 里没有这两份 → 挂 target 的模块（不是留一份随机初始化）。"""
    speculator, target_model = make_speculator(k=3, omit_embed_tokens=True,
                                               omit_lm_head=True)
    draft_model = speculator.model
    assert draft_model.has_own_embed_tokens is False
    assert draft_model.has_own_lm_head is False
    assert draft_model.model.embed_tokens is target_model.model.embed_tokens
    assert draft_model.lm_head is target_model.lm_head


@requires_cuda
def test_own_weights_are_kept_when_present():
    """checkpoint 自带这两份 → 用自己的（共享判据不是"形状相同就共享"）。"""
    speculator, target_model = make_speculator(k=3)
    draft_model = speculator.model
    assert draft_model.has_own_embed_tokens is True
    assert draft_model.has_own_lm_head is True
    assert draft_model.model.embed_tokens is not target_model.model.embed_tokens
    assert draft_model.lm_head is not target_model.lm_head


def test_missing_mask_token_is_rejected():
    """没有 noise/mask token：K−1 行会填进一个"模型没见过的"输入 —— 配置期报错。"""
    _, draft_config = model_configs()
    hf = json.loads(json.dumps(draft_config.hf_config))
    hf.pop("mask_token_id", None)
    hf.pop("dspark_noise_token_id", None)
    hf["dflash_config"] = {k: v for k, v in (hf.get("dflash_config") or {}).items()
                           if k != "mask_token_id"}
    draft = ModelConfig(model=draft_config.model, dtype="float32", max_model_len=64,
                        hf_config=hf)
    with pytest.raises(ValueError, match="mask token"):
        SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                          draft_model_config=draft)


def test_markov_rank_is_required_by_the_model():
    """模型侧也要拒绝：没有 r 就建不出 Markov 头（不是静默用某个缺省秩）。"""
    from minivllm.models.qwen3_dspark import Qwen3DSparkForCausalLM

    *_, draft_hf = tiny_pair()
    hf = {key: value for key, value in draft_hf.items() if key != "markov_rank"}
    with pytest.raises(ValueError, match="markov_rank"):
        Qwen3DSparkForCausalLM(hf)
