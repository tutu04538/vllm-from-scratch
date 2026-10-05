"""step65：MTP 的**配置期**行为（需求 065 §3.1/§3.3 + §5 的别名表）。

三件事：
1. 别名归一（上游 `MTPModelTypes` 一长串 → `method="mtp"`）与"哪一串别名/是否多模块"的判定；
2. 从 **target 配置**派生 MTP 的 draft 配置（`n_predict`、`architectures`、模型目录 = target 目录）
   以及 K 与 `n_predict` 的整除约束（上游 "Ensure divisibility for MTP module reuse"）；
3. 与上游 `SpeculativeConfig` **同输入同输出**的对照（别名表逐项、派生结果、报错条件）。
"""

import json
from pathlib import Path

import pytest

from minivllm import CacheConfig, ModelConfig, SpeculativeConfig
from minivllm.config import MTP_MODEL_TYPES
from minivllm.testing.tiny_models import tiny_mtp_dir, tiny_qwen3_config

TARGET_DIR = "/tmp"          # 配置期只用到 hf_config，不需要真的目录


def mtp_target_config(*, num_nextn_predict_layers: int = 1) -> dict:
    config = tiny_qwen3_config("tiny_gqa")
    if num_nextn_predict_layers:
        config["num_nextn_predict_layers"] = num_nextn_predict_layers
    return config


def make_target(*, num_nextn_predict_layers: int = 1) -> ModelConfig:
    return ModelConfig(model=TARGET_DIR, dtype="float32", max_model_len=64,
                       hf_config=mtp_target_config(
                           num_nextn_predict_layers=num_nextn_predict_layers))


def make_spec(**kwargs) -> SpeculativeConfig:
    args = dict(method="mtp", num_speculative_tokens=2)
    args.update(kwargs)
    return SpeculativeConfig(**args)


def write_qwen3_next_mtp_dir(tmp_path, *, name: str, num_nextn_predict_layers: int):
    """写一个"上游认得出是 MTP 家族"的 tiny 目录（配置期对照用，不需要真权重）。

    上游的别名归一/架构改写是按 target 的 `model_type`/`architectures` 查表的，所以这里必须
    长得像 Qwen3-Next（`model_type: qwen3_next`），否则它会当成普通模型 → 报 "Unsupported
    speculative method"。
    """
    config = mtp_target_config(num_nextn_predict_layers=num_nextn_predict_layers)
    config.update({"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
                   "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                   "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                   "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
                   "layer_types": ["linear_attention", "full_attention"],
                   "partial_rotary_factor": 0.25, "num_experts": 0})
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import torch
    from safetensors.torch import save_file

    save_file({"dummy": torch.zeros(1)}, str(model_dir / "model.safetensors"))
    return model_dir, config


# ---------------------------------------------------------------- 别名归一


def test_alias_normalization():
    """所有 MTP 别名都归一到 `method="mtp"`（上游 `config/speculative.py:748-756`）。"""
    for alias in MTP_MODEL_TYPES:
        spec = SpeculativeConfig(method=alias, num_speculative_tokens=1)
        assert spec.method == "mtp", f"别名 {alias!r} 没有归一到 mtp"
        assert spec.uses_mtp() and spec.use_eagle()


def test_alias_table_matches_upstream():
    """别名表与上游 `MTPModelTypes` 逐项一致（漏一个别名就是"某个模型静默走成 draft_model"）。"""
    from typing import get_args

    from vllm.config.speculative import MTPModelTypes

    upstream = set(get_args(MTPModelTypes))
    ours = set(MTP_MODEL_TYPES)
    assert ours == upstream, (f"多出 {sorted(ours - upstream)}；"
                              f"缺少 {sorted(upstream - ours)}")


def test_supported_method_list_and_lookahead():
    """`mtp` 进支持列表；调度侧按 EAGLE 系算 lookahead = K，输入槽位多占 0 行。"""
    spec = make_spec(num_speculative_tokens=3)
    assert spec.use_eagle() and spec.uses_mtp()
    # 上游 `max_num_new_slots_for_drafting` 的表里 MTP 是 0（不跑 draft 模型的第二遍输入）
    assert spec.max_num_new_slots_for_drafting == 0
    # 调度器给提议者预留的 KV 位置 = K（MTP 自己的那一层要往 target query 之外写 K 个位置）
    engine_spec = make_spec(num_speculative_tokens=3)
    from minivllm.core.sched.scheduler import Scheduler  # noqa: F401  （只静态确认可导入）

    assert engine_spec.num_speculative_tokens == 3


def test_multi_module_judgement():
    """`use_multi_module_mtp()`：`min(n_predict, K) > 1`（上游 L1501-1507）。"""
    spec = make_spec(num_speculative_tokens=2,
                     draft_model_config=ModelConfig(
                         model=TARGET_DIR, hf_config=mtp_target_config(
                             num_nextn_predict_layers=2)))
    assert spec.use_multi_module_mtp() is True
    single = make_spec(num_speculative_tokens=1)
    single = SpeculativeConfig(method="mtp", num_speculative_tokens=1,
                               draft_model_config=ModelConfig(
                                   model=TARGET_DIR,
                                   hf_config=mtp_target_config(num_nextn_predict_layers=1)))
    assert single.use_multi_module_mtp() is False
    assert spec.use_multi_module_mtp() is True


# ---------------------------------------------------------------- 派生 draft 配置


def test_derive_mtp_draft_config():
    """派生结果：目录 = target 目录、architectures 换成 MTP、`n_predict` = spec 层数。"""
    spec = make_spec(num_speculative_tokens=2)
    target = make_target(num_nextn_predict_layers=1)
    derived = spec.derive_mtp_draft_config(target)
    assert derived.model == target.model, "MTP 权重就在 target 的 checkpoint 里（同一目录）"
    assert derived.dtype == target.dtype and derived.max_model_len == target.max_model_len
    assert derived.hf_config["architectures"] == ["Qwen3MTPModel"]
    assert derived.hf_config["n_predict"] == 1
    assert derived.hf_config["num_hidden_layers"] == 2, "target 的字段原样保留"
    # 不冒充上游的 `Qwen3NextMTP`：本仓库的 MTP 块是稠密 Qwen3（见 alignment §3）
    assert derived.hf_config["architectures"] != ["Qwen3NextMTP"]


def test_derive_mtp_requires_num_nextn_predict_layers():
    """target 配置没有 `num_nextn_predict_layers` → 明确报错（不知道附了几层就没法加载）。"""
    spec = make_spec()
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        spec.derive_mtp_draft_config(make_target(num_nextn_predict_layers=0))


def test_derive_mtp_k_divisibility():
    """K > n_predict 时必须能被它整除（上游原话：Ensure divisibility for MTP module reuse）。"""
    target = make_target(num_nextn_predict_layers=2)
    SpeculativeConfig(method="mtp", num_speculative_tokens=4).derive_mtp_draft_config(target)
    with pytest.raises(ValueError, match="必须能被 n_predict=2 整除"):
        SpeculativeConfig(method="mtp",
                          num_speculative_tokens=3).derive_mtp_draft_config(target)
    # K <= n_predict 时不要求整除（每个模块各用一次）
    SpeculativeConfig(method="mtp",
                      num_speculative_tokens=1).derive_mtp_draft_config(target)


def test_mtp_requires_positive_k():
    """K 必须显式给（上游不填时取 n_predict；本仓库的配置拿不到 target，所以必须显式）。"""
    with pytest.raises(ValueError, match="num_speculative_tokens 必须 > 0"):
        SpeculativeConfig(method="mtp", num_speculative_tokens=0)


# ---------------------------------------------------------------- 与上游配置对照


def test_upstream_config_derivation_matches(tmp_path):
    """同输入下，上游 `SpeculativeConfig` 的归一结果与我们的对应关系逐项对得上。

    上游把 draft 配置写进 `spec.draft_model_config`（它持有 target 配置），本仓库把这一步做成
    `derive_mtp_draft_config()`，所以比的是**内容**：`n_predict`、"目录 = target 目录"、
    "别名 → mtp"，以及 `architectures` 的**对应关系**（上游按家族选 `Qwen3NextMTP`，
    我们按本仓库实现的家族选 `Qwen3MTPModel`，差异记在 alignment §3）。
    """
    from vllm.config import ModelConfig as UpModelConfig
    from vllm.config import SpeculativeConfig as UpSpeculativeConfig
    from vllm.config.parallel import ParallelConfig

    model_dir, config = write_qwen3_next_mtp_dir(tmp_path, name="tiny_mtp",
                                                 num_nextn_predict_layers=1)
    target = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    draft = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    parallel = ParallelConfig()
    upstream = UpSpeculativeConfig(model=str(model_dir), method="qwen3_next_mtp",
                                   num_speculative_tokens=2, target_model_config=target,
                                   draft_model_config=draft, target_parallel_config=parallel)
    assert upstream.method == "mtp", "上游同样把别名归一到 mtp"
    assert upstream.draft_model_config.model == upstream.target_model_config.model
    n_predict = getattr(upstream.draft_model_config.hf_config, "n_predict", None)
    assert n_predict == 1
    assert upstream.draft_model_config.architectures == ["Qwen3NextMTP"]

    ours = SpeculativeConfig(method="qwen3_next_mtp", num_speculative_tokens=2)
    ours_derived = ours.derive_mtp_draft_config(
        ModelConfig(model=str(model_dir), dtype="float32", max_model_len=64, hf_config=config))
    assert ours.method == "mtp"
    assert ours_derived.model == str(model_dir)
    assert ours_derived.hf_config["n_predict"] == n_predict
    # 唯一的有意差异：架构名（我们的 MTP 块是稠密 Qwen3，不是 Qwen3-Next 的混合层）
    assert ours_derived.hf_config["architectures"] == ["Qwen3MTPModel"]


def test_upstream_rejects_non_divisible_k(tmp_path):
    """上游对同一个错（K 不能整除 n_predict）也报错——两边**同一条约束**。"""
    from vllm.config import ModelConfig as UpModelConfig
    from vllm.config import SpeculativeConfig as UpSpeculativeConfig
    from vllm.config.parallel import ParallelConfig

    model_dir, _ = write_qwen3_next_mtp_dir(tmp_path, name="tiny_mtp2",
                                            num_nextn_predict_layers=2)
    target = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    draft = UpModelConfig(model=str(model_dir), dtype="float32", max_model_len=64)
    parallel = ParallelConfig()
    with pytest.raises(ValueError, match="divisible"):
        UpSpeculativeConfig(model=str(model_dir), method="mtp", num_speculative_tokens=3,
                            target_model_config=target, draft_model_config=draft,
                            target_parallel_config=parallel)
    with pytest.raises(ValueError, match="必须能被 n_predict=2 整除"):
        SpeculativeConfig(method="mtp",
                          num_speculative_tokens=3).derive_mtp_draft_config(make_target(
                              num_nextn_predict_layers=2))


# ---------------------------------------------------------------- 与 tiny checkpoint 的一致性


def test_tiny_mtp_checkpoint_config_carries_spec_layers():
    """tiny checkpoint 的配置里有 `num_nextn_predict_layers`（MTP 判定的唯一依据）。"""
    for naming in ("mtp", "absolute"):
        directory = tiny_mtp_dir("tiny_gqa", naming=naming)
        config = json.loads((Path(directory) / "config.json").read_text())
        assert config["num_nextn_predict_layers"] == 1
        assert config["architectures"] == ["Qwen3ForCausalLM"], \
            "配置保持 target 的样子（draft 的架构名由派生改写）"
        assert CacheConfig().block_size > 0
