"""step66：Medusa 多头提议（需求 066 §2/§4）。

分五层，与需求 §4 的验收条目一一对应：

1. **配置层**：旧 FasterDecoding checkpoint 的 `medusa_num_heads/medusa_num_layers` 改名、
   `model_type`/`architectures` 推断、缺省值与上游 `MedusaConfig` 逐个字段对齐，
   以及"**K 就是 head 数**"这条约束（上游 `num_lookahead_tokens` 的 setter）；
2. **模型/加载层**：残差块的形状与"并行读同一份 hidden"、三种权重命名、bias 的加载/记账、
   共享 lm_head + `token_map` 截断词表、未知与缺失权重都**明确报错**；
3. **上游数值对照**：用**同一份 tiny 权重**实例化 site-packages 里真的
   `model_executor/models/medusa.py::Medusa` + `v1/spec_decode/medusa.py::MedusaProposer`，
   比每个 head 的 logits 与最终候选列顺序（需求 §4 第一条）；
4. **Runner 行选择**：`select_target_hidden_states()` 的算式、上游 stride 在混合
   prefill 批次下错位的反证、中间 prefill 块不参与；
5. **端到端**：不同 K / 三种命名的 greedy 输出与非投机逐 token 相同、草稿真的按请求各提 K 枚、
   行选错时草稿必变（反证）、小块数（抢占恢复）下仍一致、提议不含 `draft_probs`（argmax = 点质量 q）。
"""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.config import MLP_SPECULATOR_MODEL_TYPE, medusa_hf_config  # noqa: F401
from minivllm.models.medusa import Medusa, ResidualBlock
from minivllm.spec_decode.medusa import MedusaProposer
from minivllm.testing.tiny_models import (tiny_medusa_dir, tiny_qwen3_config,
                                          tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 2
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]
# 本机实测的真实旧 config.json（FasterDecoding/medusa-vicuna-7b-v1.3）：
# 连 model_type / vocab_size / hidden_size / architectures 都没有
OLD_CHECKPOINT_CONFIG = {
    "base_model_name_or_path": "lmsys/vicuna-7b-v1.3",
    "medusa_num_heads": 2,
    "medusa_num_layers": 1,
    "transformers_version": "4.31.0",
}


@lru_cache(maxsize=None)
def draft_dir(naming="old", num_heads=K, num_layers=1, fc_bias=False,
              original_lm_head=False, truncated_vocab=None):
    """tiny Medusa head 目录（进程内缓存：同一套规格只生成一次）。"""
    return tiny_medusa_dir("tiny_gqa", num_heads=num_heads, num_layers=num_layers,
                           naming=naming, fc_bias=fc_bias,
                           original_lm_head=original_lm_head,
                           truncated_vocab=truncated_vocab)


def draft_hf(directory) -> dict:
    return json.loads((Path(directory) / "config.json").read_text())


def draft_model_config(directory, *, k=K, cfg=None) -> ModelConfig:
    return ModelConfig(model=directory, dtype="float32", max_model_len=64,
                       hf_config=draft_hf(directory) if cfg is None else cfg)


def medusa_spec(directory, *, k=K, cfg=None) -> SpeculativeConfig:
    return SpeculativeConfig(method="medusa", num_speculative_tokens=k,
                             draft_model_config=draft_model_config(
                                 directory, k=k, cfg=cfg))


def medusa_model(directory, *, k=K, cfg=None, extra=None) -> Medusa:
    """按**生产路径**造一个 Medusa 模型：配置归一 → 与 target 对齐词表（不加载权重）。

    单测里不能手搓 hf 配置：那会绕开"词表对齐"这一步，于是 lm_head 的宽度是
    `MedusaConfig` 的默认值 32001，跟检查点里 11 行的权重对不上。
    """
    config = dict(cfg) if cfg is not None else {**draft_hf(directory), **(extra or {})}
    spec = medusa_spec(directory, k=k, cfg=config)
    derived = spec.derive_medusa_draft_config(target_config().model_config)
    return Medusa(derived.hf_config)


def target_config(*, block_size=4, num_gpu_blocks=32, max_num_seqs=2, budget=64,
                  max_model_len=64, spec=None):
    directory = tiny_qwen3_dir("tiny_gqa")
    return VllmConfig(
        model_config=ModelConfig(model=directory, dtype="float32",
                                 max_model_len=max_model_len,
                                 hf_config=tiny_qwen3_config("tiny_gqa")),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)


_UNSET = object()


def make_engine(**kwargs):
    """起一个 tiny 引擎（不传 `spec` 时默认带 Medusa 投机 K=2）。

    **`spec=None` 表示真的不开投机**（基线引擎），所以这里用哨兵区分"没传"与"传了 None"——
    早先写成 `kwargs.pop("spec", None)` 会让 `spec=None` 也拿到默认投机配置，
    "与非投机一致"的对照就变成了"与另一个投机配置一致"（断言强度会悄悄变弱）。
    """
    spec = kwargs.pop("spec", _UNSET)
    if spec is _UNSET:
        spec = medusa_spec(draft_dir())
    config = target_config(spec=spec, **kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, prompts, *, max_tokens=6):
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs = {}
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    return outputs


def install_draft_spy(runner):
    """记录每轮交给调度器的草稿（`take_draft_token_ids()` 是协议出口）。"""
    log = []
    original = runner.take_draft_token_ids

    def spy():
        drafts = original()
        if drafts is not None:
            log.append({"req_ids": list(drafts.req_ids),
                        "drafts": [list(tokens) for tokens in drafts.draft_token_ids],
                        "has_probs": drafts.draft_probs is not None})
        return drafts

    runner.take_draft_token_ids = spy
    return log


def install_propose_spy(runner):
    """记录 `MedusaProposer.propose()` 收到的 hidden 行与交回的候选。"""
    proposer = runner.proposer
    calls = []
    original = proposer.propose

    def spy(num_speculative_tokens, target_hidden_states, sampling_metadata=None,
            slot_mappings=None):
        result = original(num_speculative_tokens, target_hidden_states, sampling_metadata,
                          slot_mappings)
        calls.append({"num_speculative_tokens": num_speculative_tokens,
                      "hidden": target_hidden_states.detach().clone(),
                      "tokens": result.detach().clone()})
        return result

    proposer.propose = spy
    return calls


# ================================================================ 1. 配置层


def test_old_checkpoint_keys_are_renamed():
    """旧 checkpoint 的 `medusa_num_heads/medusa_num_layers` → `num_heads/num_hidden_layers`。"""
    config = medusa_hf_config(OLD_CHECKPOINT_CONFIG, K)
    assert config["num_hidden_layers"] == 1          # 来自 medusa_num_layers
    assert config["num_heads"] == K                  # 被 K 覆盖（见下一条）
    assert config["model_type"] == "medusa"
    assert config["architectures"] == ["MedusaModel"]
    # 旧文件缺的字段落回 MedusaConfig 的默认值
    assert config["hidden_size"] == 4096 and config["vocab_size"] == 32001
    assert config["truncated_vocab_size"] == 32001   # 没写 = 不截断
    assert config["max_paths"] == 64 and config["topk"] == 10
    assert config["max_seq_len"] == 2 ** 20
    # 与 target 无关的原始字段原样保留（上游会丢给 PretrainedConfig 当普通属性）
    assert config["base_model_name_or_path"] == "lmsys/vicuna-7b-v1.3"


def test_defaults_match_upstream_medusa_config():
    """缺省值逐个字段对上游 `MedusaConfig()` 的同一份默认值（差分，不靠记忆）。"""
    from vllm.transformers_utils.configs.medusa import MedusaConfig

    ours = medusa_hf_config({}, K)
    upstream = MedusaConfig()
    for field in ("hidden_size", "vocab_size", "num_hidden_layers", "max_paths", "topk",
                  "max_seq_len", "truncated_vocab_size"):
        assert ours[field] == getattr(upstream, field), field
    assert upstream.architectures == ["MedusaModel"]
    assert upstream.model_type == "medusa"


def test_k_overrides_head_count_from_checkpoint():
    """`num_heads` 恒等于 K（上游 `num_lookahead_tokens` 的 setter 就是 `num_heads = K`）。"""
    assert medusa_hf_config({"medusa_num_heads": 2}, 5)["num_heads"] == 5
    assert medusa_hf_config({"num_heads": 7}, 3)["num_heads"] == 3
    spec = medusa_spec(draft_dir(), k=2)
    assert spec.draft_model_config.hf_config["num_heads"] == 2


def test_method_inferred_from_draft_model_type():
    """draft 配置自己声明 `model_type="medusa"` 时不用显式给 method（上游 :958-960）。"""
    spec = SpeculativeConfig(num_speculative_tokens=K,
                             draft_model_config=draft_model_config(
                                 draft_dir(), cfg={**draft_hf(draft_dir()),
                                                   "model_type": "medusa"}))
    assert spec.method == "medusa" and spec.uses_medusa()


def test_derive_aligns_vocab_with_target():
    """旧 checkpoint 缺 vocab_size（默认 32001）→ 对齐 target 的词表（上游同款）。"""
    spec = medusa_spec(draft_dir())          # tiny target 的词表是 11
    assert spec.draft_model_config.hf_config["vocab_size"] == 32001
    derived = spec.derive_medusa_draft_config(
        target_config().model_config)
    assert derived.hf_config["vocab_size"] == 11
    assert derived.hf_config["truncated_vocab_size"] == 11
    # draft 的目录/精度不变（只有 hf 配置被补齐）
    assert derived.model == spec.draft_model_config.model
    assert derived.dtype == spec.draft_model_config.dtype


def test_derive_rejects_hidden_size_mismatch():
    """head 吃的就是 target 的 hidden：对不上时配置期就拒绝（上游会在前向里炸形状错）。"""
    config = {**draft_hf(draft_dir()), "hidden_size": 64}
    spec = medusa_spec(draft_dir(), cfg=config)
    with pytest.raises(ValueError, match="hidden_size"):
        spec.derive_medusa_draft_config(target_config().model_config)


def test_community_architectures_rejected():
    """社区版把基座的 architectures 抄进 config.json → 明确报错并给改法（不静默建基座模型）。"""
    config = {**draft_hf(draft_dir()), "architectures": ["Qwen2ForCausalLM"]}
    with pytest.raises(ValueError, match="MedusaModel"):
        medusa_hf_config(config, K)


def test_missing_draft_config_and_k_zero_error():
    with pytest.raises(ValueError, match="draft_model_config"):
        SpeculativeConfig(method="medusa", num_speculative_tokens=K)
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        SpeculativeConfig(method="medusa", num_speculative_tokens=0,
                          draft_model_config=draft_model_config(draft_dir()))


def test_scheduler_reserves_no_lookahead_for_medusa():
    """`num_lookahead_tokens` / `draft_slots` 都是 0（上游 `VllmConfig.num_lookahead_tokens`
    只给 `use_eagle()` 与 `uses_draft_model()` 留 K；Medusa 不写 KV、也不吃额外输入行）。"""
    from minivllm.core.sched.scheduler import Scheduler

    spec = medusa_spec(draft_dir())
    assert spec.max_num_new_slots_for_drafting == 0
    assert not spec.use_eagle() and spec.uses_medusa()

    class _FakeKVCacheManager:            # Scheduler 只读 `enable_caching` 这一个属性
        enable_caching = False

    scheduler = Scheduler(SchedulerConfig(), _FakeKVCacheManager(), max_model_len=64,
                          speculative_config=spec)
    assert scheduler.num_lookahead_tokens == 0 and scheduler.draft_slots == 0


# ================================================================ 2. 模型与加载


def test_forward_shape_and_parallel_heads():
    """N 个 head **并行**读同一份 hidden：head 0 的输出不会流进 head 1。"""
    model = medusa_model(draft_dir())
    hidden = torch.randn(3, model.config["hidden_size"])

    seen_inputs = []
    original = model.blocks[0].forward

    def spy(x):
        seen_inputs.append(x.detach().clone())
        return original(x) * 0 + 1e6        # 把 head 0 的输出改成垃圾（链式实现必然被污染）

    model.blocks[0].forward = spy
    blocks = model(hidden)
    assert len(blocks) == K
    assert torch.equal(seen_inputs[0], hidden)          # head 0 读的是入参
    assert torch.allclose(blocks[0], torch.full_like(blocks[0], 1e6))
    logits = model.compute_logits(blocks)
    assert [tuple(row.shape) for row in logits] == [(3, 11)] * K
    # head 1 的输出与垃圾无关 → 说明它读的也是原始 hidden，而不是 head 0 的结果
    model.blocks[0].forward = original
    clean = model(hidden)
    for index in range(1, K):
        assert torch.allclose(blocks[index], clean[index])


def test_residual_block_is_identity_plus_silu():
    """残差块的公式：`x = x + SiLU(Linear(x))`（逐层叠加）——用小手算的权重验证。"""
    config = {"medusa_fc_bias": False}
    block = ResidualBlock(config, hidden_size=2, num_layers=2)
    with torch.no_grad():
        block.layers[0].weight.copy_(torch.eye(2))
        block.layers[1].weight.copy_(torch.eye(2))
    x = torch.tensor([[0.5, -1.0]])
    step = x + torch.nn.functional.silu(x)
    expected = step + torch.nn.functional.silu(step)
    assert torch.allclose(block(x), expected)


@pytest.mark.parametrize("naming", ["old", "medusa_heads", "vllm"])
def test_three_weight_namings_load_identical_values(naming):
    """旧名字 / 带 `medusa_heads.` 前缀 / 已经是本模型名字：三种都要认，且值逐位相同。"""
    from safetensors.torch import load_file

    model = medusa_model(draft_dir(naming=naming))
    state = load_file(str(Path(draft_dir(naming=naming)) / "model.safetensors"))
    loaded = model.load_weights(iter(state.items()))
    params = dict(model.named_parameters())
    assert loaded == set(params)
    # 逐参数与检查点里的源张量相同（不是"加载成功了但装错了行"）
    reference = load_file(str(Path(draft_dir()) / "model.safetensors"))
    assert torch.equal(params["blocks.1.layers.0.weight"], reference["1.0.linear.weight"])
    assert torch.equal(params["lm_heads.1.weight"], reference["1.1.weight"])


def test_fc_bias_is_loaded_or_recorded_as_dropped():
    """`medusa_fc_bias`：开了就加载 bias，没开就把检查点里的 bias **显式记账后丢掉**。"""
    from safetensors.torch import load_file

    state = load_file(str(Path(draft_dir(fc_bias=True)) / "model.safetensors"))

    # 检查点里有 bias、但配置**没声明** medusa_fc_bias → 模型不建 bias，权重记账后丢掉
    without = medusa_model(draft_dir(fc_bias=True), cfg={
        key: value for key, value in draft_hf(draft_dir(fc_bias=True)).items()
        if key != "medusa_fc_bias"})
    loaded = without.load_weights(iter(state.items()))
    assert "blocks.0.layers.0.bias" not in dict(without.named_parameters())
    assert "blocks.0.layers.0.bias" in without.dropped_weights
    assert "lm_heads.0.bias" in without.dropped_weights
    assert loaded == set(dict(without.named_parameters()))

    with_bias = medusa_model(draft_dir(fc_bias=True), extra={"medusa_fc_bias": True})
    with_bias.load_weights(iter(state.items()))
    params = dict(with_bias.named_parameters())
    assert torch.equal(params["blocks.0.layers.0.bias"], state["0.0.linear.bias"])


def test_head_count_trimmed_when_k_is_smaller():
    """K < 检查点里的 head 数：只建 K 个 head，多出来的**记账**后丢掉（上游静默丢）。"""
    from safetensors.torch import load_file

    directory = draft_dir(num_heads=5)
    state = load_file(str(Path(directory) / "model.safetensors"))
    model = medusa_model(directory, k=2)
    loaded = model.load_weights(iter(state.items()))
    assert len(model.blocks) == 2
    assert loaded == set(dict(model.named_parameters()))
    assert sorted(model.dropped_weights) == ["blocks.2.layers.0.weight", "blocks.3.layers.0.weight",
                                            "blocks.4.layers.0.weight", "lm_heads.2.weight",
                                            "lm_heads.3.weight", "lm_heads.4.weight"]


def test_missing_weights_error_when_k_is_larger():
    """K > 检查点里的 head 数：缺的参数没法凭空造 → 加载期报错（不留随机初始化的参数）。"""
    from safetensors.torch import load_file

    directory = draft_dir(num_heads=2)
    state = load_file(str(Path(directory) / "model.safetensors"))
    model = medusa_model(directory, k=4)
    with pytest.raises(ValueError, match="没有被加载"):
        model.load_weights(iter(state.items()))


def test_unknown_weight_name_errors():
    """认不出的参数名当场报错（上游静默丢弃）——这是"块没实现"的唯一信号。"""
    from safetensors.torch import load_file

    state = load_file(str(Path(draft_dir()) / "model.safetensors"))
    state["0.5.linear.weight"] = torch.zeros(32, 32)
    model = medusa_model(draft_dir())
    with pytest.raises(ValueError, match="认不出的参数"):
        model.load_weights(iter(state.items()))


def test_original_lm_head_shares_one_head_and_remaps_truncated_vocab():
    """`original_lm_head` + `token_map`：所有 head 共享一个 lm_head，草稿词表被截断。

    截断后 `compute_logits()` 要把 k 宽的 logits 散回原词表、其余位置 `-inf`，
    于是 argmax 永远落在 `token_map` 里（这正是"截断词表不影响正确性"的依据）。
    """
    from safetensors.torch import load_file

    directory = draft_dir(original_lm_head=True, truncated_vocab=5)
    state = load_file(str(Path(directory) / "model.safetensors"))
    assert "token_map" in state
    model = medusa_model(directory)
    assert model.truncated_vocab_size == 5 and model.orig_vocab_size == 11
    loaded = model.load_weights(iter(state.items()))
    params = dict(model.named_parameters())
    assert "lm_head.weight" in params and "lm_heads.0.weight" not in params
    assert loaded == set(params) | {"token_map"}
    assert model.token_map.shape == (5,)
    # 检查点里的 lm_head 是整份词表（11 行）→ 加载时按 token_map 选出 5 行
    assert torch.equal(params["lm_head.weight"], state["0.1.weight"][state["token_map"]])
    # 逐 head 的 lm 权重里只有第 0 份被用上，其余记账丢掉（共享 head 的意思）
    assert "lm_heads.1.weight" in model.dropped_weights

    hidden = torch.randn(2, model.config["hidden_size"])
    logits = model.compute_logits(model(hidden))
    for row in logits:
        assert row.shape == (2, 11)
        outside = torch.ones(11, dtype=torch.bool)
        outside[state["token_map"]] = False
        assert torch.all(row[:, outside] == -torch.inf)
        assert bool(torch.isin(row.argmax(dim=-1), state["token_map"]).all())


def test_truncated_vocab_without_token_map_errors():
    """声明了截断词表却没有 `token_map` → 明确报错（上游是 assert）。

    `vocab_size` 必须写成 target 的词表（11）：否则词表对齐那一步会把 `truncated_vocab_size`
    也一起改成 target 的词表（下一条用例钉的就是这个上游行为）。
    """
    from safetensors.torch import load_file

    model = medusa_model(draft_dir(),
                         extra={"vocab_size": 11, "truncated_vocab_size": 5})
    assert model.truncated_vocab_size == 5 and model.orig_vocab_size == 11
    state = load_file(str(Path(draft_dir()) / "model.safetensors"))
    with pytest.raises(ValueError, match="token_map"):
        model.load_weights(iter(state.items()))


def test_vocab_alignment_overrides_explicit_truncated_vocab():
    """上游的对齐规则是"不等就**两个都**改成 target 的"：显式截断会被一起冲掉。

    想保留截断词表，就得让配置里的 `vocab_size` 与 target 一致（社区版 config 抄了基座的
    `vocab_size`，正是这个原因）。本仓库照抄这条规则，不"顺手修好"。
    """
    spec = medusa_spec(draft_dir(), cfg={
        **draft_hf(draft_dir()), "truncated_vocab_size": 5})       # 没写 vocab_size
    derived = spec.derive_medusa_draft_config(target_config().model_config)
    assert derived.hf_config["truncated_vocab_size"] == 11        # 被 target 的词表冲掉

    spec = medusa_spec(draft_dir(), cfg={
        **draft_hf(draft_dir()), "vocab_size": 11, "truncated_vocab_size": 5})
    derived = spec.derive_medusa_draft_config(target_config().model_config)
    assert derived.hf_config["truncated_vocab_size"] == 5         # vocab_size 一致 → 不动


def test_dummy_run_and_eplb_guard():
    """`dummy_run()` 按最大工作区跑一遍；MoE+EPLB 的组合明确拒绝（上游的 assert）。"""
    config = target_config(spec=medusa_spec(draft_dir()))
    proposer = MedusaProposer(config, DEVICE)
    proposer.load_model()
    proposer.dummy_run(num_tokens=8)                 # 不抛异常即形状/设备自检通过
    assert proposer.model is not None
    # 本仓库既没有 MoE 也没有 EPLB，所以这条组合只能"人为造出来"：
    # 造出来之后必须报错，而不是静默跑
    proposer.model.is_mixture_of_experts = True

    class _ParallelConfig:
        enable_eplb = True

    # `VllmConfig` 是 frozen dataclass，本仓库也没有并行配置：直接挂一个临时字段
    object.__setattr__(config, "parallel_config", _ParallelConfig())
    try:
        with pytest.raises(ValueError, match="EPLB"):
            proposer._reject_unsupported_eplb()
    finally:
        object.__delattr__(config, "parallel_config")
    assert not hasattr(config, "parallel_config")


# ================================================================ 3. 上游数值对照


def build_upstream_pair():
    """同一份 tiny 权重，两边各建一次；返回 `(ours, upstream, upstream_proposer, state)`。

    写成普通函数（fixture 只是缓存它）是为了让 `benchmarks/check_step66_medusa.py` 也能调同
    一段对照逻辑，不必复制一份上游夹具。
    """
    from safetensors.torch import load_file

    directory = draft_dir(num_layers=2)
    state = load_file(str(Path(directory) / "model.safetensors"))
    target_hf = tiny_qwen3_config("tiny_gqa")
    target_dir = tiny_qwen3_dir("tiny_gqa")

    spec = medusa_spec(directory)
    derived = spec.derive_medusa_draft_config(
        ModelConfig(model=target_dir, dtype="float32", max_model_len=64, hf_config=target_hf))
    ours = Medusa(derived.hf_config)
    ours.load_weights(iter(state.items()))
    ours = ours.float().eval()

    from vllm.config import (DeviceConfig as UpDeviceConfig, ModelConfig as UpModelConfig,
                             ParallelConfig, SpeculativeConfig as UpSpeculativeConfig,
                             VllmConfig as UpVllmConfig, set_current_vllm_config)
    from vllm.config.cache import CacheConfig as UpCacheConfig
    from vllm.config.scheduler import SchedulerConfig as UpSchedulerConfig
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.model_executor.models.medusa import Medusa as UpstreamMedusa
    from vllm.v1.spec_decode.medusa import MedusaProposer as UpstreamProposer

    target = UpModelConfig(model=target_dir, dtype="float32", max_model_len=64)
    parallel = ParallelConfig()
    upstream_spec = UpSpeculativeConfig(model=directory, method="medusa",
                                        num_speculative_tokens=K, target_model_config=target,
                                        target_parallel_config=parallel)
    vllm_config = UpVllmConfig(
        model_config=target, speculative_config=upstream_spec,
        cache_config=UpCacheConfig(block_size=16, gpu_memory_utilization=0.1),
        parallel_config=parallel, device_config=UpDeviceConfig("cpu"),
        scheduler_config=UpSchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64,
                                          max_model_len=64, is_encoder_decoder=False))
    with set_current_vllm_config(vllm_config):
        # 幂等：同一进程里别的用例可能已经初始化过（vLLM 重复初始化会 assert 失败）
        if not model_parallel_is_initialized():
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method="tcp://127.0.0.1:29773",
                                         backend="gloo")
            initialize_model_parallel(tensor_model_parallel_size=1,
                                      pipeline_model_parallel_size=1)
        upstream = UpstreamMedusa(vllm_config=vllm_config, prefix="")
        upstream.load_weights(iter(state.items()))
        proposer = UpstreamProposer(vllm_config=vllm_config, device=torch.device("cpu"))
        proposer.model = upstream                     # 跳过上游的加载器（权重已经灌好）
    return ours, upstream.float().eval(), proposer, state


@pytest.fixture(scope="module")
def upstream_models():
    return build_upstream_pair()


def test_upstream_per_head_logits_and_column_order_match(upstream_models):
    """需求 §4 第一条：每个 head 的 logits 与**最终候选列顺序**都一致。"""
    ours, upstream, proposer, _ = upstream_models
    torch.manual_seed(0)
    hidden = torch.randn(5, ours.config["hidden_size"], dtype=torch.float32)
    with torch.inference_mode():
        ours_blocks = ours(hidden)
        upstream_blocks = upstream(hidden)
        ours_logits = ours.compute_logits(ours_blocks)
        upstream_logits = upstream.compute_logits(upstream_blocks)
        ours_columns = torch.stack([row.argmax(dim=-1) for row in ours_logits], dim=1)
        upstream_columns = proposer.propose(K, hidden, None)

    assert len(ours_blocks) == len(upstream_blocks) == K
    for ours_block, upstream_block in zip(ours_blocks, upstream_blocks):
        assert torch.equal(ours_block, upstream_block)
    for ours_row, upstream_row in zip(ours_logits, upstream_logits):
        assert torch.equal(ours_row, upstream_row)      # 实测 max|Δ| = 0.0（同一份权重）
    assert torch.equal(ours_columns, upstream_columns)
    assert tuple(upstream_columns.shape) == (5, K)


def test_upstream_config_normalization_matches(upstream_models):
    """本仓库的配置归一结果与上游 `MedusaConfig` 的同一组字段逐个相同。"""
    _, _, _, _ = upstream_models
    ours = medusa_spec(draft_dir(num_layers=2)).derive_medusa_draft_config(
        ModelConfig(model=tiny_qwen3_dir("tiny_gqa"), dtype="float32", max_model_len=64,
                    hf_config=tiny_qwen3_config("tiny_gqa"))).hf_config
    from vllm.config import ModelConfig as UpModelConfig, ParallelConfig
    from vllm.config import SpeculativeConfig as UpSpeculativeConfig

    target = UpModelConfig(model=tiny_qwen3_dir("tiny_gqa"), dtype="float32",
                           max_model_len=64)
    upstream = UpSpeculativeConfig(model=draft_dir(num_layers=2), method="medusa",
                                   num_speculative_tokens=K, target_model_config=target,
                                   target_parallel_config=ParallelConfig())
    hf = upstream.draft_model_config.hf_config
    for field in ("hidden_size", "vocab_size", "truncated_vocab_size", "num_heads",
                  "num_hidden_layers", "max_paths", "topk"):
        assert ours[field] == getattr(hf, field), field
    assert upstream.draft_model_config.architectures == ["MedusaModel"]
    assert ours["architectures"] == upstream.draft_model_config.architectures


# ================================================================ 4. Runner 行选择


def test_select_last_hidden_rows():
    """`[b][d1]…[dK]` 里"产出 bonus 的那一行"= 每请求块内第 `采样数 - 1` 行。"""
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)     # 2 请求 × 3 行（K=2）
    # 请求 0 全接受（采出 3 枚 → 块内第 2 行）；请求 1 首枚被拒（采出 1 枚 → 块内第 0 行）
    selected = MedusaProposer.select_target_hidden_states(hidden, [3, 3], [3, 1])
    assert torch.equal(selected, hidden[[2, 3]])
    # 没有草稿的批（每请求一行）→ 行号就是 0..B-1（与上游第一条分支逐值相同）
    one_row = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(
        MedusaProposer.select_target_hidden_states(one_row, [1, 1], [1, 1]), one_row)


def test_upstream_stride_misaligns_mixed_prefill_batch():
    """反证：上游 `offset += num_draft + 1` 在一批里混进中间 prefill 块时会整体错位。"""
    rows_per_request = [5, 3, 3]          # 请求 0 是 5 行的 prefill 块，请求 1/2 各有 2 枚草稿
    num_sampled = [1, 3, 1]               # prefill 块没有采到 token（这里按"1 枚"占位算）
    hidden = torch.arange(44, dtype=torch.float32).reshape(11, 4)

    # 本仓库：只给"采到了 token"的请求选行，stride 用调度快照的行数
    ours = MedusaProposer.select_target_hidden_states(hidden[5:], rows_per_request[1:],
                                                      num_sampled[1:])
    assert torch.equal(ours, hidden[[5 + 2, 8 + 0]])  # 块 1 的第 2 行、块 2 的第 0 行

    # 上游算式：stride 用 num_draft + 1（对 prefill 块只前进 1 行）
    upstream_indices, offset = [], 0
    for num_draft, tokens in zip([0, 2, 2], num_sampled):
        upstream_indices.append(offset + tokens - 1)
        offset += num_draft + 1
    assert upstream_indices == [0, 3, 4]              # 第 0 个还落在 prefill 块里
    assert not torch.equal(hidden[upstream_indices[1:]], ours)


def test_select_rejects_prefill_row():
    """中间 prefill 块（没有采到 token）不允许进这套算式（上游会算出 -1）。"""
    hidden = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="prefill"):
        MedusaProposer.select_target_hidden_states(hidden, [1, 3], [1, 0])
    with pytest.raises(RuntimeError, match="行数"):
        MedusaProposer.select_target_hidden_states(hidden, [1, 1], [1, 1])


# ================================================================ 5. 端到端


@pytest.mark.parametrize("k", [1, 2, 3])
def test_greedy_matches_non_speculative(k):
    """不同 K 下 greedy 输出与非投机逐 token 相同（草稿只是候选）。"""
    baseline_engine, _, _ = make_engine(spec=None)
    try:
        baseline = run(baseline_engine, PROMPTS)
    finally:
        baseline_engine.shutdown()
    engine, _, _ = make_engine(spec=medusa_spec(draft_dir(num_heads=k), k=k))
    try:
        assert run(engine, PROMPTS) == baseline
    finally:
        engine.shutdown()


@pytest.mark.parametrize("naming", ["old", "vllm", "medusa_heads"])
def test_greedy_matches_non_speculative_other_namings(naming):
    engine, _, _ = make_engine(spec=medusa_spec(draft_dir(naming=naming)))
    try:
        assert run(engine, PROMPTS) == {"a": [4, 4, 4, 6, 0, 6], "b": [10, 4, 4, 4, 4, 4]}
    finally:
        engine.shutdown()


def test_drafts_are_k_wide_per_request_and_prob_free():
    """每条请求各提 K 枚、列顺序就是 head 顺序；argmax 提议不带 `draft_probs`（点质量 q）。"""
    engine, _, runner = make_engine()
    drafts_log = install_draft_spy(runner)
    calls = install_propose_spy(runner)
    try:
        outputs = run(engine, PROMPTS)
    finally:
        engine.shutdown()
    assert outputs == {"a": [4, 4, 4, 6, 0, 6], "b": [10, 4, 4, 4, 4, 4]}
    assert drafts_log, "一轮都没提草稿？"
    for entry in drafts_log:
        assert not entry["has_probs"]                 # 不带 q（NO_DRAFT_PROBS 分支）
        for tokens in entry["drafts"]:
            assert len(tokens) in (0, K)              # 0 = 中间 prefill 块（不提）
    # 至少有一轮两条请求都拿到了草稿，而且**各自的行**不同（行映射没串）
    both = [e for e in drafts_log if all(len(t) == K for t in e["drafts"])]
    assert both and all(e["drafts"][0] != e["drafts"][1] for e in both)
    # propose() 收到的行数 == 本轮真正提草稿的请求数；交回的候选与按该 hidden 重算的一致
    for call in calls:
        assert call["num_speculative_tokens"] == K
        assert tuple(call["tokens"].shape)[1] == K
        with torch.inference_mode():
            recomputed = torch.stack(
                [row.argmax(dim=-1) for row in
                 runner.proposer.model.compute_logits(
                     runner.proposer.model(call["hidden"]))], dim=1)
        assert torch.equal(call["tokens"].cpu(), recomputed.cpu())


def test_wrong_hidden_row_changes_drafts():
    """反证：把行选择整体错开一位，草稿必变（说明"取对行"是真在起作用，不是装饰）。"""
    engine, _, runner = make_engine()
    correct_log = install_draft_spy(runner)
    try:
        run(engine, PROMPTS)
    finally:
        engine.shutdown()

    engine, _, runner = make_engine()
    shifted_log = install_draft_spy(runner)
    original = MedusaProposer.select_target_hidden_states

    def shifted(hidden, rows_per_request, num_sampled_tokens):
        indices = []
        offset = 0
        for rows, num_tokens in zip(rows_per_request, num_sampled_tokens):
            # 故意取"块内最后一行"而不是"产出 bonus 的那一行"
            indices.append(min(offset + rows, hidden.shape[0] - 1))
            offset += rows
        return hidden[torch.tensor(indices, dtype=torch.int64)]

    MedusaProposer.select_target_hidden_states = staticmethod(shifted)
    try:
        run(engine, PROMPTS)
    finally:
        # 恢复时也要包一层 `staticmethod`：直接赋函数会变成实例方法（多收一个 self）
        MedusaProposer.select_target_hidden_states = staticmethod(original)
        engine.shutdown()
    assert correct_log and shifted_log
    differed = any(a["drafts"] != b["drafts"]
                   for a, b in zip(correct_log, shifted_log))
    assert differed, "行选错之后草稿居然没变：说明那份 hidden 根本没被用上"


def test_chunked_prefill_mixes_do_not_break_medusa():
    """一个批里混进中间 prefill 块（预算很小、prompt 很长）时不串行、不报错、greedy 仍一致。"""
    prompts = [("a", [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7]), ("b", [5, 6, 5, 6])]
    baseline_engine, _, _ = make_engine(spec=None, budget=8)
    try:
        baseline = run(baseline_engine, prompts)
    finally:
        baseline_engine.shutdown()
    engine, _, runner = make_engine(budget=8)
    drafts_log = install_draft_spy(runner)
    try:
        outputs = run(engine, prompts)
    finally:
        engine.shutdown()
    assert outputs == baseline
    # prefill 轮里没有任何请求拿到草稿（不提无意义的草稿），decode 轮里长度是 K
    assert all(len(tokens) in (0, K) for entry in drafts_log for tokens in entry["drafts"])
    assert any(len(tokens) == 0 for entry in drafts_log for tokens in entry["drafts"])


def test_preemption_with_tiny_kv_pool_still_matches():
    """块池很小（会抢占/恢复）时 greedy 仍与非投机一致。"""
    baseline_engine, _, _ = make_engine(spec=None, num_gpu_blocks=6)
    try:
        baseline = run(baseline_engine, PROMPTS)
    finally:
        baseline_engine.shutdown()
    engine, _, _ = make_engine(num_gpu_blocks=6)
    try:
        assert run(engine, PROMPTS) == baseline
    finally:
        engine.shutdown()


def install_oracle_drafts(runner, baseline_tokens, *, break_at=None):
    """把 Medusa 的草稿换成"非投机基线接下来的 K 枚"（= target 自己的 greedy 续写）。

    为什么这样能稳定跑出三种拒绝模式：**greedy 验证是确定性的**（`rejection_greedy_sample_kernel`
    逐位置比 `草稿 == target 的 argmax`），草稿正好等于 target 的续写就一定被接受。
    `break_at=j` 时把第 j 枚换成别的 token，于是第 j 个位置必被拒：

        break_at=None → 全接受        break_at=0 → 首拒        break_at=1（K>1）→ 中拒

    "target 的续写"取自**非投机基线**：它的 token 序列与投机路径逐 token 相同（本关一直在验证
    这条不变量），所以基线在该请求已生成 n 枚之后的第 j 枚就是 target 在第 n+j 个位置的 argmax。
    """
    original = runner._propose_medusa
    log = []

    def patched(state, sampled_by_row):
        drafts = original(state, sampled_by_row)          # 先走真实路径（模型/行选择都被调用）
        for index, req_id in enumerate(drafts.req_ids):
            tokens = drafts.draft_token_ids[index]
            if not tokens:
                continue                                   # 中间 prefill 块：保持空
            request = runner.requests[req_id]
            generated = len(request.all_token_ids) - len(request.prompt_token_ids)
            oracle = list(baseline_tokens[req_id][generated:generated + len(tokens)])
            if len(oracle) < len(tokens):
                continue                                   # 基线到尾部了：这一轮用真实草稿
            if break_at is not None and break_at < len(tokens):
                vocab = runner.input_batch.vocab_size
                oracle[break_at] = (oracle[break_at] + 1) % vocab
            drafts.draft_token_ids[index] = oracle
        log.append([[list(tokens) for tokens in drafts.draft_token_ids], len(drafts.req_ids)])
        return drafts

    runner._propose_medusa = patched
    return log


@pytest.mark.parametrize("break_at,expected_accepted", [(None, 3), (0, 0), (1, 1)])
def test_first_middle_all_accepted_go_through_same_verifier(break_at, expected_accepted):
    """首拒 / 中拒 / 全接受三种模式都走同一个 verifier（59 关那套），greedy 输出仍与基线一致。

    用**注入的 oracle 草稿**把三种模式各自稳定地造出来（见 `install_oracle_drafts`），
    再断言：每一轮"满宽草稿"的接受数正好是预期的那个值，且最终输出与非投机逐 token 相同。
    """
    prompt = [PROMPTS[0]]
    baseline_engine, _, _ = make_engine(spec=None)
    try:
        baseline = run(baseline_engine, prompt)
    finally:
        baseline_engine.shutdown()
    engine, core, runner = make_engine(spec=medusa_spec(draft_dir(num_heads=3), k=3))
    injected = install_oracle_drafts(runner, baseline, break_at=break_at)
    first_full_round = None
    try:
        for req_id, tokens in prompt:
            engine.add_request(req_id, list(tokens),
                               SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
        outputs = {}
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                outputs[out.request_id] = list(out.token_ids)
            stats = core.scheduler.spec_decoding_stats
            # **只断言第一轮满宽验证**：它的草稿一定来自注入（`gen=1`、基线还有 6 枚），
            # 后面的轮次在基线快结束时可能退回真实草稿（那时接受数不受控，不该拿来断言）。
            if stats is not None and first_full_round is None and stats.num_draft_tokens == 3:
                first_full_round = (stats.num_draft_tokens, stats.num_accepted_tokens)
    finally:
        engine.shutdown()
    assert outputs == baseline, "三种拒绝模式下 greedy 输出都必须与非投机一致"
    assert any(any(tokens) for tokens, _ in injected), "注入没生效：草稿还是模型自己的"
    assert first_full_round == (3, expected_accepted), first_full_round


def test_real_checkpoint_format_pt_is_rejected_loudly(tmp_path):
    """真实旧 checkpoint 是 `medusa_lm_head.pt`（torch pickle）：本仓库只读 safetensors，
    必须**明确报错**（不是"文件读不出来"这种模糊失败），且注释里给出转换方向。"""
    from minivllm.model_loader import get_model

    directory = tmp_path / "medusa_real"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(OLD_CHECKPOINT_CONFIG) + "\n")
    (directory / "medusa_lm_head.pt").write_bytes(b"not-a-real-checkpoint")
    config = ModelConfig(model=str(directory), dtype="float32", max_model_len=64,
                         hf_config=OLD_CHECKPOINT_CONFIG)
    with pytest.raises(NotImplementedError, match="safetensors"):
        get_model(config, "cpu")
