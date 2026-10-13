"""step75 的测试夹具：三条验证方式（standard / synthetic / block）各自的驱动方式。

三块内容，互不依赖：

**A. V2 内核夹具**（`V2SpecInputs` / `build_v2_inputs` / `assert_same_outputs`）
   直接喂 `minivllm/worker/gpu/spec_decode/rejection_sampler_utils.py::rejection_sample`，
   并能与上游 `vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils` 用**同一批张量**做差分。
   `block_cases_to_v2(...)` 把 `minivllm/testing/torch_block_verification.py` 的 `BlockCase`
   摊成内核要的扁平布局 —— 保证"Torch 数学参考"和"真内核"吃的是同一组数字。

**B. V1 内核夹具**（`make_metadata` / `sampling_metadata` / `run_v1` / `run_upstream_v1`）
   V1 的 `rejection_sample` 吃 `SpecDecodeMetadata` + `SamplingMetadata`（形态见 59 关）。
   `patch_uniform_probs` / `patch_recovered_tokens` 把两条通路里的随机源钉死，于是
   "本项目 vs 上游"可以逐值差分（总纲 §5：比较源码算法时固定随机源）。

**C. tiny 引擎夹具**（`make_v2_engine` / `make_v1_engine`）
   synthetic 的端到端只跑 tiny 模型：V2 需要 UVA（`VLLM_WSL2_ENABLE_PIN_MEMORY=1`），
   V1 不需要。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,  # noqa: E402
                      SamplingParams, SchedulerConfig, SpeculativeConfig,
                      UniProcExecutor, VllmConfig, Worker)
from minivllm.sample import SamplingMetadata  # noqa: E402
from minivllm.testing.spec_metadata import make_metadata  # noqa: E402
from minivllm.testing.tiny_models import (  # noqa: E402
    tiny_eagle3_dir,
    tiny_qwen3_config,
    tiny_qwen3_dir,
)
from minivllm.testing.torch_block_verification import BlockCase  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: 常驻 slot 表的行数（`temperature` / `seeds` / `draft_logits` 的第 0 维）
MAX_NUM_REQS = 8
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="拒绝采样是 Triton 内核，只在 CUDA 上可执行（上游同款边界）")

INT32 = torch.int32
INT64 = torch.int64
FP32 = torch.float32

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
DRAFT = tiny_eagle3_dir("tiny_gqa")
DRAFT_HF = json.loads((Path(DRAFT) / "config.json").read_text())
VOCAB = int(HF["vocab_size"])


# ---------------------------------------------------------------------------
# A. V2 内核夹具
# ---------------------------------------------------------------------------


def layout_of(drafts_per_req: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """"每请求几枚草稿" → `(cu_num_logits, expanded_local_pos, 每请求起始行)`。

    每请求行数 = 草稿数 + 1（最后一行永远是 bonus 行）。
    """
    rows_per_req = [n + 1 for n in drafts_per_req]
    cu = np.zeros(len(drafts_per_req) + 1, dtype=np.int32)
    np.cumsum(rows_per_req, out=cu[1:])
    local_pos = np.concatenate(
        [np.arange(n + 1, dtype=np.int32) for n in drafts_per_req]
    ).astype(np.int32)
    return cu, local_pos, [int(cu[r]) for r in range(len(drafts_per_req))]


@dataclass
class V2SpecInputs:
    """一次 V2 `rejection_sample` 调用的全部输入（本项目与上游共用同一批张量对象）。

    字段与张量布局（上游 `rejection_sample` 的 docstring）：

        target_logits               [num_logits, V]     **已按温度缩放**
        draft_logits                [max_num_reqs, K, V] 或 None（one-hot 草稿）
        draft_sampled               [num_logits]        草稿 token；`draft_sampled[start+i+1]`
                                                        才是第 i 步要验证的草稿（`start+i` 那行
                                                        是块外/上一枚 token，内核不读）
        cu_num_logits               [num_reqs + 1]      前闭后开
        pos                         [num_logits]        RNG 流按 `(seed, pos)` 索引
        idx_mapping                 [num_reqs]          batch 行 → 常驻 slot
        expanded_idx_mapping        [num_logits]        logits 行 → slot
        expanded_local_pos          [num_logits]        该请求的第几行（== K 的是 bonus 行）
        temperature / seeds         [max_num_reqs]      按 slot 取
        num_speculative_steps       K                   条件率表的长度也是它
    """

    target_logits: torch.Tensor
    draft_logits: torch.Tensor | None
    draft_sampled: torch.Tensor
    cu_num_logits: torch.Tensor
    pos: torch.Tensor
    idx_mapping: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor
    num_speculative_steps: int
    synthetic_conditional_rates: torch.Tensor | None
    use_fp64: bool
    drafts_per_req: list[int]
    use_block_verification: bool = False
    call_only: tuple[str, ...] = ()

    @property
    def num_reqs(self) -> int:
        return len(self.drafts_per_req)

    def call(self, module, **overrides):
        """调 `rejection_sample`（本项目/上游同签名；`call_only` 可只跑某几段做分段对照）。

        `call_only` 是给"逐 kernel 差分"用的**测试接缝**：正常调用不留任何缺口。
        """
        kwargs = dict(
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64=self.use_fp64,
            use_block_verification=self.use_block_verification,
        )
        kwargs.update(overrides)
        return module.rejection_sample(
            self.target_logits,
            self.draft_logits,
            self.draft_sampled,
            self.cu_num_logits,
            self.pos,
            self.idx_mapping,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.seeds,
            self.num_speculative_steps,
            **kwargs,
        )


def build_v2_inputs(
    *,
    drafts_per_req: list[int],
    slots: list[int],
    num_speculative_steps: int,
    target_logits: torch.Tensor,          # [num_logits, V] CPU（已按温度缩放）
    draft_sampled: torch.Tensor,          # [num_logits] CPU int32
    temperature_by_slot: dict[int, float],
    seed_by_slot: dict[int, int],
    draft_logits: torch.Tensor | None = None,   # [max_num_reqs, K, V] CPU（未除温度）
    synthetic_conditional_rates: torch.Tensor | None = None,   # [K] CPU
    use_fp64: bool = False,
    use_block_verification: bool = False,
    max_num_reqs: int = 8,
    pos_start: int = 100,
) -> V2SpecInputs:
    """构造自洽的输入；任何不自洽当场断言（避免用错布局跑出假绿）。"""
    num_reqs = len(drafts_per_req)
    assert len(slots) == num_reqs
    assert all(0 <= n <= num_speculative_steps for n in drafts_per_req)
    assert len(set(slots)) == num_reqs and all(0 <= s < max_num_reqs for s in slots)

    cu_np, local_pos_np, _starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    vocab = target_logits.shape[1]
    assert target_logits.shape == (num_logits, vocab), target_logits.shape
    assert draft_sampled.shape == (num_logits,), draft_sampled.shape
    assert draft_sampled.dtype == INT32
    # 草稿 token 必须落在词表内（-1 占位除外），否则内核会越界读概率表
    assert int(draft_sampled.min()) >= -1 and int(draft_sampled.max()) < vocab
    if draft_logits is not None:
        assert draft_logits.shape == (max_num_reqs, num_speculative_steps, vocab)

    idx_mapping = torch.tensor(slots, dtype=INT64)
    expanded_idx_mapping = idx_mapping.repeat_interleave(
        torch.tensor([n + 1 for n in drafts_per_req], dtype=INT64))
    temperature = torch.zeros(max_num_reqs, dtype=FP32)
    for slot, t in temperature_by_slot.items():
        temperature[slot] = t
    seeds = torch.zeros(max_num_reqs, dtype=INT64)
    for slot, s in seed_by_slot.items():
        seeds[slot] = s

    def to_dev(t):
        return None if t is None else t.to(DEVICE).contiguous()

    return V2SpecInputs(
        target_logits=to_dev(target_logits),
        draft_logits=to_dev(draft_logits),
        draft_sampled=to_dev(draft_sampled),
        cu_num_logits=torch.from_numpy(cu_np).to(DEVICE).contiguous(),
        pos=torch.arange(pos_start, pos_start + num_logits, dtype=INT64).to(DEVICE),
        idx_mapping=idx_mapping.to(DEVICE),
        expanded_idx_mapping=expanded_idx_mapping.to(DEVICE),
        expanded_local_pos=torch.from_numpy(local_pos_np).to(DEVICE).contiguous(),
        temperature=temperature.to(DEVICE),
        seeds=seeds.to(DEVICE),
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=to_dev(synthetic_conditional_rates),
        use_fp64=use_fp64,
        drafts_per_req=list(drafts_per_req),
        use_block_verification=use_block_verification,
    )


def block_cases_to_v2(
    cases: list[BlockCase],
    *,
    slots: list[int],
    seeds: list[int],
    pos_starts: list[int] | None = None,
    max_num_reqs: int = 8,
    for_upstream: bool = False,
) -> V2SpecInputs:
    """把 `BlockCase`（Torch 数学参考的输入）摊成 V2 内核的扁平布局。

    布局要点（与内核的 `draft_sampled[logit_idx + 1]` 约定一致）：
    第 r 条请求占 `[start, start+K]` 共 K+1 行；`target_logits[start+i]` 是第 i 行的 target
    logits（**已按温度缩放**，所以这里不除温度）；`draft_sampled[start+i+1] = d_i`；
    `draft_sampled[start]` 是块外令牌（内核不读，填 0）。bonus 行 `start+K` 也属于该请求
    （`expanded_local_pos = K`）。

    `for_upstream=True` 时把 K 取成"批内最大草稿数"（上游的 `draft_logits` 第 1 维是 K）。
    """
    num_reqs = len(cases)
    assert len(slots) == num_reqs == len(seeds)
    k_max = max(c.num_drafts for c in cases)
    vocab = cases[0].vocab_size
    assert all(c.vocab_size == vocab for c in cases)
    drafts_per_req = [c.num_drafts for c in cases]
    cu_np, _local_pos, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])

    target_logits = torch.zeros(num_logits, vocab, dtype=FP32)
    draft_logits = (
        torch.zeros(max_num_reqs, k_max, vocab, dtype=FP32)
        if any(c.draft_logits is not None for c in cases) else None
    )
    assert all(c.draft_logits is not None for c in cases) or draft_logits is None, (
        "同一个批里不能有的请求带草稿分布、有的不带（HAS_DRAFT_LOGITS 是整批常量）")
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    temperature_by_slot: dict[int, float] = {}
    seed_by_slot: dict[int, int] = {}
    for r, case in enumerate(cases):
        start = starts[r]
        slot = slots[r]
        # 内核把 target_logits 当"已缩放"，而 BlockCase 存的也是缩放后的 → 直接搬
        target_logits[start:start + case.num_drafts + 1] = case.target_logits.to(FP32)
        for i, token in enumerate(case.draft_tokens):
            draft_sampled[start + i + 1] = token
        if draft_logits is not None:
            draft_logits[slot, :case.num_drafts] = case.draft_logits.to(FP32)
        temperature_by_slot[slot] = case.temperature
        seed_by_slot[slot] = seeds[r]

    inputs = build_v2_inputs(
        drafts_per_req=drafts_per_req,
        slots=slots,
        num_speculative_steps=k_max,
        target_logits=target_logits,
        draft_sampled=draft_sampled,
        temperature_by_slot=temperature_by_slot,
        seed_by_slot=seed_by_slot,
        draft_logits=draft_logits,
        use_block_verification=True,
        max_num_reqs=max_num_reqs,
    )
    if pos_starts is not None:
        pos = np.concatenate([
            np.arange(pos_starts[r], pos_starts[r] + drafts_per_req[r] + 1)
            for r in range(num_reqs)
        ])
        inputs.pos = torch.from_numpy(pos.astype(np.int64)).to(DEVICE)
    return inputs


def assert_same_outputs(inputs: V2SpecInputs, a, b) -> None:
    """逐值比较 `(sampled, num_sampled)`；只比"有效前缀"（未写过的显存没有语义）。

    有效长度 = `num_sampled[r]`：被接受的第 `0..num_sampled-2` 枚 + 被拒/bonus 位重采样出的
    最后一枚。请求草稿数 < K 时 `sampled[r, num_sampled[r]:]` 两边都没被写过（不是"写了但不等"）。
    """
    a_sampled, a_num = a
    b_sampled, b_num = b
    k = inputs.num_speculative_steps
    assert a_sampled.shape == b_sampled.shape == (inputs.num_reqs, k + 1)
    assert a_num.shape == b_num.shape == (inputs.num_reqs,)
    assert a_sampled.dtype == b_sampled.dtype == INT64
    assert a_num.dtype == b_num.dtype == INT32
    assert a_sampled.device.type == "cuda"

    assert torch.equal(a_num, b_num), f"num_sampled 不一致:\n{a_num}\n{b_num}"
    assert bool(((a_num >= 1) & (a_num <= k + 1)).all()), a_num
    for r in range(inputs.num_reqs):
        n = int(a_num[r])
        assert torch.equal(a_sampled[r, :n], b_sampled[r, :n]), (
            f"请求 {r} 的有效前缀不一致:\n{a_sampled[r, :n]}\n{b_sampled[r, :n]}")
    if bool((a_num == k + 1).all()):
        assert torch.equal(a_sampled, b_sampled), "整行都定义时仍未逐值相等"


def peak_logits(row_tokens: list[int], vocab: int, peak: float = 6.0) -> torch.Tensor:
    """`[len(row_tokens), vocab]`：第 j 行的 argmax 固定为 `row_tokens[j]`（其余 0）。"""
    logits = torch.zeros(len(row_tokens), vocab, dtype=FP32)
    for j, token in enumerate(row_tokens):
        logits[j, token] = peak
    return logits


# ---------------------------------------------------------------------------
# B. V1 内核夹具（形态与 59 关一致）
# ---------------------------------------------------------------------------


def sampling_metadata(temperatures, drafts, *, device=DEVICE, generators=None,
                      top_k=None, top_p=None, output_token_ids=None,
                      no_penalties=True) -> SamplingMetadata:
    """按行造一份 V1 `SamplingMetadata`（与 Runner 那份同构，只是行数由用例决定）。"""

    def tensor(values, dtype=torch.float32):
        return None if values is None else torch.tensor(values, dtype=dtype, device=device)

    count = len(temperatures)
    all_greedy = all(value < 1e-5 for value in temperatures)
    return SamplingMetadata(
        temperature=None if all_greedy else tensor(temperatures),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_k=tensor(top_k, torch.int64), top_p=tensor(top_p),
        generators=dict(generators or {}),
        no_penalties=no_penalties,
        prompt_token_ids=[[] for _ in range(count)],
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        min_tokens=[0] * count,
        stop_token_ids=[[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts],
        presence_penalties=tensor(None),
        frequency_penalties=tensor(None),
        repetition_penalties=tensor(None),
    )


def upstream_sampling_metadata(temperatures, drafts, *, device=DEVICE):
    """上游 V1 的 `SamplingMetadata`（只为差分构造，生产路径不 import 它）。"""
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata as VllmSamplingMetadata

    all_greedy = all(value < 1e-5 for value in temperatures)
    count = len(temperatures)
    return VllmSamplingMetadata(
        temperature=None if all_greedy else torch.tensor(
            temperatures, dtype=torch.float32, device=device),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_p=None, top_k=None, generators={}, max_num_logprobs=None, no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(count, device=device),
        presence_penalties=torch.zeros(count, device=device),
        repetition_penalties=torch.ones(count, device=device),
        output_token_ids=[[] for _ in range(count)],
        allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[list(draft) for draft in drafts])


def run_v1(meta, logits, draft_probs, bonus, sm, *, synthetic_mode=False,
           synthetic_conditional_rates=None):
    """本项目 V1 的 Triton 通路（`minivllm/sample/rejection_sampler.py`）。"""
    from minivllm.sample import rejection_sampler as mine

    return mine.rejection_sample(
        meta.draft_token_ids, meta.num_draft_tokens, meta.max_spec_len,
        meta.cu_num_draft_tokens, draft_probs, logits[meta.target_logits_indices],
        bonus, sm, synthetic_mode=synthetic_mode,
        synthetic_conditional_rates=synthetic_conditional_rates)


def run_upstream_v1(meta, logits, draft_probs, bonus, temperatures, *,
                    synthetic_mode=False, synthetic_conditional_rates=None, device=DEVICE):
    """上游 V1 的 `vllm.v1.sample.rejection_sampler.rejection_sample`（只用于差分）。"""
    from vllm.v1.sample.rejection_sampler import rejection_sample as upstream

    drafts = []
    tokens = meta.draft_token_ids.tolist()
    start = 0
    for num in meta.num_draft_tokens:
        drafts.append(tokens[start:start + num])
        start += num
    return upstream(
        meta.draft_token_ids, meta.num_draft_tokens, meta.max_spec_len,
        meta.cu_num_draft_tokens, draft_probs, logits[meta.target_logits_indices],
        bonus, upstream_sampling_metadata(temperatures, drafts, device=device),
        synthetic_mode=synthetic_mode,
        synthetic_conditional_rates=synthetic_conditional_rates)


class patch_random_sources:
    """把 V1 两条通路里的随机源钉成同一份（只用于差分测试）。

    `rejection_sample` 内部会自己抽两类随机数：`generate_uniform_probs()`（接受判定）与
    `sample_recovered_tokens()`（拒绝后的恢复 token）。上游与本项目是两份独立实现，直接跑必然
    拿到不同的随机数 —— 那比的是随机源不是算法。这里把**两个模块里的同名函数**都换成
    "返回同一份预先算好的张量"，于是同输入必须逐值同输出（总纲 §5：固定随机源）。
    """

    def __init__(self, uniform_probs: torch.Tensor, recovered_token_ids: torch.Tensor):
        self.uniform_probs = uniform_probs
        self.recovered_token_ids = recovered_token_ids
        self._saved: list[tuple[object, str, object]] = []

    def __enter__(self):
        import minivllm.sample.rejection_sampler as mine
        import vllm.v1.sample.rejection_sampler as up

        for module in (mine, up):
            for name, value in (("generate_uniform_probs", self.uniform_probs),
                                ("sample_recovered_tokens", self.recovered_token_ids)):
                self._saved.append((module, name, getattr(module, name)))
                setattr(module, name, (lambda _v: (lambda *a, **k: _v))(value))
        return self

    def __exit__(self, *exc):
        for module, name, original in self._saved:
            setattr(module, name, original)
        self._saved.clear()
        return False


# ---------------------------------------------------------------------------
# C. tiny 引擎夹具（synthetic 的端到端）
# ---------------------------------------------------------------------------


def make_v2_engine(*, spec_k: int = 3, method: str = "none", rates=None,
                   length=None, budget: int = 64, blocks: int = 64,
                   max_num_seqs: int = 2, max_model_len: int = 64,
                   capture_sizes=(1, 2, 3, 4, 8), mode: str = "full_decode_only"):
    """V2 + EAGLE3 tiny 引擎；`method` 是 `rejection_sample_method`。

    `method="none"` 表示不带投机配置（不给 `speculative_config`）；synthetic 必须给
    `rates` 或 `length` 之一。
    """
    from minivllm.config import CompilationConfig

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    spec = None
    if spec_k is not None:
        kwargs = {}
        # `method="none"` = 不显式给验证方式（走配置默认 standard），便于对照用例
        if method != "none":
            kwargs["rejection_sample_method"] = method
        if rates is not None:
            kwargs["synthetic_acceptance_rates"] = list(rates)
        if length is not None:
            kwargs["synthetic_acceptance_length"] = length
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=spec_k,
            draft_model_config=ModelConfig(model=DRAFT, dtype="float32",
                                           max_model_len=max_model_len, hf_config=DRAFT_HF),
            **kwargs)
    config = VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                                 hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
        compilation_config=CompilationConfig(
            cudagraph_mode=mode, cudagraph_capture_sizes=list(capture_sizes)),
    )
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def make_v1_engine(*, spec_k: int = 3, method: str = "none", rates=None, length=None,
                   budget: int = 64, blocks: int = 64, max_num_seqs: int = 2,
                   max_model_len: int = 64):
    """V1 + draft_model 的 tiny 引擎（synthetic 的 V1 回归；不需要 UVA）。"""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    spec = None
    if spec_k is not None:
        kwargs = {}
        if method != "none":
            kwargs["rejection_sample_method"] = method
        if rates is not None:
            kwargs["synthetic_acceptance_rates"] = list(rates)
        if length is not None:
            kwargs["synthetic_acceptance_length"] = length
        spec = SpeculativeConfig(
            method="draft_model", num_speculative_tokens=spec_k,
            draft_model_config=ModelConfig(model=TINY, dtype="float32",
                                           max_model_len=max_model_len, hf_config=HF),
            **kwargs)
    config = VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                                 hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_prompts(engine, prompts, *, max_tokens=8, temperature=0.0, seed=None):
    """跑完一批请求，返回 `{req_id: token_ids}`。"""
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                          seed=seed, eos_token_id=VOCAB + 7))
    outputs: dict[str, list[int]] = {}
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
    engine.shutdown()
    return outputs
