"""DFlash2 的 V2 提议者（对应 vLLM `v1/worker/gpu/spec_decode/dflash2/speculator.py`）。

DFlash2 = **DFlash 的并行主干**（订阅上下文 KV、一次 query 前向）+ **候选关联 walk**：

    一次 query 前向 → 每个草稿位置的 hidden（`[B·K, hidden]`）
        ↓
    `compute_candidates()`        每个位置取词表 top-k：`candidate_ids [B·K, k]`、
                                  `unary_logits [B·K, k]`（**不是整词表分布**）
        ↓
    `candidate_selector()`        `[B, K, k, k]` 的边分：`unary[c] + pred[p]·hidden·succ[c]`
        ↓
    `_selector_walk_kernel`       从左到右逐步在**上一位置选中的候选**那些边里挑一个
                                  （Gumbel-max，噪声按候选 token id 取；temperature 来自 slot）
        ↓
    `_cache_draft_logits_kernel`  把实际走到的那些候选写进 `draft_logits[slot, step, :]`
                                  （fp32、其余列 `-inf`）——这就是验证器读到的 q

**为什么 q 不能是 unary logits**（需求 078 §1）：真正决定草稿的是"沿哪条路径走过候选图"，
那条路径的分数里含前驱项；拿 unary 当 q 会让拒绝采样按一个**根本没用来采样**的分布算接受
概率——不报错，只是交付分布不再等于 target 的分布。

**为什么旧候选列必须先清成 `-inf`**：`draft_logits` 是按**持久 slot**复用的缓冲。slot 换了
请求、或候选集变了之后，只写新候选列的话，上一轮的旧列还留着有限值 → q 多出几个"幽灵候选"，
而 walk 根本没有考虑它们。清理依据是 `_cached_candidate_ids`（每 slot 每步上一次写了哪些列）。

**图**：主干前向 + 候选选择 + walk + 落盘全部进 FULL 图（与 DFlash 的 `1 + K` 布局同形）。
walk 的 grid 是 `(num_reqs,)`，重放时按捕获时的大小跑；补齐行的 `sample_idx_mapping = -1`
让两个内核都跳过它们（上游同款）。
"""

from typing import Any

import torch

from .....config import CUDAGraphMode, VllmConfig
from .....triton_utils import tl, triton
from ...sample.gumbel import gumbel_noised_argmax
from ..dflash.speculator import DFlashSpeculator


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """沿候选图做条件 walk（上游同名内核，逐行照抄）。

    每个请求一个 program，`num_steps` 步**串行**（第 `step` 步只在上一步选中的候选所对应的
    那一片 `[k]` 边分里挑）。`previous` 是上一步在候选表里的**列号**（0..k−1），不是 token id；
    `scores` 的布局是 `[B·K, k_prev, k_cur]`，所以 `score_base = (flat*k + previous)*k`。

    Gumbel 噪声的 key 用的是 **candidate token id**（不是列号）：`gumbel_noised_argmax`
    的契约是"同一个 token 在任何位置都抽到同一个噪声"，这样草稿和验证侧对同一个 token 的
    采样才一致（上游注释：Candidate ids key the noise, matching the target's own sampling）。
    采样位置按 `sample_pos - 1`（`sample_pos` 是"被预测 token 的位置 Q"，验证侧按 Q−1 取噪声）。

    greedy（`SAMPLE_PROBABILISTIC=False`）：温度按 0 传，退化成 argmax；`realized_scores`
    仍然落盘（它就是路径分数，greedy 下没有验证器读它，但保留与上游一致的副作用）。
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float64 if USE_FP64 else tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        # Candidate ids key the noise, matching the target's own sampling.
        position = tl.load(sample_pos_ptr + flat) - 1
        _, index = gumbel_noised_argmax(
            scores,
            candidates,
            mask & valid,
            seed,
            position,
            temperature if SAMPLE_PROBABILISTIC else 0.0,
            USE_FP64=USE_FP64,
        )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


@triton.jit
def _cache_draft_logits_kernel(
    draft_logits_ptr,
    cached_candidate_ptr,
    candidate_ptr,
    scores_ptr,
    req_state_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """把实际路径的候选分数写进 q，并把**旧候选列清成 `-inf`**（上游同名内核）。

    执行顺序（不能换）：
        1. 读 `cached_candidate_ids[slot, step, :]`（上一次写过哪些列）
        2. 把这些旧列写成 `-inf`
        3. 写这一轮的新候选列 = 路径分数
        4. 更新 `cached_candidate_ids`

    旧列与新列有交集时，先 `-inf` 再写分数，结果仍然正确；反过来（先写新、后清旧）会把
    本轮刚写进去的分数清掉——候选集完全不相交时两种顺序看不出差别，**相交时**才会错。
    `flat` 的排布是 `[req, step]` 行主序，`req_state` 来自 `sample_idx_mapping`（补齐行为 -1，
    整条跳过）。
    """
    flat = tl.program_id(0)
    req_state = tl.load(req_state_ptr + flat)
    step = flat % num_steps
    offsets = tl.arange(0, BLOCK_K)
    mask = (req_state >= 0) & (offsets < top_k)
    candidate_base = flat * top_k
    cache_base = (req_state * num_steps + step) * top_k
    old_token_ids = tl.load(cached_candidate_ptr + cache_base + offsets, mask=mask)
    logits_base = (
        draft_logits_ptr
        + req_state * draft_logits_stride_0
        + step * draft_logits_stride_1
    )
    tl.store(logits_base + old_token_ids, -float("inf"), mask=mask)
    token_ids = tl.load(candidate_ptr + candidate_base + offsets, mask=mask)
    scores = tl.load(scores_ptr + candidate_base + offsets, mask=mask)
    tl.store(logits_base + token_ids, scores, mask=mask)
    tl.store(cached_candidate_ptr + cache_base + offsets, token_ids, mask=mask)


class DFlash2Speculator(DFlashSpeculator):
    """DFlash2 的 V2 提议者（上游 `DFlash2Speculator(DFlashSpeculator)`）。"""

    _speculator_name = "DFlash2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        draft_hf = self.speculative_config.draft_model_config.hf_config or {}
        draft_config = draft_hf.get("dflash_config") or {}
        # `selector_top_k` 是 **checkpoint 字段**（训练时定死的候选图宽度），不是采样器的 top_k。
        # 配置期已校验存在且落在 [1, vocab_size]（`_resolve_dflash2`）。
        self.selector_top_k = int(draft_config["selector_top_k"])
        # 每请求 query 块第 0 行（anchor = target 刚采出的 bonus）在 input_buffers 里的下标。
        # 与 DSpark 的 `_anchor_idx` 同义；**捕获期也要用**，所以是常驻张量。
        self._anchor_indices = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        # 每 (slot, step) 的 `[k_prev, k_cur]` 边分；walk 的输入。
        self._selector_scores = torch.empty(
            self.max_num_reqs,
            self.num_speculative_steps,
            self.selector_top_k,
            dtype=torch.float32,
            device=device,
        )
        # 上一次写进 `draft_logits` 的候选 id（清理旧列的依据）。初值 0 = 第一轮先清 0 号列
        # （`draft_logits` 本来就是 -inf，清它无害；上游同一初值）。
        self._cached_candidate_ids = torch.zeros(
            self._selector_scores.shape, dtype=torch.int64, device=device
        )

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        """q 必须是 fp32 + `-inf` 初值（需求 078 §3，上游同名覆盖）。

        - **fp32**：walk 读的边分是 fp32，验证器必须读到同一份分布；用 bf16 会先把分数舍掉
          再算接受概率（两处看到的 q 不是同一个数）。
        - **-inf**：缓存内核只写 k 个候选列，其余列必须一开始就无概率；写 0 会让 softmax
          多出一堆"均匀幽灵候选"。
        """
        return torch.float32, -float("inf")

    def _sample_path(self, candidate_ids: torch.Tensor, scores: torch.Tensor,
                     num_reqs: int) -> None:
        block_k = triton.next_power_of_2(self.selector_top_k)
        _selector_walk_kernel[(num_reqs,)](
            scores.contiguous(),
            candidate_ids.contiguous(),
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self._selector_scores,
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            SAMPLE_PROBABILISTIC=self.draft_logits is not None,
            USE_FP64=self.use_fp64_gumbel,
            num_warps=1,
        )

    def _cache_draft_logits(self, candidate_ids: torch.Tensor, num_sample: int) -> None:
        draft_logits = self.draft_logits
        assert draft_logits is not None
        block_k = triton.next_power_of_2(self.selector_top_k)
        _cache_draft_logits_kernel[(num_sample,)](
            draft_logits,
            self._cached_candidate_ids,
            candidate_ids,
            self._selector_scores,
            self.sample_idx_mapping,
            draft_logits.stride(0),
            draft_logits.stride(1),
            num_steps=self.num_speculative_steps,
            top_k=self.selector_top_k,
            BLOCK_K=block_k,
            num_warps=1,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """一次 query 前向 → 候选 → 边分 → 条件 walk → 落 q（上游同名方法）。"""
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        # 候选是**每个草稿位置**的 top-k（`[B·K, k]`），不是整词表分布。
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.selector_top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        # 第 0 步的前驱：anchor 行喂进去的 token（target 刚采出的 bonus）。
        anchor_token_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_sample)
