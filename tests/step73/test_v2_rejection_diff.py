"""73 关：V2 拒绝采样内核的**逐值差分**——本项目移植 vs 上游 vLLM 0.28.0。

被测对象（两个文件都是"逐行移植"，只改了 import 前缀）：

- 本项目：``minivllm/worker/gpu/spec_decode/rejection_sampler_utils.py``
  （对应上游 ``vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py``）
- 对照：``vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils``（site-packages 里的真 vLLM）

**为什么可以要求逐值相等**：两边拿到的是同一批张量、同一份 ``seeds``/``pos``，内核里的
随机数发生器是 ``tl.rand(seed, pos)``（无全局状态、无隐藏流），所以同输入必然同输出；
只要移植没有改语义，``sampled`` 与 ``num_sampled`` 就必须逐值相等。**不放宽断言**：
差分不过就是移植错了（或输入构造得不自洽，那种情况要修输入而不是放宽断言）。

张量布局（形状全部用具名常量写清，与上游 ``rejection_sample`` 的 docstring 一致）::

    target_logits             [num_logits, vocab]                          # 已按温度处理过的 target logits
    draft_logits              [max_num_reqs, num_speculative_steps, vocab] # 或 None（one-hot 草稿）
    draft_sampled             [num_logits]                                 # 草稿 token（-1 = 占位）
    cu_num_logits             [num_reqs + 1]                               # 每请求的 logits 行区间（前闭后开）
    pos                       [num_logits]
    idx_mapping               [num_reqs]
    expanded_idx_mapping      [num_logits]
    expanded_local_pos        [num_logits]
    temperature               [max_num_reqs]
    seeds                     [max_num_reqs]
    synthetic_conditional_rates [num_speculative_steps] 或 None
    返回                      (sampled [num_reqs, num_speculative_steps + 1] int64,
                               num_sampled [num_reqs] int32)

三张映射表的关系（上游 ``model_runner.py`` + ``input_batch.expand_idx_mapping`` 的约定）：

- ``idx_mapping``：**batch 行 → 请求常驻 slot**（`slot` 是 ``temperature``/``seeds`` 的下标，
  也叫 ``req_state_idx``；同一个请求在不同轮次复用同一个 slot，与 batch 行号无关）。
- ``expanded_idx_mapping``：**logits 行 → slot**，即 ``idx_mapping`` 按每请求行数展开：
  ``expanded_idx_mapping[cu[r] + i] = idx_mapping[r]``。
- ``expanded_local_pos``：**这是该请求的第几行 logits**，即 ``expanded_local_pos[cu[r] + i] = i``。
  它的第 ``i`` 行取 ``draft_logits[slot, i]`` 当草稿分布 q；``>= num_speculative_steps`` 的行是
  **bonus 行**（不参与草稿验证，只用来在尾部采一枚 token）。

另外一条容易踩的错位：内核用 ``draft_sampled[logit_idx + 1]`` 取"第 ``logit_idx`` 行要验证的
草稿 token"，所以 ``draft_sampled[cu[r] + i + 1]`` 才是配 ``draft_logits[slot, i]`` 的那枚草稿，
``draft_sampled[cu[r]]`` 是本请求块外的上一枚 token（内核不使用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

# 与仓库其它 CUDA 用例一致：没有 CUDA 就整模块跳过（Triton 内核跑不了）。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    pytest.skip("拒绝采样差分需要 CUDA（Triton 内核）", allow_module_level=True)

import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs  # noqa: E402
import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rs  # noqa: E402

# ---------------------------------------------------------------- 具名形状常量
VOCAB = 64  # 小词表，快跑；内核按 8192 分块，所以只有 1 个词表块
MAX_NUM_REQS = 8  # temperature / seeds / draft_logits 的第 0 维（常驻 slot 数）
INT32 = torch.int32
INT64 = torch.int64
FP32 = torch.float32


def _layout(drafts_per_req: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """由"每请求几枚草稿"推出 ``cu_num_logits`` / ``expanded_local_pos`` / 每请求起始行。

    每请求行数 = 草稿数 + 1（最后一个永远是 bonus 行）。
    """
    rows_per_req = [n + 1 for n in drafts_per_req]
    cu = np.zeros(len(drafts_per_req) + 1, dtype=np.int32)
    np.cumsum(rows_per_req, out=cu[1:])
    local_pos = np.concatenate(
        [np.arange(n + 1, dtype=np.int32) for n in drafts_per_req]
    ).astype(np.int32)
    starts = [int(cu[r]) for r in range(len(drafts_per_req))]
    return cu, local_pos, starts


@dataclass
class SpecInputs:
    """一次 ``rejection_sample`` 调用的全部输入（两边共用同一批张量对象）。"""

    target_logits: torch.Tensor  # [num_logits, VOCAB]
    draft_logits: torch.Tensor | None  # [MAX_NUM_REQS, K, VOCAB] 或 None
    draft_sampled: torch.Tensor  # [num_logits]
    cu_num_logits: torch.Tensor  # [num_reqs + 1]
    pos: torch.Tensor  # [num_logits]
    idx_mapping: torch.Tensor  # [num_reqs]
    expanded_idx_mapping: torch.Tensor  # [num_logits]
    expanded_local_pos: torch.Tensor  # [num_logits]
    temperature: torch.Tensor  # [MAX_NUM_REQS]
    seeds: torch.Tensor  # [MAX_NUM_REQS]
    num_speculative_steps: int  # K
    synthetic_conditional_rates: torch.Tensor | None  # [K] 或 None
    use_fp64: bool
    drafts_per_req: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.drafts_per_req)

    def call(self, module) -> tuple[torch.Tensor, torch.Tensor]:
        # 参数顺序与上游 rejection_sample 的签名逐位一致（use_block_verification 固定 False：
        # block verification（Sun et al. 2024）是另一条通路，本关只差分默认通路）。
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
            self.synthetic_conditional_rates,
            use_fp64=self.use_fp64,
            use_block_verification=False,
        )


def build_spec_inputs(
    *,
    drafts_per_req: list[int],
    slots: list[int],
    num_speculative_steps: int,
    target_logits: torch.Tensor,  # [num_logits, VOCAB] CPU
    draft_sampled: torch.Tensor,  # [num_logits] CPU
    temperature_by_slot: dict[int, float],
    seed_by_slot: dict[int, int],
    draft_logits: torch.Tensor | None = None,  # [MAX_NUM_REQS, K, VOCAB] CPU
    synthetic_conditional_rates: torch.Tensor | None = None,  # [K] CPU
    use_fp64: bool = False,
) -> SpecInputs:
    """构造自洽的输入，并把"不自洽就报错"写死在断言里（避免用错布局跑出假绿）。"""
    num_reqs = len(drafts_per_req)
    assert len(slots) == num_reqs
    assert all(0 <= n <= num_speculative_steps for n in drafts_per_req)
    assert len(set(slots)) == num_reqs and all(0 <= s < MAX_NUM_REQS for s in slots)

    cu_np, local_pos_np, starts = _layout(drafts_per_req)
    num_logits = int(cu_np[-1])
    assert target_logits.shape == (num_logits, VOCAB), target_logits.shape
    assert draft_sampled.shape == (num_logits,), draft_sampled.shape
    assert draft_sampled.dtype == INT32
    # token 必须落在词表内（-1 占位除外），否则内核会越界读 target_logits。
    assert int(draft_sampled.min()) >= -1 and int(draft_sampled.max()) < VOCAB

    idx_mapping = torch.tensor(slots, dtype=INT64)
    expanded_idx_mapping = idx_mapping.repeat_interleave(
        torch.tensor([n + 1 for n in drafts_per_req], dtype=INT64)
    )
    if draft_logits is not None:
        assert draft_logits.shape == (MAX_NUM_REQS, num_speculative_steps, VOCAB)

    temperature = torch.zeros(MAX_NUM_REQS, dtype=FP32)
    for slot, t in temperature_by_slot.items():
        temperature[slot] = t
    seeds = torch.zeros(MAX_NUM_REQS, dtype=INT64)
    for slot, s in seed_by_slot.items():
        seeds[slot] = s

    return SpecInputs(
        target_logits=target_logits.to(DEVICE).contiguous(),
        draft_logits=None if draft_logits is None else draft_logits.to(DEVICE).contiguous(),
        draft_sampled=draft_sampled.to(DEVICE).contiguous(),
        cu_num_logits=torch.from_numpy(cu_np).to(DEVICE).contiguous(),
        # pos 是"该行 query 在序列里的位置"，逐行递增即可（RNG 流按 (seed, pos) 索引）。
        pos=torch.arange(100, 100 + num_logits, dtype=INT64).to(DEVICE).contiguous(),
        idx_mapping=idx_mapping.to(DEVICE),
        expanded_idx_mapping=expanded_idx_mapping.to(DEVICE),
        expanded_local_pos=torch.from_numpy(local_pos_np).to(DEVICE).contiguous(),
        temperature=temperature.to(DEVICE),
        seeds=seeds.to(DEVICE),
        num_speculative_steps=num_speculative_steps,
        synthetic_conditional_rates=(
            None
            if synthetic_conditional_rates is None
            else synthetic_conditional_rates.to(DEVICE).contiguous()
        ),
        use_fp64=use_fp64,
        drafts_per_req=list(drafts_per_req),
    )


def assert_same_outputs(
    inputs: SpecInputs, a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """逐值比较 ``(sampled, num_sampled)``。

    ``sampled`` 是 ``new_empty``，内核只在"有效前缀"里写值：有效长度 = ``num_sampled``
    （被接受的第 0..num_sampled-2 枚 + 被拒/bonus 位上重采样出来的最后一枚）。请求草稿数 < K 时，
    该请求行里 ``sampled[r, num_sampled[r]:]`` 两边都**从未被写**（不是"两边都写了但不等"），
    比较未初始化显存没有语义。所以：先要求 ``num_sampled`` 整体逐值相等，再逐请求比较
    ``[:num_sampled[r]]`` 这整段有效前缀；当每个请求都写满（``num_sampled == K + 1``）时，
    额外要求整张 ``sampled`` 逐值相等，保证"全定义"场景下没有任何位置被漏比。
    """
    a_sampled, a_num = a
    b_sampled, b_num = b
    K = inputs.num_speculative_steps
    num_reqs = inputs.num_reqs

    # 形状 / dtype / device 也要一致（移植时改错 dtype 会静默改变索引宽度）。
    assert a_sampled.shape == b_sampled.shape == (num_reqs, K + 1)
    assert a_num.shape == b_num.shape == (num_reqs,)
    assert a_sampled.dtype == b_sampled.dtype == INT64
    assert a_num.dtype == b_num.dtype == INT32
    assert a_sampled.device == b_sampled.device
    assert a_sampled.device.type == "cuda"  # 模块级 skip 已保证；这里防"悄悄跑在 CPU 上"

    assert torch.equal(a_num, b_num), f"num_sampled 不一致:\n{a_num}\n{b_num}"
    assert bool(((a_num >= 1) & (a_num <= K + 1)).all()), a_num
    for r in range(num_reqs):
        n = int(a_num[r])
        assert torch.equal(a_sampled[r, :n], b_sampled[r, :n]), (
            f"请求 {r} 的有效前缀不一致:\n{a_sampled[r, :n]}\n{b_sampled[r, :n]}"
        )
    if bool((a_num == K + 1).all()):
        assert torch.equal(a_sampled, b_sampled), "整行都定义时仍未逐值相等"


def _target_logits_with_peaks(row_tokens: list[int], peak: float = 6.0) -> torch.Tensor:
    """构造 ``[len(row_tokens), VOCAB]`` 的 target logits：第 j 行的 argmax 固定为 row_tokens[j]。

    其余位置为 0，峰值位置为 ``peak``，于是 softmax 后峰值 token 概率 ≈ 0.99、
    其它 token ≈ 1e-4 —— 接受/拒绝两条分支都能被稳定触发（非 greedy 时由 u 决定）。
    """
    logits = torch.zeros(len(row_tokens), VOCAB, dtype=FP32)
    for j, tok in enumerate(row_tokens):
        logits[j, tok] = peak
    return logits


def _uniform_case(
    drafts_per_req: list[int],
    slots: list[int],
    temperature_by_slot: dict[int, float],
    seed_by_slot: dict[int, int],
    *,
    use_fp64: bool,
    reject_steps: set[tuple[int, int]] = frozenset(),
    placeholder_at: dict[int, int] | None = None,
    synthetic_conditional_rates: torch.Tensor | None = None,
) -> SpecInputs:
    """所有请求草稿数相同、且接受/拒绝可以手算的 one-hot 草稿批次。

    第 j 行的 target argmax 固定为 ``(7j + 3) % VOCAB``（峰值 logits）。

    - ``reject_steps`` 里的 ``(r, i)``：请求 r 第 i 步的草稿故意取另一个 token，
      并且把**该 token 的 target logit 设成 -inf** → 非 greedy 下 ``p(x) = 0``，
      ``p(x) > u * q(x)`` 恒不成立（``u`` 由 ``includes_zero=False`` 保证 > 0），
      于是"拒绝"是确定的，不受随机数影响。
    - ``placeholder_at[r] = i``：请求 r 第 i 步的草稿是 -1 占位（内核必须当成拒绝）。
    """
    K = max(drafts_per_req)
    placeholders = placeholder_at or {}
    cu_np, _, starts = _layout(drafts_per_req)
    num_logits = int(cu_np[-1])
    row_tokens = [(7 * j + 3) % VOCAB for j in range(num_logits)]
    target_logits = _target_logits_with_peaks(row_tokens)

    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    for r, start in enumerate(starts):
        # draft_sampled[start] 是块外上一枚 token，内核不读；给个合法值即可。
        draft_sampled[start] = row_tokens[start]
        for i in range(drafts_per_req[r]):
            row = start + i
            if placeholders.get(r) == i:
                draft_sampled[row + 1] = -1
            elif (r, i) in reject_steps:
                wrong = (row_tokens[row] + 1) % VOCAB
                draft_sampled[row + 1] = wrong
                target_logits[row, wrong] = float("-inf")
            else:
                draft_sampled[row + 1] = row_tokens[row]  # 与 target argmax 相同 → 被接受
    return build_spec_inputs(
        drafts_per_req=drafts_per_req,
        slots=slots,
        num_speculative_steps=K,
        target_logits=target_logits,
        draft_sampled=draft_sampled,
        temperature_by_slot=temperature_by_slot,
        seed_by_slot=seed_by_slot,
        synthetic_conditional_rates=synthetic_conditional_rates,
        use_fp64=use_fp64,
    )


# ---------------------------------------------------------------------------
# 用例 ①：K=1、全 greedy（temperature 全 0）、batch=3，draft_logits=None（one-hot 草稿）
# ---------------------------------------------------------------------------
def test_case1_greedy_k1_matches_upstream() -> None:
    inputs = _uniform_case(
        drafts_per_req=[1, 1, 1],
        slots=[5, 0, 3],
        temperature_by_slot={5: 0.0, 0: 0.0, 3: 0.0},
        seed_by_slot={5: 11, 0: 22, 3: 33},
        use_fp64=False,
        # 请求 1 的草稿不是 target argmax → 必被拒。
        reject_steps={(1, 0)},
        # 请求 2 的草稿是 -1 占位：greedy 下必须走 `accepted &= is_valid_draft` 被拒。
        placeholder_at={2: 0},
    )
    # cu_num_logits = [0, 2, 4, 6]：每请求 1 枚草稿 + 1 个 bonus 行。
    assert inputs.cu_num_logits.tolist() == [0, 2, 4, 6]
    assert inputs.target_logits.shape == (6, VOCAB)

    mv_sampled, mv_num = inputs.call(mv_rs)
    up_sampled, up_num = inputs.call(up_rs)
    assert_same_outputs(inputs, (mv_sampled, mv_num), (up_sampled, up_num))

    # 语义钉死（证明上面的"逐值相等"不是恒真）：greedy 下接受 ⟺ 草稿 == target argmax。
    row_tokens = [(7 * j + 3) % VOCAB for j in range(6)]
    # 请求 0：草稿 = target argmax(行 0) → 接受；bonus 行（行 1）采出它的 argmax。
    # 请求 1：草稿 ≠ target argmax(行 2) → 拒绝，该位置直接写 target argmax(行 2)。
    # 请求 2：草稿 = -1 → 拒绝，该位置写 target argmax(行 4)。
    # 请求 0 三项全定义（num_sampled == K + 1 == 2）→ 上面的比较退化成"整行逐值相等"。
    assert mv_num.tolist() == [2, 1, 1]
    assert mv_sampled[0, :2].tolist() == [row_tokens[0], row_tokens[1]]
    assert mv_sampled[1, 0].item() == row_tokens[2]
    assert mv_sampled[2, 0].item() == row_tokens[4]


# ---------------------------------------------------------------------------
# 用例 ②：K=3、temperature 有 0 有非 0、显式 seeds、batch=4，one-hot 草稿
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_fp64", [False, True])
def test_case2_mixed_temperature_k3_matches_upstream(use_fp64: bool) -> None:
    inputs = _uniform_case(
        drafts_per_req=[3, 3, 3, 3],
        slots=[2, 6, 1, 4],
        # slot 2/1 → greedy；slot 6 → 1.0；slot 4 → 2.0（非 1 的温度会缩放 logits）。
        temperature_by_slot={2: 0.0, 6: 1.0, 1: 0.0, 4: 2.0},
        seed_by_slot={2: 999983, 6: 12345, 1: 424242, 4: 7},
        use_fp64=use_fp64,
        # 请求 1（温度 1.0）第 1 步的草稿 p = 0 → 非 greedy 下必被拒。
        # 请求 2（greedy）第 1 步草稿与 argmax 不同 → 必被拒，且第 2 步不再验证。
        reject_steps={(1, 1), (2, 1)},
        # 请求 3（温度 2.0）第 1 步是 -1 占位：非 greedy 下 `verifying &= is_valid_draft`
        # 必须截断验证，后面的草稿不再参与。
        placeholder_at={3: 1},
    )
    # 每请求 3 枚草稿 + 1 个 bonus 行 → cu_num_logits = [0, 4, 8, 12, 16]。
    assert inputs.cu_num_logits.tolist() == [0, 4, 8, 12, 16]
    assert inputs.target_logits.shape == (16, VOCAB)

    mv_sampled, mv_num = inputs.call(mv_rs)
    up_sampled, up_num = inputs.call(up_rs)
    assert_same_outputs(inputs, (mv_sampled, mv_num), (up_sampled, up_num))

    # 请求 0（greedy，3 枚草稿全部匹配）→ 全接受，num_sampled == K + 1 == 4（整行都定义）；
    # 请求 2（greedy）→ 第 0 步接受、第 1 步拒绝 → 只有 2 项有效。
    assert mv_num[0].item() == 4
    assert mv_num[2].item() == 2


# ---------------------------------------------------------------------------
# 用例 ③：带 draft_logits（概率提议，[MAX_NUM_REQS, K, VOCAB]）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_fp64", [False, True])
def test_case3_probabilistic_draft_logits_matches_upstream(use_fp64: bool) -> None:
    K = 2
    drafts_per_req = [2, 2, 2]
    slots = [0, 4, 7]
    cu_np, _, starts = _layout(drafts_per_req)
    num_logits = int(cu_np[-1])

    gen = torch.Generator().manual_seed(20261009)
    row_tokens = [(5 * j + 1) % VOCAB for j in range(num_logits)]
    target_logits = _target_logits_with_peaks(row_tokens)

    # 概率草稿 q：draft_logits[slot, i]，只填用到的 slot。
    draft_logits = torch.zeros(MAX_NUM_REQS, K, VOCAB, dtype=FP32)
    for slot in slots:
        draft_logits[slot] = torch.randn(MAX_NUM_REQS, K, VOCAB, generator=gen)[slot] * 2.0

    # 草稿 token 必须真的从 q 里采（"概率提议"），固定 generator → 可重复。
    temperature_by_slot = {0: 1.0, 4: 0.5, 7: 1.0}
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    for r, start in enumerate(starts):
        slot = slots[r]
        temp = temperature_by_slot[slot]
        draft_sampled[start] = row_tokens[start]
        for i in range(drafts_per_req[r]):
            probs = torch.softmax(draft_logits[slot, i] / temp, dim=-1)
            tok = int(torch.multinomial(probs, 1, generator=gen))
            assert tok < VOCAB
            # 请求 2 的第 1 步换成 -1 占位：验证"q 是概率分布 + 草稿是占位"这条组合。
            draft_sampled[start + i + 1] = -1 if (r == 2 and i == 1) else tok

    inputs = build_spec_inputs(
        drafts_per_req=drafts_per_req,
        slots=slots,
        num_speculative_steps=K,
        target_logits=target_logits,
        draft_sampled=draft_sampled,
        temperature_by_slot=temperature_by_slot,
        seed_by_slot={0: 2024, 4: 4096, 7: 8192},
        draft_logits=draft_logits,
        use_fp64=use_fp64,
    )
    assert inputs.draft_logits.shape == (MAX_NUM_REQS, K, VOCAB)
    assert inputs.cu_num_logits.tolist() == [0, 3, 6, 9]

    mv_sampled, mv_num = inputs.call(mv_rs)
    up_sampled, up_num = inputs.call(up_rs)
    assert_same_outputs(inputs, (mv_sampled, mv_num), (up_sampled, up_num))


# ---------------------------------------------------------------------------
# 用例 ④：混合批——有的请求 K 枚草稿、有的 0 枚（cu_num_logits 不均等）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_draft_logits", [False, True])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_case4_mixed_draft_counts_matches_upstream(
    use_fp64: bool, with_draft_logits: bool
) -> None:
    K = 3
    # 请求 1 完全没有草稿（只 1 个 bonus 行）→ cu_num_logits 步长不均等。
    drafts_per_req = [3, 0, 2, 1]
    slots = [6, 2, 5, 0]
    temperature_by_slot = {6: 0.0, 2: 1.0, 5: 0.0, 0: 1.5}
    seed_by_slot = {6: 31337, 2: 2718, 5: 1414, 0: 1732}

    cu_np, _, starts = _layout(drafts_per_req)
    num_logits = int(cu_np[-1])
    row_tokens = [(3 * j + 2) % VOCAB for j in range(num_logits)]
    target_logits = _target_logits_with_peaks(row_tokens)

    draft_logits = None
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    if with_draft_logits:
        gen = torch.Generator().manual_seed(73)
        draft_logits = torch.zeros(MAX_NUM_REQS, K, VOCAB, dtype=FP32)
        for slot in slots:
            draft_logits[slot] = torch.randn(MAX_NUM_REQS, K, VOCAB, generator=gen)[slot]
    for r, start in enumerate(starts):
        slot = slots[r]
        temp = temperature_by_slot[slot]
        draft_sampled[start] = row_tokens[start]
        for i in range(drafts_per_req[r]):
            if with_draft_logits:
                if temp == 0.0:
                    # greedy 请求：草稿按 argmax 取（点质量 q）。内核的 greedy 分支根本不读
                    # draft_logits，但 HAS_DRAFT_LOGITS 是**整批**的常量（`draft_logits is not None`），
                    # 所以混批里 greedy 请求也必须带一份合法的 q（不能除 0 温度）。
                    tok = int(draft_logits[slot, i].argmax())
                else:
                    probs = torch.softmax(draft_logits[slot, i] / temp, dim=-1)
                    tok = int(torch.multinomial(probs, 1, generator=gen))
            elif (r + i) % 2 == 0:
                tok = row_tokens[start + i]
            else:
                tok = (row_tokens[start + i] + 1) % VOCAB
            # 短请求（1 枚草稿）的第 0 步用 -1 占位，覆盖"未满 K + 占位"的组合。
            draft_sampled[start + i + 1] = -1 if (r == 3 and i == 0) else tok

    inputs = build_spec_inputs(
        drafts_per_req=drafts_per_req,
        slots=slots,
        num_speculative_steps=K,
        target_logits=target_logits,
        draft_sampled=draft_sampled,
        temperature_by_slot=temperature_by_slot,
        seed_by_slot=seed_by_slot,
        draft_logits=draft_logits,
        use_fp64=use_fp64,
    )
    # 不均等的 cu_num_logits：[0, 4, 5, 8, 10]（+1 / +1 / +3 / +2）。
    assert inputs.cu_num_logits.tolist() == [0, 4, 5, 8, 10]

    mv_sampled, mv_num = inputs.call(mv_rs)
    up_sampled, up_num = inputs.call(up_rs)
    assert_same_outputs(inputs, (mv_sampled, mv_num), (up_sampled, up_num))

    # 0 草稿的请求只有 bonus 行：必然只采出 1 枚 token；
    # 1 枚草稿且第 0 步是 -1 占位的请求同样只有 1 枚有效 token（占位必被拒）。
    assert mv_num[1].item() == 1
    assert mv_num[3].item() == 1


# ---------------------------------------------------------------------------
# 用例 ⑤：synthetic_conditional_rates（合成接受率模式 = SYNTHETIC_MODE）
# ---------------------------------------------------------------------------
def test_case5_synthetic_conditional_rates_matches_upstream() -> None:
    K = 2
    inputs = _uniform_case(
        drafts_per_req=[2, 2],
        slots=[3, 1],
        temperature_by_slot={3: 1.0, 1: 1.0},
        seed_by_slot={3: 555, 1: 777},
        use_fp64=False,
        synthetic_conditional_rates=torch.tensor([0.25, 0.75], dtype=FP32),  # [K]
    )
    # SYNTHETIC_MODE 下接受与否只看 `u < rate`，与 p/q 无关（但内核仍会算 logprob）。
    mv_sampled, mv_num = inputs.call(mv_rs)
    up_sampled, up_num = inputs.call(up_rs)
    assert_same_outputs(inputs, (mv_sampled, mv_num), (up_sampled, up_num))


# ---------------------------------------------------------------------------
# 反证：差分断言不是恒真——故意改一个输入，输出必须不同（且断言必须报错）
# ---------------------------------------------------------------------------
def test_diff_assertion_is_not_vacuous() -> None:
    # 只有请求 0 的草稿 token 不同：命中 target argmax（接受）vs 错开 1（拒绝）。
    row_tokens = [(7 * j + 3) % VOCAB for j in range(6)]
    target_logits = _target_logits_with_peaks(row_tokens)

    def make(draft_token_row0: int) -> SpecInputs:
        draft_sampled = torch.zeros(6, dtype=INT32)
        draft_sampled[0] = row_tokens[0]
        draft_sampled[1] = draft_token_row0  # 请求 0 的第 0 步草稿
        draft_sampled[3] = (row_tokens[2] + 1) % VOCAB
        draft_sampled[5] = (row_tokens[4] + 1) % VOCAB
        return build_spec_inputs(
            drafts_per_req=[1, 1, 1],
            slots=[5, 0, 3],
            num_speculative_steps=1,
            target_logits=target_logits,
            draft_sampled=draft_sampled,
            temperature_by_slot={5: 0.0, 0: 0.0, 3: 0.0},
            seed_by_slot={5: 11, 0: 22, 3: 33},
        )

    hit = make(row_tokens[0])  # 接受：num_sampled[0] == 2
    miss = make((row_tokens[0] + 1) % VOCAB)  # 拒绝：num_sampled[0] == 1
    hit_mv = hit.call(mv_rs)
    miss_up = miss.call(up_rs)
    assert hit_mv[1][0].item() == 2 and miss_up[1][0].item() == 1
    with pytest.raises(AssertionError):
        assert_same_outputs(hit, hit_mv, miss_up)
