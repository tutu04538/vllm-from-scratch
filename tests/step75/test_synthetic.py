"""75 关：**synthetic 接受率**（`rejection_sample_method="synthetic"`），V2 为主、V1 回归。

synthetic 的语义（075 §3.4）：不做概率比检验，按配置给的**条件**接受率 `u < c_i` 接受草稿。
配置里写的是**无条件**概率（"前 i+1 枚全被接受的概率"），要经
`unconditional_to_conditional_rates()`（`c_i = p_i / p_{i-1}`）换算。需求点名的算例：

    K=3、rates=[0.8, 0.4, 0.2] → 条件率 [0.8, 0.5, 0.5]
    验算：P(τ≥1)=0.8、P(τ≥2)=0.8·0.5=0.4、P(τ≥3)=0.8·0.5·0.5=0.2 ✓ 与配置逐项一致

### 本关必须写明的一条限制（075 §4）

**synthetic 不证明模型分布无损**：它绕过了 `p/q` 检验，"接受长度"是人为设定的，交付的 token
不再保证服从 target 分布。`test_synthetic_is_not_distribution_preserving` 用一个可算的例子把它
钉出来：target 把概率几乎全给了 token A、草稿给的是 token B，只要 rates 全 = 1.0，内核照样接受
B —— 交付的 token 在 target 分布下概率只有 ~3.3e-4（甚至概率为 0 也照收）。所以 synthetic 只能用
来做"接受长度 → 吞吐/调度/显存"的可复现实验，**不能**用它验收正确性。

### 与上游的关系

- `unconditional_to_conditional_rates`：上游 `v1/spec_decode/utils.py:598-601`，逐行同义。
- V2 内核的 synthetic 分支（`SYNTHETIC_MODE`）：上游
  `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` —— 逐值差分。
- V1 内核的 synthetic 分支：上游 `v1/sample/rejection_sampler.py`（两条内核各一处）——
  也逐值差分（随机源用 `patch_random_sources` 钉死后才要求逐值相等，总纲 §5）。
- ⚠️ **本仓库与上游有意不同的一处**：上游 V1 收到 `rejection_sample_method="block"` 时会**静默
  按 standard 跑**；本仓库 V1 在构造 `RejectionSampler` 时明确报错（075 §4）。
"""

from __future__ import annotations

import torch
import pytest

from spec75_helpers import (  # noqa: E402
    DEVICE,
    INT32,
    MAX_NUM_REQS,
    VOCAB,
    build_v2_inputs,
    layout_of,
    make_metadata,
    patch_random_sources,
    requires_cuda,
    run_upstream_v1,
    run_v1,
    sampling_metadata,
)

RATES_EXAMPLE = [0.8, 0.4, 0.2]
CONDITIONAL_EXAMPLE = [0.8, 0.5, 0.5]
V1_VOCAB = 16


# ---------------------------------------------------------------------------
# 一、换算与配置（纯 CPU）
# ---------------------------------------------------------------------------


def test_conversion_matches_the_requirement_example_and_upstream() -> None:
    """需求 075 §3.4 的原例 + 与上游同名函数逐值一致 + `q == 0` 的退化分支。"""
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates as upstream_cvt

    from minivllm.spec_decode.utils import unconditional_to_conditional_rates as mine

    assert mine(RATES_EXAMPLE) == CONDITIONAL_EXAMPLE
    # 等价说法：条件率的连乘 == 配置里的无条件率（这正是"逐位置以原数组抽签"会错的原因）
    prefix = 1.0
    for conditional, unconditional in zip(CONDITIONAL_EXAMPLE, RATES_EXAMPLE):
        prefix *= conditional
        assert prefix == pytest.approx(unconditional)

    for rates in ([1.0], [0.0], [1.0, 1.0, 1.0], [0.5, 0.5], [0.0, 0.0, 0.0],
                  [0.9, 0.8, 0.7, 0.6, 0.5], [0.3, 0.0, 0.0]):
        assert mine(list(rates)) == upstream_cvt(list(rates)), rates


def test_config_resolution_matches_upstream() -> None:
    """长度 / 单调 / 取值 / 二选一：本仓库与上游逐项同判（连错误原文都一样）。"""
    from vllm.config.speculative import SpeculativeConfig as UpstreamSpecConfig

    from minivllm.config import SpeculativeConfig

    cases = [
        (3, [0.8, 0.4, 0.2], None),
        (3, [0.8, 0.4], None),           # 长度不对
        (3, [0.8, 0.9, 0.2], None),      # 非单调
        (3, [1.2, 0.4, 0.2], None),      # 越界（>1）
        (3, [-0.1, 0.4, 0.2], None),     # 越界（<0）
        (3, None, None),                 # 二选一都没给
        (3, [0.8, 0.4, 0.2], 2.0),       # 两个都给了
        (3, None, 0.5),                  # length 越界（<1）
        (3, None, 4.5),                  # length 越界（>K+1）
        (4, None, 2.5),                  # 合法 length
        (1, None, 2.0),                  # K=1 的上界（K+1=2）
        (1, [1.0], None),                # K=1 的合法 rates
    ]
    for n, rates, length in cases:
        mine_error = upstream_error = None
        mine_rates = upstream_rates = None
        try:
            mine_rates = SpeculativeConfig._resolve_synthetic_acceptance_rates(
                n, rates, length)
        except ValueError as exc:
            mine_error = str(exc)
        try:
            upstream_rates = UpstreamSpecConfig._resolve_synthetic_acceptance_rates(
                n, rates, length)
        except ValueError as exc:
            upstream_error = str(exc)
        assert (mine_error is None) == (upstream_error is None), (n, rates, length)
        if mine_error is not None:
            assert mine_error == upstream_error, (n, rates, length)
        else:
            assert mine_rates == upstream_rates, (n, rates, length)


def test_config_normalizes_length_to_rates_and_clears_it() -> None:
    """`synthetic_acceptance_length` 在配置期换算成 rates 并清空（下游只认 rates 一种口径）。"""
    from minivllm.config import SpeculativeConfig

    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=4,
                             rejection_sample_method="synthetic",
                             synthetic_acceptance_length=2.5)
    assert spec.synthetic_acceptance_rates == [1.0, 0.5, 0.0, 0.0]
    assert spec.synthetic_acceptance_length is None
    # 期望接受枚数 = Σ rates = length - 1（"最小方差档位表"的定义）
    assert sum(spec.synthetic_acceptance_rates) == pytest.approx(1.5)


def test_config_synthetic_with_zero_k_is_a_no_op() -> None:
    """K=0（本仓库 `method="ngram"` 不给 K 的缺省值）时 synthetic 退化成空表，边界自洽。

    上游在更早的地方就要求显式给 K，所以这条边界是本仓库特有的；钉住它是为了"K=0 + synthetic"
    报的不是一句莫名其妙的"length 必须在 [1,1] 里"，而是可解释的语义。
    """
    from minivllm.config import SpeculativeConfig

    zero = SpeculativeConfig(method="ngram", rejection_sample_method="synthetic",
                             synthetic_acceptance_length=1.0)
    assert zero.num_speculative_tokens == 0
    assert zero.synthetic_acceptance_rates == []
    with pytest.raises(ValueError, match=r"must be in \[1, 1\]"):
        SpeculativeConfig(method="ngram", rejection_sample_method="synthetic",
                          synthetic_acceptance_length=2.0)
    with pytest.raises(ValueError, match="must have length 0"):
        SpeculativeConfig(method="ngram", rejection_sample_method="synthetic",
                          synthetic_acceptance_rates=[0.5])


def test_config_rejects_synthetic_fields_outside_synthetic_mode() -> None:
    """两个字段只在 synthetic 下合法（上游原话同款），未知方法名也当场报错。"""
    from minivllm.config import SpeculativeConfig

    with pytest.raises(ValueError, match="only valid with rejection_sample_method='synthetic'"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                          synthetic_acceptance_rates=RATES_EXAMPLE)
    with pytest.raises(ValueError, match="only valid with rejection_sample_method='synthetic'"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                          synthetic_acceptance_length=2.0)
    with pytest.raises(ValueError, match="rejection_sample_method 只能是"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                          rejection_sample_method="synthetic_v2")


def test_sampler_mode_selection_from_config() -> None:
    """两个采样器的 `__init__` 是**唯一的模式分派点**：synthetic 建条件率表、block 置布尔。

    V2 用一个只暴露 `logprobs_mode` 的替身：本用例只验"读配置 → 建表/置位"，不碰 `_verify`
    （真实采样器的接线由端到端用例覆盖）。
    """
    from minivllm.config import SpeculativeConfig
    from minivllm.sample import Sampler
    from minivllm.sample.rejection_sampler import RejectionSampler as V1RejectionSampler
    from minivllm.worker.gpu.spec_decode.rejection_sampler import (
        RejectionSampler as V2RejectionSampler,
    )

    class _StubSampler:
        logprobs_mode = "raw_logprobs"

    def spec(method, **kwargs):
        return SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                                 rejection_sample_method=method, **kwargs)

    device = torch.device(DEVICE)
    standard = spec("standard")
    synthetic = spec("synthetic", synthetic_acceptance_rates=RATES_EXAMPLE)
    block = spec("block")

    v2_sampler = _StubSampler()
    v2_std = V2RejectionSampler(v2_sampler, standard, device)
    assert v2_std.use_block_verification is False
    assert v2_std.synthetic_conditional_rates is None
    v2_syn = V2RejectionSampler(v2_sampler, synthetic, device)
    assert v2_syn.use_block_verification is False
    # 配置里是**无条件**率，采样器里必须是**条件**率（075 §3.4）
    assert v2_syn.synthetic_conditional_rates.tolist() == pytest.approx(CONDITIONAL_EXAMPLE)
    v2_blk = V2RejectionSampler(v2_sampler, block, device)
    assert v2_blk.use_block_verification is True
    assert v2_blk.synthetic_conditional_rates is None

    v1_sampler = Sampler("raw_logprobs")
    v1_std = V1RejectionSampler(v1_sampler, standard, device)
    assert v1_std.synthetic_mode is False
    assert v1_std.synthetic_conditional_rates is None
    v1_syn = V1RejectionSampler(v1_sampler, synthetic, device)
    assert v1_syn.synthetic_mode is True
    assert v1_syn.synthetic_conditional_rates.tolist() == pytest.approx(CONDITIONAL_EXAMPLE)
    assert v1_syn.synthetic_conditional_rates.device.type == device.type


# ---------------------------------------------------------------------------
# 二、V2 内核：逐值差分 + 经验接受率
# ---------------------------------------------------------------------------


def _synthetic_v2_inputs(*, rates, drafts_per_req, slots, temperatures, seeds,
                         with_draft_logits=False, k=None):
    k = max(drafts_per_req) if k is None else k
    cu_np, _, starts = layout_of(drafts_per_req)
    num_logits = int(cu_np[-1])
    gen = torch.Generator().manual_seed(7500)
    target_logits = torch.randn(num_logits, VOCAB, generator=gen) * 2.0
    draft_logits = None
    draft_sampled = torch.zeros(num_logits, dtype=INT32)
    if with_draft_logits:
        draft_logits = torch.randn(MAX_NUM_REQS, k, VOCAB, generator=gen) * 2.0
    for r, start in enumerate(starts):
        for i in range(drafts_per_req[r]):
            # 请求 1 的第 0 步留成 -1 占位：synthetic 下草案位必拒（`draft_token_id >= 0`）
            if r == 1 and i == 0:
                draft_sampled[start + i + 1] = -1
            elif with_draft_logits:
                draft_sampled[start + i + 1] = int(draft_logits[slots[r], i].argmax())
            else:
                draft_sampled[start + i + 1] = int(target_logits[start + i].argmax())
    return build_v2_inputs(
        drafts_per_req=drafts_per_req, slots=slots, num_speculative_steps=k,
        target_logits=target_logits, draft_sampled=draft_sampled,
        temperature_by_slot=dict(zip(slots, temperatures)),
        seed_by_slot=dict(zip(slots, seeds)), draft_logits=draft_logits,
        synthetic_conditional_rates=torch.tensor(rates, dtype=torch.float32),
        max_num_reqs=MAX_NUM_REQS)


@requires_cuda
@pytest.mark.parametrize("with_draft_logits", [False, True])
def test_v2_synthetic_matches_frozen_kernel(with_draft_logits: bool) -> None:
    """V2 内核 synthetic 分支：本项目 vs 冻结实现逐值相等（含贪心/随机混批与 -1 占位）。"""
    import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rs

    from spec75_helpers import assert_same_outputs

    inputs = _synthetic_v2_inputs(
        rates=CONDITIONAL_EXAMPLE, drafts_per_req=[3, 3, 1], slots=[5, 2, 0],
        temperatures=[1.0, 0.0, 2.5], seeds=[11, 22, 33],
        with_draft_logits=with_draft_logits)
    assert_same_outputs(inputs, inputs.call(mv_rs), inputs.call(up_rs))


def _run_v2_trials(cond_rates, n_trials: int, k: int, *, temperature=1.0):
    """一次调用塞 `n_trials` 条请求（每请求一个 seed）→ 返回 `(sampled, num_sampled)`。

    内核的随机数是 `tl_rand32(seed, pos)`，所以"不同 seed"等价于独立重复试验。
    """
    import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs

    rows = k + 1
    gen = torch.Generator().manual_seed(4242)
    target_logits = (torch.randn(n_trials * rows, VOCAB, generator=gen) * 1.5).to(DEVICE)
    draft_sampled = torch.zeros(n_trials * rows, dtype=INT32, device=DEVICE)
    draft_sampled[1::rows] = torch.randint(0, VOCAB, (n_trials,), generator=gen).to(DEVICE)
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=INT32, device=DEVICE)
    idx = torch.arange(n_trials, dtype=torch.int64, device=DEVICE)
    return mv_rs.rejection_sample(
        target_logits, None, draft_sampled, cu,
        torch.arange(n_trials * rows, dtype=torch.int64, device=DEVICE) + 100,
        idx, idx.repeat_interleave(rows),
        torch.arange(rows, dtype=INT32, device=DEVICE).repeat(n_trials),
        torch.full((n_trials,), temperature, dtype=torch.float32, device=DEVICE),
        torch.arange(n_trials, dtype=torch.int64, device=DEVICE),
        k, torch.tensor(cond_rates, dtype=torch.float32, device=DEVICE),
        use_fp64=False, use_block_verification=False)


@requires_cuda
def test_v2_synthetic_prefix_acceptance_matches_config() -> None:
    """经验前缀接受率 == 配置的无条件率（`P(τ≥i+1) = rates[i]`），并且有判别力。

    判别力：如果把**无条件数组**当条件率用（075 §3.4 点名要避免的错），
    `P(τ≥2)` 会从 0.4 掉到 0.8×0.4=0.32 —— 差 0.08，是本用例容差（0.025）的 3 倍以上。
    """
    k = 3
    n_trials = 8000
    _sampled, num_sampled = _run_v2_trials(CONDITIONAL_EXAMPLE, n_trials, k)
    tau = num_sampled.to(torch.int64) - 1
    for i, unconditional in enumerate(RATES_EXAMPLE):
        empirical = float((tau >= i + 1).double().mean())
        assert abs(empirical - unconditional) < 0.025, (
            f"P(τ≥{i + 1}) 实测 {empirical:.4f}，配置 {unconditional}")
    # 反面：用错口径（原数组当条件率）算出来的 P(τ≥2) 必须落在容差之外
    wrong = RATES_EXAMPLE[0] * RATES_EXAMPLE[1]
    assert abs(wrong - RATES_EXAMPLE[1]) > 0.025
    # 平均接受长度 = Σ 无条件率
    mean_tau = float(tau.double().mean())
    assert abs(mean_tau - sum(RATES_EXAMPLE)) < 0.06, mean_tau


@requires_cuda
def test_v2_synthetic_all_accept_and_all_reject() -> None:
    """边界：rates 全 1 → 每轮全接受（τ=K）；全 0 → 必拒（τ=0，因为 `u > 0`）。"""
    k = 3
    _s, n_all = _run_v2_trials([1.0] * k, 512, k)
    assert bool((n_all == k + 1).all()), n_all.unique()
    _s2, n_none = _run_v2_trials([0.0] * k, 512, k)
    assert bool((n_none == 1).all()), n_none.unique()


@requires_cuda
def test_v2_synthetic_ignores_draft_probabilities() -> None:
    """synthetic 的接受判定**不看** p/q：把 target 换成极不均匀的分布，接受率不动。"""
    k = 2
    n_trials = 4000
    rows = k + 1
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=INT32, device=DEVICE)
    idx = torch.arange(n_trials, dtype=torch.int64, device=DEVICE)
    draft_sampled = torch.zeros(n_trials * rows, dtype=INT32, device=DEVICE)
    draft_sampled[1::rows] = 0
    cond = torch.tensor([0.5, 0.25], dtype=torch.float32, device=DEVICE)
    import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs

    def measure(peak: float) -> float:
        target = torch.zeros(n_trials * rows, VOCAB, dtype=torch.float32, device=DEVICE)
        target[:, 0] = peak
        _s, num_sampled = mv_rs.rejection_sample(
            target, None, draft_sampled, cu,
            torch.arange(n_trials * rows, dtype=torch.int64, device=DEVICE) + 100,
            idx, idx.repeat_interleave(rows),
            torch.arange(rows, dtype=INT32, device=DEVICE).repeat(n_trials),
            torch.ones(n_trials, dtype=torch.float32, device=DEVICE),
            torch.arange(n_trials, dtype=torch.int64, device=DEVICE), k, cond)
        return float(((num_sampled.to(torch.int64) - 1) >= 1).double().mean())

    # 平坦 target（草稿 token 0 的概率 1/8）与极尖 target（≈1）下，接受率都是 0.5
    assert abs(measure(0.0) - 0.5) < 0.03
    assert abs(measure(8.0) - 0.5) < 0.03
    # 对照：standard 下这两个 target 的接受概率天差地别（这正是 synthetic 与 standard 的分界）
    peaked_target = torch.zeros(3, VOCAB)
    peaked_target[:, 0] = 8.0
    assert float(torch.softmax(torch.zeros(VOCAB), dim=-1)[0]) < 0.2
    assert float(torch.softmax(peaked_target[0], dim=-1)[0]) > 0.99


@requires_cuda
def test_synthetic_is_not_distribution_preserving() -> None:
    """**synthetic 不证明分布无损**（075 §4 要求测试报告写明）——这里给出可算的证据。

    构造：target 把概率几乎全给 token 0（logit 8.0，其余 0），草稿给的是 token 1/2，
    rates 全 = 1.0。synthetic 下 `u < 1` 恒成立 → 草稿**必被接受**；
    而 token 1 在 target 分布下的概率只有 `e^0/(e^8+7) ≈ 3.3e-4`。把它的 target logit 改成
    `-inf`（概率恰好 0）也照样被接受。所以这个模式**只能**用来做接受长度实验。
    """
    k = 2
    n_trials = 256
    rows = k + 1
    cu = torch.arange(0, (n_trials + 1) * rows, rows, dtype=INT32, device=DEVICE)
    idx = torch.arange(n_trials, dtype=torch.int64, device=DEVICE)
    target = torch.zeros(n_trials * rows, VOCAB, dtype=torch.float32, device=DEVICE)
    target[:, 0] = 8.0
    draft_sampled = torch.zeros(n_trials * rows, dtype=INT32, device=DEVICE)
    draft_sampled[1::rows] = 1                      # 草稿 token = 1（target 概率 ~2e-4）
    draft_sampled[2::rows] = 2
    import minivllm.worker.gpu.spec_decode.rejection_sampler_utils as mv_rs

    def call(target_logits):
        return mv_rs.rejection_sample(
            target_logits, None, draft_sampled, cu,
            torch.arange(n_trials * rows, dtype=torch.int64, device=DEVICE) + 100,
            idx, idx.repeat_interleave(rows),
            torch.arange(rows, dtype=INT32, device=DEVICE).repeat(n_trials),
            torch.ones(n_trials, dtype=torch.float32, device=DEVICE),
            torch.arange(n_trials, dtype=torch.int64, device=DEVICE), k,
            torch.ones(k, dtype=torch.float32, device=DEVICE))

    sampled, num_sampled = call(target)
    assert bool((num_sampled == k + 1).all())        # 全接受
    assert bool((sampled[:, :k] == torch.tensor([[1, 2]], device=DEVICE)).all())
    # 解析值 = e^0/(e^8+7) ≈ 3.35e-4（实测 3.34e-4，差在 float32 的 exp/softmax 上）
    target_prob_1 = float(torch.softmax(target[0], dim=-1)[1])
    assert 2e-4 < target_prob_1 < 4e-4, target_prob_1

    target[:, 1] = float("-inf")                    # 概率恰好 0 也一样收
    sampled2, num2 = call(target)
    assert bool((num2 == k + 1).all())
    assert bool((sampled2[:, 0] == 1).all())


# ---------------------------------------------------------------------------
# 三、V1 内核：逐值差分 + 经验接受率（synthetic 同时回归 V1）
# ---------------------------------------------------------------------------


def _v1_rows(num_rows: int, *, peak: float = 0.0) -> torch.Tensor:
    """`[num_rows, V1_VOCAB]` 的概率行（`logits = log(p)`）：**前 ΣK 行验证、后 B 行 bonus**。"""
    rows = torch.full((num_rows, V1_VOCAB), 1.0 / V1_VOCAB)
    if peak:
        rows[:, 0] = 1.0
    return rows


def _v1_case(*, batch: int, temperatures, generators, drafts, peak=0.0):
    num_tokens = sum(len(d) for d in drafts)
    meta = make_metadata(drafts, device=DEVICE)
    logits = torch.log(_v1_rows(num_tokens + batch, peak=peak)).to(DEVICE)
    bonus = torch.zeros(batch, 1, dtype=INT32, device=DEVICE)
    sm = sampling_metadata(temperatures, drafts, generators=generators)
    return meta, logits, bonus, sm


@requires_cuda
@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_v1_synthetic_matches_frozen_kernel(with_draft_probs: bool) -> None:
    """V1 synthetic：本项目 vs 上游逐值相等（贪心 + 随机混批、含 -1 占位草稿）。

    随机源必须钉死（`patch_random_sources`）：两条通路各自抽均匀数与指数噪声，不钉的话比的是
    随机源。钉死后同输入必须逐值同输出。
    """
    k = 3
    batch = 3
    drafts = [[1, 2, 3], [-1, 0, 4], [5, 6, 7]]   # 请求 1 的第 0 步是 -1 占位
    meta, logits, bonus, sm = _v1_case(
        batch=batch, temperatures=[1.0, 0.0, 2.0], generators={}, drafts=drafts, peak=0.5)
    num_tokens = sum(len(d) for d in drafts)
    draft_probs = None
    if with_draft_probs:
        gen = torch.Generator().manual_seed(75)
        draft_probs = torch.softmax(
            torch.randn(num_tokens, V1_VOCAB, generator=gen), dim=-1).to(DEVICE).contiguous()
    uniform = (torch.arange(1, num_tokens + 1, dtype=torch.float64) / (num_tokens + 1)).to(DEVICE)
    recovered = (torch.arange(num_tokens, dtype=torch.int32, device=DEVICE) % V1_VOCAB)
    cond = torch.tensor(CONDITIONAL_EXAMPLE, dtype=torch.float32, device=DEVICE)

    with patch_random_sources(uniform, recovered):
        mine = run_v1(meta, logits, draft_probs, bonus, sm,
                      synthetic_mode=True, synthetic_conditional_rates=cond)
        up = run_upstream_v1(meta, logits, draft_probs, bonus, [1.0, 0.0, 2.0],
                             synthetic_mode=True, synthetic_conditional_rates=cond)
    assert torch.equal(mine, up), f"\n本项目:\n{mine}\n上游:\n{up}"


@requires_cuda
def test_v1_synthetic_prefix_acceptance_matches_config() -> None:
    """V1 的经验前缀接受率同样 == 配置的无条件率（每个请求一个独立 generator）。"""
    k = 3
    batch = 6000
    drafts = [[1, 2, 3]] * batch
    cond = torch.tensor(CONDITIONAL_EXAMPLE, dtype=torch.float32, device=DEVICE)

    def run(offset: int):
        generators = {r: torch.Generator(device=DEVICE).manual_seed(offset + r)
                      for r in range(batch)}
        meta, logits, bonus, sm = _v1_case(
            batch=batch, temperatures=[1.0] * batch, generators=generators,
            drafts=drafts, peak=0.5)
        return run_v1(meta, logits, None, bonus, sm,
                      synthetic_mode=True, synthetic_conditional_rates=cond)

    out = run(100_000)
    # 交付的 token 数 = τ + 1（最后一枚是 recovered token 或全接受时的 bonus）
    tau = (out != -1).sum(dim=1) - 1
    for i, unconditional in enumerate(RATES_EXAMPLE):
        empirical = float((tau >= i + 1).double().mean())
        assert abs(empirical - unconditional) < 0.03, (
            f"P(τ≥{i + 1}) 实测 {empirical:.4f}，配置 {unconditional}")
    # 同一个 seed 下可复现（generator 是每请求独立的 → 与批内顺序无关）
    assert torch.equal(out, run(100_000))
    # 换 seed 就会变（否则上面的"经验分布"就是假的）
    assert not torch.equal(out, run(200_000))


@requires_cuda
def test_v1_and_v2_synthetic_have_the_same_semantics() -> None:
    """V1 与 V2 的 synthetic 是同一套语义：τ 分布（不是逐值）必须一致。"""
    k = 3
    from minivllm.testing.torch_block_verification import total_variation

    _s, n_v2 = _run_v2_trials(CONDITIONAL_EXAMPLE, 8000, k)
    tau_v2 = n_v2.to(torch.int64) - 1
    dist_v2 = torch.bincount(tau_v2.long(), minlength=k + 1).double() / tau_v2.numel()

    batch = 8000
    cond = torch.tensor(CONDITIONAL_EXAMPLE, dtype=torch.float32, device=DEVICE)
    generators = {r: torch.Generator(device=DEVICE).manual_seed(700_000 + r)
                  for r in range(batch)}
    meta, logits, bonus, sm = _v1_case(
        batch=batch, temperatures=[1.0] * batch, generators=generators,
        drafts=[[1, 2, 3]] * batch, peak=0.5)
    out = run_v1(meta, logits, None, bonus, sm,
                 synthetic_mode=True, synthetic_conditional_rates=cond)
    tau_v1 = (out != -1).sum(dim=1) - 1
    dist_v1 = torch.bincount(tau_v1.long(), minlength=k + 1).double() / batch

    reference = torch.tensor([
        1.0 - RATES_EXAMPLE[0],
        RATES_EXAMPLE[0] - RATES_EXAMPLE[1],
        RATES_EXAMPLE[1] - RATES_EXAMPLE[2],
        RATES_EXAMPLE[2],
    ], dtype=torch.float64)
    assert total_variation(dist_v1, reference) < 0.02, dist_v1.tolist()
    assert total_variation(dist_v2, reference) < 0.02, dist_v2.tolist()
    assert total_variation(dist_v1, dist_v2) < 0.02


# ---------------------------------------------------------------------------
# 四、V1 的 block 边界（不得静默按 standard 跑）
# ---------------------------------------------------------------------------


def test_v1_rejects_block_explicitly_but_accepts_synthetic() -> None:
    """上游 V1 会**静默按 standard 跑** block；本仓库明确报错（075 §4 点名要求）。"""
    from minivllm.config import SpeculativeConfig
    from minivllm.sample import Sampler
    from minivllm.sample.rejection_sampler import RejectionSampler

    def spec(method, **kwargs):
        return SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                                 rejection_sample_method=method, **kwargs)

    sampler = Sampler("raw_logprobs")
    device = torch.device(DEVICE)
    with pytest.raises(NotImplementedError) as excinfo:
        RejectionSampler(sampler, spec("block"), device)
    # 报错必须说清"上游会静默按 standard"与"该走 V2"，否则用户会以为拿到了块验证的结果
    assert "静默按 standard" in str(excinfo.value) and "V2" in str(excinfo.value)

    synthetic = RejectionSampler(
        sampler, spec("synthetic", synthetic_acceptance_rates=RATES_EXAMPLE), device)
    assert synthetic.synthetic_mode is True
    assert synthetic.synthetic_conditional_rates.tolist() == pytest.approx(CONDITIONAL_EXAMPLE)
    standard = RejectionSampler(sampler, spec("standard"), device)
    assert standard.synthetic_mode is False
    assert standard.synthetic_conditional_rates is None


# ---------------------------------------------------------------------------
# 五、端到端（tiny 引擎）：配置 → 采样器 → 接受长度
# ---------------------------------------------------------------------------


def _run_and_collect(engine, core, prompt, *, max_tokens: int = 80, temperature: float = 1.0):
    """跑完一条请求，累加调度器每步的投机统计，返回 `(token_ids, accepted, drafts)`。"""
    from minivllm import SamplingParams

    engine.add_request("r0", list(prompt),
                       SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                      seed=7, eos_token_id=VOCAB + 7))
    tokens: list[int] = []
    accepted = drafts = 0
    for _ in range(300):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            tokens = list(out.token_ids)
        stats = core.scheduler.spec_decoding_stats
        if stats is not None and stats.num_drafts:
            accepted += stats.num_accepted_tokens
            drafts += stats.num_drafts
    engine.shutdown()
    return tokens, accepted, drafts


@requires_cuda
def test_v2_synthetic_end_to_end_acceptance_length_follows_config() -> None:
    """V2 引擎端到端：synthetic 的接受长度跟着**配置**走，与草稿模型质量无关。

    这正是这个模式存在的意义：接受长度由**配置**决定而不是由草稿模型质量决定——配置说
    `[0.8, 0.4, 0.2]`，实测每轮接受枚数就收敛到 `Σ rates = 1.4`。
    ⚠️ 夹具的 `max_model_len` 只能是 64：tiny 模型的位置表按 64 建（写 128 会在 RoPE 的
    gather 上触发 device-side assert），所以 `max_tokens=80` 实际会被上下文截到 60 枚。
    """
    from spec75_helpers import make_v2_engine

    engine, core, runner = make_v2_engine(spec_k=3, method="synthetic",
                                          rates=RATES_EXAMPLE, budget=32,
                                          max_num_seqs=1, max_model_len=64)
    # 接线检查：条件率表已经按配置建好、在 device 上（这是 synthetic 生效的唯一来源）
    rates = runner.rejection_sampler.synthetic_conditional_rates
    assert rates is not None and rates.device.type == "cuda"
    assert rates.tolist() == pytest.approx(CONDITIONAL_EXAMPLE)

    tokens, accepted, drafts = _run_and_collect(engine, core, [1, 2, 3, 4])
    assert drafts > 20, drafts
    mean = accepted / drafts
    assert abs(mean - sum(RATES_EXAMPLE)) < 0.5, (accepted, drafts, mean)
    assert len(tokens) > 8


@requires_cuda
def test_v1_synthetic_end_to_end_and_standard_control() -> None:
    """V1 回归（075 §4）：synthetic 的接受长度跟着配置；standard 作对照同 workload 跑得起来。"""
    from spec75_helpers import make_v1_engine

    engine, core, _runner = make_v1_engine(spec_k=3, method="synthetic",
                                           rates=RATES_EXAMPLE, budget=32,
                                           max_num_seqs=1, max_model_len=64)
    tokens, accepted, drafts = _run_and_collect(engine, core, [1, 2, 3, 4])
    assert drafts > 20, drafts
    mean = accepted / drafts
    assert abs(mean - sum(RATES_EXAMPLE)) < 0.5, (accepted, drafts, mean)
    assert len(tokens) > 8

    # standard 对照：同一 workload 下接受长度跟着**模型**走，与本关配置无关。
    # 这个夹具里 draft 就是 target（同一份 tiny 权重）→ 标准验证几乎全接受（τ≈K=3），
    # 远高于 synthetic 配置出来的 1.4；换成真实 EAGLE3 权重时它会掉到 1.2~1.5。
    # 这正是 synthetic 存在的意义：把"接受长度"从模型质量里解耦出来当实验输入。
    engine_std, core_std, _ = make_v1_engine(spec_k=3, method="standard", budget=32,
                                            max_num_seqs=1, max_model_len=64)
    _tokens, accepted_std, drafts_std = _run_and_collect(engine_std, core_std, [1, 2, 3, 4])
    assert drafts_std > 10, drafts_std
    std_mean = accepted_std / drafts_std
    assert std_mean > sum(RATES_EXAMPLE) + 0.5, (accepted_std, drafts_std, std_mean)
