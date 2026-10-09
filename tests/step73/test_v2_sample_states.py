"""73 关：V2 采样状态 / min_p / 惩罚 / 结构化输出四个移植文件的实测与逐值差分。

被测对象（都是"逐行移植"，只改 import 与点名的等价替换）：

- 本项目 ``minivllm/worker/gpu/sample/states.py`` ↔ 上游 ``vllm/v1/worker/gpu/sample/states.py``
- 本项目 ``minivllm/worker/gpu/sample/min_p.py`` ↔ 上游 ``vllm/v1/worker/gpu/sample/min_p.py``
- 本项目 ``minivllm/worker/gpu/sample/penalties.py`` ↔ 上游 ``vllm/v1/worker/gpu/sample/penalties.py``
- 本项目 ``minivllm/worker/gpu/structured_outputs.py`` ↔ 上游 ``vllm/v1/worker/gpu/structured_outputs.py``

四类用例：

1. ``SamplingStates.add_request`` 的**字段归一化**（`top_k<=0 或 > vocab_size → vocab_size`、
   `logprobs=-1 → vocab_size`、`logprobs=None → NO_LOGPROBS(-1)`、`seed=None` 走全局 RNG 且
   `seeds_set=False`），并与上游同名方法逐值对照。
2. ``apply_min_p`` / ``apply_temperature``：同一份 logits / 同一份参数张量分别喂给两个移植版本，
   断言 ``torch.equal``（内核里没有隐藏随机流，同输入必然同输出；不相等就是移植错了）。
3. ``PenaltiesState``：`use_penalty()` 真值表；`add_request` 后的 `_new_penalties_reqs` 与 UVA 值；
   `apply_staged_writes()` 之后 `prompt_bin_mask` / `output_bin_counts` 与**手工数出来的次数**逐值相等
   —— 这一条钉住 `prompt_len != prefill_len` 的意义：prompt 段进位图（只置一次），已生成段进计数（累加）。
4. 语法掩码：``_apply_grammar_bitmask_kernel`` 与上游 kernel 逐值差分 + 语义断言
   （bit=0 → `-inf`、bit=1 → 不动、`position_is_active` 为假的行完全不动），
   外加 ``StructuredOutputsWorker.apply_grammar_bitmask`` 的端到端走查（映射 + 异步拷贝 + 内核）。

**为什么可以要求逐值相等**：两边拿到同一批张量、同一份参数，内核是纯函数式的
（`tl.range` / `atomic_or` / `atomic_add` 里没有随机源；`atomic_*` 只作用在互不重叠的地址上），
所以"不等"只可能是移植改了语义。**不放宽断言**：差分不过就是移植错了。
"""

from __future__ import annotations

import os

# ⚠️ 必须在 import 任何 minivllm 之前设置。
# 本机是 WSL2，本项目 `minivllm/utils/platform_utils.py::is_pin_memory_available()` 按上游 vLLM 的
# 规则读环境变量 `VLLM_WSL2_ENABLE_PIN_MEMORY`（**默认关**）。不设的话 UVA 支架
# （`UvaBackedTensor` / `StagedWriteTensor`，V2 常驻状态的载体）会直接
# `RuntimeError: UVA is not available` —— 这与上游 vLLM 在本机的行为完全一致，
# 不是本仓库额外加的门槛（AGENTS.md §3：上游引擎在本机也要 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`）。
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import ast  # noqa: E402
import inspect  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.sampling_params import SamplingParams as UpstreamSamplingParams  # noqa: E402

# 对照侧：site-packages 里的真 vLLM 0.28.0（测试允许 import 上游，生产路径禁止）。
import vllm.v1.worker.gpu.sample.min_p as up_min_p  # noqa: E402
import vllm.v1.worker.gpu.sample.penalties as up_penalties  # noqa: E402
import vllm.v1.worker.gpu.sample.states as up_states  # noqa: E402
import vllm.v1.worker.gpu.structured_outputs as up_structured  # noqa: E402

# `gumbel.py` 由另一个并行 agent 移植（本文件不创建、不修改它）。若它还没就绪，
# 只有"温度"那一项跳过，其余用例照跑（`min_p` 必须真跑通过）。
# ⚠️ 必须放在 import `minivllm.worker.gpu.sample.states` 之前：那个模块在 import 期就
# `from minivllm.worker.gpu.sample.gumbel import apply_temperature`，缺文件时是 ImportError
# 而不是"可跳过"。
gumbel = pytest.importorskip(
    "minivllm.worker.gpu.sample.gumbel",
    reason="并行 agent 的 minivllm/worker/gpu/sample/gumbel.py 尚未就绪",
)
import vllm.v1.worker.gpu.sample.gumbel as up_gumbel  # noqa: E402

from minivllm.sampling_params import SamplingParams  # noqa: E402
from minivllm.triton_utils import triton  # noqa: E402
from minivllm.utils.math_utils import cdiv  # noqa: E402
from minivllm.worker.gpu import structured_outputs as mv_structured  # noqa: E402
from minivllm.worker.gpu.sample import min_p as mv_min_p  # noqa: E402
from minivllm.worker.gpu.sample import penalties as mv_penalties  # noqa: E402
from minivllm.worker.gpu.sample import states as mv_states  # noqa: E402
from minivllm.worker.gpu.states import RequestState  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="UVA / Triton 内核需要 CUDA"
)

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max

# 与上游 structured_outputs.py 的 launch 参数一致（`BLOCK_SIZE = 8192`）。
_BITMASK_BLOCK_SIZE = 8192


def _f32(values, device=DEVICE) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# 0. 机械移植的静态守卫：生产模块里不许出现真 vLLM 的 import
# ---------------------------------------------------------------------------

_PORTED_MODULES = [mv_states, mv_min_p, mv_penalties, mv_structured]


@pytest.mark.parametrize("mod", _PORTED_MODULES, ids=lambda m: m.__name__)
def test_ported_modules_do_not_import_installed_vllm(mod):
    """只看真正的 import 语句（docstring 里引用的上游 import 行不算）。

    这是"机械移植"的守卫：只要有人把 `from vllm...` 粘回来，这条就红。
    """
    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported, f"{mod.__name__} 解析不到任何 import，静态守卫失效"
    bad = [m for m in imported if m == "vllm" or m.startswith("vllm.")]
    assert not bad, f"{mod.__name__} 仍然 import 了真 vLLM：{bad}"


def test_structured_outputs_pin_memory_replacement():
    """`PIN_MEMORY` 常量 → `is_pin_memory_available()`：替换必须真的落地。"""
    assert not hasattr(mv_structured, "PIN_MEMORY")
    from minivllm.utils.platform_utils import is_pin_memory_available

    src = inspect.getsource(mv_structured)
    assert "pin_memory=is_pin_memory_available()" in src
    tree = ast.parse(src)
    names = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "minivllm.utils.platform_utils"
        for alias in node.names
    ]
    assert names == ["is_pin_memory_available"]
    # 语义前提：本用例所在的环境确实"能用 pinned memory"（否则 UVA 支架根本起不来）。
    assert is_pin_memory_available() is True


# ---------------------------------------------------------------------------
# 1. SamplingStates.add_request 的字段归一化
# ---------------------------------------------------------------------------

_MAX_NUM_REQS = 4
_VOCAB_SIZE = 32


@requires_cuda
def test_sampling_states_add_request_normalization():
    states = mv_states.SamplingStates(max_num_reqs=_MAX_NUM_REQS, vocab_size=_VOCAB_SIZE)

    # 构造期：top_k / top_p 必须手工填（0 对它们不是合法值），logprobs 默认 -1。
    assert states.top_k.np.tolist() == [_VOCAB_SIZE] * _MAX_NUM_REQS
    assert states.top_p.np.tolist() == [1.0] * _MAX_NUM_REQS
    assert states.num_logprobs.tolist() == [mv_states.NO_LOGPROBS] * _MAX_NUM_REQS
    assert mv_states.NO_LOGPROBS == up_states.NO_LOGPROBS == -1

    # seed=None 走**全局 numpy RNG**：先把全局 RNG 钉死，就能对"随机"逐值断言，
    # 而不是只写一句"看起来是随机的"。
    np.random.seed(20261009)
    expected_random_seeds = [
        np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX) for _ in range(2)
    ]
    np.random.seed(20261009)

    # slot 0：全默认 → top_k=-1 归一化成 vocab_size；seed=None；logprobs=None → -1
    states.add_request(0, SamplingParams())
    # slot 1：top_k=0（"不筛"的另一种写法）→ vocab_size
    states.add_request(1, SamplingParams(top_k=0))
    # slot 2：top_k > vocab_size → vocab_size
    states.add_request(2, SamplingParams(top_k=1000))
    # slot 3：显式 seed + logprobs=-1（全词表）+ 合法 top_k
    states.add_request(3, SamplingParams(top_k=5, seed=1234, logprobs=-1))

    assert states.top_k.np.tolist() == [_VOCAB_SIZE, _VOCAB_SIZE, _VOCAB_SIZE, 5]
    assert states.top_p.np.tolist() == [1.0] * _MAX_NUM_REQS
    assert states.temperature.np.tolist() == [1.0] * _MAX_NUM_REQS
    assert states.min_p.np.tolist() == [0.0] * _MAX_NUM_REQS

    # logprobs：None → -1；-1 → vocab_size。
    assert states.num_logprobs.tolist() == [
        mv_states.NO_LOGPROBS,
        mv_states.NO_LOGPROBS,
        mv_states.NO_LOGPROBS,
        _VOCAB_SIZE,
    ]

    # seed：显式 seed 原样落盘且标记 seeds_set；None 时取全局 RNG 的下一个 int64 且不标记。
    assert states.seeds_set.tolist() == [False, False, False, True]
    assert states.seeds.np[3] == 1234
    assert states.seeds.np[0] == expected_random_seeds[0]
    assert states.seeds.np[1] == expected_random_seeds[1]
    for idx in (0, 1, 2):
        assert _NP_INT64_MIN <= int(states.seeds.np[idx]) < _NP_INT64_MAX

    # 覆盖写同一个 slot：top_k=-1 与 logprobs=k>0 都要按上游口径归一化/透传。
    states.add_request(1, SamplingParams(top_k=-1, logprobs=3))
    assert states.top_k.np[1] == _VOCAB_SIZE
    assert states.num_logprobs[1] == 3

    # apply_staged_writes：把 CPU 侧真值搬进 GPU 可见的 UVA 视图。
    states.apply_staged_writes()
    assert states.top_k.gpu.tolist() == [_VOCAB_SIZE, _VOCAB_SIZE, _VOCAB_SIZE, 5]
    assert states.seeds.gpu.tolist() == [int(s) for s in states.seeds.np]

    # 三个按 batch 的读取口径。
    idx_mapping_np = np.array([0, 1, 2, 3], dtype=np.int64)
    assert states.any_greedy(idx_mapping_np) is False  # 没有 temperature == 0
    assert states.any_explicit_seed(idx_mapping_np) is True  # slot 3 有显式 seed
    # slot 3 的 logprobs=-1 被归一化成 vocab_size（32），所以整批最大值是 32；单独看 slot 1 是 3。
    assert states.max_num_logprobs(idx_mapping_np) == _VOCAB_SIZE
    assert states.max_num_logprobs(np.array([1])) == 3
    # 全默认（top_k == vocab_size、top_p == 1.0）时跳过 top_k/top_p 两个内核。
    assert states.get_top_k_top_p(torch.tensor([0, 1], device=DEVICE), np.array([0, 1])) == (
        None,
        None,
    )


@requires_cuda
def test_sampling_states_add_request_matches_upstream():
    """同一批采样参数分别喂给两个实现，逐字段比对（`seeds` 只在显式 seed 时可比）。"""
    cases: list[dict] = [
        dict(),
        dict(top_k=0),
        dict(top_k=-1),
        dict(top_k=1000),
        dict(top_k=5),
        dict(top_p=0.9, temperature=0.7, min_p=0.1),
        dict(seed=987654321, logprobs=-1),
        dict(logprobs=7, temperature=2.0),
    ]
    mine = mv_states.SamplingStates(max_num_reqs=_MAX_NUM_REQS, vocab_size=_VOCAB_SIZE)
    theirs = up_states.SamplingStates(max_num_reqs=_MAX_NUM_REQS, vocab_size=_VOCAB_SIZE)
    assert theirs.num_logprobs.tolist() == mine.num_logprobs.tolist()

    for i, kwargs in enumerate(cases):
        slot = i % _MAX_NUM_REQS  # 4 个 slot 轮着用：第二轮是对同一 slot 的覆盖写
        mine.add_request(slot, SamplingParams(**kwargs))
        theirs.add_request(slot, UpstreamSamplingParams(**kwargs))
        assert mine.temperature.np[slot] == theirs.temperature.np[slot], kwargs
        assert mine.top_p.np[slot] == theirs.top_p.np[slot], kwargs
        assert mine.top_k.np[slot] == theirs.top_k.np[slot], kwargs
        assert mine.min_p.np[slot] == theirs.min_p.np[slot], kwargs
        assert int(mine.num_logprobs[slot]) == int(theirs.num_logprobs[slot]), kwargs
        assert bool(mine.seeds_set[slot]) == bool(theirs.seeds_set[slot]), kwargs
        if kwargs.get("seed") is not None:
            assert mine.seeds.np[slot] == theirs.seeds.np[slot] == kwargs["seed"], kwargs

    mine.apply_staged_writes()
    theirs.apply_staged_writes()
    assert mine.top_k.gpu.tolist() == theirs.top_k.gpu.tolist()
    assert mine.top_p.gpu.tolist() == theirs.top_p.gpu.tolist()
    assert mine.temperature.gpu.tolist() == theirs.temperature.gpu.tolist()
    assert mine.min_p.gpu.tolist() == theirs.min_p.gpu.tolist()


# ---------------------------------------------------------------------------
# 2. min_p / temperature 内核：与上游逐值差分
# ---------------------------------------------------------------------------


def _random_logits(num_tokens: int, vocab_size: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_tokens, vocab_size, generator=gen, dtype=torch.float32)


@requires_cuda
@pytest.mark.parametrize("vocab_size", [1024, 10000])
def test_apply_min_p_matches_upstream(vocab_size):
    # 8 行 logits，其中 1/2/3/4/5/6/7 行喂**同一份数据**，只是所在请求的 min_p 不同
    # （0.05 / 0.5 / 1.0 / 0.05 / 0.5 / 1.0），于是可以拿"保留集合的包含关系"当独立语义检查：
    # 阈值越高保留越少，且都包含 argmax。这样不需要复刻 `tl.log` 的最后一两位精度。
    num_tokens = 8
    min_p = _f32([0.0, 0.05, 0.5, 1.0])
    expanded_idx_mapping = torch.tensor(
        [0, 1, 2, 3, 1, 2, 3, 0], dtype=torch.int32, device=DEVICE
    )
    base = _random_logits(num_tokens, vocab_size, seed=7)
    for row in (2, 3, 4, 5, 6, 7):
        base[row] = base[1]

    mine = base.clone().to(DEVICE)
    theirs = base.clone().to(DEVICE)
    mv_min_p.apply_min_p(mine, expanded_idx_mapping, min_p)
    up_min_p.apply_min_p(theirs, expanded_idx_mapping, min_p)

    # ① 与上游逐值相等（-inf 也要求逐位相同，torch.equal 对 -inf==-inf 为 True）。
    assert torch.equal(mine, theirs)

    # ② min_p == 0.0 的行必须**一个字节都没动**（内核提前 return，不加载 logits）。
    for row in (0, 7):
        assert torch.equal(mine[row].cpu(), base[row])

    def kept(row: int) -> set[int]:
        return set(torch.nonzero(torch.isfinite(mine[row])).flatten().tolist())

    argmax = int(base[1].argmax())
    kept_0p05, kept_0p5, kept_1p0 = kept(1), kept(2), kept(3)
    # 同一份数据 + 同一个 min_p 的两个请求行必须筛出完全相同的集合（内核无隐藏状态）。
    assert kept(4) == kept_0p05 and kept(5) == kept_0p5 and kept(6) == kept_1p0
    # ③ min_p 越大筛掉越多：0.05 ⊇ 0.5 ⊇ 1.0，且严格变小。
    assert kept_0p05 > kept_0p5 > kept_1p0
    # ④ min_p == 1.0 时 threshold == max，只留最大值（本例唯一）。
    assert int((base[1] == base[1].max()).sum()) == 1
    assert kept_1p0 == {argmax}
    # ⑤ 最大值在任何 min_p 下都必须留下；0.05 也确实筛掉了东西。
    assert argmax in kept_0p05 and 0 < len(kept_0p05) < vocab_size


@requires_cuda
@pytest.mark.parametrize("vocab_size", [1024, 10000])
def test_apply_temperature_matches_upstream(vocab_size):
    num_tokens = 6
    # slot 0 -> 0.0（greedy，内核提前 return）；slot 1 -> 1.0（不缩放，提前 return）
    temperature = _f32([0.0, 1.0, 0.7, 2.5])
    expanded_idx_mapping = torch.tensor([0, 1, 2, 3, 2, 0], dtype=torch.int32, device=DEVICE)
    base = _random_logits(num_tokens, vocab_size, seed=11)

    mine = base.clone().to(DEVICE)
    theirs = base.clone().to(DEVICE)
    gumbel.apply_temperature(mine, expanded_idx_mapping, temperature)
    up_gumbel.apply_temperature(theirs, expanded_idx_mapping, temperature)

    # ① 与上游逐值相等。
    assert torch.equal(mine, theirs)

    # ② temperature ∈ {0.0, 1.0} 的行必须原样不动。
    for row in (0, 1, 5):
        assert torch.equal(mine[row].cpu(), base[row])

    # ③ 其余行按 logits / temperature 缩放。
    #    这里用 rtol 而不是 torch.equal：Triton 的 fp32 `/` 降级成 `div.full.f32`（快除法，
    #    ~1 ulp，非 IEEE 正确舍入），与 torch 的 IEEE 除法逐位不同（实测 930/10000 个元素差 1 ulp，
    #    最大相对误差 1.1e-7 ≈ 2^-23）。**严格的那条断言是与上游的 torch.equal**，
    #    这里只钉住"确实是按本请求的温度缩放"（错槽位会差 3 倍以上）。
    for row in (2, 4):
        assert torch.allclose(
            mine[row].cpu(), base[row] / temperature[2].cpu(), rtol=1e-6, atol=0
        )
    assert torch.allclose(
        mine[3].cpu(), base[3] / temperature[3].cpu(), rtol=1e-6, atol=0
    )


# ---------------------------------------------------------------------------
# 3. PenaltiesState
# ---------------------------------------------------------------------------


def test_use_penalty_truth_table():
    """三个惩罚项都是默认值 → False；任一非默认 → True；并与上游判定逐值一致。"""
    cases = [
        {},
        dict(repetition_penalty=1.1),
        dict(frequency_penalty=0.5),
        dict(presence_penalty=0.5),
        dict(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0),
        dict(top_k=5, temperature=0.7, logprobs=3),  # 与惩罚无关的字段不影响判定
    ]
    assert mv_penalties.use_penalty(SamplingParams()) is False
    for kwargs in cases:
        mine = mv_penalties.use_penalty(SamplingParams(**kwargs))
        theirs = up_penalties.use_penalty(UpstreamSamplingParams(**kwargs))
        assert mine == theirs, kwargs
    assert mv_penalties.use_penalty(SamplingParams(repetition_penalty=0.9)) is True
    assert mv_penalties.use_penalty(SamplingParams(frequency_penalty=-1.0)) is True


def _make_req_state(
    max_num_reqs: int, vocab_size: int, max_model_len: int = 16
) -> RequestState:
    return RequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        num_speculative_steps=1,
        vocab_size=vocab_size,
        device=torch.device(DEVICE),
    )


@requires_cuda
def test_penalties_state_bincount_prompt_vs_output():
    """`prompt_len != prefill_len` 的意义：prompt 段进位图、已生成段进计数。"""
    max_num_reqs, vocab_size = 4, 64
    req_states = _make_req_state(max_num_reqs, vocab_size)
    penalties = mv_penalties.PenaltiesState(req_states)

    # A：prompt 4 枚 + 已生成 2 枚（prefill_len > prompt_len）
    #    prompt = [5, 7, 5, 40]：5 出现两次 → 位图**只置一次**；40 落在第 1 个 32 位字里
    #    生成 = [7, 7]：token 7 计数 = 2（累加），prompt 里出现过的 5/40 计数必须为 0
    req_states.add_request(
        "A", prompt_len=4, all_token_ids=[5, 7, 5, 40, 7, 7], num_computed_tokens=6, max_tokens=4
    )
    # B：只有 prompt（prefill_len == prompt_len），prompt 里 1 出现两次 → 位图置一次、计数为 0
    req_states.add_request(
        "B", prompt_len=3, all_token_ids=[1, 1, 2], num_computed_tokens=3, max_tokens=4
    )
    # C：有历史但**不需要惩罚**（全默认采样参数）→ 不该被 bincount，两张表必须保持全零
    req_states.add_request(
        "C", prompt_len=3, all_token_ids=[3, 4, 4, 4], num_computed_tokens=4, max_tokens=4
    )
    req_states.apply_staged_writes()

    slot_a = req_states.req_id_to_index["A"]
    slot_b = req_states.req_id_to_index["B"]
    slot_c = req_states.req_id_to_index["C"]

    penalties.add_request(slot_a, SamplingParams(repetition_penalty=1.5))
    penalties.add_request(slot_b, SamplingParams(frequency_penalty=0.5))
    penalties.add_request(slot_c, SamplingParams())

    # 只有"要惩罚"的请求进队列；use_penalty 逐 slot 正确。
    assert penalties._new_penalties_reqs == [slot_a, slot_b]
    assert penalties.use_penalty[slot_a] and penalties.use_penalty[slot_b]
    assert not penalties.use_penalty[slot_c]

    penalties.apply_staged_writes()
    assert penalties._new_penalties_reqs == []  # 队列被清空，不会重复 bincount

    # UVA 视图里就是刚写进去的三个惩罚系数。
    assert penalties.repetition_penalty.np[slot_a] == np.float32(1.5)
    assert penalties.repetition_penalty.gpu[slot_a].item() == np.float32(1.5)
    assert penalties.frequency_penalty.gpu[slot_b].item() == np.float32(0.5)
    assert penalties.presence_penalty.gpu[slot_a].item() == 0.0
    # repetition_penalty 的构造期默认值是 1.0（0 对它不是合法值）。
    assert penalties.repetition_penalty.np[slot_c] == np.float32(1.0)

    mask = penalties.prompt_bin_mask.cpu().numpy()
    counts = penalties.output_bin_counts.cpu().numpy()
    assert mask.shape == (max_num_reqs, cdiv(vocab_size, 32))

    def bit(row: int, token: int) -> int:
        return int((mask[row, token // 32] >> np.int32(token % 32)) & np.int32(1))

    # A 的 prompt 段 {5, 7, 40}：5 重复只置一次、40 在第 1 个字里。
    assert bit(slot_a, 5) == 1 and bit(slot_a, 7) == 1 and bit(slot_a, 40) == 1
    assert int(mask[slot_a, 0]) == (1 << 5) | (1 << 7)
    assert int(mask[slot_a, 1]) == 1 << 8  # 40 = 32 + 8
    # A 的已生成段 {7, 7}：计数累加为 2；prompt 段不计数（5/40 必须是 0）。
    assert int(counts[slot_a, 7]) == 2
    assert int(counts[slot_a].sum()) == 2
    assert int(counts[slot_a, 5]) == 0 and int(counts[slot_a, 40]) == 0
    # B 只有 prompt：位图置位、计数全 0（这正是 prompt_len 与 prefill_len 分开的意义）。
    assert int(mask[slot_b, 0]) == (1 << 1) | (1 << 2)
    assert int(counts[slot_b].sum()) == 0
    # C 没进 bincount：两张表整行全零。
    assert int(mask[slot_c].sum()) == 0
    assert int(counts[slot_c].sum()) == 0

    # 第二次 apply_staged_writes 是空操作（队列已清空），表不会被清掉或重复累加。
    penalties.apply_staged_writes()
    mask2 = penalties.prompt_bin_mask.cpu().numpy()
    counts2 = penalties.output_bin_counts.cpu().numpy()
    assert np.array_equal(mask, mask2) and np.array_equal(counts, counts2)


@requires_cuda
def test_apply_penalties_kernel_matches_upstream_and_semantics():
    """`_penalties_kernel` 逐值差分 + 一个只含 frequency / presence 的手算对照。"""
    vocab_size = 2 * 8192  # 恰好两个 BLOCK_SIZE，避免掩码分支掺进来
    num_reqs, num_tokens = 2, 6
    # 请求 0 的行：0,1,2（local pos 0,1,2）；请求 1 的行：3,4,5。
    expanded_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE)
    expanded_local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=DEVICE)
    # `expanded_local_pos` 的语义：token_ids[token_idx - pos + 1 : token_idx + 1] 是"该行之前的 token"。
    token_ids = torch.tensor([5, 7, 5, 9, 9, 2], dtype=torch.int32, device=DEVICE)

    # 请求 0 只用 presence（+1.0），请求 1 只用 frequency（-0.5）；两者都不动 prompt 位图。
    repetition_penalty = _f32([1.0, 1.0])
    frequency_penalty = _f32([0.0, 0.5])
    presence_penalty = _f32([1.0, 0.0])
    prompt_bin_mask = torch.zeros(
        num_reqs, cdiv(vocab_size, 32), dtype=torch.int32, device=DEVICE
    )
    # 请求 0 的历史计数：token 5 -> 1、token 7 -> 2；请求 1 全 0（计数全来自行内历史）。
    output_bin_counts = torch.zeros(num_reqs, vocab_size, dtype=torch.int32, device=DEVICE)
    output_bin_counts[0, 5] = 1
    output_bin_counts[0, 7] = 2

    base = _random_logits(num_tokens, vocab_size, seed=23)
    args = (
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )
    mine = base.clone().to(DEVICE)
    theirs = base.clone().to(DEVICE)
    mv_penalties.apply_penalties(mine, *args)
    up_penalties.apply_penalties(theirs, *args)
    assert torch.equal(mine, theirs)

    # 手算：请求 0（presence=1.0、freq=0、rep=1.0）逐行
    #   row 0: pos=0 → 历史为空 → 只有 output_bin_counts 里的 {5, 7} 命中
    #   row 1: pos=1 → 多出 token_ids[1] = 7
    #   row 2: pos=2 → 多出 token_ids[1..2] = 7, 5
    for row, hit in ((0, {5, 7}), (1, {5, 7}), (2, {5, 7})):
        expected = base[row].clone()
        for token in hit:
            expected[token] -= 1.0
        assert torch.equal(mine[row].cpu(), expected), row

    # 手算：请求 1（freq=0.5、pres=0、rep=1.0）→ logits -= 0.5 * 行内历史计数
    for row, pos in ((3, 0), (4, 1), (5, 2)):
        expected = base[row].clone()
        start = row - pos
        for prev in range(pos):
            expected[int(token_ids[start + prev + 1])] -= 0.5
        assert torch.equal(mine[row].cpu(), expected), row


# ---------------------------------------------------------------------------
# 4. 结构化输出：语法掩码
# ---------------------------------------------------------------------------


def _pack_allowed(allowed: np.ndarray, vocab_size: int) -> np.ndarray:
    """`allowed [num_masks, vocab]`（True = 允许）→ int32 位图 `[num_masks, cdiv(vocab,32)]`。"""
    num_masks, cols = allowed.shape[0], cdiv(vocab_size, 32)
    padded = np.zeros((num_masks, cols * 32), dtype=np.uint32)
    padded[:, :vocab_size] = allowed.astype(np.uint32)
    weights = np.uint32(1) << np.arange(32, dtype=np.uint32)
    packed = (padded.reshape(num_masks, cols, 32) * weights).sum(axis=2, dtype=np.uint32)
    return packed.view(np.int32)


def _run_bitmask_kernel(
    kernel,
    logits: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    bitmask: torch.Tensor,
    mask_stride: int,
) -> None:
    """完全照抄上游 `apply_grammar_bitmask` 的 launch（grid / 参数顺序 / BLOCK_SIZE）。"""
    vocab_size = logits.shape[-1]
    grid = (bitmask.shape[0], triton.cdiv(vocab_size, _BITMASK_BLOCK_SIZE))
    kernel[grid](
        logits,
        logits.stride(0),
        logits_indices,
        cu_num_logits,
        bitmask,
        bitmask.stride(0),
        vocab_size,
        MASK_STRIDE=mask_stride,
        BLOCK_SIZE=_BITMASK_BLOCK_SIZE,
    )


def _grammar_case(vocab_size: int, seed: int = 3):
    """构造"每请求 1 + K 行"的语法掩码输入。

    ``mask_stride = 4``；``cu_num_logits = [0, 3, 4, 6]`` → 有效行：r0 = 0,1,2；r1 = 3；r2 = 4,5。
    每个请求都比有效行**多**给一个位置（模拟自适应验证把 CPU 侧 logits 偏移压成 bonus-only 后，
    语法掩码仍按调度布局走的情形）：r0 的 pos3、r1 的 pos1、r2 的 pos2 都是
    ``position_is_active == False``，它们指向 logits 行 3 / 4 / 6。这些"非活跃"位置的位图故意给
    **全 0**——一旦 `position_is_active` 守卫失效，目标行会被打成整行 `-inf`，一眼可辨
    （行 3 / 4 同时是活跃位置的目标，所以判据是"内容恰为活跃位图的图案"；行 6 / 7 没有活跃程序
    指向，判据是"一个字节都没动"）。
    """
    num_logits_alloc = 8  # 留出 6/7 两行当"非活跃位置指向的行"
    cu_num_logits_np = np.array([0, 3, 4, 6], dtype=np.int32)
    mapping: list[int] = []
    for req_idx, num_positions in ((0, 4), (1, 2), (2, 3)):
        mapping.extend(
            req_idx * 4 + position for position in range(num_positions)
        )
    logits_indices = torch.tensor(mapping, dtype=torch.int32, device=DEVICE)
    cu_num_logits = torch.tensor(cu_num_logits_np, dtype=torch.int32, device=DEVICE)

    tokens = np.arange(vocab_size, dtype=np.int64)
    allowed = np.zeros((len(mapping), vocab_size), dtype=bool)
    # 活跃位置的允许集合：确定性的稀疏图案（含第 1 个位图字，确保跨字分支被走到）。
    active_mask_rows = [0, 1, 2, 4, 6, 7]
    for mask_idx in active_mask_rows:
        allowed[mask_idx] = (tokens * (mask_idx + 3)) % 7 == 0
    assert all(allowed[idx].any() for idx in active_mask_rows), "活跃位置的允许集合不能为空"
    assert not allowed[3].any() and not allowed[5].any() and not allowed[8].any()

    bitmask_np = _pack_allowed(allowed, vocab_size)
    base = _random_logits(num_logits_alloc, vocab_size, seed=seed)
    return mapping, logits_indices, cu_num_logits, bitmask_np, allowed, base


@requires_cuda
@pytest.mark.parametrize("vocab_size", [64, 10000])
def test_apply_grammar_bitmask_kernel(vocab_size):
    mapping, logits_indices, cu_num_logits, bitmask_np, allowed, base = _grammar_case(
        vocab_size
    )
    bitmask_mine = torch.tensor(bitmask_np, dtype=torch.int32, device=DEVICE)
    bitmask_theirs = bitmask_mine.clone()
    mine = base.clone().to(DEVICE)
    theirs = base.clone().to(DEVICE)

    _run_bitmask_kernel(
        mv_structured._apply_grammar_bitmask_kernel,
        mine,
        logits_indices,
        cu_num_logits,
        bitmask_mine,
        mask_stride=4,
    )
    _run_bitmask_kernel(
        up_structured._apply_grammar_bitmask_kernel,
        theirs,
        logits_indices,
        cu_num_logits,
        bitmask_theirs,
        mask_stride=4,
    )

    # ① 与上游逐值相等（含 `-inf`）。
    assert torch.equal(mine, theirs)

    # ② 语义：bit=0 的位置必须变 `-inf`，bit=1 的位置必须原样。
    #    mapping 的第 0/1/2 行 → logits 行 0/1/2；第 4 行 → 行 3；第 6/7 行 → 行 4/5。
    row_of_mask = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4, 7: 5}
    for mask_idx, logits_row in row_of_mask.items():
        row = mine[logits_row].cpu()
        allow = torch.from_numpy(allowed[mask_idx])
        assert torch.equal(torch.isfinite(row), allow), (mask_idx, logits_row)
        assert torch.equal(row[allow], base[logits_row][allow]), (mask_idx, logits_row)
        assert torch.isinf(row[~allow]).all() and (row[~allow] < 0).all()

    # ③ `position_is_active == False` 的位置**只能由活跃程序写**：
    #    行 3 / 行 4 同时是"非活跃位置"（mask 3 / mask 5，位图全 0）与活跃位置（mask 4 / mask 6）
    #    的目标，② 已经证明它们的内容**恰是活跃位图的图案**——若守卫失效，它们会被全 0 位图打成
    #    整行 `-inf`（那时 ② 的 `torch.equal(isfinite(row), allow)` 就挂了）。
    #    行 6（非活跃位置 mask 8 的目标）与行 7（无人指向）则必须**一个字节都没动**。
    for row in (3, 4):
        assert torch.isfinite(mine[row]).any(), row
    for row in (6, 7):
        assert torch.equal(mine[row].cpu(), base[row]), row


@dataclass
class _WorkerInputBatch:
    """只提供 `StructuredOutputsWorker.apply_grammar_bitmask` 真正读的四个字段。

    真实的 `InputBatch`（`minivllm/worker/gpu/input_batch.py`，由 73 关主 agent 维护）字段更多，
    这里刻意用最小替身：本用例只钉住**本文件移植的代码**（映射 + 异步拷贝 + 内核 + `mask_stride`
    /`num_bonus_tokens` 的口径），不因为 `input_batch.py` 的字段增删而假失败。
    """

    req_ids: list[str]
    cu_num_logits_np: np.ndarray
    num_draft_tokens_per_req: np.ndarray | None
    cu_num_logits: torch.Tensor


@requires_cuda
def test_structured_outputs_worker_apply_grammar_bitmask():
    """端到端走查：K=2/0/1 三条请求，CPU 偏移被压成 bonus-only（每请求只有 1 行有效）。"""
    vocab_size = 64
    mask_stride, num_bonus_tokens = 4, 1
    req_ids = ["r0", "r1", "r2"]
    # 草稿数：r0=1、r1=0、r2=2 → 位图位置数 = 草稿数 + bonus = 2 / 1 / 3。
    num_draft_tokens_per_req = np.array([1, 0, 2], dtype=np.int32)
    # 自适应验证把 CPU 侧 logits 偏移压成 bonus-only：每请求只剩 1 行。
    cu_num_logits_np = np.array([0, 1, 2, 3], dtype=np.int32)
    cu_num_logits = torch.tensor(cu_num_logits_np, dtype=torch.int32, device=DEVICE)

    worker = mv_structured.StructuredOutputsWorker(
        max_num_logits=8,
        vocab_size=vocab_size,
        device=torch.device(DEVICE),
        mask_stride=mask_stride,
        num_bonus_tokens=num_bonus_tokens,
    )
    input_batch = _WorkerInputBatch(
        req_ids=req_ids,
        cu_num_logits_np=cu_num_logits_np,
        num_draft_tokens_per_req=num_draft_tokens_per_req,
        cu_num_logits=cu_num_logits,
    )
    grammar_req_ids = ["r0", "r2"]

    # 映射口径由 `_build_grammar_mapping` 决定，先与上游逐值核对。
    mine_mapping = mv_structured._build_grammar_mapping(
        req_ids, grammar_req_ids, cu_num_logits_np, num_draft_tokens_per_req,
        num_bonus_tokens, mask_stride,
    )
    theirs_mapping = up_structured._build_grammar_mapping(
        req_ids, grammar_req_ids, cu_num_logits_np, num_draft_tokens_per_req,
        num_bonus_tokens, mask_stride,
    )
    assert mine_mapping == theirs_mapping == [0, 1, 8, 9, 10]

    # 允许集合：mask 行 0 允许 {1, 2, 33}；行 2 允许 {0, 63}；其余（非活跃位置）全 0。
    allowed = np.zeros((len(mine_mapping), vocab_size), dtype=bool)
    allowed[0, [1, 2, 33]] = True
    allowed[2, [0, 63]] = True
    bitmask_np = _pack_allowed(allowed, vocab_size)

    logits = torch.full((3, vocab_size), 3.5, dtype=torch.float32, device=DEVICE)
    worker.apply_grammar_bitmask(
        logits, input_batch, grammar_req_ids, bitmask_np
    )

    # 行 0 = r0 的 bonus 行；行 1 = r1 的 bonus 行（被 r0 的非活跃位置 pos1 指向，必须不动）；
    # 行 2 = r2 的 bonus 行。
    assert torch.equal(torch.isfinite(logits[0]).cpu(), torch.from_numpy(allowed[0]))
    assert torch.equal(torch.isfinite(logits[2]).cpu(), torch.from_numpy(allowed[2]))
    assert torch.equal(logits[1].cpu(), torch.full((vocab_size,), 3.5))
    assert (logits[0][~torch.from_numpy(allowed[0])] == -float("inf")).all()
    # 行 1 是全 0 位图的目标（pos1 非活跃）——守卫失效就会整行 `-inf`。
    assert not torch.isinf(logits[1]).any()

    # 同一个映射喂给上游内核：逐值相等。
    theirs = torch.full((3, vocab_size), 3.5, dtype=torch.float32, device=DEVICE)
    _run_bitmask_kernel(
        up_structured._apply_grammar_bitmask_kernel,
        theirs,
        torch.tensor(mine_mapping, dtype=torch.int32, device=DEVICE),
        cu_num_logits,
        torch.tensor(bitmask_np, dtype=torch.int32, device=DEVICE),
        mask_stride=mask_stride,
    )
    assert torch.equal(logits, theirs)
