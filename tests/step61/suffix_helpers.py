"""step61 测试的共享工具：InputBatch 替身 + "同一事件序列喂两个提议者"的驱动器。

差分思路（本关最有价值的一条）：上游 `vllm/v1/spec_decode/suffix_decoding.py` 的
`SuffixDecodingProposer` 只读 vLLM config 的 6 个字段与 InputBatch 的 5 个属性，**不依赖
Runner**，所以可以用鸭子类型的 config/批把它**原样实例化**，与我们的实现吃同一串事件：
同样的请求进入/离开批、同样的采样、同样的行序，逐步比对

    - `draft_token_ids`（每条请求的候选，**不定长**）
    - `suffix_cache.active_requests` / `cached_requests`（活跃集与全局缓存集）

输入缓冲的**类型差异照实还原**：上游读的是 vLLM InputBatch 的 **int32 numpy 视图**，
我们读的是本仓库 InputBatch 的 **int64 torch 张量**——两份替身只是 dtype/后端不同，
逻辑内容由同一个驱动器写，所以差异只能来自实现本身。
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_step59_helpers():
    """按路径加载 `tests/step59/spec_helpers.py`（不能写 `import helpers`：会重名）。"""
    path = ROOT / "tests" / "step59" / "spec_helpers.py"
    spec = importlib.util.spec_from_file_location("step59_spec_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step59_helpers = _load_step59_helpers()


# ---------------------------------------------------------------------------
# 提议者：我们的 vs 上游的（同一组配置）
# ---------------------------------------------------------------------------


def upstream_proposer(*, num_speculative_tokens, max_model_len, max_tree_depth=24,
                      max_cached_requests=10000, max_spec_factor=1.0, min_token_prob=0.1):
    """实例化**上游那份** `SuffixDecodingProposer`（差分基准，只在测试里用）。

    AGENTS §2：生产路径禁止 import 上游 Runner/proposer；差分测试允许（同 tests/step58 直接
    调上游内核的做法）。上游的 `__init__` 只读 config 的 6 个字段，所以用 SimpleNamespace
    喂鸭子类型即可——传真的 `VllmConfig` 反而要拉起整个 vLLM 平台初始化。
    """
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            suffix_decoding_max_tree_depth=max_tree_depth,
            suffix_decoding_max_cached_requests=max_cached_requests,
            suffix_decoding_max_spec_factor=max_spec_factor,
            suffix_decoding_min_token_prob=min_token_prob),
        model_config=SimpleNamespace(max_model_len=max_model_len))
    return SuffixDecodingProposer(config)


def our_proposer(*, num_speculative_tokens, max_model_len, max_tree_depth=24,
                 max_cached_requests=10000, max_spec_factor=1.0, min_token_prob=0.1):
    """实例化本仓库的 `SuffixDecodingProposer`（配置口径与上面逐字段相同）。"""
    from minivllm.config import ModelConfig, SpeculativeConfig, VllmConfig
    from minivllm.spec_decode.suffix_decoding import SuffixDecodingProposer

    config = VllmConfig(
        model_config=ModelConfig(model="fake/tiny", dtype="float32",
                                 max_model_len=max_model_len),
        speculative_config=SpeculativeConfig(
            method="suffix", num_speculative_tokens=num_speculative_tokens,
            suffix_decoding_max_tree_depth=max_tree_depth,
            suffix_decoding_max_cached_requests=max_cached_requests,
            suffix_decoding_max_spec_factor=max_spec_factor,
            suffix_decoding_min_token_prob=min_token_prob))
    return SuffixDecodingProposer(config)


def make_pair(**kwargs):
    """返回 `(ours, upstream)`：同一组配置的两个提议者。"""
    return our_proposer(**kwargs), upstream_proposer(**kwargs)


# ---------------------------------------------------------------------------
# InputBatch 替身
# ---------------------------------------------------------------------------


class FakeBatch:
    """InputBatch 的最小替身：行是**前压实**的（`req_ids[i]` ↔ `req_id_to_index[req_id] == i`）。

    `numpy_view=False` → 暴露 int64 torch 缓冲（本仓库 InputBatch 的形态，我们的提议者读它）；
    `numpy_view=True`  → 暴露 int32 numpy 视图（vLLM InputBatch 的形态，上游提议者读它）。
    """

    def __init__(self, max_num_reqs: int, max_model_len: int, *, numpy_view: bool = False):
        dtype = torch.int32 if numpy_view else torch.int64
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._tokens = torch.zeros((max_num_reqs, max_model_len), dtype=dtype)
        self._num_tokens_no_spec = torch.zeros(max_num_reqs, dtype=dtype)
        self._num_prompt_tokens = torch.zeros(max_num_reqs, dtype=dtype)
        if numpy_view:
            self.token_ids_cpu = self._tokens.numpy()
            self.num_tokens_no_spec = self._num_tokens_no_spec.numpy()
            self.num_prompt_tokens = self._num_prompt_tokens.numpy()
        else:
            self.token_ids_cpu = self._tokens
            self.num_tokens_no_spec = self._num_tokens_no_spec
            self.num_prompt_tokens = self._num_prompt_tokens
        self.req_ids: list[str] = []
        self.req_id_to_index: dict[str, int] = {}

    # 与本仓库 InputBatch 同名的两个便利方法（测试断言用）
    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    def num_tokens(self, row: int) -> int:
        return int(self._num_tokens_no_spec[row])

    def layout(self, order, histories, prompt_lens, visible=None) -> None:
        """把本步的批摆成 `order` 的行序，并把每行的历史写进缓冲。

        `histories[req_id]` 是"prompt + 已提交输出"的当前值（**含本步刚采到的**：vLLM 的
        `_bookkeeping_sync` 在提议之前就把采样结果写进缓冲，所以提议者看到的
        `num_tokens_no_spec` 已经包含本步的采样）。`visible` 用来模拟中间 prefill 块：
        该行只写到第 n 个 token（`num_tokens_no_spec = n`），采样为空、会被提议者跳过。
        """
        visible = visible or {}
        assert len(order) <= self.max_num_reqs
        self.req_ids = list(order)
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(order)}
        for row in range(self.max_num_reqs):
            self._tokens[row].zero_()
            self._num_tokens_no_spec[row] = 0
            self._num_prompt_tokens[row] = 0
        for row, req_id in enumerate(order):
            tokens = histories[req_id]
            length = min(visible.get(req_id, len(tokens)), self.max_model_len)
            if length:
                self._tokens[row, :length] = torch.tensor(tokens[:length], dtype=self.dtype)
            self._num_tokens_no_spec[row] = length
            self._num_prompt_tokens[row] = min(prompt_lens[req_id], self.max_model_len)


# ---------------------------------------------------------------------------
# 驱动器：同一串事件喂两个提议者
# ---------------------------------------------------------------------------


def run_trace(proposer, *, max_model_len: int, prompts: dict, steps: list[dict],
              num_speculative_tokens: int, max_num_reqs: int | None = None,
              numpy_view: bool = False, history: dict | None = None) -> dict:
    """按 `steps` 驱动一个提议者，返回 `drafts`（逐步草稿）与 `states`（逐步缓存状态）。

    `steps` 的每一项：`{"order": [批行序的 req_id], "sampled": {req_id: [新采到的 token]}}`，
    可选 `"visible": {req_id: n}`（中间 prefill：这一行只算到第 n 个 token）。
    请求**不在** `order` 里 = 这一步没进批（被抢占 / 预算不够 / 已结束）。

    `prompts[req_id]` 是**本轮请求的 prompt**（决定 `num_prompt_tokens`，即建树范围）；
    `history[req_id]` 可以给出"批缓冲里当前已有的历史"（默认 = prompt 本身），
    用来表达"同 ID 的新请求"或"已经产出了几枚 token 但驱动器没记录"的起点。

    `numpy_view=True` 时批缓冲按 vLLM 的 **int32 numpy 视图**暴露（上游提议者读的那种）；
    否则按本仓库的 **int64 torch 缓冲**（我们的提议者读的那种）。两者逻辑内容完全一样，
    由同一段代码写入，所以比对结果只反映实现差异。
    """
    histories = {req_id: list((history or {}).get(req_id, tokens))
                 for req_id, tokens in prompts.items()}
    prompt_lens = {req_id: len(tokens) for req_id, tokens in prompts.items()}
    batch = FakeBatch(max_num_reqs or max(len(step["order"]) for step in steps) or 1,
                      max_model_len, numpy_view=numpy_view)
    drafts_trace: list[list[list[int]]] = []
    states: list[tuple[set[str], set[str]]] = []
    for step in steps:
        sampled = step.get("sampled", {})
        for req_id, tokens in sampled.items():
            assert req_id in step["order"], "不在批里的请求不会有采样结果"
            histories[req_id] = histories[req_id] + list(tokens)
        batch.layout(step["order"], histories, prompt_lens, step.get("visible"))
        rows = [list(sampled.get(req_id, [])) for req_id in step["order"]]
        drafts = proposer.propose(num_speculative_tokens, batch, rows)
        drafts_trace.append([[int(token) for token in draft] for draft in drafts])
        states.append((set(proposer.suffix_cache.active_requests),
                       set(proposer.suffix_cache.cached_requests)))
    return {"drafts": drafts_trace, "states": states, "batch": batch, "histories": histories}


def build_proposer(proposer_kind: str, *, num_speculative_tokens: int, max_model_len: int,
                   **config):
    factory = our_proposer if proposer_kind == "ours" else upstream_proposer
    return factory(num_speculative_tokens=num_speculative_tokens,
                   max_model_len=max_model_len, **config)


def run_trace_on(proposer_kind: str, *, max_model_len: int, prompts: dict, steps: list[dict],
                 num_speculative_tokens: int, max_num_reqs: int | None = None,
                 history: dict | None = None, **config) -> dict:
    """`proposer_kind` 取 `"ours"` / `"upstream"`：建对应提议者并跑同一串事件。"""
    proposer = build_proposer(proposer_kind, num_speculative_tokens=num_speculative_tokens,
                              max_model_len=max_model_len, **config)
    return run_trace(proposer, max_model_len=max_model_len, prompts=prompts, steps=steps,
                     num_speculative_tokens=num_speculative_tokens, max_num_reqs=max_num_reqs,
                     history=history, numpy_view=(proposer_kind == "upstream"))


def run_segments_on(proposer_kind: str, *, max_model_len: int, segments: list[dict],
                    num_speculative_tokens: int, **config) -> dict:
    """**同一个提议者**依次吃多段事件，拼成一条时间线（用来表达"请求结束 → 同 ID 新请求"）。

    每段：`{"prompts": ..., "steps": [...], "history": 可选, "stop": [可选，**本段跑完后**停掉的 req_id]}`。
    `stop` 走 `suffix_cache.stop_request`（两个实现都有的公共接口——上游那份提议者没有
    `remove_requests`；我们那层的 `remove_requests` 由 `test_remove_requests_*` 单独覆盖）。
    """
    proposer = build_proposer(proposer_kind, num_speculative_tokens=num_speculative_tokens,
                              max_model_len=max_model_len, **config)
    drafts_trace: list[list[list[int]]] = []
    states: list[tuple[set[str], set[str]]] = []
    for segment in segments:
        result = run_trace(proposer, max_model_len=max_model_len, prompts=segment["prompts"],
                           steps=segment["steps"], num_speculative_tokens=num_speculative_tokens,
                           history=segment.get("history"), numpy_view=(proposer_kind == "upstream"))
        drafts_trace.extend(result["drafts"])
        states.extend(result["states"])
        for req_id in segment.get("stop", ()):
            proposer.suffix_cache.stop_request(req_id)
    return {"drafts": drafts_trace, "states": states}


# ---------------------------------------------------------------------------
# 真引擎（e2e）
# ---------------------------------------------------------------------------


def make_suffix_engine(*, tiny_dir, hf_config, spec_k=8, budget=64, blocks=32,
                       max_model_len=64, max_num_seqs=2, device=None, **suffix_kwargs):
    """按 method="suffix" 建引擎；返回 `(engine, core, runner)`（调用方负责 shutdown）。"""
    from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SchedulerConfig,
                          SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VllmConfig(
        model_config=ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                                 hf_config=hf_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device),
        speculative_config=SpeculativeConfig(method="suffix",
                                             num_speculative_tokens=spec_k,
                                             **suffix_kwargs))
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner
