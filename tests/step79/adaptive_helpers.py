"""step79 的测试夹具：真实 DSpark tiny 引擎 + 不建引擎的管理器。

三块：

    ManagerFixture   直接建 `RequestState` + `AdaptiveVerificationManager`，手工摆置信度/成本表，
                     用于预算与布局的手算对照。
    make_adaptive_engine  真正的 V2 DSpark 引擎（confidence head 打开，可选 adaptive），
                     用来跑端到端、CPU stale vs GPU 当前值、取消/slot 复用。
    RecordBatch      包住 `runner.prepare_inputs`，把每轮真实的 `num_draft_tokens` /
                     `cu_num_logits` / `query_start_loc` 抓下来（测试里断言用）。

`DSpark` 只有 V2；环境的开关沿用 step77 的 `v2_env()`（模块顶层不能写死，见其注释）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STEP77 = Path(__file__).resolve().parents[1] / "step77"
if str(STEP77) not in sys.path:
    sys.path.insert(0, str(STEP77))

from dspark_helpers import (DEVICE, make_vllm_config, model_configs,  # noqa: E402
                            run_prompts, v2_env)
from minivllm import (LLMEngine, SpeculativeConfig, UniProcExecutor,  # noqa: E402
                      Worker)
from minivllm.worker.gpu.spec_decode.adaptive_verification import (  # noqa: E402
    AdaptiveVerificationManager,
)
from minivllm.worker.gpu.states import RequestState  # noqa: E402


class ManagerFixture:
    """不建引擎的 manager：状态/成本表/置信度都由测试显式给。"""

    def __init__(self, req_ids=("a", "b"), *, k=3, max_num_reqs=4,
                 max_num_batched_tokens=64, max_total_logits=64,
                 prefill_len=1, num_bonus_tokens=1):
        self.k = k
        self.req_ids = list(req_ids)
        self.num_bonus_tokens = num_bonus_tokens
        self.state = RequestState(
            max_num_reqs=max_num_reqs,
            max_model_len=64,
            max_num_batched_tokens=max_num_batched_tokens,
            num_speculative_steps=k,
            vocab_size=32,
            device=torch.device(DEVICE),
        )
        for rid in self.req_ids:
            self.state.add_request(rid, 1, [1], 0, 16)
        self.state.apply_staged_writes()
        # 默认都是 decode（已过 prefill）：computed + 非草稿行 >= prefill_len。
        self.state.prefill_len.np[:] = prefill_len
        self.state.num_computed_tokens_np[:] = prefill_len
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=DEVICE)
        self.manager = AdaptiveVerificationManager(
            self.state, self.query_start_loc, num_bonus_tokens, max_total_logits)

    def slot(self, req_id: str) -> int:
        return self.state.req_id_to_index[req_id]

    def set_stale(self, confidences: dict[str, list[float]]) -> None:
        """直接写"上一轮已经落地"的 CPU 置信度（不走 record 的异步拷贝）。"""
        slot_np = self.manager._stale_confidences[self.manager._stale_idx].np
        for rid, row in confidences.items():
            slot_np[self.slot(rid)] = np.asarray(row, dtype=np.float32)

    def set_tokens(self, tokens: dict[str, int] | None = None,
                   drafts: dict[str, list[int]] | None = None):
        if tokens is None:
            tokens = {rid: 1 + self.k for rid in self.req_ids}
        if drafts is None:
            drafts = {rid: list(range(1, self.k + 1)) for rid in self.req_ids}
        return tokens, drafts

    def select(self, tokens=None, drafts=None) -> int:
        tokens, drafts = self.set_tokens(tokens, drafts)
        return self.manager.get_num_tokens(tokens, drafts)

    def indices(self) -> torch.Tensor:
        return torch.tensor(
            [self.slot(rid) for rid in self.req_ids], dtype=torch.int32, device=DEVICE)


def make_adaptive_engine(*, k=3, draft_sample_method="greedy", adaptive=True,
                         confidence_head=True, max_num_seqs=2, mode="none",
                         capture_sizes=(6, 12), budget=64, spec_kwargs=None,
                         attention_backend="torch", block_size=4, **kwargs):
    """真正的 V2 引擎；`adaptive=False` 用于固定枚数对照。"""
    _, draft_config = model_configs(confidence_head=confidence_head, **kwargs)
    spec = SpeculativeConfig(
        method="dspark",
        num_speculative_tokens=k,
        draft_model_config=draft_config,
        draft_sample_method=draft_sample_method,
        enable_adaptive_verification=adaptive,
        **(spec_kwargs or {}),
    )
    config = make_vllm_config(
        spec, max_num_seqs=max_num_seqs, budget=budget, mode=mode,
        capture_sizes=capture_sizes, block_size=block_size)
    object.__setattr__(config, "attention_backend", attention_backend)
    with v2_env():
        engine = LLMEngine(
            config, UniProcExecutor(config, Worker(config)), tokenizer=None)
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    return engine, core, runner


class RecordBatch:
    """记录每轮 `prepare_inputs` 后的真实布局（测试结束后调用 `stop()` 还原）。"""

    def __init__(self, runner):
        self.runner = runner
        self.rows: list[dict] = []
        self._orig = runner.prepare_inputs

        def wrapped(scheduler_output, batch_req_state, *args, **kwargs):
            input_batch = self._orig(
                scheduler_output, batch_req_state, *args, **kwargs)
            num_reqs = input_batch.num_reqs
            self.rows.append({
                "req_ids": list(input_batch.req_ids),
                "num_reqs": num_reqs,
                "num_tokens": input_batch.num_tokens,
                "num_draft_tokens": input_batch.num_draft_tokens,
                "scheduled_drafts": [
                    len(v) for v in scheduler_output.scheduled_spec_decode_tokens.values()
                ],
                "per_req_drafts": (
                    None if input_batch.num_draft_tokens_per_req is None
                    else input_batch.num_draft_tokens_per_req.tolist()
                ),
                "cu_num_logits": input_batch.cu_num_logits[: num_reqs + 1].tolist(),
                "query_start_loc": input_batch.query_start_loc[: num_reqs + 1].tolist(),
                # CPU 侧可能是上界；测试用它证明"没有拿它当真实 offsets"。
                "cu_num_logits_np": input_batch.cu_num_logits_np.tolist(),
                "cu_num_logits_np_exact": input_batch.cu_num_logits_np_exact,
                "query_start_loc_np": input_batch.query_start_loc_np.tolist(),
            })
            return input_batch

        runner.prepare_inputs = wrapped

    def stop(self):
        self.runner.prepare_inputs = self._orig


def run_adaptive(engine, prompts, *, max_tokens=4, shutdown=True):
    return run_prompts(engine, prompts, max_tokens=max_tokens,
                       shutdown=shutdown)
