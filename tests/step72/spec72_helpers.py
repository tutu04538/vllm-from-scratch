"""step72 测试夹具：PARD / P-EAGLE 的 tiny 引擎与"第一遍真实缓冲区"观测。

72 关要验的三件事：

    输入布局    [有效行][锚点][K−1 个 mask][被拒行] —— 与上游 Triton kernel **逐值**一致
    一次 forward 并行模式下每轮只跑 1 次 draft 前向（串行是 K 次）
    槽位/预算   PARD 净增 K 行、P-EAGLE 净增 K−1 行（K=1 时 P-EAGLE 是 0）

夹具提供：

    make_config / make_engine     可指定 method / K / parallel_drafting 的 tiny 引擎
    ParallelRunRecorder           记录每轮第一遍**真正写进工作区**的那些行
                                  （input_ids / positions / 两个 mask / 采样行 / hidden / 槽位）
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.tiny_models import (tiny_eagle3_dir, tiny_pard_dir,  # noqa: E402
                                          tiny_qwen3_config, tiny_qwen3_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = __import__("pytest").mark.skipif(
    not torch.cuda.is_available(),
    reason="并行提议的输入布局差分要对上游 Triton kernel 跑（CPU 上待验，不是通过）")

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")
PARD_TOKEN = 5
MASK_TOKEN = 7


def draft_dir(method: str, *, parallel: bool, mask_token: int = MASK_TOKEN,
              pard_token: int = PARD_TOKEN) -> tuple[str, dict]:
    """造一份**对应格式**的 tiny draft 并返回 `(目录, hf 配置)`。

    PARD：普通 tiny qwen3 + `pard_token`（不吃 hidden、不左移）
    P-EAGLE：tiny eagle3 + `mask_token_id` + 权重里的 `mask_hidden`（并行第一遍要它）
    """
    if method == "eagle3":
        path = tiny_eagle3_dir("tiny_gqa", parallel_drafting=parallel,
                               mask_token_id=mask_token)
    else:
        path = tiny_pard_dir("tiny_gqa", pard_token=pard_token)
    return path, json.loads((Path(path) / "config.json").read_text())


def make_config(*, method="eagle3", spec_k=3, parallel=False, mode="none", budget=32,
                blocks=32, max_num_seqs=2, max_model_len=64, draft=None):
    """`spec_k=None` = **不开投机**（对照组的非投机引擎）；否则按 method/parallel 建 draft。"""
    spec = None
    if spec_k is not None:
        draft_path, draft_hf = draft if draft is not None else draft_dir(method, parallel=parallel)
        spec = SpeculativeConfig(
            method=method, num_speculative_tokens=spec_k, parallel_drafting=parallel,
            draft_model_config=ModelConfig(model=draft_path, dtype="float32",
                                           max_model_len=max_model_len, hf_config=draft_hf))
    return VllmConfig(
        model_config=ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                                 hf_config=HF),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode=mode))


def make_engine(**kwargs):
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def greedy(*, req_ids=("A",), prompt=(1, 2, 3, 4, 5, 6), max_tokens=6, **kwargs):
    engine, _core, _runner = make_engine(**kwargs)
    try:
        for req_id in req_ids:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                              eos_token_id=999))
        return run_to_end(engine)
    finally:
        engine.shutdown()


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


class ParallelRunRecorder:
    """记录并行提议每一轮**真正写进工作区**的东西（提议者只读，测试自己算参考值对比）。"""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.proposer = runner.proposer
        self.rounds: list[dict] = []
        self.forwards = 0
        self._patched = []
        proposer = self.proposer

        original_first = proposer.set_inputs_first_pass

        def first(*args, **kwargs):
            plan = original_first(*args, **kwargs)
            n = plan.num_tokens
            self.rounds.append({
                "num_tokens": n,
                "input_ids": proposer.input_ids_cpu[:n].tolist(),
                "positions": proposer.positions_cpu[:n].tolist(),
                "is_rejected": [int(x) for x in
                                proposer.is_rejected_token_mask_cpu[:n].tolist()],
                "is_masked": [int(x) for x in
                              proposer.is_masked_token_mask_cpu[:n].tolist()],
                "sample_rows": list(plan.sample_rows),
                "sample_positions": list(plan.sample_positions),
                "seq_lens": list(plan.seq_lens),
                "query_start_loc": [int(x) for x in
                                    proposer.query_start_loc_cpu[:plan.num_reqs + 1].tolist()],
                "slot_mapping": [int(x) for x in proposer.slot_mapping_cpu[:n].tolist()],
                "hidden": (proposer.hidden_states_cpu[:n].clone()
                           if proposer.pass_hidden_states_to_model else None),
                "drafts": [],
            })
            return plan

        proposer.set_inputs_first_pass = first
        self._patched.append(("set_inputs_first_pass", original_first))

        original_forward = proposer._forward

        def forward(*args, **kwargs):
            self.forwards += 1
            return original_forward(*args, **kwargs)

        proposer._forward = forward
        self._patched.append(("_forward", original_forward))

        original_sample = proposer._sample_draft_tokens

        def sample(hidden, row_refs, input_batch, drafts, probs):
            original_sample(hidden, row_refs, input_batch, drafts, probs)
            if self.rounds:
                self.rounds[-1]["drafts"] = [len(drafts[req_id])
                                             for req_id, _ in row_refs]

        proposer._sample_draft_tokens = sample
        self._patched.append(("_sample_draft_tokens", original_sample))

    def restore(self) -> None:
        for name, original in self._patched:
            setattr(self.proposer, name, original)


__all__ = ["DEVICE", "HF", "MASK_TOKEN", "PARD_TOKEN", "ParallelRunRecorder", "TINY",
           "draft_dir", "greedy", "make_config", "make_engine", "requires_cuda", "run_to_end"]
