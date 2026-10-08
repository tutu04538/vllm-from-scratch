"""step71 测试的共享夹具（名字唯一：一次 pytest 跑多个 `tests/stepNN` 时不能同名）。

71 关（动态投机长度）要验证的三层：

    查找表      区间表 → `dense_schedule[批大小]`（与上游工具函数逐值差分）
    配置改写    full graph → PIECEWISE；DP>1 → 关表；方法边界（固定 K 的提议者不允许）
    两轮时序    本轮 K 控制的是"本轮采完后要提的草稿"，本轮验证的是**上一轮**提的候选

所以夹具提供：

    make_config / make_engine   可指定区间表、图模式的 tiny 引擎
    spy_runner                  记录每一轮**真实**发出的 `SchedulerOutput`
                                （K、被调度的请求数、本轮验证的旧候选）+ 提议宽度 + 第一遍次数
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.config import CompilationConfig  # noqa: E402
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir  # noqa: E402

# 投机验证在 CUDA 上走 Triton 内核（59 关起与上游一致），CPU 上走参考实现
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
requires_cuda = __import__("pytest").mark.skipif(
    not torch.cuda.is_available(),
    reason="这一项要看 CUDA Graph / Triton 拒绝采样的真实行为（CPU 上待验，不是通过）")

TINY = tiny_qwen3_dir("tiny_gqa")
HF = tiny_qwen3_config("tiny_gqa")

#: 需求 071 §3 的那个例子：最大 K=4，`[(1,2,4),(5,8,1)]`
DOC_TABLE = [(1, 2, 4), (5, 8, 1)]


def make_config(*, table=None, spec_k=None, method="draft_model", mode=None, budget=32,
                blocks=32, max_model_len=64, max_num_seqs=2, enforce_eager=False,
                block_size=4, device=DEVICE, capture_sizes=None, draft_sample_method="greedy",
                draft_hf=None, async_scheduling=None):
    """造一份可指定"区间表 / 图模式 / 并发上限"的配置。

    `table=None`（且 `spec_k` 给了）= 静态 K 的对照组；`table` 给了就是动态投机长度。
    """
    model = ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                        hf_config=HF, enforce_eager=enforce_eager)
    spec = None
    if spec_k is not None:
        spec = SpeculativeConfig(
            method=method, num_speculative_tokens=spec_k,
            num_speculative_tokens_per_batch_size=table,
            draft_sample_method=draft_sample_method,
            draft_model_config=ModelConfig(model=TINY, dtype="float32",
                                           max_model_len=max_model_len,
                                           hf_config=draft_hf or HF))
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget,
                                         async_scheduling=async_scheduling),
        device_config=DeviceConfig(device=device), speculative_config=spec,
        compilation_config=CompilationConfig(cudagraph_mode=mode,
                                             cudagraph_capture_sizes=capture_sizes))



def make_engine(**kwargs):
    """返回 `(engine, core, runner)`；调用方负责 `engine.shutdown()`。"""
    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def make_prompt_requests(engine, req_ids, prompt=(1, 2, 3, 4, 5, 6), max_tokens=6):
    for req_id in req_ids:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))


def run_to_end(engine, limit=400):
    final = {}
    for _ in range(limit):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            final[out.request_id] = list(out.token_ids)
    return final


def greedy(*, req_ids=("A",), max_tokens=6, **kwargs):
    engine, _core, _runner = make_engine(**kwargs)
    make_prompt_requests(engine, req_ids, max_tokens=max_tokens)
    outputs = run_to_end(engine)
    engine.shutdown()
    return outputs


class RoundRecorder:
    """记录每一轮**真实发生**的四件事（测试只读这里，不猜内部字段）。

        k            调度器本轮选出的 K（`SchedulerOutput.num_spec_tokens_to_schedule`）
        num_reqs     本轮**实际被调度**的请求数（查表的输入，需求 071 §3.2）
        adopted      本轮要**验证**的旧候选（`scheduled_spec_decode_tokens`，按 req_id）
        proposed     本轮采完之后**提**的草稿（提议者返回值，按 req_id）
        first_passes 本轮 draft 第一遍跑了几次（K=0 轮的"仍要跑第一遍"靠它证明）
        drafts_gpu   本轮提的草稿带不带 q（概率草稿），形状 `[P, V]`
    """

    def __init__(self, runner, core) -> None:
        self.runner = runner
        self.core = core
        self.rounds: list[dict] = []
        self._patched: list[tuple[object, str, object]] = []
        self._patch_execute(runner)
        self._patch_proposer(runner)

    def _patch_execute(self, runner) -> None:
        original = runner.execute_model
        self._patched.append((runner, "execute_model", original))

        def spy(packet, non_block: bool = False):
            self.rounds.append({
                "k": packet.num_spec_tokens_to_schedule,
                "num_reqs": len(packet.num_scheduled_tokens),
                "num_scheduled": dict(packet.num_scheduled_tokens),
                "adopted": {req_id: list(tokens)
                            for req_id, tokens in packet.scheduled_spec_decode_tokens.items()},
                "proposed": {},
                "first_passes": 0,
                "drafts_gpu": None,
            })
            return original(packet, non_block=non_block)

        runner.execute_model = spy

    def _patch_proposer(self, runner) -> None:
        proposer = runner.proposer
        self.proposer = proposer
        original_propose = proposer.propose
        self._patched.append((proposer, "propose", original_propose))

        def propose_spy(*args, **kwargs):
            result = original_propose(*args, **kwargs)
            if self.rounds:
                if hasattr(result, "req_ids"):
                    # 本仓库的统一协议：`DraftTokenIds`
                    req_ids, drafts = list(result.req_ids), result.draft_token_ids
                    probs = getattr(result, "draft_probs", None)
                else:
                    # 上游形态（ngram 等）：`list[list[int]]`，按**批行序**对齐
                    req_ids = list(runner.input_batch.req_ids)
                    drafts, probs = result, None
                self.rounds[-1]["proposed"] = {req_id: list(tokens)
                                               for req_id, tokens in zip(req_ids, drafts)}
                self.rounds[-1]["drafts_gpu"] = (None if probs is None
                                                 else tuple(probs.shape))
            return result

        proposer.propose = propose_spy

        if hasattr(proposer, "_forward"):
            original_forward = proposer._forward
            self._patched.append((proposer, "_forward", original_forward))

            def forward_spy(*args, **kwargs):
                if self.rounds:
                    self.rounds[-1]["first_passes"] += 1
                return original_forward(*args, **kwargs)

            proposer._forward = forward_spy

    @property
    def ks(self) -> list[int]:
        return [round_["k"] for round_ in self.rounds]

    def restore(self) -> None:
        """把被观察的方法原样装回去（引擎可能还要继续跑）。"""
        for target, name, original in self._patched:
            setattr(target, name, original)


def table_lookup(table, max_batch_size, max_k, batch_size):
    """测试里独立算一遍 `dense_schedule[batch_size]`（用生产实现，但只走这一条公共入口）。"""
    from minivllm.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

    return build_dynamic_sd_schedule_lookup(table, max_batch_size, max_k)[batch_size]


__all__ = ["DEVICE", "DOC_TABLE", "HF", "TINY", "RoundRecorder", "greedy", "make_config",
           "make_engine", "make_prompt_requests", "requires_cuda", "run_to_end",
           "table_lookup"]
