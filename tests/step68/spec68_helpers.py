"""step68 的共享工具：造小词表张量、跑我们的/上游的采样与拒绝采样、造引擎。

命名带 `68` / `spec68` 是为了满足 AGENTS §6 的"每个 tests/stepNN 目录的辅助模块名要唯一"：
一次 pytest 跑多个目录时，同名的 `helpers` 会互相覆盖。
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.outputs import LogprobsLists, ModelRunnerOutput  # noqa: E402
from minivllm.sample import Sampler, SamplingMetadata  # noqa: E402
from minivllm.testing.spec_metadata import make_metadata  # noqa: E402


def sampling_metadata(temperatures, drafts, *, device="cpu", max_num_logprobs=None,
                      logprobs_mode="raw_logprobs", output_token_ids=None,
                      prompt_token_ids=None, top_k=None, top_p=None, min_p=None,
                      no_penalties=True, presence_penalties=None,
                      frequency_penalties=None, repetition_penalties=None,
                      min_tokens=None, stop_token_ids=None, generators=None):
    """按行造一份 `SamplingMetadata`（与 Runner 里那份同构，行数由用例决定）。"""

    def tensor(values, dtype=torch.float32):
        return None if values is None else torch.tensor(values, dtype=dtype, device=device)

    count = len(temperatures)
    all_greedy = all(value < 1e-5 for value in temperatures)
    return SamplingMetadata(
        temperature=None if all_greedy else tensor(temperatures),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_k=tensor(top_k, torch.int64), top_p=tensor(top_p), min_p=tensor(min_p),
        generators=dict(generators or {}),
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids or [[] for _ in range(count)],
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        min_tokens=min_tokens or [0] * count,
        stop_token_ids=stop_token_ids or [[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts],
        presence_penalties=tensor(presence_penalties),
        frequency_penalties=tensor(frequency_penalties),
        repetition_penalties=tensor(repetition_penalties),
        max_num_logprobs=max_num_logprobs,
        logprobs_mode=logprobs_mode,
    )


def prob_rows(rows, device="cpu"):
    """概率行 → logits 行（`softmax(log(p)) == p`，用于精确控制 p）。"""
    return torch.tensor(rows, dtype=torch.float32, device=device).log()


# ---------------------------------------------------------------------------
# 上游同层对象（只给差分测试用；生产路径不 import 它们）
# ---------------------------------------------------------------------------


def upstream_sampling_metadata(temperatures, drafts, *, device="cpu",
                               max_num_logprobs=None, min_p=None, top_k=None,
                               top_p=None, output_token_ids=None,
                               prompt_token_ids=None, no_penalties=True,
                               presence_penalties=None, frequency_penalties=None,
                               repetition_penalties=None, generators=None):
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata as VllmSamplingMetadata

    count = len(temperatures)
    all_greedy = all(value < 1e-5 for value in temperatures)

    def tensor(values, dtype=torch.float32):
        return None if values is None else torch.tensor(values, dtype=dtype, device=device)

    return VllmSamplingMetadata(
        temperature=None if all_greedy else tensor(temperatures),
        all_greedy=all_greedy, all_random=all(value >= 1e-5 for value in temperatures),
        top_p=tensor(top_p) if top_p is not None else None,
        top_k=tensor(top_k, torch.int64) if top_k is not None else None,
        generators=dict(generators or {}), max_num_logprobs=max_num_logprobs,
        no_penalties=no_penalties,
        # 上游这里要的是 [num_reqs, max_prompt_len] 的 int64 张量（它按张量做计数掩码）；
        # 本仓库存的是逐行 list。差分用例两边给同一批 prompt token。
        prompt_token_ids=(torch.tensor(prompt_token_ids, dtype=torch.int64, device=device)
                          if prompt_token_ids is not None else None),
        frequency_penalties=(tensor(frequency_penalties) if frequency_penalties is not None
                             else torch.zeros(count, device=device)),
        presence_penalties=(tensor(presence_penalties) if presence_penalties is not None
                            else torch.zeros(count, device=device)),
        repetition_penalties=(tensor(repetition_penalties)
                              if repetition_penalties is not None
                              else torch.ones(count, device=device)),
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        allowed_token_ids_mask=None, bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=[list(draft) for draft in drafts] if drafts is not None else None)


def upstream_spec_metadata(drafts, device="cpu"):
    """上游 `SpecDecodeMetadata`（本仓库 `testing/spec_metadata.py` 的同一布局）。"""
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    mine = make_metadata(drafts, device=device)
    return SpecDecodeMetadata(
        draft_token_ids=mine.draft_token_ids,
        num_draft_tokens=mine.num_draft_tokens,
        cu_num_draft_tokens=mine.cu_num_draft_tokens,
        cu_num_sampled_tokens=mine.cu_num_sampled_tokens,
        target_logits_indices=mine.target_logits_indices,
        bonus_logits_indices=mine.bonus_logits_indices,
        logits_indices=mine.logits_indices,
    )


def ours_sampler_output(logits, sm, *, predict_bonus_token=False, mode=None):
    """跑我们的 `Sampler.forward`，返回 `SamplerOutput`。

    **先 clone**：`apply_logits_processors` 与温度缩放都是**原地**改 logits（上游同款，
    为了省显存）。差分用例如果把同一份张量先后喂给两个实现，第二个实现会在"已经被改过的"
    数据上再算一遍（实测：温度被除了两次）——那是用例的错，不是实现的错。
    """
    return Sampler(sm.logprobs_mode if mode is None else mode).forward(
        logits.clone(), sm, predict_bonus_token=predict_bonus_token)


def upstream_sampler_output(logits, sm, *, mode="raw_logprobs", predict_bonus_token=False,
                            override=None):
    from vllm.v1.sample.sampler import Sampler as UpstreamSampler

    return UpstreamSampler(logprobs_mode=mode).forward(
        logits.clone(), sm, predict_bonus_token=predict_bonus_token,
        logprobs_mode_override=override)


# ---------------------------------------------------------------------------
# 引擎（端到端）
# ---------------------------------------------------------------------------


def make_config(*, model_dir, hf_config, spec_k=None, method="ngram", budget=32, blocks=32,
                max_model_len=64, max_num_seqs=4, device="cpu", logprobs_mode="raw_logprobs",
                backend="auto", max_logprobs=20, structured_outputs_config=None):
    from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                          SpeculativeConfig, VllmConfig)

    model = ModelConfig(model=model_dir, dtype="float32", max_model_len=max_model_len,
                        hf_config=hf_config, logprobs_mode=logprobs_mode,
                        max_logprobs=max_logprobs, tokenizer=model_dir)
    spec = None if spec_k is None else SpeculativeConfig(
        method=method, num_speculative_tokens=spec_k)
    kwargs = {}
    if structured_outputs_config is not None:
        kwargs["structured_outputs_config"] = structured_outputs_config
    elif backend != "auto":
        from minivllm.config import StructuredOutputsConfig

        kwargs["structured_outputs_config"] = StructuredOutputsConfig(backend=backend)
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device), speculative_config=spec, **kwargs)


def make_engine(**kwargs):
    """返回 `(engine, core, runner)`；调用方负责 `engine.shutdown()`。"""
    from minivllm import LLMEngine, UniProcExecutor, Worker

    config = make_config(**kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)),
                       tokenizer=_tokenizer_for(config))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def _tokenizer_for(config):
    try:
        from minivllm.tokenizer_utils import cached_tokenizer_from_config

        return cached_tokenizer_from_config(config.model_config)
    except Exception:            # noqa: BLE001 —— tiny 模型目录可能没有 tokenizer
        return None


def run_requests(*, model_dir, hf_config, requests, max_steps=200, collect=None, **kwargs):
    """跑完一批请求，返回 `(每请求的 RequestOutput 列表, core)`。

    `requests` 是 `(req_id, prompt_token_ids, sampling_params)` 三元组。
    """
    engine, core, _runner = make_engine(model_dir=model_dir, hf_config=hf_config, **kwargs)
    for req_id, prompt, params in requests:
        engine.add_request(req_id, list(prompt), params)
    collected: dict[str, list] = {}
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            collected.setdefault(output.request_id, []).append(output)
            if collect is not None:
                collect.append(output)
    engine.shutdown()
    return collected, core


def logprobs_lists(token_ids, logprobs, ranks, cu=None) -> LogprobsLists:
    """手工拼一份 `LogprobsLists`（用例里逐值可控）。"""
    return LogprobsLists(np.asarray(token_ids, dtype=np.int32),
                         np.asarray(logprobs, dtype=np.float32),
                         np.asarray(ranks, dtype=np.int32), cu)


def runner_output(req_ids, tokens, logprobs=None):
    return ModelRunnerOutput(req_ids=list(req_ids),
                             req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                             sampled_token_ids=[list(t) for t in tokens],
                             logprobs=logprobs)
