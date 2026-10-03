"""step59 测试的共享工具：造 metadata / 造采样参数 / 调上游同层函数做差分。"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm.sample import Sampler, SamplingMetadata  # noqa: E402
from minivllm.sample import rejection_sampler as mine  # noqa: E402
from minivllm.testing.spec_metadata import make_metadata  # noqa: E402
from minivllm.testing.torch_rejection_sampler import TorchRejectionSampler  # noqa: E402

# 需求 059 §2 的算例：B=3、候选数 [3,0,1]、D=4
DRAFTS_ABC = [[1, 2, 3], [], [4]]


def metadata_for(drafts, device="cuda"):
    return make_metadata(drafts, device=device)


def sampling_metadata(temperatures, drafts, *, device="cuda", output_token_ids=None,
                      top_k=None, top_p=None, min_tokens=None, stop_token_ids=None,
                      no_penalties=True, prompt_token_ids=None,
                      presence_penalties=None, frequency_penalties=None,
                      repetition_penalties=None, generators=None):
    """按行造一份 `SamplingMetadata`（与 Runner 里那份同构，只是行数由用例决定）。"""

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
        prompt_token_ids=prompt_token_ids or [[] for _ in range(count)],
        output_token_ids=output_token_ids or [[] for _ in range(count)],
        min_tokens=min_tokens or [0] * count,
        stop_token_ids=stop_token_ids or [[] for _ in range(count)],
        spec_token_ids=[list(draft) for draft in drafts],
        presence_penalties=tensor(presence_penalties),
        frequency_penalties=tensor(frequency_penalties),
        repetition_penalties=tensor(repetition_penalties),
    )


def prob_rows(rows):
    """概率行 → logits 行（`softmax(log(p)) == p`，用来精确控制 p）。"""
    return torch.tensor(rows, dtype=torch.float32).log()


def compact_logits(rows, device="cuda"):
    """`rows` 是**取完之后**的紧凑行序：前 ΣK 行是验证行、后 B 行是 bonus 行。"""
    return prob_rows(rows).to(device)


def bonus_from(rows, batch, device="cuda"):
    bonus = [int(np.argmax(row)) for row in rows[len(rows) - batch:]]
    return torch.tensor(bonus, dtype=torch.int32, device=device).unsqueeze(1)


def run_ours(meta, logits, draft_probs, bonus, sm):
    """我们的生产路径（Triton 内核），返回 `[B, K+1]` 的 padded 结果。"""
    target = logits[meta.target_logits_indices]
    return mine.rejection_sample(
        meta.draft_token_ids, meta.num_draft_tokens, meta.max_spec_len,
        meta.cu_num_draft_tokens, draft_probs, target, bonus, sm)


def meta_drafts(meta):
    """从 metadata 里还原"逐请求草稿"（按 `cu_num_draft_tokens` 切）。"""
    tokens = meta.draft_token_ids.tolist()
    out, start = [], 0
    for num in meta.num_draft_tokens:
        out.append(tokens[start:start + num])
        start += num
    return out


def run_reference(meta, logits, draft_probs, bonus, sm_cpu, **kwargs):
    """Torch 参考实现（CPU；`sm_cpu` 必须是 CPU 的采样参数）。

    参考实现自己会重算 bonus（`forward` 里有 bonus 采样那一步），所以这里只用来对 token。
    """
    meta_cpu = make_metadata(meta_drafts(meta), device="cpu")
    return TorchRejectionSampler(Sampler()).forward(
        meta_cpu, None if draft_probs is None else draft_probs.cpu(),
        logits.cpu(), sm_cpu, **kwargs).sampled_token_ids


# ---------------------------------------------------------------------------
# 上游同层函数（差分用；生产路径不 import 它）
# ---------------------------------------------------------------------------


def upstream_sampling_metadata(temperatures, drafts, *, device="cuda"):
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


def run_upstream(meta, logits, draft_probs, bonus, temperatures, *, device="cuda"):
    """上游 `vllm.v1.sample.rejection_sampler.rejection_sample`（只用于差分对照）。"""
    from vllm.v1.sample.rejection_sampler import rejection_sample as upstream

    drafts = meta_drafts(meta)
    return upstream(meta.draft_token_ids, meta.num_draft_tokens, meta.max_spec_len,
                    meta.cu_num_draft_tokens, draft_probs,
                    logits[meta.target_logits_indices], bonus,
                    upstream_sampling_metadata(temperatures, drafts, device=device))


# ---------------------------------------------------------------------------
# 端到端：tiny 模型 + 真引擎（GPU 上的投机验证）
# ---------------------------------------------------------------------------


def make_config(*, tiny_dir, hf_config, spec_k=3, method="draft_model", budget=16, blocks=32,
                max_model_len=64, max_num_seqs=2, device="cuda"):
    from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                          SpeculativeConfig, VllmConfig)

    model = ModelConfig(model=tiny_dir, dtype="float32", max_model_len=max_model_len,
                        hf_config=hf_config)
    spec = None if spec_k is None else SpeculativeConfig(
        method=method, num_speculative_tokens=spec_k,
        draft_model_config=(ModelConfig(model=tiny_dir, dtype="float32",
                                        max_model_len=max_model_len, hf_config=hf_config)
                            if method == "draft_model" else None))
    return VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=device), speculative_config=spec)


def make_engine(*, tiny_dir, hf_config, **kwargs):
    """返回 `(engine, core, runner)`；调用方负责 `engine.shutdown()`。"""
    from minivllm import LLMEngine, UniProcExecutor, Worker

    config = make_config(tiny_dir=tiny_dir, hf_config=hf_config, **kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run_prompts(*, tiny_dir, hf_config, prompts, max_tokens=8, temperature=0.0, seed=None,
                collect_stats=False, **kwargs):
    """跑完一批请求，返回 `(每条请求的 token, 每步的 SpecDecodingStats 列表)`。"""
    from minivllm import SamplingParams

    engine, core, _runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, **kwargs)
    for req_id, prompt in prompts:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                          seed=seed, eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        if collect_stats:
            stats.append(core.scheduler.spec_decoding_stats)
    engine.shutdown()
    return outputs, stats
