"""78 关验收探针：DFlash2 候选关联选择与 LiLiCorr 版本边界（跑法见文末）。

跑法：`python benchmarks/check_step78_dflash2.py`（退出码非 0 = 失败）。

分段（对应需求 078 §3/§5）：

    A 配置边界     architectures=["DFlash2DraftModel"] 被接受、K 写回 draft 配置、
                   V1 明确拒绝、缺 selector/conv 字段/缩减词表/非法 top-k 全部拒绝、
                   分派依据是 architecture（不是 method/directory）
    B 公式与内核   分组卷积 / 边分 / 候选 top-k 与上游真实模块对照；
                   walk 与 cache 两个内核在 greedy/概率、K=1/top-k=1 下与上游**逐位相等**
    C realized q   q 只在候选列有值、旧候选列被清成 -inf、交集列更新、slot 复用、
                   补齐行不写；2000 个种子的经验分布 vs softmax(q/temp)
    D 模型/端到端  解码器层与参考公式逐值相等；greedy 无损；图/eager 逐轮草稿相同；
                   greedy vs probabilistic（温度 0）同输出；slot 复用后仍无损

⚠️ 本机**没有真实 DFlash2 权重**（只有 tiny 模型），所以这里验的是"协议 / 数学 / 图"，
不是草稿质量与加速比（需求 078 §5：真实权重与 tiny 结果分开报告）。
"""

import copy
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step78"))

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from dflash2_helpers import (DEVICE, make_baseline_engine, make_engine,  # noqa: E402
                             make_spec_config, make_speculator, make_upstream_conv,
                             make_upstream_selector, model_configs, run_prompts,
                             set_walk_inputs, tiny_pair, walk_case)
from minivllm import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,  # noqa: E402
                      SpeculativeConfig, VllmConfig)
from minivllm.models.qwen3_dflash2 import (CandidateSelector,  # noqa: E402
                                           DFlashGroupedConv, _grouped_conv,
                                           _score_edges)
from minivllm.testing.torch_dflash2 import (reference_decoder_layer,  # noqa: E402
                                            reference_greedy_walk,
                                            reference_grouped_conv,
                                            reference_score_edges)
from minivllm.testing.tiny_models import tiny_dflash_dir  # noqa: E402

PASSED, FAILED = 0, []
CUDA = torch.cuda.is_available()
VOCAB = 11
TOP_K = 3


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")


def _vllm_config(spec, max_num_seqs=2):
    target_dir, target_hf, _, _ = tiny_pair()
    return VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=target_hf),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=64),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=64),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)


def _draft_hf(**kwargs):
    return copy.deepcopy(tiny_pair(**kwargs)[3])


def _model_config(hf, directory):
    return ModelConfig(model=directory, dtype="float32", max_model_len=64, hf_config=hf)


# ================================================ A 配置

print("== A 配置边界 ==")
spec = make_spec_config(k=3)
check("A1. architectures=['DFlash2DraftModel'] 被接受，method 仍是 'dflash'",
      spec.method == "dflash" and spec.uses_dflash2(),
      f"method={spec.method}")
check("A2. parallel_drafting=True、槽位预算=K",
      spec.parallel_drafting is True and spec.max_num_new_slots_for_drafting == 3)
check("A3. K 写回 draft hf_config（模型构造只拿得到 hf_config）",
      spec.draft_model_config.hf_config["num_speculative_tokens"] == 3)

previous = os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
try:
    try:
        _vllm_config(spec)
        v1_ok = False
    except NotImplementedError:
        v1_ok = True
finally:
    if previous is not None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous
check("A4. 不加 VLLM_USE_V2_MODEL_RUNNER 时配置期明确拒绝（V1 会静默退化成 DFlash1）",
      v1_ok)

_, _, draft_dir, _ = tiny_pair()
bad = 0
for field in ("selector_top_k", "selector_rank", "conv_kernel_size", "conv_group_size"):
    hf = _draft_hf()
    del hf["dflash_config"][field]
    try:
        SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                          draft_model_config=_model_config(hf, draft_dir))
    except ValueError:
        bad += 1
check("A5. 缺任一 selector/conv 字段都拒绝（4/4）", bad == 4, f"{bad}/4")

hf = _draft_hf()
hf["dflash_config"]["conv_group_size"] = 7
try:
    SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                      draft_model_config=_model_config(hf, draft_dir))
    group_ok = False
except ValueError:
    group_ok = True
hf = _draft_hf()
hf["dflash_config"]["selector_top_k"] = VOCAB + 1
try:
    SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                      draft_model_config=_model_config(hf, draft_dir))
    topk_ok = False
except ValueError:
    topk_ok = True
hf = _draft_hf()
hf["dflash_config"]["output_multiplier"] = 0.0
try:
    SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                      draft_model_config=_model_config(hf, draft_dir))
    mult_ok = False
except ValueError:
    mult_ok = True
check("A6. conv_group_size 不整除 / selector_top_k 越界 / output_multiplier<=0 都拒绝",
      group_ok and topk_ok and mult_ok, (group_ok, topk_ok, mult_ok))

hf = _draft_hf()
hf["draft_vocab_size"] = VOCAB - 2
try:
    SpeculativeConfig(method="dflash", num_speculative_tokens=3,
                      draft_model_config=_model_config(hf, draft_dir))
    reduced_ok = False
except NotImplementedError:
    reduced_ok = True
check("A7. 缩减词表（d2t）在 DFlash2 上明确拒绝（候选 id 与 q 宽度会不一致）",
      reduced_ok)

from minivllm.worker.gpu.spec_decode import init_speculator  # noqa: E402
from minivllm.worker.gpu.spec_decode.dflash.speculator import (  # noqa: E402
    DFlashSpeculator)
from minivllm.worker.gpu.spec_decode.dflash2.speculator import (  # noqa: E402
    DFlash2Speculator)

speculator = init_speculator(_vllm_config(spec), torch.device(DEVICE))
plain_dir = tiny_dflash_dir("tiny_gqa")
plain_hf = json.loads((Path(plain_dir) / "config.json").read_text())
plain_spec = SpeculativeConfig(
    method="dflash", num_speculative_tokens=3,
    draft_model_config=ModelConfig(model=plain_dir, dtype="float32", max_model_len=64,
                                   hf_config=plain_hf))
plain_speculator = init_speculator(_vllm_config(plain_spec), torch.device(DEVICE))
check("A8. init_speculator 按 architecture 分派（同一 method='dflash'）",
      isinstance(speculator, DFlash2Speculator)
      and isinstance(plain_speculator, DFlashSpeculator)
      and not isinstance(plain_speculator, DFlash2Speculator))

# ================================================ B 公式与内核

print("== B 公式与内核 ==")
generator = torch.Generator().manual_seed(0)
hidden = torch.randn(12, 32, generator=generator)
delta = torch.randn(12, 3, 4, generator=generator)
base = torch.randn(3, 32, generator=generator)
ours = _grouped_conv(hidden, delta, base, 4, 4, 8, 3)
reference = reference_grouped_conv(hidden, delta, base, 4, 4, 8, 3)
check("B1. 分组卷积 vs 逐 token 参考", torch.equal(ours, reference))

mine_conv = DFlashGroupedConv(32, 3, 8, 4)
up_conv = make_upstream_conv(hidden_size=32, taps=3, group_size=8, block_size=4)
with torch.no_grad():
    mine_conv.base_kernel.copy_(torch.randn(2, 3, 32, generator=generator) * 0.1)
    mine_conv.kernel_projection.weight.copy_(
        torch.randn(24, 32, generator=generator) * 0.1)
    up_conv.base_kernel.copy_(mine_conv.base_kernel)
    up_conv.kernel_projection.weight.copy_(mine_conv.kernel_projection.weight)
with torch.no_grad():
    mine_prepare, mine_coeff = mine_conv.prepare(hidden)
    up_prepare, up_coeff = up_conv.prepare(hidden)
conv_diff = max((mine_prepare - up_prepare).abs().max().item(),
                (mine_coeff - up_coeff).abs().max().item())
check("B2. 分组卷积 vs 上游 DFlashGroupedConv",
      conv_diff < 1e-6, f"max|Δ|={conv_diff:.2e}")

candidate_ids = torch.stack([
    torch.randperm(VOCAB, generator=generator)[:TOP_K] for _ in range(2 * 3)
]).view(2, 3, TOP_K)
unary = torch.randn(2, 3, TOP_K, generator=generator)
hidden_proj = torch.randn(2, 3, 3, generator=generator) * 0.1
pred = torch.randn(VOCAB, 3, generator=generator) * 0.1
succ = torch.randn(VOCAB, 3, generator=generator) * 0.1
anchors = torch.tensor([1, 5])
mine_scores = _score_edges(pred, succ, candidate_ids, unary, hidden_proj, anchors, TOP_K)
ref_scores = reference_score_edges(pred, succ, candidate_ids, unary, hidden_proj,
                                   anchors, TOP_K)
check("B3. 边分 vs 逐元素参考", (mine_scores - ref_scores).abs().max().item() < 1e-6,
      f"max|Δ|={(mine_scores - ref_scores).abs().max().item():.2e}")

mine_selector = CandidateSelector(32, VOCAB, 3, TOP_K)
up_selector = make_upstream_selector(hidden_size=32, vocab_size=VOCAB, rank=3,
                                     top_k=TOP_K)
with torch.no_grad():
    mine_selector.predecessor_codebook.copy_(
        torch.randn(VOCAB, 3, generator=generator) * 0.2)
    mine_selector.successor_codebook.copy_(
        torch.randn(VOCAB, 3, generator=generator) * 0.2)
    mine_selector.hidden_projection.weight.copy_(
        torch.randn(3, 32, generator=generator) * 0.2)
    up_selector.predecessor_codebook.copy_(mine_selector.predecessor_codebook)
    up_selector.successor_codebook.copy_(mine_selector.successor_codebook)
    up_selector.hidden_projection.weight.copy_(mine_selector.hidden_projection.weight)
states = torch.randn(2, 3, 32, generator=generator)
with torch.no_grad():
    selector_diff = (mine_selector(candidate_ids, unary, states, anchors)
                     - up_selector(candidate_ids, unary, states, anchors)).abs().max()
check("B4. CandidateSelector vs 上游模块", selector_diff.item() < 1e-6,
      f"max|Δ|={selector_diff.item():.2e}")

if CUDA:
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (  # noqa: E402
        _cache_draft_logits_kernel, _selector_walk_kernel)

    def run_walk(speculator, candidate_ids, scores, temperature, k, top_k, slots):
        set_walk_inputs(speculator, anchors=list(range(len(slots))), temperature=temperature,
                        seeds=13, slot_ids=slots)
        speculator._sample_path(candidate_ids, scores, len(slots))
        return (speculator.draft_tokens[:len(slots)].clone(),
                speculator._selector_scores[:len(slots)].clone())

    def upstream_walk(speculator, candidate_ids, scores, k, top_k, slots, probabilistic):
        tokens = torch.zeros(len(slots), k, dtype=torch.int64, device=DEVICE)
        realized = torch.zeros(len(slots), k, top_k, dtype=torch.float32, device=DEVICE)
        _selector_walk_kernel[(len(slots),)](
            scores.contiguous(), candidate_ids.contiguous(), speculator.sample_pos,
            speculator.sample_idx_mapping, speculator.temperature, speculator.seeds,
            tokens, realized, num_steps=k, top_k=top_k,
            BLOCK_K=max(1, 1 << (top_k - 1).bit_length()),
            SAMPLE_PROBABILISTIC=probabilistic, USE_FP64=False, num_warps=1)
        return tokens, realized

    walk_speculator, _ = make_speculator(k=3, draft_sample_method="probabilistic",
                                         max_num_seqs=4)
    cands, scores = walk_case(batch=3, k=3, seed=1)
    slots = [2, 0, 1]
    ours_tokens, ours_realized = run_walk(walk_speculator, cands, scores, 0.0, 3, TOP_K,
                                          slots)
    up_tokens, up_realized = upstream_walk(walk_speculator, cands, scores, 3, TOP_K,
                                           slots, False)
    check("B5. walk（greedy）vs 上游内核逐位相等",
          torch.equal(ours_tokens, up_tokens) and torch.equal(ours_realized, up_realized))

    ours_tokens, ours_realized = run_walk(walk_speculator, cands, scores, 0.7, 3, TOP_K,
                                          slots)
    up_tokens, up_realized = upstream_walk(walk_speculator, cands, scores, 3, TOP_K,
                                           slots, True)
    check("B6. walk（概率，temp=0.7）vs 上游内核逐位相等",
          torch.equal(ours_tokens, up_tokens) and torch.equal(ours_realized, up_realized))

    ref_tokens, ref_realized = reference_greedy_walk(scores, cands)
    greedy_tokens, _ = upstream_walk(walk_speculator, cands, scores, 3, TOP_K, slots, False)
    check("B7. greedy walk 与 Torch 参考实现一致",
          torch.equal(ref_tokens, greedy_tokens))

    # K=1 / top-k=1 边界
    small, _ = make_speculator(k=1, draft_sample_method="probabilistic",
                               selector_top_k=1, max_num_seqs=2)
    cands1, scores1 = walk_case(batch=2, k=1, top_k=1, seed=2)
    ours_tokens, ours_realized = run_walk(small, cands1, scores1, 0.0, 1, 1, [1, 0])
    up_tokens, up_realized = upstream_walk(small, cands1, scores1, 1, 1, [1, 0], False)
    check("B8. K=1、top-k=1 边界与上游逐位相等",
          torch.equal(ours_tokens, up_tokens) and torch.equal(ours_realized, up_realized))

    # cache 两轮（disjoint）与上游逐位相等
    cache_spec, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    up_logits = torch.full_like(cache_spec.draft_logits, float("-inf"))
    up_cached = torch.zeros_like(cache_spec._cached_candidate_ids)
    same = True
    for rows in ([[0, 1, 2], [3, 4, 5]], [[2, 6, 7], [8, 9, 10]]):
        cands2 = torch.tensor([rows], dtype=torch.int64, device=DEVICE)
        set_walk_inputs(cache_spec, anchors=[3], temperature=0.0, seeds=5, slot_ids=[0])
        scores2 = torch.randn(1, 2, TOP_K, TOP_K, device=DEVICE)
        cache_spec._sample_path(cands2, scores2, 1)
        num_sample = 2
        cache_spec._cache_draft_logits(cands2, num_sample)
        _cache_draft_logits_kernel[(num_sample,)](
            up_logits, up_cached, cands2, cache_spec._selector_scores[:1],
            cache_spec.sample_idx_mapping, up_logits.stride(0), up_logits.stride(1),
            num_steps=2, top_k=TOP_K, BLOCK_K=4, num_warps=1)
        same = same and torch.equal(cache_spec.draft_logits, up_logits) \
            and torch.equal(cache_spec._cached_candidate_ids, up_cached)
    check("B9. cache 内核（两轮、候选集换掉）vs 上游逐位相等", same)
else:
    print("SKIP  B5~B9（没有 CUDA：walk/cache 是 Triton 内核）")

# ================================================ C realized q

print("== C realized q ==")
if CUDA:
    q_spec, _ = make_speculator(k=1, draft_sample_method="probabilistic")
    first = torch.tensor([[[0, 1, 2]]], dtype=torch.int64, device=DEVICE)
    set_walk_inputs(q_spec, anchors=[1], temperature=0.0, seeds=1, slot_ids=[0])
    q_spec._sample_path(first, torch.randn(1, 1, TOP_K, TOP_K, device=DEVICE), 1)
    q_spec._cache_draft_logits(first, 1)
    q_first = q_spec.draft_logits[0, 0].clone()
    second = torch.tensor([[[3, 4, 5]]], dtype=torch.int64, device=DEVICE)
    set_walk_inputs(q_spec, anchors=[1], temperature=0.0, seeds=2, slot_ids=[0])
    q_spec._sample_path(second, torch.randn(1, 1, TOP_K, TOP_K, device=DEVICE), 1)
    realized = q_spec._selector_scores[0, 0].clone()
    q_spec._cache_draft_logits(second, 1)
    q_second = q_spec.draft_logits[0, 0].clone()
    check("C1. q 初值 fp32/-inf，且只写 k 个候选列",
          q_first.dtype == torch.float32
          and int(torch.isfinite(q_first).sum()) == TOP_K
          and set(torch.nonzero(torch.isfinite(q_second)).flatten().tolist()) == {3, 4, 5})
    check("C2. 新候选列的值 == 实际路径的 realized scores",
          set(q_second[torch.isfinite(q_second)].tolist()) == set(realized.tolist()))
    check("C3. 新旧候选集不相交 → 旧列恢复 -inf",
          torch.all(q_second[:3] == float("-inf")))
    check("C4. softmax(q) 是合法分布",
          abs(torch.softmax(q_second, -1).sum().item() - 1.0) < 1e-6)

    third = torch.tensor([[[2, 4, 6]]], dtype=torch.int64, device=DEVICE)
    set_walk_inputs(q_spec, anchors=[1], temperature=0.0, seeds=3, slot_ids=[0])
    q_spec._sample_path(third, torch.randn(1, 1, TOP_K, TOP_K, device=DEVICE), 1)
    new_realized = q_spec._selector_scores[0, 0].clone()
    q_spec._cache_draft_logits(third, 1)
    q_third = q_spec.draft_logits[0, 0].clone()
    check("C5. 交集列（2/4）被更新为新分数、非交集旧列清空",
          q_third[0].item() == float("-inf") and q_third[2].item() != q_second[2].item()
          and set(q_third[torch.isfinite(q_third)].tolist()) == set(new_realized.tolist()))

    # padding 行：mapping=-1 的请求不写 token/分数
    pad_spec, _ = make_speculator(k=2, draft_sample_method="probabilistic")
    cands, scores = walk_case(batch=2, k=2, seed=4)
    set_walk_inputs(pad_spec, anchors=[1, 2], temperature=0.0, slot_ids=[0, -1])
    pad_spec.draft_tokens.fill_(777)
    pad_spec._selector_scores.fill_(123.0)
    pad_spec._sample_path(cands, scores, 2)
    check("C6. 补齐行（mapping=-1）不写 token / 不写分数",
          torch.all(pad_spec.draft_tokens[1] == 777)
          and torch.all(pad_spec._selector_scores[1] == 123.0))

    # 2000 种子经验分布
    import torch as _torch

    dist_spec, _ = make_speculator(k=1, draft_sample_method="probabilistic")
    one = torch.tensor([[[0, 5, 9]]], dtype=torch.int64, device=DEVICE)
    scores_one = _torch.tensor([1.0, 0.2, -0.7], device=DEVICE).view(1, 1, 1, 3).expand(
        1, 1, 3, 3).contiguous()
    counts = torch.zeros(TOP_K, dtype=torch.float64, device=DEVICE)
    for seed in range(2000):
        set_walk_inputs(dist_spec, anchors=[3], temperature=2.0, seeds=seed, slot_ids=[0])
        dist_spec._sample_path(one, scores_one, 1)
        counts[int((one[0, 0] == dist_spec.draft_tokens[0, 0]).nonzero()[0])] += 1
    dist_spec._cache_draft_logits(one, 1)
    expected = torch.softmax(dist_spec.draft_logits[0, 0] / 2.0, -1)[one[0, 0]]
    tv = 0.5 * (counts / counts.sum() - expected).abs().sum().item()
    check("C7. 2000 个种子的经验分布 vs softmax(q/temp)（TV<0.1）", tv < 0.1,
          f"TV={tv:.4f}")
else:
    print("SKIP  C1~C7（没有 CUDA）")


# ================================================ D 模型与端到端

print("== D 模型与端到端 ==")
from minivllm.model_loader import get_model  # noqa: E402

_, draft_config = model_configs(k=3, final_logit_softcapping=5.0, output_multiplier=1.7)
model = get_model(draft_config, "cpu")
model.eval()
layer = model.model.layers[0]


def stub_attention(positions, hidden_states):
    return hidden_states * 0.5 + 0.1


layer.self_attn.forward = stub_attention
try:
    x = torch.randn(8, model.model.hidden_size, generator=generator)
    positions = torch.arange(8)
    out, residual = layer(positions, x, None)
    ref_out, ref_residual = reference_decoder_layer(
        layer, lambda value, pos: stub_attention(pos, value), positions, x, None)
finally:
    del layer.self_attn.forward
layer_diff = max((out - ref_out).abs().max().item(),
                 (residual - ref_residual).abs().max().item())
check("D1. 解码器层（残差/两侧卷积/MLP）vs 参考公式", layer_diff < 1e-6,
      f"max|Δ|={layer_diff:.2e}")

ids, values = model.compute_candidates(
    torch.randn(5, model.model.hidden_size, generator=generator))
check("D2. compute_candidates 的形状与 top-k 来源",
      ids.shape == (5, 3) and values.dtype == torch.float32
      and model.model.candidate_selector.top_k == 3)

if CUDA:
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
    baseline_engine, _, _ = make_baseline_engine()
    baseline = run_prompts(baseline_engine, prompts, max_tokens=6, temperature=0.0)
    engine, _core, runner = make_engine(k=3, draft_sample_method="greedy")
    speculative = run_prompts(engine, prompts, max_tokens=6, temperature=0.0)
    check("D3. greedy 投机输出 == 不投机的 greedy 输出（无损）",
          speculative == baseline, f"{speculative}")

    rounds, finals = {}, {}
    for mode in ("none", "full_decode_only"):
        engine, _core, runner = make_engine(k=3, draft_sample_method="probabilistic",
                                            mode=mode, capture_sizes=(8, 16))
        try:
            for req_id, prompt in prompts:
                from minivllm import SamplingParams
                engine.add_request(req_id, list(prompt), SamplingParams(
                    max_tokens=6, temperature=0.7, seed=1234, eos_token_id=99))
            snapshot, final = [], {}
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
                snapshot.append(runner.req_states.draft_tokens[:2].clone())
        finally:
            engine.shutdown()
        rounds[mode] = snapshot
        finals[mode] = final
        if mode == "full_decode_only":
            stats = runner.cudagraph_capture_stats or {}
            selections = runner.speculator.query_cudagraph_manager.selections
    same_rounds = len(rounds["none"]) == len(rounds["full_decode_only"]) > 1 and all(
        torch.equal(a, b) for a, b in zip(rounds["none"], rounds["full_decode_only"]))
    check("D4. 图/eager 逐轮草稿 token 相同（候选+walk+落 q 都在图里）",
          same_rounds and finals["none"] == finals["full_decode_only"],
          f"{len(rounds['none'])} 轮")
    check("D5. 图确实捕获并重放过",
          stats.get("captured", 0) >= 1 and bool(selections),
          f"captured={stats.get('captured')}，selections={len(selections)}")

    outputs = {}
    for method in ("greedy", "probabilistic"):
        engine, _core, _runner = make_engine(k=3, draft_sample_method=method)
        outputs[method] = run_prompts(engine, prompts, max_tokens=6, temperature=0.0)
    check("D6. greedy 与 probabilistic（温度 0）输出相同",
          outputs["greedy"] == outputs["probabilistic"])

    baseline_engine, _, _ = make_baseline_engine(max_num_seqs=1)
    reuse_baseline = run_prompts(
        baseline_engine, [("r0", [1, 2]), ("r1", [3, 4]), ("r2", [5, 6])],
        max_tokens=5, temperature=0.0)
    engine, _core, _runner = make_engine(k=2, draft_sample_method="probabilistic",
                                        max_num_seqs=1)
    reuse_spec = run_prompts(
        engine, [("r0", [1, 2]), ("r1", [3, 4]), ("r2", [5, 6])],
        max_tokens=5, temperature=0.0)
    check("D7. slot 复用（请求数 > max_num_seqs）后仍与 greedy 对照相同",
          reuse_spec == reuse_baseline, f"{reuse_spec}")
else:
    print("SKIP  D3~D7（没有 CUDA）")


# ================================================ 汇总

print()
if FAILED:
    print(f"失败 {len(FAILED)} 项：" + "、".join(FAILED))
    raise SystemExit(1)
print(f"全部通过（{PASSED} 项）")
