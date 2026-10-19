"""77 关验收探针：DSpark 条件采样与实际 proposal 分布（跑法见文末）。

跑法：`python benchmarks/check_step77_dspark.py`（退出码非 0 = 失败）。

分段（对应需求 077 §2/§3/§4）：

    A 配置边界     method=dspark 被接受、parallel_drafting 强制打开、槽位预算 = K−1；
                   两种 anchor 布局都被接受；非 Qwen3 架构 / 非法 top-k / 自适应验证 /
                   PP / V1 通路全部明确拒绝
    B 布局逐值     两种布局各与上游 `_prepare_dflash_inputs_kernel` 逐值相等（12 个输出）；
                   两布局的草稿位置相同；K=1 边界
    C 条件采样     q == 沿实际路径复算的参考；改前一个 token → q 变；缩减词表（非 d2t 列 -inf、
                   偏移生效、无残留）；top-k 截断；2000 个种子的经验分布 vs softmax(q)；
                   confidence 落盘；greedy 是点质量
    D 端到端       两种布局都能跑；图/eager 逐轮草稿相同（且确实捕获了图）；
                   greedy vs probabilistic（温度 0）同输出；缩减词表 + top-k 组合能跑

⚠️ 本机**没有真实 DSpark 权重**（只有 tiny 模型），所以这里验的是"协议 / 数学 / 分布"，
不是草稿质量与加速比（需求 077 §5：预算决策属 79 关，本关不设阈值）。
"""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step76"))
sys.path.insert(0, str(ROOT / "tests" / "step77"))

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from spec76_helpers import run_our_v2_kernel, run_upstream_v2_kernel  # noqa: E402

from dspark_helpers import (DEVICE, fake_pp_config, make_engine,  # noqa: E402
                            make_spec_config, make_speculator, run_prompts,
                            set_sampling, tiny_pair)
from minivllm import ModelConfig, SamplingParams, SpeculativeConfig  # noqa: E402
from minivllm.config import VllmConfig  # noqa: E402
from minivllm.testing.torch_dspark import reference_sequential  # noqa: E402
from minivllm.worker.gpu.spec_decode.dspark.utils import load_dspark_model  # noqa: E402

PASSED, FAILED, TRACE = 0, [], []
CUDA = torch.cuda.is_available()
VOCAB = 11


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


def spec(**overrides):
    kwargs = dict(k=3)
    kwargs.update(overrides)
    return make_spec_config(**kwargs)


# ================================================ A 配置

print("== A 配置边界 ==")
dspark = spec()
check("A1. method='dspark' 被接受，parallel_drafting 被强制打开",
      dspark.method == "dspark" and dspark.parallel_drafting is True)
check("A2. 槽位预算 = K−1（anchor 那一格复用调度器已给的 query 槽位）",
      dspark.max_num_new_slots_for_drafting == 2, dspark.max_num_new_slots_for_drafting)
check("A3. use_eagle()/aux 特征采集都认 dspark",
      dspark.use_eagle() and dspark.eagle3_use_aux_hidden_state())
check("A4. 两种 anchor 布局都能过配置期校验",
      spec(anchor=True).draft_model_config.hf_config["sample_from_anchor"] is True
      and spec(anchor=False).draft_model_config.hf_config["sample_from_anchor"] is False)

_, _, base_draft_dir, base_draft = tiny_pair()
hf = dict(base_draft)
hf["architectures"] = ["Gemma4DSparkModel"]
try:
    SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                      draft_model_config=ModelConfig(
                          model=base_draft_dir, dtype="float32", max_model_len=64,
                          hf_config=hf))
    arch_ok = False
except NotImplementedError:
    arch_ok = True
check("A5. 非 Qwen3 架构（Gemma4/K3/DSV4）配置期拒绝", arch_ok)

bad_topk = 0
for value in (0, VOCAB + 1):
    try:
        spec(draft_topk=value)
    except ValueError:
        bad_topk += 1
topk_spec = spec(draft_topk=3)
check("A6. 非法 top-k 拒绝、合法值写进 draft 配置",
      bad_topk == 2 and topk_spec.draft_model_config.hf_config["dspark_draft_topk"] == 3)

_, _, conf_dir, conf_hf = tiny_pair(confidence_head=True)
conf_draft = ModelConfig(model=conf_dir, dtype="float32", max_model_len=64,
                         hf_config=conf_hf)
try:
    SpeculativeConfig(method="dspark", num_speculative_tokens=3,
                      draft_model_config=conf_draft, enable_adaptive_verification=True)
    adaptive_ok = False
except NotImplementedError:
    adaptive_ok = True
check("A7. enable_adaptive_verification（预算属 79 关）配置期拒绝", adaptive_ok)

try:
    load_dspark_model(None, fake_pp_config(dspark, pp_size=2))
    pp_ok = False
except NotImplementedError:
    pp_ok = True
check("A8. PP>1 拒绝（上游同一条 NotImplementedError）", pp_ok)

_target_dir, _target_hf, _, _ = tiny_pair()
target_mc = ModelConfig(model=_target_dir, dtype="float32", max_model_len=64,
                        hf_config=_target_hf)
old = os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
try:
    try:
        VllmConfig(model_config=target_mc, speculative_config=dspark)
        v1_ok = False
    except NotImplementedError:
        v1_ok = True
finally:
    if old is not None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old
check("A9. DSpark 只有 V2：不设 VLLM_USE_V2_MODEL_RUNNER 时配置期拒绝", v1_ok)


# ================================================ B 布局逐值

print()
print("== B 布局逐值 ==")
if CUDA:
    from spec76_helpers import layout_case

    def dspark_case(anchor, k=3):
        case = layout_case(DEVICE)
        case.num_speculative_steps = k
        case.num_query_per_req = k if anchor else 1 + k
        return case

    for anchor in (True, False):
        case = dspark_case(anchor)
        upstream = run_upstream_v2_kernel(case, sample_from_anchor=anchor)
        ours = run_our_v2_kernel(case, sample_from_anchor=anchor)
        same = all(torch.equal(upstream[name], ours[name]) and
                   upstream[name].dtype == ours[name].dtype for name in upstream)
        check(f"B{1 if anchor else 2}. {'anchor-as-first' if anchor else 'fill-in'} 布局"
              f"与上游内核 12/12 逐值相等", same,
              f"query_start_loc={upstream['query_start_loc'][:3].tolist()}")

    a = run_our_v2_kernel(dspark_case(True), sample_from_anchor=True)
    f = run_our_v2_kernel(dspark_case(False), sample_from_anchor=False)
    same_pos = a["sample_pos"][:6].tolist() == f["sample_pos"][:6].tolist()
    check("B3. 两种布局的草稿位置相同（只有 query 行怎么排不同）", same_pos,
          a["sample_pos"][:3].tolist())

    k1_a = run_upstream_v2_kernel(dspark_case(True, k=1), sample_from_anchor=True)
    k1_f = run_upstream_v2_kernel(dspark_case(False, k=1), sample_from_anchor=False)
    ours_a = run_our_v2_kernel(dspark_case(True, k=1), sample_from_anchor=True)
    ours_f = run_our_v2_kernel(dspark_case(False, k=1), sample_from_anchor=False)
    check("B4. K=1 边界两种布局都与上游逐值相等",
          all(torch.equal(k1_a[n], ours_a[n]) for n in k1_a)
          and all(torch.equal(k1_f[n], ours_f[n]) for n in k1_f),
          f"anchor seq_lens={k1_a['seq_lens'][:2].tolist()}，"
          f"fill-in seq_lens={k1_f['seq_lens'][:2].tolist()}")
else:
    print("SKIP  B1~B4（没有 CUDA：布局内核是 Triton）")


# ================================================ C 条件采样

print()
print("== C 条件采样与 q ==")


def markov_weights(speculator):
    head = speculator.model.model.markov_head
    return head.markov_w1.weight, head.markov_w2.weight


def draft_ids_of(model, target_ids):
    d2t = model.draft_id_to_target_id
    if d2t is None:
        return target_ids
    draft = torch.arange(d2t.shape[0], device=d2t.device)
    table = draft + d2t
    return (target_ids.unsqueeze(-1) == table.unsqueeze(0)).to(
        torch.int64).argmax(dim=-1)


if CUDA:
    # C1：q == 沿实际路径的参考（含 logit_scale）
    speculator, _ = make_speculator(k=3, logit_scale=1.7)
    n = set_sampling(speculator, [1, 4], temperature=0.7, seeds=7)
    hidden = torch.randn(n, 32, device=DEVICE)
    base = speculator.model.compute_draft_logits(hidden).view(2, 3, -1)
    speculator._sample_sequential(2, hidden)
    tokens, q = speculator.draft_tokens[:2].clone(), speculator.draft_logits[:2].clone()
    w1, w2 = markov_weights(speculator)
    ref_tokens, ref_q = reference_sequential(
        base, torch.tensor([1, 4], device=DEVICE), w1, w2,
        choose=lambda logits, step: draft_ids_of(speculator.model, tokens[:, step]),
        logit_scale=speculator.model.logits_processor.scale)
    check("C1. 缓存的 q == 沿实际采样路径复算的条件分布（含 logit_scale）",
          torch.allclose(ref_q, q, atol=1e-6) and not torch.allclose(q, base),
          f"max|Δ|={float((ref_q - q).abs().max()):.2e}")
    check("C1b. 参考路径与生产路径的草稿 token 相同", torch.equal(ref_tokens, tokens),
          tokens.tolist())

    # C2：同一份 base logits，改"前一个 token" → q 变
    speculator2, _ = make_speculator(k=3, logit_scale=1.0)
    head = speculator2.model.model.markov_head
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w2.weight.zero_()
        head.markov_w1.weight[1] = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
        head.markov_w1.weight[2] = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
        head.markov_w2.weight[0] = torch.tensor([3.0, 0.0, 0.0], device=DEVICE)
        head.markov_w2.weight[1] = torch.tensor([0.0, 3.0, 0.0], device=DEVICE)
    base2 = torch.zeros(1, 3, VOCAB, device=DEVICE)
    base2[0, 1, :2] = torch.tensor([2.0, 1.0], device=DEVICE)
    qs, paths = {}, {}
    flat = base2.reshape(-1, VOCAB)
    original = speculator2.model.compute_draft_logits
    speculator2.model.compute_draft_logits = lambda hidden: flat
    for name, anchor in (("A", 2), ("B", 1)):
        n2 = set_sampling(speculator2, [anchor], temperature=0.0)
        speculator2._sample_sequential(1, torch.zeros(n2, 32, device=DEVICE))
        qs[name] = speculator2.draft_logits[0].clone()
        paths[name] = speculator2.draft_tokens[0].clone()
    speculator2.model.compute_draft_logits = original
    check("C2. 同一份 base logits、前一 token 不同 → 后一位置 q 不同，且选择跟着变",
          not torch.allclose(qs["A"][1], qs["B"][1])
          and paths["A"][1].item() == 0 and paths["B"][1].item() == 1,
          f"qA1={qs['A'][1][:2].tolist()} qB1={qs['B'][1][:2].tolist()}")

    # C3：缩减词表（非零偏移）
    reduced, _ = make_speculator(k=3, draft_vocab_size=7, d2t_offset=2)
    n3 = set_sampling(reduced, [1, 4], temperature=0.0)
    hidden3 = torch.randn(n3, 32, device=DEVICE)
    reduced._sample_sequential(2, hidden3)
    q3 = reduced.draft_logits[:2].clone()
    forbidden = [0, 1, 9, 10]
    probs = torch.softmax(q3, dim=-1)
    check("C3. 缩减词表：非 d2t 列恒 -inf、偏移生效、q 是完整 target 分布",
          all(torch.isinf(q3[:, :, c]).all() and (q3[:, :, c] < 0).all() for c in forbidden)
          and torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6)
          and set(reduced.draft_tokens[:2].flatten().tolist()) <= {2, 3, 4, 5, 6, 7, 8},
          f"d2t={reduced.model.draft_id_to_target_id.tolist()[:3]}…")

    # C4：重复 batch 不残留
    hidden4 = torch.randn(n3, 32, device=DEVICE)
    reduced._sample_sequential(2, hidden4)
    q4 = reduced.draft_logits[:2].clone()
    w1r, w2r = markov_weights(reduced)
    _, ref_q4 = reference_sequential(
        reduced.model.compute_draft_logits(hidden4).view(2, 3, -1),
        torch.tensor([1, 4], device=DEVICE), w1r, w2r,
        choose=lambda logits, step: draft_ids_of(reduced.model,
                                                 reduced.draft_tokens[:2][:, step]),
        logit_scale=reduced.model.logits_processor.scale,
        d2t=reduced.model.draft_id_to_target_id, target_vocab=VOCAB)
    check("C4. 同一 slot 换一批输入：q 整份被覆盖（不残留上一批）",
          not torch.allclose(q3, q4) and torch.allclose(ref_q4, q4, atol=1e-6))

    # C5：top-k 截断
    topk_speculator, _ = make_speculator(k=3, draft_topk=3)
    n5 = set_sampling(topk_speculator, [1, 4], temperature=0.0)
    hidden5 = torch.randn(n5, 32, device=DEVICE)
    base5 = topk_speculator.model.compute_draft_logits(hidden5).view(2, 3, -1)
    topk_speculator._sample_sequential(2, hidden5)
    q5 = topk_speculator.draft_logits[:2].clone()
    finite = torch.isfinite(q5).sum(-1)
    cand = base5.topk(3, dim=-1).indices
    inside = all(topk_speculator.draft_tokens[r, s].item() in cand[r, s].tolist()
                 for r in range(2) for s in range(3))
    check("C5. top-k：每步只有 k 个有限列，采样 id 一定在候选里",
          finite.tolist() == [[3, 3, 3], [3, 3, 3]] and inside, finite.tolist())

    # C6：采样确实从 q 抽（2000 个种子）
    sampler, _ = make_speculator(k=3)
    logits6 = torch.randn(1, VOCAB, device=DEVICE) * 2.0
    expected = torch.softmax(logits6[0], dim=-1).double()
    counts = torch.zeros(VOCAB, dtype=torch.float64, device=DEVICE)
    draws = 2000
    idx_map = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    pos = torch.tensor([10], dtype=torch.int64, device=DEVICE)
    for seed in range(draws):
        sampler.temperature.fill_(1.0)
        sampler.seeds.fill_(seed)
        counts[sampler._sample_logits(logits6, idx_map, pos, 0).item()] += 1
    tv = float(0.5 * (counts / draws - expected).abs().sum())
    check("C6. 2000 个种子的经验分布 ≈ softmax(q)（TV < 0.1）", tv < 0.1, f"TV={tv:.4f}")

    # C7：confidence 落盘
    conf_spec, _ = make_speculator(k=3, confidence_head=True)
    conf_spec.enable_adaptive_verification = True
    n7 = set_sampling(conf_spec, [1, 4], temperature=0.0)
    hidden7 = torch.randn(n7, 32, device=DEVICE)
    conf_spec._sample_sequential(2, hidden7)
    probs7 = conf_spec.draft_token_confidence_probs[:2]
    head7 = conf_spec.model.model.confidence_head
    w1c, _ = markov_weights(conf_spec)
    prev = torch.cat([torch.tensor([1, 4], device=DEVICE).unsqueeze(1),
                      conf_spec.draft_tokens[:2, :-1]], dim=1)
    x = torch.cat([hidden7.view(2, 3, -1), w1c[prev]], dim=-1).reshape(
        -1, head7.proj.in_features)
    want7 = torch.sigmoid(torch.nn.functional.linear(
        x.float(), head7.proj.weight, head7.proj.bias)).view(2, 3)
    check("C7. confidence 每位置落盘 == sigmoid(proj([hidden, markov_embed]))",
          torch.allclose(probs7, want7, atol=1e-6),
          f"{probs7.flatten()[:3].tolist()}")

    # C8：greedy 是点质量（q 不缓存）
    greedy_spec, _ = make_speculator(k=3, draft_sample_method="greedy")
    n8 = set_sampling(greedy_spec, [1, 4], temperature=0.0)
    hidden8 = torch.randn(n8, 32, device=DEVICE)
    greedy_spec._sample_sequential(2, hidden8)
    ref_hidden = greedy_spec.model.compute_draft_logits(hidden8).view(2, 3, -1)
    w1g, w2g = markov_weights(greedy_spec)
    _, ref_q8 = reference_sequential(
        ref_hidden, torch.tensor([1, 4], device=DEVICE), w1g, w2g,
        choose=lambda logits, step: logits.argmax(dim=-1),
        logit_scale=greedy_spec.model.logits_processor.scale)
    check("C8. greedy：q 为点质量（draft_logits=None）且 token == 参考 argmax",
          greedy_spec.draft_logits is None
          and torch.equal(ref_q8.argmax(-1), greedy_spec.draft_tokens[:2]),
          greedy_spec.draft_tokens[:2].tolist())
else:
    print("SKIP  C1~C8（没有 CUDA：Gumbel 采样是 Triton 内核）")


# ================================================ D 端到端

print()
print("== D 端到端 ==")
if CUDA:
    prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
    outputs = {}
    for anchor in (True, False):
        engine, _core, _runner = make_engine(k=3, draft_sample_method="greedy", anchor=anchor)
        outputs[anchor] = run_prompts(engine, prompts, max_tokens=6)
    check("D1. 两种布局都能跑通（贪心输出一致）",
          all(len(t) == 6 for t in outputs[True].values())
          and outputs[True] == outputs[False], outputs[True])

    rounds, finals = {}, {}
    for mode in ("none", "full_decode_only"):
        engine, _core, runner = make_engine(k=3, draft_sample_method="probabilistic",
                                           mode=mode, capture_sizes=[6, 12])
        try:
            for req_id, prompt in prompts:
                engine.add_request(req_id, list(prompt),
                                   SamplingParams(max_tokens=6, temperature=0.7,
                                                  seed=1234, eos_token_id=99))
            snapshot, final = [], {}
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                for out in engine.step():
                    final[out.request_id] = list(out.token_ids)
                snapshot.append(runner.req_states.draft_tokens[:2].clone())
        finally:
            engine.shutdown()
        rounds[mode], finals[mode] = snapshot, final
        if mode == "full_decode_only":
            stats = runner.cudagraph_capture_stats or {}
            selections = runner.speculator.query_cudagraph_manager.selections
    same_rounds = len(rounds["none"]) == len(rounds["full_decode_only"]) > 1 and all(
        torch.equal(a, b) for a, b in zip(rounds["none"], rounds["full_decode_only"]))
    check("D2. 图/eager 逐轮草稿 token 相同（串行采样也在图里）",
          same_rounds and finals["none"] == finals["full_decode_only"],
          f"{len(rounds['none'])} 轮")
    check("D2b. 图确实捕获并重放过",
          stats.get("captured", 0) >= 1 and bool(selections),
          f"captured={stats.get('captured')}，selections={len(selections)}")

    out = {}
    for method in ("greedy", "probabilistic"):
        engine, _core, runner = make_engine(k=3, draft_sample_method=method)
        out[method] = run_prompts(engine, prompts, max_tokens=6, temperature=0.0)
    check("D3. greedy 与 probabilistic（温度 0）输出相同", out["greedy"] == out["probabilistic"])

    engine, _core, _runner = make_engine(k=3, draft_sample_method="probabilistic",
                                        draft_vocab_size=7, d2t_offset=2, draft_topk=3)
    mixed = run_prompts(engine, prompts, max_tokens=6)
    check("D4. 缩减词表 + top-k + 概率草稿的组合能跑",
          all(len(t) == 6 for t in mixed.values()), mixed)

    live_rows = []
    engine, _core, runner = make_engine(k=3, draft_sample_method="probabilistic")
    try:
        for req_id, prompt in prompts:
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=99))
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            for _out in engine.step():
                pass
            mapping = runner.speculator.sample_idx_mapping
            live_rows.append(int((mapping >= 0).sum()))
    finally:
        engine.shutdown()
    total = runner.speculator.sample_idx_mapping.numel()
    check("D5. 采样映射按 max_num_reqs×K 铺满，每轮只点亮 num_reqs×K 行（其余惰性 −1）",
          total == runner.max_num_reqs * 3 and set(live_rows) <= {3, 6} and 6 in live_rows,
          f"buffer={total}，每轮有效行={sorted(set(live_rows))}")
else:
    print("SKIP  D1~D5（没有 CUDA：V2 需要 UVA / Triton）")


# ================================================ 汇总

print()
if FAILED:
    print(f"失败 {len(FAILED)} 项：" + "、".join(FAILED))
    raise SystemExit(1)
print(f"全部通过（{PASSED} 项）")
