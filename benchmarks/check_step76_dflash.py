"""76 关验收探针：DFlash 上下文 KV + 并行 query（跑法见文末）。

跑法：`python benchmarks/check_step76_dflash.py`（退出码非 0 = 失败）。

分段（对应需求 076 §3/§4）：

    A 配置边界     method=dflash 被接受；parallel_drafting 强制打开、槽位预算 = K；
                   sample_from_anchor / 混合滑窗 / 缺 mask token 全部配置期拒绝
    B 布局逐值     V1 展开与上游 `copy_and_expand_dflash_inputs_kernel` 逐值相等（6 个输出）；
                   V2 内核与上游 `_prepare_dflash_inputs_kernel` 逐值相等（12 个输出）；
                   补齐行惰性（映射 −1）；"漏减被拒行"的反证
    C 模型与 KV    融合上下文 KV == 逐层参考；query 前向 == 冻结参考（因果/非因果两种）；
                   因果性确实改变输出；显式槽位写入；d2t / midlayer / mask_hidden 拒绝
    D 端到端       V1 跑通；V1 与 V2 贪心逐 token 相同；V2 图与 eager 相同且确实捕获了图；
                   每轮 context 槽位数 = target 行数 − 被拒数（被拒尾部不进草稿缓存）；
                   probabilistic 草稿模式两条路径都能跑

⚠️ 本机**没有真实 DFlash 权重**（只有 Qwen3-1.7B 与它的 EAGLE3 草稿），所以这里验的是
"布局 / 数学 / 协议"，不是草稿质量与加速比（需求 076 §5 明确不许把"减少 target 调用"写成
"更快"，本关只做一次 context/query 的 trace 验证）。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step76"))

from spec76_helpers import (DEVICE, MASK_TOKEN_ID, fake_rows,  # noqa: E402
                            layout_case, make_v1_engine, make_v2_engine, model_configs,
                            run_our_v1_expansion, run_our_v2_kernel, run_prompts,
                            run_upstream_v1_kernel, run_upstream_v2_kernel,
                            tiny_dflash_model)

from minivllm.attention.backends.torch_sdpa import TorchAttentionBackend  # noqa: E402
from dataclasses import replace  # noqa: E402

from minivllm.config import ModelConfig, SpeculativeConfig  # noqa: E402
from minivllm.forward_context import set_forward_context  # noqa: E402
from minivllm.spec_decode.utils import (compute_slot_mapping_for_rows,  # noqa: E402
                                        expand_dflash_inputs)
from minivllm.testing.torch_dflash import reference_context_kv, reference_forward  # noqa: E402

PASSED, FAILED, TRACE = 0, [], []
CUDA = torch.cuda.is_available()


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")
    TRACE.append({"item": name, "ok": bool(ok), "detail": str(detail)[:300]})


def dflash_spec(**overrides):
    _, draft = model_configs()
    kwargs = dict(method="dflash", num_speculative_tokens=3, draft_model_config=draft)
    kwargs.update(overrides)
    return SpeculativeConfig(**kwargs)


# ================================================ A 配置

print("== A 配置边界 ==")
spec = dflash_spec()
check("A1. method='dflash' 被接受，且 parallel_drafting 被强制打开",
      spec.method == "dflash" and spec.parallel_drafting is True)
check("A2. 槽位预算 = K（1 个 bonus + K 个 mask 全在 target query 之外）",
      spec.max_num_new_slots_for_drafting == 3, spec.max_num_new_slots_for_drafting)
check("A3. use_eagle()/uses_dflash() 都认 dflash（KV 预留与特征采集依赖它们）",
      spec.use_eagle() and spec.uses_dflash())
check("A4. 辅助层取值链：eagle_aux_hidden_state_layer_ids 优先",
      spec.eagle_aux_hidden_state_layers() == (0, 1), spec.eagle_aux_hidden_state_layers())

_, draft_config = model_configs()
hf = dict(draft_config.hf_config)
hf["dflash_config"] = {**hf["dflash_config"], "sample_from_anchor": True}
try:
    dflash_spec(draft_model_config=ModelConfig(
        model=draft_config.model, dtype="float32", max_model_len=64, hf_config=hf))
    anchor_ok = False
except NotImplementedError:
    anchor_ok = True
check("A5. sample_from_anchor=True（DSpark 形状）配置期拒绝", anchor_ok)

hf = dict(draft_config.hf_config)
hf["dflash_config"] = {**hf["dflash_config"], "use_swa": True}
try:
    dflash_spec(draft_model_config=ModelConfig(
        model=draft_config.model, dtype="float32", max_model_len=64, hf_config=hf))
    swa_ok = False
except NotImplementedError:
    swa_ok = True
check("A6. 混合滑窗（需要多 KV group）配置期拒绝", swa_ok)

hf = dict(draft_config.hf_config)
hf["dflash_config"] = {k: v for k, v in hf["dflash_config"].items() if k != "mask_token_id"}
try:
    dflash_spec(draft_model_config=ModelConfig(
        model=draft_config.model, dtype="float32", max_model_len=64, hf_config=hf))
    mask_ok = False
except ValueError:
    mask_ok = True
check("A7. 缺 mask token 配置期拒绝（不猜占位符）", mask_ok)

# ================================================ B 布局

print()
print("== B 布局逐值 ==")
case = layout_case("cuda" if CUDA else "cpu")
rows = fake_rows(case)
expanded = expand_dflash_inputs(
    [int(v) for v in case.target_positions.tolist()],
    rows, num_query_per_req=case.num_query_per_req, mask_token_id=MASK_TOKEN_ID)
check("B1. CPU 规则：位置/槽位/采样行/seq_lens",
      expanded.positions == [9, 10, 11, 12, 22, 23, 24, 25]
      and expanded.seq_lens == [13, 26]
      and expanded.token_indices_to_sample == [1, 2, 3, 5, 6, 7]
      and expanded.context_positions == [7, 8, 9, 10, 20, 21])

positions = torch.tensor([7, 9, 63, 64], dtype=torch.int64, device=case.target_positions.device)
req = torch.zeros(4, dtype=torch.int64, device=positions.device)
slots = compute_slot_mapping_for_rows(positions, req, case.block_table, 4, 64)
check("B2. 越界位置打哨兵（不写缓存）", slots.tolist() == [7, 9, 63, -1], slots.tolist())

# 反证：漏减被拒行 → 位置整体后移
wrong = expand_dflash_inputs(
    [int(v) for v in case.target_positions.tolist()],
    [replace(rows[0], num_rejected=0), rows[1]],
    num_query_per_req=case.num_query_per_req, mask_token_id=MASK_TOKEN_ID)
check("B3. 反证：漏减被拒行时 query 位置整体后移（9→11）",
      wrong.positions[:4] == [11, 12, 13, 14] and wrong.seq_lens[0] == 15)

if CUDA:
    up_v1 = run_upstream_v1_kernel(layout_case("cuda"))
    my_v1 = run_our_v1_expansion(layout_case("cuda"))
    same_v1 = all(torch.equal(up_v1[name], my_v1[name]) for name in up_v1)
    check("B4. V1 展开 vs 上游 kernel（6 个输出逐值相等）", same_v1,
          {name: bool(torch.equal(up_v1[name], my_v1[name])) for name in up_v1})

    up_v2 = run_upstream_v2_kernel(layout_case("cuda"))
    my_v2 = run_our_v2_kernel(layout_case("cuda"))
    same_v2 = all(torch.equal(up_v2[name], my_v2[name]) for name in up_v2)
    check("B5. V2 内核 vs 上游 kernel（12 个输出逐值相等）", same_v2,
          {name: bool(torch.equal(up_v2[name], my_v2[name])) for name in up_v2})
    check("B6. V2 被拒尾部：position 0 / slot −1（不写草稿缓存）",
          up_v2["context_positions"][:6].tolist() == [7, 8, 0, 0, 20, 21]
          and up_v2["context_slot_mapping"][:6].tolist() == [7, 8, -1, -1, 84, 85])
    check("B7. 补齐行惰性：sample_idx_mapping=−1、slot=−1、seq_lens=0",
          up_v2["sample_idx_mapping"][6:].tolist() == [-1] * 18
          and up_v2["query_slot_mapping"][8:].tolist() == [-1] * 56
          and up_v2["seq_lens"][2:].tolist() == [0] * 6)
else:
    print("SKIP  B4~B7（没有 CUDA：Triton 内核不可执行）")

# ================================================ C 模型与 KV

print()
print("== C 模型与 KV ==")
model, hf = tiny_dflash_model()
names = model.get_draft_kv_cache_layer_names()
num_ctx, num_query = 5, 4
context = torch.randn(num_ctx, hf["hidden_size"], device=DEVICE)
context_positions = torch.arange(0, num_ctx, device=DEVICE)
query_ids = torch.tensor([3, hf["dflash_config"]["mask_token_id"],
                          hf["dflash_config"]["mask_token_id"], 5], device=DEVICE)
query_positions = torch.arange(num_ctx, num_ctx + num_query, device=DEVICE)
slots = torch.arange(num_ctx, device=DEVICE)
model.precompute_and_store_context_kv(context, context_positions, slots)
modules = dict(model.named_modules())
reference = reference_context_kv(model, context, context_positions)
kv_ok = True
for index, name in enumerate(names):
    module = modules[name]
    got_k = module.kv_cache[0].reshape(-1, module.num_kv_heads, module.head_size)[slots]
    want_k, _ = reference[index]
    kv_ok = kv_ok and bool(torch.allclose(got_k, want_k, atol=1e-6))
check("C1. 融合上下文 KV == 逐层参考（K 逐值）", kv_ok)

builder = TorchAttentionBackend.get_builder_cls()(4)


def run_query(causal: bool):
    model.precompute_and_store_context_kv(context, context_positions, slots)
    metadata = builder.build(
        query_start_loc=torch.tensor([0, num_query], device=DEVICE),
        seq_lens=torch.tensor([num_ctx + num_query], device=DEVICE),
        block_table=torch.arange(8, device=DEVICE).view(1, -1),
        slot_mapping=torch.arange(num_ctx, num_ctx + num_query, device=DEVICE),
        num_reqs=1, causal=causal)
    with set_forward_context({name: metadata for name in names}, num_tokens=num_query):
        return model(query_ids, query_positions)


out_causal = run_query(True)
out_free = run_query(False)
ref_causal = reference_forward(model, context, context_positions, query_ids, query_positions,
                               causal=True)
ref_free = reference_forward(model, context, context_positions, query_ids, query_positions,
                             causal=False)
check("C2. query 前向 == 冻结参考（因果 / 非因果两种）",
      float((out_causal - ref_causal).abs().max()) < 1e-6
      and float((out_free - ref_free).abs().max()) < 1e-6,
      f"causal {float((out_causal - ref_causal).abs().max()):.2e} / "
      f"free {float((out_free - ref_free).abs().max()):.2e}")
check("C3. 因果开关确实改变输出（不是两边都没跑对）",
      float((out_causal - out_free).abs().max()) > 1e-3,
      float((out_causal - out_free).abs().max()))

reduced, reduced_hf = tiny_dflash_model(draft_vocab_size=6)
logits = reduced.compute_logits(torch.randn(3, reduced_hf["hidden_size"], device=DEVICE))
check("C4. 缩小词表：scatter 回 target 宽度，交集外 −inf",
      logits.shape == (3, reduced_hf["vocab_size"]) and torch.isinf(logits[:, 6:]).all()
      and logits[:, 6:].max() < 0)
midlayer, _ = tiny_dflash_model(naming="midlayer", num_layers=1)
check("C5. 历史命名 midlayer.* 能加载（映射到 layers.0.*）",
      midlayer.model.layers[0].self_attn.qkv_proj.weight.isfinite().all())
try:
    model.load_weights(iter([("mask_hidden", torch.zeros(1, hf["hidden_size"]))]))
    mask_hidden_ok = False
except ValueError:
    mask_hidden_ok = True
check("C6. 权重里带 mask_hidden（P-EAGLE 形状）加载期拒绝", mask_hidden_ok)

# ================================================ D 端到端

print()
print("== D 端到端 ==")
prompts = [("a", [1, 2, 3, 4]), ("b", [5, 6])]
v1_engine, _, _ = make_v1_engine()
v1_out = run_prompts(v1_engine, prompts, max_tokens=8)
check("D1. V1 跑通（每请求 8 个 token）",
      all(len(tokens) == 8 for tokens in v1_out.values()), v1_out)

if CUDA:
    v2_engine, _, v2_runner = make_v2_engine()
    v2_out = run_prompts(v2_engine, prompts, max_tokens=8)
    check("D2. V1 与 V2 贪心逐 token 相同", v1_out == v2_out, v2_out)

    eager_engine, _, _ = make_v2_engine(mode="none")
    eager_out = run_prompts(eager_engine, [("a", [1, 2, 3, 4])], max_tokens=8)
    graph_engine, _, graph_runner = make_v2_engine(capture_sizes=[8, 16])
    graph_out = run_prompts(graph_engine, [("a", [1, 2, 3, 4])], max_tokens=8)
    stats = graph_runner.cudagraph_capture_stats or {}
    selections = graph_runner.speculator.query_cudagraph_manager.selections
    check("D3. V2 图与 eager 逐 token 相同，且确实捕获/重放了图",
          eager_out == graph_out and stats.get("captured", 0) >= 1
          and any(entry["replayed"] for entry in selections),
          f"captured={stats.get('captured')} selections={len(selections)}")

    # 被拒尾部不进草稿缓存：记录每轮的 num_scheduled / num_rejected / context 槽位
    from minivllm.worker.gpu.spec_decode.dflash import speculator as module

    engine2, _, runner2 = make_v2_engine()
    speculator2 = runner2.speculator
    real_prepare = module.prepare_dflash_inputs
    recorded = []

    def spy_prepare(*args, **kwargs):
        input_batch, num_rejected = args[9], args[11]
        result = real_prepare(*args, **kwargs)
        recorded.append((input_batch.num_scheduled_tokens.copy(),
                         num_rejected.detach().cpu().numpy().copy(),
                         speculator2._context_slot_mapping.clone()))
        return result

    module.prepare_dflash_inputs = spy_prepare
    try:
        run_prompts(engine2, [("c", [1, 2, 3, 4]), ("d", [5, 6])], max_tokens=10)
    finally:
        module.prepare_dflash_inputs = real_prepare
    tail_ok, saw_reject = True, False
    for scheduled, rejected, context_slots in recorded:
        rows_total = int(scheduled.sum())
        if int((context_slots[:rows_total] == -1).sum()) != int(rejected.sum()):
            tail_ok = False
            break
        cursor = 0
        for length, count in zip(scheduled, rejected):
            row_slots = context_slots[cursor:cursor + length]
            if count:
                saw_reject = True
                if not (row_slots[length - count:] == -1).all():
                    tail_ok = False
                if not (row_slots[:length - count] != -1).all():
                    tail_ok = False
            cursor += length
    check("D4. 每轮 context 槽位数 = target 行数 − 被拒数（被拒尾部不写草稿缓存）",
          tail_ok and saw_reject, f"轮数 {len(recorded)}，出现被拒 {saw_reject}")

    prob_engine, _, _ = make_v2_engine(draft_sample_method="probabilistic")
    prob_out = run_prompts(prob_engine, [("p", [1, 2, 3, 4])], max_tokens=4, seed=7)
    check("D5. probabilistic 草稿模式（V2）能跑", all(len(t) == 4 for t in prob_out.values()),
          prob_out)
else:
    print("SKIP  D2~D5（没有 CUDA：V2 需要 UVA / Triton）")

# V1 的 probabilistic 模式与 K=0（动态投机长度）都要能跑
prob_v1, _, _ = make_v1_engine(draft_sample_method="probabilistic")
out = run_prompts(prob_v1, [("p", [1, 2, 3, 4])], max_tokens=4, seed=7)
check("D5b. probabilistic 草稿模式（V1）能跑", all(len(t) == 4 for t in out.values()), out)

check("D6. K=0 的展开也能构造（1 行 query、0 个采样行）",
      expand_dflash_inputs([7], [replace(rows[0], target_rows=1, num_rejected=0)],
                           num_query_per_req=1, mask_token_id=MASK_TOKEN_ID
                           ).token_indices_to_sample == [])

# ================================================ 汇总

print()
if FAILED:
    print(f"失败 {len(FAILED)} 项：" + "、".join(FAILED))
    raise SystemExit(1)
print(f"全部通过（{PASSED} 项）")
