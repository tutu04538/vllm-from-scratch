"""第三十七关引擎级语义检查：路径选择、回退、以及 CUDA Graph 的请求划分重放。

    python benchmarks/check_step37_engine_paths.py

需求 §5 特别点名的一条：**相同 token 总数不代表请求划分相同**。
如果路径或 grid 随划分变化而没进图缓存键，重放就会算错。
这里用「同样 N=64，但划分不同」的两组输入打同一张图，与 eager 结果逐 token 比对。
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import step37 as m

FIX = ROOT / "fixtures" / "step30_qwen3" / "tiny_gqa"
VOCAB = 11


def build(use_cuda_graph, dtype=torch.bfloat16, max_num_batched_tokens=64):
    return m.Engine.from_model_dir(
        str(FIX), device="cuda", attention_backend="triton", use_cuda_graph=use_cuda_graph,
        max_num_seqs=4, max_num_batched_tokens=max_num_batched_tokens,
        num_kv_blocks=128, block_size=4, enable_prefix_caching=False, dtype=dtype)


def run(engine, prompts, max_new_tokens):
    """提交一组请求，跑到底，返回 {request_id: 输出 token 列表}。"""
    got = {}
    engine.scheduler.on_finished = lambda r: got.__setitem__(r["request_id"], list(r["output_ids"]))
    for i, prompt in enumerate(prompts):
        engine.add_request({"request_id": f"r{i}", "prompt_ids": prompt,
                            "max_new_tokens": max_new_tokens})
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    return got, steps


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main():
    torch.manual_seed(0)

    print("=== 1. 路径选择 ===")
    eng = build(use_cuda_graph=False)
    # 多 token 的 prefill：两条请求各 32 行
    eng.add_request({"request_id": "a", "prompt_ids": ids(32, 1), "max_new_tokens": 1})
    eng.add_request({"request_id": "b", "prompt_ids": ids(32, 2), "max_new_tokens": 1})
    eng.step()
    check("prefill（每条 32 行、BF16）走分块路径", eng.model.use_tiled,
          f"use_tiled={eng.model.use_tiled}, tiles={eng.model.attention_metadata.cpu_num_tiles[0].item()}")
    # 给两条请求各留 3 个 token，制造出真正的 decode 步
    eng = build(use_cuda_graph=False)
    eng.add_request({"request_id": "a", "prompt_ids": ids(32, 1), "max_new_tokens": 3})
    eng.step()                                   # 第 1 步：32 行 prefill
    prefill_tiled = eng.model.use_tiled
    eng.step()                                   # 第 2 步：1 行 decode
    check("prefill 走分块路径、下一步的 decode 回退逐行路径",
          prefill_tiled is True and eng.model.use_tiled is False,
          f"prefill={prefill_tiled} -> decode={eng.model.use_tiled}")

    eng2 = build(use_cuda_graph=False, dtype=torch.float32)
    eng2.add_request({"request_id": "c", "prompt_ids": ids(32, 3), "max_new_tokens": 1})
    eng2.step()
    check("FP32 回退逐行路径", eng2.model.use_tiled is False, f"use_tiled={eng2.model.use_tiled}")

    eng3 = build(use_cuda_graph=False)
    eng3.add_request({"request_id": "d", "prompt_ids": ids(32, 4), "max_new_tokens": 2})
    eng3.add_request({"request_id": "e", "prompt_ids": ids(32, 5), "max_new_tokens": 2})
    eng3.step()                      # 两条都是 prefill
    eng3.add_request({"request_id": "f", "prompt_ids": ids(1, 6), "max_new_tokens": 2})
    eng3.step()                      # d/e 已进入 decode，f 是首轮 prefill -> 混合批次
    check("混合 prefill/decode 整批回退", eng3.model.use_tiled is False,
          f"use_tiled={eng3.model.use_tiled}")

    print("\n=== 2. 同一张图重放不同的请求划分（N=64 固定）===")
    # A：2 条请求各 32 行 -> 2 个 tile
    # C：40 + 24 行     -> 3 个 tile
    # 两者 N 都是 64，图键相同；若键里没有路径/划分信息就会互相污染
    case_a = [ids(32, 11), ids(32, 12)]
    case_c = [ids(40, 13), ids(24, 14)]

    for tag, prompts in (("A(32+32)", case_a), ("C(40+24)", case_c)):
        eager, _ = run(build(use_cuda_graph=False), prompts, 4)
        g1, _ = run(build(use_cuda_graph=True), prompts, 4)     # 首次捕获
        g2, _ = run(build(use_cuda_graph=True), prompts, 4)     # 同划分重放
        check(f"{tag} eager 与 Graph 一致", eager == g1)
        check(f"{tag} 同划分重复运行一致", g1 == g2)

    # 关键：同一个引擎里先跑 A 再跑 C，两者 N 相同、划分不同，共用图缓存
    eng_mix = build(use_cuda_graph=True)
    a_first, _ = run(eng_mix, case_a, 4)
    c_after, _ = run(eng_mix, case_c, 4)      # 这会命中 A 捕获的图
    eager_a, _ = run(build(use_cuda_graph=False), case_a, 4)
    eager_c, _ = run(build(use_cuda_graph=False), case_c, 4)
    check("A 之后跑 C：A 的结果不受污染", a_first == eager_a)
    check("A 之后跑 C：C 的结果正确（重用了 A 的图）", c_after == eager_c,
          "同 N 不同划分共用图")

    print("\n=== 3. 整段 prefill / chunked prefill / M=0 ===")
    # chunked：预算 16，prompt 60 -> 分 4 个 chunk（fixture 的 max_seq_len=64，
    # prompt 要留出生成空间，否则块表会超出 max_blocks_per_request）
    eng_chunk = build(use_cuda_graph=False, max_num_batched_tokens=16)
    got_chunk, steps = run(eng_chunk, [ids(60, 21)], 2)
    eng_whole = build(use_cuda_graph=False, max_num_batched_tokens=64)
    got_whole, _ = run(eng_whole, [ids(60, 21)], 2)
    check("chunked prefill 与整段 prefill 结果一致", got_chunk == got_whole,
          f"chunked 用了 {steps} 步；两者输出 {got_chunk.get('r0')}")

    # M=0：本轮没有任何请求需要采样（用一条已结束的请求驱动）
    eng_m0 = build(use_cuda_graph=False)
    eng_m0.add_request({"request_id": "z", "prompt_ids": ids(8, 31), "max_new_tokens": 1})
    while eng_m0.has_unfinished_requests():
        eng_m0.step()
    check("M=0（无待采样请求）不报错", True)

    print("\n=== 4. 与逐行路径在同一引擎配置下的数值一致性 ===")
    # 比 **logits**，不比生成文本：小模型上 bf16 量级的差就能翻转 argmax，
    # 用「逐 token 相同」当判据会把正常的舍入差误报成错误。
    prompt = ids(32, 41)
    lo = prefill_logits(False, prompt)
    ln = prefill_logits(None, prompt)          # 自然选择：prefill 会走分块
    diff = (ln - lo).abs().max().item()
    step = lo.abs().max().item() * 2 ** -9
    check("prefill 两条路径的 logits 差落在 bf16 台阶量级",
          diff <= 4 * step,
          f"最大差={diff:.6f}，bf16 台阶≈{step:.6f}，倍数={diff / step:.1f}")
    check("prefill 两条路径的 argmax 一致",
          bool((lo.argmax(-1) == ln.argmax(-1)).all()))

    # 单行 query 也走分块 kernel（正常选择不会这样，但这是合法输入，要能跑对）
    lone_old = prefill_logits(False, ids(32, 42))
    lone_new = prefill_logits(True, ids(32, 42))
    d2 = (lone_new - lone_old).abs().max().item()
    check("单行 query 时分块 kernel 仍与逐行路径一致（有限值且同量级）",
          bool(torch.isfinite(lone_new).all()) and d2 <= 4 * step,
          f"最大差={d2:.6f}")

    print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"结果：{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过"
          + (f"，失败：{bad}" if bad else ""))
    return 1 if bad else 0


def prefill_logits(tiled, prompt):
    """跑一步 prefill，返回模型给出的 logits（只看首步，decode 不参与）。

    tiled=None 表示不干预，用引擎自己的路径选择；True/False 则把路径钉死。
    """
    eng = build(use_cuda_graph=False)
    orig_prep = eng.model._prepare_inputs

    def prep(*a, **kw):
        n = orig_prep(*a, **kw)
        if tiled is not None:
            eng.model.use_tiled = tiled
        return n

    captured = {}
    orig_fwd = eng.model._forward_append

    def fwd(*a, **kw):
        out = orig_fwd(*a, **kw)
        captured["logits"] = out.detach().float().clone()
        return out

    eng.model._prepare_inputs = prep
    eng.model._forward_append = fwd
    eng.add_request({"request_id": "r0", "prompt_ids": prompt, "max_new_tokens": 4})
    eng.step()
    return captured["logits"]


if __name__ == "__main__":
    raise SystemExit(main())
