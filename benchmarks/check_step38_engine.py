"""第三十八关引擎级检查：后端接线、Graph 位置重放、两条路径的数值一致。

    python benchmarks/check_step38_engine.py
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import step38 as m

FIX = ROOT / "fixtures" / "step30_qwen3" / "tiny_gqa"
VOCAB = 11
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def build(rope_backend, use_cuda_graph=False, dtype=torch.bfloat16, max_batched=64):
    return m.Engine.from_model_dir(
        str(FIX), device="cuda", attention_backend="triton", use_cuda_graph=use_cuda_graph,
        max_num_seqs=4, max_num_batched_tokens=max_batched, num_kv_blocks=128, block_size=4,
        enable_prefix_caching=False, dtype=dtype, rope_backend=rope_backend)


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


def run(engine, prompts, max_new):
    got = {}
    engine.scheduler.on_finished = lambda r: got.__setitem__(r["request_id"], list(r["output_ids"]))
    for i, p in enumerate(prompts):
        engine.add_request({"request_id": f"r{i}", "prompt_ids": p, "max_new_tokens": max_new})
    while engine.has_unfinished_requests():
        engine.step()
    return got


def last_logits(engine, prompt, gen=4):
    """跑一步 prefill，取该步的 logits（用于两条后端对照）"""
    cap = {}
    orig = engine.model._forward_append

    def fwd(*a, **kw):
        out = orig(*a, **kw)
        cap["logits"] = out.detach().float().clone()
        return out

    engine.model._forward_append = fwd
    engine.add_request({"request_id": "r0", "prompt_ids": prompt, "max_new_tokens": gen})
    engine.step()
    engine.model._forward_append = orig
    return cap["logits"]


def main():
    torch.manual_seed(0)

    print("=== 1. 后端接线 ===")
    for backend in ("torch", "triton"):
        e = build(backend)
        check(f"rope_backend={backend} 传到 RotaryEmbedding",
              e.model.rotary.backend == backend and e.model.rope_backend == backend)
    e_t = build("triton", dtype=torch.float32)
    check("FP32 也支持 triton RoPE", e_t.model.rotary.backend == "triton")
    try:
        build("nope")
        check("未知 rope_backend 被拒绝", False, "没有报错")
    except ValueError:
        check("未知 rope_backend 被拒绝", True)

    print("\n=== 2. 两条后端的 logits 一致（BF16 / FP32）===")
    for dt in (torch.bfloat16, torch.float32):
        prompt = ids(32, 7)
        lo = last_logits(build("torch", dtype=dt), prompt)
        ln = last_logits(build("triton", dtype=dt), prompt)
        scale = lo.abs().max().item()
        eps = 2 ** -9 if dt == torch.bfloat16 else 2 ** -25
        d = (ln - lo).abs().max().item()
        check(f"{dt} 下两条后端 logits 一致", d <= 4 * scale * eps,
              f"最大差={d:.3e} 容限={4 * scale * eps:.3e}")

    print("\n=== 3. Graph 重放：相同 shape、不同 positions 必须生效 ===")
    # decode 步的 RoPE 输入 shape 都是 [1, H, D]，但位置随请求不同。
    # 若图把首次捕获时的 positions 固化下来，第二个请求就会算错。
    prompts = [ids(5, 11), ids(23, 12), ids(40, 13)]
    eager = run(build("torch"), prompts, 6)
    eager_tri = run(build("triton"), prompts, 6)
    graph_tri = run(build("triton", use_cuda_graph=True), prompts, 6)
    check("三个不同长度的请求：Graph 与 eager 输出一致", graph_tri == eager_tri,
          f"graph={graph_tri}")
    check("Torch 与 Triton 后端输出一致（Graph）", graph_tri == eager,
          f"torch={eager}")

    # 同一个引擎里连续跑不同位置的请求，确保图被复用而不是各自重捕获
    e = build("triton", use_cuda_graph=True)
    a = run(e, [ids(5, 21)], 6)
    keys_a = set(e.model.graphs)
    b = run(e, [ids(37, 22)], 6)
    keys_b = set(e.model.graphs)
    ea = run(build("triton"), [ids(5, 21)], 6)
    eb = run(build("triton"), [ids(37, 22)], 6)
    check("先跑位置短的、再跑位置长的：两者都正确",
          a == ea and b == eb, f"a={a} b={b}")
    # prompt 长度不同（5 vs 37），prefill 的 N 不同，本来就该各有一张 prefill 图；
    # 但两侧的 decode 步 shape 相同（N=1），必须复用同一张图。
    # 若按位置重新捕获，多出来的就不止一张。
    added = keys_b - keys_a
    check("只多出 B 的 prefill 图，decode 图被复用（未按位置重捕获）",
          len(added) == 1 and added == {(37, 1, True, True)},
          f"新增图 {sorted(str(k) for k in added)}")

    print("\n=== 4. 真实 Qwen3-0.6B 端到端 ===")
    from benchmarks.bench_step36_vllm_compare import MODEL_DIR
    out = {}
    for backend in ("torch", "triton"):
        import importlib
        eng = m.Engine.from_model_dir(
            MODEL_DIR, device="cuda", dtype=torch.bfloat16, attention_backend="triton",
            norm_backend="triton", rope_backend=backend, use_cuda_graph=True,
            max_num_seqs=8, max_num_batched_tokens=2048, block_size=16, num_kv_blocks=512,
            enable_prefix_caching=False)
        eng.model.eos_token_ids = ()
        eng.scheduler.eos_token_ids = set()
        got = {}
        eng.scheduler.on_finished = lambda r: got.__setitem__(r["request_id"], list(r["output_ids"]))
        eng.add_request({"request_id": "x", "prompt_ids": ids(64, 99), "max_new_tokens": 8})
        while eng.has_unfinished_requests():
            eng.step()
        out[backend] = got["x"]
        check(f"真实模型 rope={backend} 生成 8 token", len(got["x"]) == 8, str(got["x"]))
    check("真实模型两条后端的 greedy 输出一致", out["torch"] == out["triton"],
          f"torch={out['torch']} triton={out['triton']}")

    print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"结果：{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过"
          + (f"，失败：{bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
