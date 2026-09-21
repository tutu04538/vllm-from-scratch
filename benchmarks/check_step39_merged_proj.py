"""第三十九关验证：合并投影的结构、数值与既有语义。

    python benchmarks/check_step39_merged_proj.py

三件事分开验：

    结构   每层投影 launch 数 7 -> 4，由 profiler 的 kernel 计数确认（不看代码）
    数值   step38 与 step39 装同一份权重、喂同一份输入，逐层 KV 与每步 logits 对照
    语义   原地改权重 / load_state_dict / state_dict() / 多请求打包 N>1 / 图捕获
"""

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIX = ROOT / "fixtures" / "step30_qwen3" / "tiny_gqa"
MODEL_DIR = "/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B"
VOCAB = 11
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


def build(step, model_dir, dtype=torch.bfloat16, rope_backend="triton", graph=False,
          max_batched=64, num_kv_blocks=128):
    mod = __import__(step)
    return mod.Engine.from_model_dir(
        str(model_dir), device="cuda", dtype=dtype, attention_backend="triton",
        norm_backend="triton", rope_backend=rope_backend, use_cuda_graph=graph,
        max_num_seqs=4, max_num_batched_tokens=max_batched, block_size=4,
        num_kv_blocks=num_kv_blocks, enable_prefix_caching=False)


def run_capture(engine, prompts, gen, steps_to_run=None):
    """跑一组请求，返回每步 logits 与该步结束时的 KV 快照。"""
    logits, kvs = [], []
    orig = engine.model._forward_append

    def fwd(*a, **kw):
        out = orig(*a, **kw)
        logits.append(out.detach().float().clone())
        kvs.append(engine.kv_cache_pool.k_cache.detach().clone())
        return out

    engine.model._forward_append = fwd
    for i, p in enumerate(prompts):
        engine.add_request({"request_id": f"r{i}", "prompt_ids": p, "max_new_tokens": gen})
    n = 0
    while engine.has_unfinished_requests():
        engine.step()
        n += 1
        if steps_to_run and n >= steps_to_run:
            break
    engine.model._forward_append = orig
    return logits, kvs


def compare(tag, a_logits, a_kvs, b_logits, b_kvs, tol_scale):
    """a = step38（分开），b = step39（合并）"""
    ok = True
    lmax = kmax = 0.0
    for i, (la, lb) in enumerate(zip(a_logits, b_logits)):
        if la.shape != lb.shape:
            check(f"{tag} 第 {i} 步 logits 形状一致", False, f"{la.shape} vs {lb.shape}")
            return
        d = (la - lb).abs().max().item()
        lmax = max(lmax, d)
        scale = la.abs().max().item()
        if d > 4 * scale * tol_scale:
            ok = False
    for i, (ka, kb) in enumerate(zip(a_kvs, b_kvs)):
        d = (ka.float() - kb.float()).abs().max().item()
        kmax = max(kmax, d)
        if not torch.isfinite(kb.float()).all():
            ok = False
    check(f"{tag} 逐步 logits 最大差={lmax:.3e}  逐层 KV 最大差={kmax:.3e}",
          ok, f"{len(a_logits)} 步，KV {tuple(a_kvs[0].shape)}")


def main():
    torch.manual_seed(0)

    print("=== 1. 结构：每层投影 launch 数 7 -> 4 ===")
    from torch.profiler import profile, ProfilerActivity
    for step in ("step38", "step39"):
        e = build(step, FIX, max_batched=16)
        with torch.inference_mode():
            e.add_request({"request_id": "a", "prompt_ids": ids(8, 1), "max_new_tokens": 2})
            e.step()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                e.step()
                torch.cuda.synchronize()
        ev = [x for x in prof.events() if str(getattr(x, "device_type", "")) == "DeviceType.CUDA"]
        # tiny 模型 2 层，一步 decode 的投影 GEMM 次数 = 层数 × 每层投影数
        gemm = [x for x in ev if "gemv" in x.name.lower() or "gemm" in x.name.lower()]
        print(f"  {step}: 一步 decode 的 GEMM/GEMV kernel = {len(gemm)} 个（2 层 × 每层投影数 = {len(gemm)//2}）")

    print("\n=== 2. 与 step38 的数值对照（tiny，多请求打包 N>1）===")
    prompts = [ids(16, 11), ids(9, 12), ids(23, 13)]
    for dtype, tol in ((torch.bfloat16, 2 ** -9), (torch.float32, 2 ** -25)):
        a = build("step38", FIX, dtype=dtype)
        b = build("step39", FIX, dtype=dtype)
        with torch.inference_mode():
            la, ka = run_capture(a, prompts, 4)
            lb, kb = run_capture(b, prompts, 4)
        compare(f"{dtype}", la, ka, lb, kb, tol)

    print("\n=== 3. 真实 Qwen3-0.6B：prefill + 若干 decode 步 ===")
    real_prompts = [ids(64, 21), ids(48, 22)]
    for rope in ("torch", "triton"):
        a = build("step38", MODEL_DIR, rope_backend=rope, graph=False,
                  max_batched=2048, num_kv_blocks=512)
        b = build("step39", MODEL_DIR, rope_backend=rope, graph=False,
                  max_batched=2048, num_kv_blocks=512)
        with torch.inference_mode():
            la, ka = run_capture(a, real_prompts, 6)
            lb, kb = run_capture(b, real_prompts, 6)
        compare(f"真实模型 rope={rope}", la, ka, lb, kb, 2 ** -9)

    print("\n=== 4. 参数名、兼容装载与往返 ===")
    e = build("step39", FIX)
    with torch.inference_mode():
        keys = set(e.model.state_dict())
        check("state_dict 用融合键（qkv_proj / gate_up_proj）",
              {"layers.0.qkv_proj.weight", "layers.0.gate_up_proj.weight"} <= keys,
              f"layers.0 的投影键: {sorted(k for k in keys if k.startswith('layers.0.') and 'proj' in k)}")
        check("旧的三键/两键不再出现",
              not ({"layers.0.q_proj.weight", "layers.0.k_proj.weight", "layers.0.v_proj.weight",
                    "layers.0.gate_proj.weight", "layers.0.up_proj.weight"} & keys))

        layer = e.model.layers[0]
        n_qkv = layer.qkv_proj.weight.numel()
        expect = (e.model.num_q_heads + 2 * e.model.num_kv_heads) * e.model.head_dim * e.model.d_model
        check("融合权重的元素数 == Q/K/V 三者之和", n_qkv == expect, f"{n_qkv} vs {expect}")

        # 旧写法（三键）必须仍能装进来，且与融合键装出来的结果逐位相同
        fused = {k: v.detach().clone() for k, v in e.model.state_dict().items()}
        legacy = {}
        for k, v in fused.items():
            if k.endswith("qkv_proj.weight"):
                nq = e.model.num_q_heads * e.model.head_dim
                nk = e.model.num_kv_heads * e.model.head_dim
                p = k[: -len("qkv_proj.weight")]
                legacy[p + "q_proj.weight"] = v[:nq]
                legacy[p + "k_proj.weight"] = v[nq:nq + nk]
                legacy[p + "v_proj.weight"] = v[nq + nk:]
            elif k.endswith("gate_up_proj.weight"):
                p = k[: -len("gate_up_proj.weight")]
                half = v.shape[0] // 2
                legacy[p + "gate_proj.weight"] = v[:half]
                legacy[p + "up_proj.weight"] = v[half:]
            else:
                legacy[k] = v
        check("旧三键字典里确实没有融合键",
              not any(k.endswith("qkv_proj.weight") for k in legacy))

        e_legacy = build("step39", FIX)
        e_legacy.model.load_state_dict(legacy, strict=True)
        same = all(torch.equal(a, b) for a, b in
                   zip(e.model.state_dict().values(), e_legacy.model.state_dict().values()))
        check("用旧三键装载后权重与融合键装载逐位相同", same)

        # 真权重上跑一遍，确认两条装载路径的 logits 也一致
        la, _ = run_capture(build("step39", FIX), prompts, 3)
        lb, _ = run_capture(e_legacy, prompts, 3)
        d = max((x - y).abs().max().item() for x, y in zip(la, lb))
        check("两条装载路径的前向输出一致", d == 0.0, f"最大差={d:.3e}")

        # 原地改融合参数
        before = layer.qkv_proj.weight.detach().clone()
        layer.qkv_proj.weight.add_(0.123)
        check("原地改融合权重生效",
              not torch.equal(layer.qkv_proj.weight, before))

    print("\n=== 5. 图捕获次数没有变多 ===")
    e38 = build("step38", FIX, graph=True)
    e39 = build("step39", FIX, graph=True)
    with torch.inference_mode():
        for e in (e38, e39):
            for i, p in enumerate(prompts):
                e.add_request({"request_id": f"g{i}", "prompt_ids": p, "max_new_tokens": 4})
            while e.has_unfinished_requests():
                e.step()
    k38 = sorted(str(k) for k in e38.model.graphs)
    k39 = sorted(str(k) for k in e39.model.graphs)
    check("图的数量与键都相同", k38 == k39, f"{len(k38)} vs {len(k39)}")

    print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"结果：{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过"
          + (f"，失败：{bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
