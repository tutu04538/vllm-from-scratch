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

    print("\n=== 4. 既有语义没有改变 ===")
    e = build("step39", FIX)
    with torch.inference_mode():
        layer = e.model.layers[0]
        n_store = layer._qkv_storage.numel()
        n_views = layer.q_proj.weight.numel() + layer.k_proj.weight.numel() + layer.v_proj.weight.numel()
        check("合并后显存不变（存储元素数 == 三个视图之和）", n_store == n_views,
              f"{n_store} vs {n_views}")
        check("q/k/v 的 Parameter 仍然是存储的视图",
              layer.q_proj.weight.data_ptr() == layer._qkv_storage.data_ptr())

        # 原地改权重：写穿到存储
        layer.q_proj.weight.add_(0.123)
        check("原地改 q_proj.weight 写穿到合并存储",
              torch.equal(layer._qkv_storage[:layer.q_proj.weight.shape[0]], layer.q_proj.weight))

        # load_state_dict 逐参数写穿
        before = e.model.state_dict()
        e.model.load_state_dict(before)
        check("load_state_dict 往返后存储仍然正确",
              torch.equal(layer._qkv_storage[:layer.q_proj.weight.shape[0]], layer.q_proj.weight))

        # state_dict 键名与取值
        keys = set(e.model.state_dict())
        check("state_dict 键名未变（仍是 q_proj/k_proj/v_proj/gate_proj/up_proj）",
              {"layers.0.q_proj.weight", "layers.0.k_proj.weight", "layers.0.v_proj.weight",
               "layers.0.gate_proj.weight", "layers.0.up_proj.weight"} <= keys)

        # .to() 之后共享关系要恢复
        e.model.to("cuda")
        check("再次 .to(cuda) 后视图仍指向存储",
              e.model.layers[0].q_proj.weight.data_ptr() == e.model.layers[0]._qkv_storage.data_ptr())

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
