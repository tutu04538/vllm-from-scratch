"""84 关基准：共享/紧凑 RoPE 与融合 Q/K 旋转。

用法：

    python benchmarks/check_step84_rope.py --audit --micro
    python benchmarks/check_step84_rope.py --real
    python benchmarks/check_step84_rope.py --replay83 --replay-fp32
    python benchmarks/check_step84_rope.py --all

边界：
- `--real` 只跑真实 target+assistant 的 8-token 短生成（eager 与 graph 各一个
  独立子进程），不跑 B×K 性能矩阵；
- `--micro` 只测 RoPE 算子本身，不代表端到端吞吐；
- `--replay83` 复放 83 的固定中文 prompt / 64-token 历史，比较 token、top1、
  logits margin 与逐层 hidden 的变化，不承诺逐 token 等价。
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from check_step82_real_spec import (  # noqa: E402
    ASSISTANT, TARGET, _git, _package_hash, _parent_gpu_info,
)

TARGET_JSON = TARGET / "config.json"
ASSISTANT_JSON = ASSISTANT / "config.json"
PROMPT = "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")


def _worker_argv() -> list[str]:
    return [sys.executable, str(Path(__file__).resolve())]


def _engine_args(*, graph: bool = False, dtype: str = "bfloat16",
                 blocks: int = 512) -> argparse.Namespace:
    return argparse.Namespace(
        device="cuda", dtype=dtype, attention_backend="torch",
        max_model_len=2048, block_size=16, blocks=blocks,
        budget=128, batch=1, max_new_tokens=8, repeats=1,
        graph=graph, prefix=False, ignore_eos=True,
    )


# ---------------------------------------------------------------------------
# 子进程 case
# ---------------------------------------------------------------------------

def _case_audit(case: dict) -> dict:
    """不加载权重：按真实 config 建 35+4 个 RoPE 层，按唯一 storage 审计。"""
    import torch

    from minivllm.models.gemma4_mtp import (_layer_geometry, _layer_types,
                                           _make_rotary_embedding, _text_config)

    max_position = int(case.get("max_position", 0)) or None

    def build(path: Path):
        config = copy.deepcopy(json.loads(path.read_text()))
        text_config = _text_config(config)
        if max_position is not None:
            text_config["max_position_embeddings"] = max_position
        types = _layer_types(text_config, int(text_config["num_hidden_layers"]))
        modules = []
        for layer_idx, layer_type in enumerate(types):
            resolved, head_dim, _ = _layer_geometry(text_config, layer_idx)
            modules.append(_make_rotary_embedding(
                text_config, layer_idx, head_dim, resolved,
                is_sliding=resolved == "sliding_attention"))
        # 模拟真实加载后的 BF16：这一步必须仍然只留 2 份 storage。
        for module in modules:
            module.to(torch.bfloat16)
        return modules, types, text_config

    target_modules, target_types, target_text = build(TARGET_JSON)
    assistant_modules, assistant_types, assistant_text = build(ASSISTANT_JSON)
    all_modules = target_modules + assistant_modules
    unique: dict[int, dict] = {}
    for module in all_modules:
        cache = module.cos_sin_cache
        ptr = cache.untyped_storage().data_ptr()
        record = unique.setdefault(ptr, {
            "shape": list(cache.shape),
            "dtype": str(cache.dtype),
            "numel": cache.numel(),
            "bytes": cache.numel() * cache.element_size(),
            "layers": [],
        })
        assert record["shape"] == list(cache.shape)
        record["layers"].append(module.rope_type)

    nominal_bytes = sum(
        module.cos_sin_cache.numel() * module.cos_sin_cache.element_size()
        for module in all_modules)
    unique_bytes = sum(record["bytes"] for record in unique.values())
    table_shape = {record["shape"][1] for record in unique.values()}
    max_position = int(target_text["max_position_embeddings"])
    expected_bytes = max_position * (256 + 512) * 2
    return {
        "target_layers": len(target_modules),
        "assistant_layers": len(assistant_modules),
        "target_layer_types": target_types,
        "assistant_layer_types": assistant_types,
        "max_position": max_position,
        "dtype": "bfloat16",
        "unique_tables": len(unique),
        "table_widths": sorted(table_shape),
        "unique_bytes": unique_bytes,
        "unique_mib": unique_bytes / 2**20,
        "expected_unique_bytes": expected_bytes,
        "per_layer_nominal_bytes": nominal_bytes,
        "per_layer_nominal_mib": nominal_bytes / 2**20,
        "saved_mib": (nominal_bytes - unique_bytes) / 2**20,
        "tables": [
            {k: v for k, v in record.items() if k != "numel"}
            for record in unique.values()
        ],
    }


def _case_micro(case: dict) -> dict:
    """只测 RoPE 算子：fused Triton vs 显式 Torch 参考。"""
    import torch

    from minivllm.layers import get_rope
    from minivllm.layers.ops.rotary_embedding import (apply_rotary_embedding,
                                                      apply_rotary_embedding_native)

    max_position = 4096
    configs = {
        "sliding": get_rope(
            256, max_position=max_position,
            rope_parameters={"rope_theta": 10000.0},
            dtype=torch.bfloat16, device="cuda"),
        "full_proportional": get_rope(
            512, max_position=max_position,
            rope_parameters={
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            dtype=torch.bfloat16, device="cuda"),
    }
    cases = [
        # name, module, num_tokens, q_heads, k_heads, q_only, iters
        ("decode_sliding", "sliding", 1, 16, 8, False, 500),
        ("decode_sliding_qonly", "sliding", 1, 16, 0, True, 500),
        ("decode_full_qonly", "full_proportional", 1, 8, 0, True, 500),
        ("prefill_sliding", "sliding", 128, 16, 8, False, 50),
    ]
    rows = []
    for name, key, num_tokens, q_heads, k_heads, q_only, iters in cases:
        module = configs[key]
        dtype = module.cos_sin_cache.dtype
        head_size = module.head_size
        apply_dim = module.rotary_dim
        q = torch.randn(num_tokens, q_heads * head_size, dtype=dtype,
                        device="cuda")
        k = None if q_only else torch.randn(
            num_tokens, k_heads * head_size, dtype=dtype, device="cuda")
        # 位置不能从 0 开始：position 0 的 cos=1/sin=0，根本不会发生旋转，
        # 微测会假通过。用非零、且不连续的位置覆盖真实 decode 查表。
        positions = (
            torch.arange(num_tokens, dtype=torch.int64, device="cuda") * 37 + 11
        ) % 1024
        cache = module.cos_sin_cache

        expected_q, expected_k = apply_rotary_embedding_native(
            positions, q, k, cache, head_size, apply_dim, True)
        fused_q = q.clone()
        fused_k = None if k is None else k.clone()
        apply_rotary_embedding(
            positions, fused_q, fused_k, cache, head_size, apply_dim, True)
        error = (fused_q.float() - expected_q.float()).abs().max().item()
        if k is not None:
            error = max(
                error,
                (fused_k.float() - expected_k.float()).abs().max().item())

        def timed(fn, count):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(count):
                    fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000.0 / count)
            samples.sort()
            return samples[len(samples) // 2]

        fused_us = timed(
            lambda: apply_rotary_embedding(
                positions, fused_q, fused_k, cache, head_size, apply_dim, True),
            iters)
        native_us = timed(
            lambda: apply_rotary_embedding_native(
                positions, q, k, cache, head_size, apply_dim, True),
            iters)
        rows.append({
            "name": name,
            "rope_key": key,
            "num_tokens": num_tokens,
            "q_heads": q_heads,
            "k_heads": k_heads,
            "q_only": q_only,
            "head_size": head_size,
            "apply_dim": apply_dim,
            "dtype": str(dtype),
            "iters": iters,
            "fused_us_per_call": fused_us,
            "native_us_per_call": native_us,
            "fused_over_native_including_launch": (
                fused_us / native_us if native_us else None),
            "max_abs_error_vs_native": error,
            "timing": "median of 5 event loops; includes Python launch gaps",
        })
    return {
        "device": torch.cuda.get_device_name(0),
        "max_position": max_position,
        "cases": rows,
        "note": "只测 RoPE 算子；fused 原地、native 返回新张量，二者都含 launch 开销",
    }


def _audit_loaded(runner) -> dict:
    import torch

    modules = []
    for owner, model in (
        ("target", runner.model),
        ("assistant", getattr(getattr(runner, "speculator", None),
                              "model", None)),
    ):
        if model is None:
            continue
        for name, module in model.named_modules():
            cache = getattr(module, "cos_sin_cache", None)
            if cache is not None:
                modules.append((owner, name, module, cache))
    unique: dict[int, dict] = {}
    for owner, name, module, cache in modules:
        ptr = cache.untyped_storage().data_ptr()
        record = unique.setdefault(ptr, {
            "shape": list(cache.shape),
            "dtype": str(cache.dtype),
            "bytes": cache.numel() * cache.element_size(),
            "owners": [],
            "sample_layers": [],
        })
        record["owners"].append(owner)
        record["sample_layers"].append(name)
    nominal = sum(cache.numel() * cache.element_size()
                  for _, _, _, cache in modules)
    unique_bytes = sum(record["bytes"] for record in unique.values())
    return {
        "num_rope_modules": len(modules),
        "unique_tables": len(unique),
        "unique_bytes": unique_bytes,
        "unique_mib": unique_bytes / 2**20,
        "per_layer_nominal_bytes": nominal,
        "per_layer_nominal_mib": nominal / 2**20,
        "table_widths": sorted({record["shape"][1] for record in unique.values()}),
        "tables": [
            {
                **{k: v for k, v in record.items() if k != "sample_layers"},
                "sample_layers": record["sample_layers"][:3],
            }
            for record in unique.values()
        ],
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def _case_real(case: dict) -> dict:
    import torch

    from check_step82_real_spec import (_build_engine, _load_tokenizer,
                                        _prompt_ids, _sampling_params)

    graph = bool(case["graph"])
    args = _engine_args(graph=graph)
    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    engine, runner, build_s = _build_engine(
        args, k=2, batch=1, graph=graph)
    after_build = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
    }
    tokens = []
    try:
        audit = _audit_loaded(runner)
        engine.add_request(
            "r", prompt,
            _sampling_params(argparse.Namespace(
                **{**vars(args), "max_new_tokens": 8,
                   "ignore_eos": True})))
        for _ in range(20000):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                tokens = list(out.token_ids)
        after_run = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
        }
        capture_stats = getattr(runner, "cudagraph_capture_stats", None)
    finally:
        engine.shutdown()
    del runner, engine
    gc.collect()
    from minivllm.layers import get_rope_cache_stats
    cache_after_shutdown = get_rope_cache_stats()
    torch.cuda.empty_cache()
    after_shutdown = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    return {
        "graph": graph,
        "build_s": build_s,
        "prompt_len": len(prompt),
        "tokens": tokens,
        "num_tokens": len(tokens),
        "audit": audit,
        "memory": {
            "before": before,
            "after_build": after_build,
            "after_run": after_run,
            "after_shutdown": after_shutdown,
        },
        "cudagraph_capture_stats": capture_stats,
        "rope_cache_after_shutdown": cache_after_shutdown,
        "note": "单 prompt、8 token；不是 B×K 性能表",
    }


def _index_trace(trace: dict) -> dict[int, dict]:
    index = {}
    for record in trace.get("target_steps", []):
        positions = record["logits_positions"].tolist()
        top = record.get("logits_top")
        hidden = record.get("final_hidden")
        layer_hidden = record.get("layer_hidden", {})
        for row, position in enumerate(positions):
            index[int(position)] = {
                "step": record["step"],
                "top1": int(top["top_ids"][row][0]) if top is not None else None,
                "margin": float(top["margin"][row]) if top is not None else None,
                "hidden": hidden[row] if hidden is not None else None,
                "layer_hidden": {
                    name: values[row] for name, values in layer_hidden.items()
                },
            }
    return index


def _compare_one_trace(old: dict, new: dict) -> dict:
    import torch

    index_old = _index_trace(old)
    index_new = _index_trace(new)
    common_positions = sorted(set(index_old) & set(index_new))
    top1_same = 0
    max_margin_change = 0.0
    max_hidden_rel = 0.0
    max_layer_rel = 0.0
    max_layer_name = None
    layer_rel: dict[str, float] = {}
    for position in common_positions:
        left, right = index_old[position], index_new[position]
        if left["top1"] is not None and left["top1"] == right["top1"]:
            top1_same += 1
        if left["margin"] is not None and right["margin"] is not None:
            max_margin_change = max(
                max_margin_change, abs(left["margin"] - right["margin"]))
        if left["hidden"] is not None and right["hidden"] is not None:
            a, b = left["hidden"].float(), right["hidden"].float()
            rel = float((a - b).norm()) / max(float(a.norm()), 1e-12)
            max_hidden_rel = max(max_hidden_rel, rel)
        for name in set(left["layer_hidden"]) & set(right["layer_hidden"]):
            a = left["layer_hidden"][name].float()
            b = right["layer_hidden"][name].float()
            rel = float((a - b).norm()) / max(float(a.norm()), 1e-12)
            layer_rel[name] = max(layer_rel.get(name, 0.0), rel)
            if rel > max_layer_rel:
                max_layer_rel, max_layer_name = rel, name
    return {
        "common_target_positions": len(common_positions),
        "top1_same": top1_same,
        "top1_agreement": (top1_same / len(common_positions)
                           if common_positions else None),
        "max_margin_abs_change": max_margin_change,
        "max_final_hidden_relative_l2": max_hidden_rel,
        "max_layer_hidden_relative_l2": max_layer_rel,
        "max_layer_hidden_relative_l2_layer": max_layer_name,
        "first_layer_relative_l2_gt_1e-3": next(
            (name for name in sorted(
                layer_rel, key=lambda s: int(s.rsplit(".", 1)[-1]))
             if layer_rel[name] > 1e-3), None),
        "position70": {
            "old": (
                None if 70 not in index_old else {
                    "top1": index_old[70]["top1"],
                    "margin": index_old[70]["margin"],
                }),
            "new": (
                None if 70 not in index_new else {
                    "top1": index_new[70]["top1"],
                    "margin": index_new[70]["margin"],
                }),
        },
    }


def _compare_trace_pair(old_k0: dict, old_k2: dict,
                        new_k0: dict, new_k2: dict, label: str) -> dict:
    old_tokens = old_k0["tokens"]
    new_tokens = new_k0["tokens"]
    old_first_diff = next(
        (i for i, (a, b) in enumerate(zip(old_tokens, old_k2["tokens"]))
         if a != b), None)
    new_first_diff = next(
        (i for i, (a, b) in enumerate(zip(new_tokens, new_k2["tokens"]))
         if a != b), None)
    return {
        "label": label,
        "old_tokens": len(old_tokens),
        "new_tokens": len(new_tokens),
        "old_first_diff": old_first_diff,
        "new_first_diff": new_first_diff,
        "token_changes_k0": sum(
            a != b for a, b in zip(old_k0["tokens"], new_k0["tokens"])),
        "token_changes_k2": sum(
            a != b for a, b in zip(old_k2["tokens"], new_k2["tokens"])),
        "k0": _compare_one_trace(old_k0, new_k0),
        "k2": _compare_one_trace(old_k2, new_k2),
    }


def _case_replay83(case: dict) -> dict:
    import torch

    dtype = str(case.get("dtype", "bfloat16"))
    blocks = int(case.get("blocks", 512))
    name = "bf16" if dtype == "bfloat16" else dtype.replace("float", "fp")
    out_dir = Path(case["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "benchmarks" / "check_step83_numerics.py"),
        "--max-new-tokens", "64",
        "--output-dir", str(out_dir),
        "--dtype", dtype,
        "--blocks", str(blocks),
    ]
    if dtype == "bfloat16":
        command.append("--fp32-final-control")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=case.get("timeout_s", 3600))
    if completed.returncode != 0:
        raise RuntimeError(
            f"83 历史复放失败（exit {completed.returncode}）：\n"
            f"{completed.stderr[-4000:]}")
    replay_s = time.perf_counter() - started

    old_dir = ROOT / "benchmarks" / "results" / "step83" / name
    old0 = old_dir / "trace_k0.pt"
    old2 = old_dir / "trace_k2.pt"
    new0 = out_dir / "trace_k0.pt"
    new2 = out_dir / "trace_k2.pt"
    if not (old0.is_file() and old2.is_file()):
        return {
            "baseline_missing": True,
            "baseline_dir": str(old_dir),
            "new_trace_dir": str(out_dir),
            "replay_s": replay_s,
            "note": "旧 83 trace 不存在；本次只生成新 trace，不做数值变化对比",
        }
    old_k0 = torch.load(old0, map_location="cpu", weights_only=False)
    old_k2 = torch.load(old2, map_location="cpu", weights_only=False)
    new_k0 = torch.load(new0, map_location="cpu", weights_only=False)
    new_k2 = torch.load(new2, map_location="cpu", weights_only=False)
    comparison = _compare_trace_pair(
        old_k0, old_k2, new_k0, new_k2, name)
    # 把“新 K=0/K=2 在第一分歧前后的证据”也落盘，避免只看 token。
    comparison["old_summary"] = json.loads(
        (old_dir / "numerics_summary.json").read_text())[
            "comparison"] if (old_dir / "numerics_summary.json").is_file() else None
    new_summary = out_dir / "numerics_summary.json"
    if new_summary.is_file():
        comparison["new_summary"] = json.loads(
            new_summary.read_text())["comparison"]
    comparison["replay_s"] = replay_s
    _write(out_dir / "step84_comparison.json", comparison)
    return comparison


def _dispatch(case: dict) -> dict:
    kind = case["kind"]
    if kind == "audit":
        return _case_audit(case)
    if kind == "micro":
        return _case_micro(case)
    if kind in ("real_eager", "real_graph"):
        return _case_real({**case, "graph": kind == "real_graph"})
    if kind == "replay83":
        return _case_replay83(case)
    raise ValueError(f"未知 case kind={kind!r}")


# ---------------------------------------------------------------------------
# 父进程编排
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_dispatch)

    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--micro", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--replay83", action="store_true")
    parser.add_argument("--replay-fp32", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir",
                        default=str(ROOT / "benchmarks/results/step84"))
    parser.add_argument("--timeout-s", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.all:
        args.audit = args.micro = args.real = args.replay83 = True
        args.replay_fp32 = True
    if not (args.audit or args.micro or args.real or args.replay83
            or args.replay_fp32):
        args.audit = args.micro = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git": _git(),
        "package": _package_hash(),
        "gpu": _parent_gpu_info(),
        "target": {"path": str(TARGET)},
        "assistant": {"path": str(ASSISTANT)},
        "args": vars(args),
        "isolation": "one-mode-per-subprocess",
    }
    _write(out_dir / "manifest.json", manifest)

    cases = []
    if args.audit:
        cases.append({"kind": "audit", "label": "audit"})
    if args.micro:
        cases.append({"kind": "micro", "label": "micro"})
    if args.real:
        cases.append({"kind": "real_eager", "label": "real_eager"})
        cases.append({"kind": "real_graph", "label": "real_graph"})
    if args.replay83:
        cases.append({
            "kind": "replay83", "label": "replay83_bf16",
            "dtype": "bfloat16", "blocks": 512,
            "out_dir": str(out_dir / "replay83" / "bf16"),
        })
    if args.replay_fp32:
        cases.append({
            "kind": "replay83", "label": "replay83_fp32",
            "dtype": "float32", "blocks": 64,
            "out_dir": str(out_dir / "replay83" / "fp32"),
        })

    result = run_matrix(
        _worker_argv(), cases, timeout_s=args.timeout_s, cwd=ROOT)
    _write(out_dir / "results.json", result)
    for record in result["cases"]:
        status = record["status"]
        label = record["case"]["label"]
        if status == "ok":
            payload = record["result"]
            print(f"{label}: ok ({payload.get('_case_duration_s', 0):.1f}s)")
            if label == "audit":
                print(
                    "  RoPE tables:",
                    f"{payload['unique_tables']} unique / "
                    f"{payload['per_layer_nominal_mib']:.1f} MiB nominal -> "
                    f"{payload['unique_mib']:.1f} MiB unique")
            elif label.startswith("real"):
                print(
                    "  tokens:", payload.get("tokens"),
                    "tables:", payload["audit"]["unique_tables"],
                    f"({payload['audit']['unique_mib']:.1f} MiB unique)")
            elif label.startswith("replay"):
                print(
                    "  first_diff old/new:", payload.get("old_first_diff"),
                    "/", payload.get("new_first_diff"),
                    "token_changes_k0/k2:",
                    payload.get("token_changes_k0"),
                    "/", payload.get("token_changes_k2"))
        else:
            print(f"{label}: FAILED {record.get('error')}", file=sys.stderr)
    if not result["ok"]:
        print(f"failures: {len(result['failures'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
