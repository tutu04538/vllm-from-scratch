"""Step88 real-model A/B, in separate processes (one variant per invocation).

--variant baseline restores only bb1380d's attention kernel and launcher in this
process, never repository files. --compare-attention compares every eager actual
attention call with the baseline on identical Q/K/V, including prefill/SWA.
Do not interpret timings with --compare-attention as performance measurements.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]


def main():
    import torch
    from check_step88_async_stride import load_variant
    from compare_step88_execution import _run_mini_case
    from step88_common import make_case, package_hash
    from minivllm.attention.backends import triton_attn as backend

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variant", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--baseline-ref", default="bb1380d")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--compare-attention", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmarks/results/step88/async_stride/model")
    args = parser.parse_args()
    if args.compare_attention and (not args.eager or args.variant != "optimized"):
        parser.error("--compare-attention requires --eager and --variant optimized")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output([
        "git", "show", f"{args.baseline_ref}:minivllm/attention/ops/triton_unified_attention.py"
    ], cwd=ROOT, text=True)
    baseline = load_variant("model_baseline", source, args.output_dir)
    original = backend.unified_attention
    stats = {"calls": 0, "different_calls": 0, "max_abs": 0.0, "shapes": {}}

    def compare(q, k, v, out, cu, seq, table, **kw):
        original(q, k, v, out, cu, seq, table, **kw)
        reference = torch.empty_like(out)
        baseline.unified_attention(q, k, v, reference, cu, seq, table, **kw)
        difference = (out.float() - reference.float()).abs().max().item()
        equal = torch.equal(out, reference)
        key = f"q={tuple(q.shape)},window={kw.get('window_size')}"
        item = stats["shapes"].setdefault(key, {"calls": 0, "different_calls": 0, "max_abs": 0.0})
        for row in (stats, item):
            row["calls"] += 1
            row["different_calls"] += int(not equal)
            row["max_abs"] = max(row["max_abs"], difference)
        return out

    backend.unified_attention = (
        baseline.unified_attention if args.variant == "baseline" else
        compare if args.compare_attention else original)
    case = make_case("minivllm", 1, 2, not args.eager, "context")
    case["args"].update(repeats=args.repeats, max_new_tokens=args.max_new_tokens)
    try:
        result = _run_mini_case(case)
    finally:
        backend.unified_attention = original
    result["attention_diagnostic"] = {
        "variant": args.variant, "baseline_ref": args.baseline_ref,
        "other_package": package_hash(), "compare_attention": args.compare_attention,
        "performance_valid": not args.compare_attention,
        "equivalence": stats if args.compare_attention else None,
    }
    suffix = "_opcheck" if args.compare_attention else ""
    path = args.output_dir / f"{args.variant}{suffix}.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)
    print(json.dumps(stats if args.compare_attention else result["output_tokens_per_s"], indent=2))
    if args.compare_attention:
        assert stats["calls"] > 0 and stats["different_calls"] == 0, stats


if __name__ == "__main__":
    main()
