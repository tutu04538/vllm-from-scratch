"""Paired legacy/constexpr-only/optimized/vLLM attention diagnosis.

Graph-batched CUDA events remove Python submission gaps. This is a warm-cache
kernel benchmark, NOT model tok/s. Baseline sources come from our own Git commit;
no upstream production kernel is copied into minivllm. Set MLIR_ENABLE_DUMP=1
and TRITON_ALWAYS_COMPILE=1 with --variant/--cases to obtain per-pass IR.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import importlib.metadata
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]


def load_variant(name, source, directory):
    source = source.replace("from ...triton_utils import tl, triton",
                            "import triton\nimport triton.language as tl")
    path = directory / f"step88_async_{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def compiled_record(entry, prefix):
    for ext in ("ttir", "ttgir", "ptx"):
        prefix.with_suffix("." + ext).write_text(entry.asm[ext])
    copies = [line for line in entry.asm["ttgir"].splitlines()
              if "async_copy_global_to_local" in line]
    return dict(n_regs=entry.n_regs, n_spills=entry.n_spills,
                shared_bytes=entry.metadata.shared,
                num_stages=entry.metadata.num_stages,
                async_copy=len(copies),
                bf16_async_copy=sum("!tt.ptr<bf16>" in line for line in copies),
                cp_async=entry.asm["ptx"].count("cp.async"))


def main():
    import torch
    from triton.runtime.errors import OutOfResources
    from minivllm.attention.ops import triton_unified_attention as optimized
    from vllm.v1.attention.ops import triton_unified_attention as upstream
    from check_step88_attention_micro import _prepare, case_matrix
    from check_step88_attention_pipeline import _vllm_run

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline-ref", default="bb1380d")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--variant", choices=("baseline", "constexpr_only", "optimized", "vllm"))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--output-dir", default=str(ROOT / "benchmarks/results/step88/async_stride"))
    a = p.parse_args()
    assert a.warmup >= 1 and a.calls >= 1 and a.rounds >= 1
    outdir = Path(a.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rel = "minivllm/attention/ops/triton_unified_attention.py"
    original = subprocess.check_output(["git", "show", f"{a.baseline_ref}:{rel}"], cwd=ROOT, text=True)
    constants = original.replace("stride_k_cache_3: tl.int64", "stride_k_cache_3: tl.constexpr").replace(
        "stride_v_cache_3: tl.int64", "stride_v_cache_3: tl.constexpr")
    modules = {"baseline": load_variant("baseline", original, outdir),
               "constexpr_only": load_variant("constexpr_only", constants, outdir),
               "optimized": optimized, "vllm": upstream}
    if a.variant:
        modules = {a.variant: modules[a.variant]}
    cases = [c for c in case_matrix() if not a.cases or c["label"] in a.cases]
    if a.cases and set(a.cases) != {c['label'] for c in cases}:
        raise ValueError("unknown case label")
    rows = []
    manifest = dict(args=vars(a), device=torch.cuda.get_device_name(),
                    recorded_at=datetime.datetime.now().astimezone().isoformat(),
                    capability=torch.cuda.get_device_capability(),
                    baseline_ref=a.baseline_ref,
                    baseline_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    optimized_sha256=hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(),
                    metric="CUDA graph burst, warm-cache ms/kernel; no engine",
                    torch=torch.__version__,
                    triton=importlib.metadata.version("triton"),
                    vllm=importlib.metadata.version("vllm"))
    for c in cases:
        prepared = _prepare(c)
        row = dict(case=c, variants={})
        functions, outputs, graphs = {}, {}, {}
        artifact_dir = outdir / c["label"]
        artifact_dir.mkdir(exist_ok=True)
        for name, module in modules.items():
            output = torch.empty_like(prepared["q"])
            outputs[name] = output
            if name == "vllm":
                output = prepared["out_vllm"]
                outputs[name] = output
                fn = lambda: _vllm_run(c, prepared)
            else:
                def fn(module=module, output=output):
                    return module.unified_attention(
                        prepared["q"], prepared["k"], prepared["v"], output,
                        prepared["cu"], prepared["seq_lens"], prepared["block_table"],
                        softmax_scale=prepared["scale"], causal=True,
                        window_size=prepared["window"], max_seqlen_q=c["q_len"])
            # Capture THIS launch's CompiledKernel, not the first cache variant.
            original_run = module.kernel_unified_attention.run
            entries = []
            def record(*args, _run=original_run, **kwargs):
                entry = _run(*args, **kwargs)
                entries.append(entry)
                return entry
            try:
                module.kernel_unified_attention.run = record
                fn()
            except OutOfResources as exc:
                if name != "vllm":
                    raise
                # The installed upstream has a known hd512+SWA shared-memory
                # limit on this GPU. Retain the failure; never call it a pass.
                row["variants"][name] = {"unsupported": str(exc)}
                del outputs[name]
                continue
            finally:
                module.kernel_unified_attention.run = original_run
            row["variants"][name] = compiled_record(entries[-1], artifact_dir / name)
            functions[name] = fn
        torch.cuda.synchronize()
        if "baseline" in outputs:
            for name, value in outputs.items():
                row["variants"][name]["max_abs_vs_baseline"] = float(
                    (value.float() - outputs["baseline"].float()).abs().max())
        for name, fn in functions.items():
            for _ in range(a.warmup):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(a.calls):
                    fn()
            graph.replay()
            graphs[name] = graph
        torch.cuda.synchronize()
        timings = {name: [] for name in functions}
        for r in range(a.rounds):
            for name in list(functions)[::1 if r % 2 == 0 else -1]:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) / a.calls)
        for name, values in timings.items():
            row["variants"][name].update(median_ms=statistics.median(values), values_ms=values)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (outdir / "summary.json").write_text(json.dumps(dict(manifest=manifest, cases=rows), indent=2))
        graphs.clear()
    failures = [c['case']['label'] for c in rows if 'optimized' in c['variants']
                and c['variants']['optimized'].get('max_abs_vs_baseline', 0) != 0]
    if failures:
        raise AssertionError(f"nonzero optimized/baseline difference: {failures}")


if __name__ == "__main__":
    main()
