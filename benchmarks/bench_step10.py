"""CPU KV cache 对照：同权重、同 token 序列；分别测 prefill/decode_chain/total。

不执行 Engine，不采样；后续 token 固定，用于避免 EOS 改变工作量。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import step09
import step10


def param_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, choices=[3, 24], default=24)
    parser.add_argument("--decode-steps", type=int, choices=[4], default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sources = {name: hashlib.sha256((PROJECT / f"{name}.py").read_bytes()).hexdigest()
               for name in ("step09", "step10")}
    full = step09.TinyCausalLM(vocab_size=5, d_model=8, max_seq_len=32).eval()
    cached = step10.TinyCausalLM(vocab_size=5, d_model=8, max_seq_len=32).eval()
    cached.load_state_dict(full.state_dict(), strict=True)
    weight_hash = param_hash(full)
    assert param_hash(cached) == weight_hash
    p, d = args.prompt_len, args.decode_steps
    all_ids = torch.arange(p + d, dtype=torch.long).unsqueeze(0) % 5
    prompt = all_ids[:, :p].clone()
    new_tokens = [all_ids[:, p+i:p+i+1].clone() for i in range(d)]
    prefixes = [all_ids[:, :p+i+1].clone() for i in range(d)]
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "sources": {name: {"path": str(PROJECT/f"{name}.py"), "sha256": digest} for name,digest in sources.items()},
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32", "num_threads": 1,
        "seed": 0, "parameter_sha256": weight_hash, "model_config": {"vocab_size": 5, "d_model": 8, "max_seq_len": 32},
        "prompt_ids": prompt.tolist(), "new_tokens": [x.item() for x in new_tokens],
        "prompt_len": p, "decode_steps": d, "sampling": False, "eos_stopping": False,
        "min_run_time_s": 1.0,
        "boundaries": {
            "common": "initialized models, prebuilt input tensors, eval and existing inference_mode context; excludes Engine, sampling, initialization, input fixtures and checks",
            "prefill": "one prompt forward; cached variant returns initial K/V, full variant returns logits",
            "decode_chain": "four subsequent forwards; cached starts from prebuilt immutable prompt K/V, allocates concatenated K/V each step; full recomputes each complete prefix; excludes initial prefill",
            "total": "prompt forward + four subsequent forwards; cached builds fresh K/V every trial and includes all concatenations; no prebuilt cache used",
        },
    }
    with torch.inference_mode():
        initial_logits, initial_kv = cached(prompt)
        kv_snapshot = tuple(x.clone() for x in initial_kv)
        kv = initial_kv
        errors = [(initial_logits - full(prompt)).abs().max().item()]
        trace = {"full": [], "cached": []}
        handles = []
        for mode, model in (("full", full), ("cached", cached)):
            def hook(layer, inputs, mode=mode):
                trace[mode].append(list(inputs[0].shape))
            handles.append(model.k_proj.register_forward_pre_hook(hook))
        try:
            for new, prefix in zip(new_tokens, prefixes):
                got, kv = cached(new, past_kv=kv)
                want = full(prefix)[:, -1:, :]
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
                errors.append((got-want).abs().max().item())
        finally:
            for h in handles: h.remove()
        torch.testing.assert_close(initial_logits, full(prompt), atol=1e-5, rtol=1e-5)
        assert trace["cached"] == [[1, 1, 8]] * d
        assert trace["full"] == [[1, p+i+1, 8] for i in range(d)]
        assert all(torch.equal(x,y) for x,y in zip(initial_kv,kv_snapshot))
        report["workload_check"] = {"max_abs_errors": errors, "decode_k_projection_input_shapes": trace,
                                    "projected_positions_total": {"full": sum(range(p,p+d+1)), "cached": p+d},
                                    "initial_cache_unchanged": True}

        def full_prefill():
            return full(prompt)
        def cached_prefill():
            return cached(prompt)
        def full_decode():
            for ids in prefixes:
                logits = full(ids)
            return logits
        def cached_decode():
            past = initial_kv
            for ids in new_tokens:
                logits, past = cached(ids, past_kv=past)
            return logits, past
        def full_total():
            logits = full(prompt)
            for ids in prefixes:
                logits = full(ids)
            return logits
        def cached_total():
            logits, past = cached(prompt)
            for ids in new_tokens:
                logits, past = cached(ids, past_kv=past)
            return logits, past

        methods = {"prefill": {"full": full_prefill, "cached": cached_prefill},
                   "decode_chain": {"full": full_decode, "cached": cached_decode},
                   "total": {"full": full_total, "cached": cached_total}}
        measurements = []
        for phase, modes in methods.items():
            for mode, fn in modes.items():
                m = Timer(stmt="fn()", globals={"fn": fn}, num_threads=1,
                          label=f"step10: {mode} {phase}", description=f"P={p}, D={d}, CPU"
                          ).blocked_autorange(min_run_time=1.0)
                measurements.append({"phase": phase, "mode": mode, "median_us": m.median*1e6,
                                     "iqr_us": m.iqr*1e6, "has_warnings": m.has_warnings,
                                     "number_per_run": m.number_per_run, "raw_times_s": m.raw_times})
                print(m)
        for cf, ff in ((cached_decode, full_decode), (cached_total, full_total)):
            logits, kv = cf()
            torch.testing.assert_close(logits, ff()[:, -1:, :], atol=1e-5, rtol=1e-5)
            assert kv[0].shape == kv[1].shape == (1, p+d, 8)
        assert all(torch.equal(x,y) for x,y in zip(initial_kv,kv_snapshot))
        assert param_hash(cached) == param_hash(full) == weight_hash
        report.update(status="measured", measurements=measurements)
    for name,digest in sources.items():
        assert hashlib.sha256((PROJECT/f"{name}.py").read_bytes()).hexdigest()==digest, "测量期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"step10_P{p}_D{d}_{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print("Saved:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
