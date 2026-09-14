"""第八关：已初始化 TinyCausalLM 的 CPU eager forward 基线，不运行 Engine。"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/tools")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(REFERENCE))
import step08
from verify_step08_contract import reference


def check_workload(model, input_ids):
    before = input_ids.clone()
    state = copy.deepcopy(model.state_dict())
    identities = {name: id(p) for name, p in model.named_parameters()}
    for name in ("q_proj", "k_proj", "v_proj", "lm_head"):
        assert getattr(model, name).bias is None
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    actual = model(input_ids)
    expected = reference(model, input_ids)
    assert actual.shape == (*input_ids.shape, 5) and actual.dtype == torch.float32
    assert not actual.requires_grad and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual, model(input_ids))
    assert torch.equal(input_ids, before)
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    return {"input_ids": input_ids.tolist(), "output_shape": list(actual.shape),
            "max_abs_error": (actual.double() - expected).abs().max().item(),
            "parameters_reused": True, "inputs_unchanged": True, "requires_grad": actual.requires_grad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seq-len", type=int, choices=[1, 3, 32], default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = PROJECT / "step08.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = step08.TinyCausalLM(vocab_size=5, d_model=8, max_seq_len=32).eval()
    input_ids = torch.arange(args.batch_size * args.seq_len, dtype=torch.long).reshape(
        args.batch_size, args.seq_len) % 5
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": digest,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_source": str(REFERENCE / "verify_step08_contract.py"),
        "reference_sha256": hashlib.sha256((REFERENCE / "verify_step08_contract.py").read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "pid": os.getpid(), "device": "cpu", "dtype": "float32",
        "num_threads": 1, "seed": 0, "inference_mode": True, "eval": True,
        "batch_size": args.batch_size, "seq_len": args.seq_len,
        "model_config": {"vocab_size": 5, "d_model": 8, "max_seq_len": 32},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "boundary": "model(input_ids) in an existing inference_mode context; includes Module call, embeddings, positions/mask construction, attention and lm_head; excludes initialization, imports, input fixtures, correctness checks and Engine",
        "comparison_scope": "new forward-only baseline, not comparable to previous Engine lifecycle timings",
        "min_run_time_s": 1.0,
    }
    with torch.inference_mode():
        try:
            report["workload_check"] = check_workload(model, input_ids)
        except Exception as exc:
            report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
            print("负载检查失败，不测速：", report["error"])
            code = 1
        else:
            measurement = Timer(stmt="model(input_ids)", globals={"model": model, "input_ids": input_ids},
                                num_threads=1, label="step08: initialized CPU TinyCausalLM forward",
                                description=f"B={args.batch_size}, T={args.seq_len}, D=8, V=5"
                                ).blocked_autorange(min_run_time=1.0)
            report["post_check"] = check_workload(model, input_ids)
            report.update(status="measured", median_us=measurement.median * 1e6,
                          iqr_us=measurement.iqr * 1e6, number_per_run=measurement.number_per_run,
                          raw_times_s=measurement.raw_times, has_warnings=measurement.has_warnings)
            print(measurement)
            code = 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, "测量期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"step08_B{args.batch_size}_T{args.seq_len}_{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved:", path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
