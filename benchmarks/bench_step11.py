"""第十一关 CPU 完整 Engine 生命周期；固定随机初始化，三请求九个新 token。"""
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
from unittest.mock import patch

import torch
from torch.utils.benchmark import Timer

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(p) for p in PROJECT.glob("step[0-9][0-9]") if p.is_dir()]+[str(PROJECT)]
import step11

FIXTURE = Path("/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/step11_contract_20260914T133029.365676Z.json")
REQUESTS = [
    {"request_id": "A", "prompt_ids": [0], "max_new_tokens": 3},
    {"request_id": "B", "prompt_ids": [1, 2, 3], "max_new_tokens": 2},
    {"request_id": "C", "prompt_ids": [0, 1, 0], "max_new_tokens": 4},
]
TOKENS = {"A": [2, 2, 0], "B": [3, 0], "C": [0, 2, 2, 2]}
ORDER = {1: "ABC", 2: "BAC", 3: "BAC"}
STEPS = {1: 9, 2: 6, 3: 4}
RNG_STATE = None


def expected(capacity):
    return [{"request_id": name, "output_ids": TOKENS[name]} for name in ORDER[capacity]]


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_static(capacity):
    # 计入 RNG 状态恢复这项测试成本，保证每个 trial 的随机权重完全相同。
    torch.set_rng_state(RNG_STATE)
    engine = step11.Engine(max_num_seqs=capacity, vocab_size=5, d_model=8, max_seq_len=32)
    for request in REQUESTS:
        engine.add_request(request)
    completed = []
    while engine.has_unfinished_requests():
        completed.extend(engine.step())
    return completed


def check_workload(capacity, fixture, canonical_weight_hash):
    history, sampled, steps, weights = [], [], [], []
    before = copy.deepcopy(REQUESTS)
    original_init = step11.TinyCausalLM.__init__
    original_forward = step11.TinyCausalLM.forward
    original_sample = step11.Sampler.sample
    original_step = step11.Engine.step

    def init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        weights.append(parameter_hash(model))

    def forward(model, ids, past_kv=None):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not model.training
        history.append(ids.tolist())
        assert len(history) <= STEPS[capacity], "额外 forward"
        assert ids.dtype == torch.long and ids.device.type == "cpu"
        logits, present = original_forward(model, ids, past_kv=past_kv)
        assert not logits.requires_grad and all(not x.requires_grad for x in present)
        return logits, present

    def sample(sampler, logits):
        ids = original_sample(sampler, logits)
        sampled.append(ids.tolist())
        return ids

    def step(engine):
        assert len(steps) < STEPS[capacity], "额外 step 或无法结束"
        f, s = len(history), len(sampled)
        result = original_step(engine)
        steps.append([len(history) - f, len(sampled) - s])
        return result

    with patch.object(step11.TinyCausalLM, "__init__", init), \
         patch.object(step11.TinyCausalLM, "forward", forward), \
         patch.object(step11.Sampler, "sample", sample), \
         patch.object(step11.Engine, "step", step):
        result = run_static(capacity)
    assert REQUESTS == before and result == expected(capacity)
    assert weights == [canonical_weight_hash], "必须只初始化一次，并使用固定参数"
    assert steps == [[1, 1]] * STEPS[capacity]
    assert history == [r["ids"] for r in fixture["forwards"]]
    assert sampled == [r["tokens"] for r in fixture["samples"]]
    return {"outputs": result, "model_inputs": history, "sampled_tokens": sampled,
            "forward_sample_calls_per_step": steps, "initialization_count": len(weights),
            "parameter_sha256": weights[0], "processed_input_positions_including_padding":
            sum(len(row) for batch in history for row in batch)}


def main():
    global RNG_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, choices=[1], default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    RNG_STATE = torch.get_rng_state().clone()
    source = PROJECT / "step11" / f"step11.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical_weight_hash = parameter_hash(step11.TinyCausalLM(vocab_size=5, d_model=8, max_seq_len=32))
    accepted = json.loads(FIXTURE.read_text())
    assert accepted["status"] == "passed"
    fixture = next(c for c in accepted["cases"] if c["name"] == f"unequal_lengths_capacity_{args.capacity}")
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "source_sha256": digest,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": str(FIXTURE), "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32", "num_threads": 1,
        "seed": 0, "rng_state_sha256": hashlib.sha256(RNG_STATE.numpy().tobytes()).hexdigest(),
        "parameter_sha256": canonical_weight_hash, "callback": None, "max_num_seqs": args.capacity,
        "model_config": {"vocab_size": 5, "d_model": 8, "max_seq_len": 32},
        "requests": REQUESTS, "expected_new_tokens": 9, "completed_requests": 3,
        "boundary": "restore CPU RNG state + new Engine/model (random initialization and eval) + submit + query/step loop (including input preparation, inference_mode entry, forward, KV concatenation/storage/cleanup, sample, writeback) + collect results; excludes imports, input fixtures, reference/tracing/checks",
        "comparison_scope": "same boundary and workload as step09 capacity=1; includes cache construction/concatenation/cleanup; not comparable with model-only timings",
        "min_run_time_s": 1.0,
    }
    try:
        report["workload_check"] = check_workload(args.capacity, fixture, canonical_weight_hash)
    except Exception as exc:
        report.update(status="correctness_failed", error=f"{type(exc).__name__}: {exc}")
        print("负载检查失败，不测速：", report["error"])
        code = 1
    else:
        measurement = Timer(stmt="run_static(capacity)", globals={"run_static": run_static, "capacity": args.capacity},
                            num_threads=1, label="step11: CPU real-model Engine lifecycle",
                            description=f"capacity={args.capacity}, 3 requests, 9 new tokens"
                            ).blocked_autorange(min_run_time=1.0)
        report["post_check"] = check_workload(args.capacity, fixture, canonical_weight_hash)
        report.update(status="measured", median_us=measurement.median * 1e6, iqr_us=measurement.iqr * 1e6,
                      number_per_run=measurement.number_per_run, raw_times_s=measurement.raw_times,
                      has_warnings=measurement.has_warnings)
        print(measurement)
        code = 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, "测量期间源码变化"
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = directory / f"step11_cap{args.capacity}_{stamp}_{os.getpid()}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved:", target)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
