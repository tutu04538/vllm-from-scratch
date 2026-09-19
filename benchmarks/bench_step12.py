"""第十二关：同权重、固定旧缓存、单轮 decode，逐请求与批量 CPU 对照。"""
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
sys.path[:0]=[str(p) for p in PROJECT.glob("step[0-9][0-9]") if p.is_dir()]+[str(PROJECT)]
import step11
import step12

RECORDS = Path("/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录")
ACCEPTANCE = RECORDS / "step12_contract_20260914T151639.421994Z.json"
REGRESSION = RECORDS / "step11_contract_20260914T151641.440164Z.json"
LENGTHS = {1: [3], 2: [3, 1], 3: [24, 3, 1]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--order", choices=["serial-first", "batch-first"], default="serial-first")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = {name: {"path": str(PROJECT / f"{name}.py"), "sha256": sha(PROJECT / f"{name}.py")}
               for name in ("step11", "step12")}
    for name, path in (("step11", REGRESSION), ("step12", ACCEPTANCE)):
        fixture = json.loads(path.read_text())
        assert fixture["status"] == "passed", "先通过正确性验收"
        assert fixture["source_sha256"] == sources[name]["sha256"], "验收后源码变化，请先重新验收"
    torch.manual_seed(0)
    serial_model = step11.TinyCausalLM().eval()
    batch_model = step12.TinyCausalLM().eval()
    batch_model.load_state_dict(serial_model.state_dict(), strict=True)
    weight_hash = parameter_hash(serial_model)
    assert parameter_hash(batch_model) == weight_hash
    lengths = LENGTHS[args.batch_size]
    prompts = [((torch.arange(p) + i) % 5).reshape(1, p) for i, p in enumerate(lengths)]
    ids = torch.tensor([[(i + 1) % 5] for i in range(len(lengths))], dtype=torch.long)
    per_row_ids = [ids[i:i+1].clone() for i in range(len(lengths))]
    started = datetime.now(timezone.utc).isoformat()

    with torch.inference_mode():
        caches = [serial_model(prompt)[1] for prompt in prompts]
        snapshots = [(k.clone(), v.clone()) for k, v in caches]
        ids_snapshot = ids.clone()

        def serial_decode():
            # 返回列表即可；不人为给旧版添加拼接 logits 的开销。
            return [serial_model(row, past_kv=cache) for row, cache in zip(per_row_ids, caches)]

        def batch_decode():
            return batch_model.forward_decode(ids, caches)

        def check():
            serial = serial_decode()
            logits, present = batch_decode()
            assert logits.shape == (len(lengths), 1, 5)
            errors = []
            for i, (expected_logits, expected_kv) in enumerate(serial):
                torch.testing.assert_close(logits[i:i+1], expected_logits, atol=1e-5, rtol=1e-5)
                errors.append((logits[i:i+1] - expected_logits).abs().max().item())
                for actual, expected in zip(present[i], expected_kv):
                    assert actual.shape == (1, lengths[i]+1, 8)
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                assert not logits.requires_grad and all(not x.requires_grad for x in present[i])
            assert all(torch.equal(x, old) for kv, snap in zip(caches, snapshots) for x, old in zip(kv, snap))
            assert torch.equal(ids, ids_snapshot) and all(torch.equal(row, ids[i:i+1]) for i, row in enumerate(per_row_ids))
            assert parameter_hash(serial_model) == parameter_hash(batch_model) == weight_hash
            return {"logits_max_abs_errors": errors, "cache_lengths": [kv[0].shape[1] for kv in present],
                    "old_caches_and_inputs_unchanged": True, "parameters_unchanged": True}

        trace, handles = {}, []
        for mode, model in (("serial", serial_model), ("batch", batch_model)):
            for name in ("q_proj", "k_proj", "v_proj"):
                def hook(layer, inputs, mode=mode, name=name):
                    trace.setdefault(mode, {}).setdefault(name, []).append(list(inputs[0].shape))
                handles.append(getattr(model, name).register_forward_pre_hook(hook))
        try:
            before = check()
        finally:
            for handle in handles:
                handle.remove()
        for name in ("q_proj", "k_proj", "v_proj"):
            assert trace["serial"][name] == [[1, 1, 8]] * len(lengths)
            assert trace["batch"][name] == [[len(lengths), 1, 8]]

        functions = {"serial": serial_decode, "batch": batch_decode}
        order = ["serial", "batch"] if args.order == "serial-first" else ["batch", "serial"]
        measurements = []
        for mode in order:
            m = Timer(stmt="fn()", globals={"fn": functions[mode]}, num_threads=1,
                      label=f"step12 {mode} single decode", description=f"past lengths={lengths}, CPU"
                      ).blocked_autorange(min_run_time=1.0)
            measurements.append({"mode": mode, "median_us": m.median * 1e6, "iqr_us": m.iqr * 1e6,
                "has_warnings": m.has_warnings, "number_per_run": m.number_per_run, "raw_times_s": m.raw_times})
            print(m)
        after = check()
        assert before == after

    assert all(sha(Path(info["path"])) == info["sha256"] for info in sources.values())
    report = {
        "status": "measured", "started_at_utc": started, "pid": os.getpid(), "sources": sources,
        "acceptance": str(ACCEPTANCE), "acceptance_sha256": sha(ACCEPTANCE),
        "regression": str(REGRESSION), "regression_sha256": sha(REGRESSION),
        "benchmark_sha256": sha(Path(__file__)), "parameter_sha256": weight_hash,
        "python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
        "platform": platform.platform(), "device": "cpu", "dtype": "float32", "num_threads": 1,
        "seed": 0, "model_config": {"vocab_size": 5, "d_model": 8, "max_seq_len": 32},
        "batch_size": len(lengths), "past_lengths": lengths,
        "prompts": [p.tolist() for p in prompts], "input_ids": ids.tolist(),
        "boundary": "one decode round for all requests, initialized eval models in existing inference_mode; fixed prebuilt inputs and immutable prefill caches reused every trial; includes model calls, batch cache concatenation/padding/mask/output splitting and serial Python loop/result list; excludes model initialization, prefill, Engine, sampling, hooks and correctness checks; serial logits are not concatenated for timing",
        "min_run_time_s": 1.0, "measurement_order": order, "projection_trace": trace,
        "before_check": before, "after_check": after, "measurements": measurements,
    }
    directory = PROJECT / "benchmarks/results"
    directory.mkdir(exist_ok=True, parents=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"step12_B{len(lengths)}_{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved:", path)


if __name__ == "__main__":
    main()
