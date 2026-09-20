"""跑完第三十六关的六个测点：两个引擎 × 3 个独立进程 × (2 次预热 + 5 次测量)。"""
import json, os, pathlib, subprocess, sys, time

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BENCH = PROJECT / "benchmarks" / "bench_step36_vllm_compare.py"
OUT = PROJECT / "benchmarks" / "results"
CASES = ["short_c1", "short_c8", "prefill_c1", "prefill_c8", "decode_c1", "decode_c8"]
PROCS = 3

def one(engine, case, proc):
    env = dict(os.environ)
    if engine == "vllm":
        # 本机 WSL2 没有 vLLM 认定的 pinned memory，V2 runner 要求 UVA；
        # 用 V1 model runner 走通（记录在报告里）
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    out = OUT / f"step36_{engine}_{case}_p{proc}.json"
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(BENCH), "--engine", engine, "--case", case,
                        "--warmup", "2", "--reps", "5", "--out", str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not out.is_file():
        print(f"  ✗ {engine:4} {case:11} p{proc} 失败 ({time.perf_counter()-t0:.0f}s)")
        print("    " + (r.stderr.strip().splitlines() or ["(无 stderr)"])[-1][:160])
        return None
    d = json.loads(out.read_text())
    print(f"  ✓ {engine:4} {case:11} p{proc} {d['median_s']:.4f}s "
          f"({d['throughput_tok_s']:.0f} tok/s) iqr/med={d['iqr_over_median']:.2f} "
          f"load={d['load_s']:.1f}s [{time.perf_counter()-t0:.0f}s]")
    return d

def main():
    which = sys.argv[1:] or ["mine", "vllm"]
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case in CASES:
        for engine in which:
            for p in range(PROCS):
                d = one(engine, case, p)
                if d: all_rows.append(d)
    (OUT / "step36_matrix_raw.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n")
    print(f"\n原始数据 -> {OUT/'step36_matrix_raw.json'}（{len(all_rows)} 条）")

if __name__ == "__main__":
    main()
