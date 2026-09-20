"""跑六个测点：两个引擎 × 3 个独立进程 × (2 次预热 + 5 次测量)。

    python benchmarks/run_step36_matrix.py                 # 两个引擎，tag=v2
    python benchmarks/run_step36_matrix.py mine            # 只跑我们的引擎

输出写到 `step36_<tag>_<engine>_<case>_p<n>.json`。

**tag 是必须的。** 首版结果（`step36_<engine>_...`，无 tag）已经被验收方记录了 sha256，
不能被覆盖；而且首版有两个配置错误（漏传 norm_backend、max_model_len 不一致），
两组数据必须能分开，不能混成一个「新基线」。

同一进程序号内**交替两个引擎**（p0 先 mine，p1 先 vllm），避免「同一引擎连跑 3 次」
带来的顺序偏差。
"""
import json, os, pathlib, subprocess, sys, time

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BENCH = PROJECT / "benchmarks" / "bench_step36_vllm_compare.py"
OUT = PROJECT / "benchmarks" / "results"
CASES = ["short_c1", "short_c8", "prefill_c1", "prefill_c8", "decode_c1", "decode_c8"]
PROCS = 3


def one(engine, case, proc, tag):
    env = dict(os.environ)
    if engine == "vllm":
        # 本机 WSL2 没有 vLLM 认定的 pinned memory，V2 runner 要求 UVA；
        # 用 V1 model runner 走通（记录在报告里）
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    out = OUT / f"step36_{tag}_{engine}_{case}_p{proc}.json"
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(BENCH), "--engine", engine, "--case", case,
                        "--warmup", "2", "--reps", "5", "--out", str(out)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not out.is_file():
        print(f"  ✗ {engine:4} {case:11} p{proc} 失败 ({time.perf_counter()-t0:.0f}s)")
        print("    " + (r.stderr.strip().splitlines() or ["(无 stderr)"])[-1][:160])
        return None
    d = json.loads(out.read_text())
    rt = d.get("runtime", {})
    extra = f"norm={rt.get('norm_backend')}" if engine == "mine" else ""
    print(f"  ✓ {engine:4} {case:11} p{proc} {d['median_s']:.4f}s "
          f"({d['throughput_tok_s']:.0f} tok/s) iqr/med={d['iqr_over_median']:.2f} "
          f"{extra} [{time.perf_counter()-t0:.0f}s]")
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "v2"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    which = args or ["mine", "vllm"]
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for case in CASES:
        for p in range(PROCS):
            # 交替顺序：p0 按给定顺序，p1 反过来，p2 再正过来
            order = which if p % 2 == 0 else list(reversed(which))
            for engine in order:
                d = one(engine, case, p, tag)
                if d:
                    d["tag"] = tag
                    all_rows.append(d)
    path = OUT / f"step36_{tag}_matrix_raw.json"
    path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n")
    print(f"\n原始数据 -> {path}（{len(all_rows)} 条）")


if __name__ == "__main__":
    main()
