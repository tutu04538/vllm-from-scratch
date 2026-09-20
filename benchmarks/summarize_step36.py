"""把 step36 的原始结果聚合成对照表。

    python benchmarks/summarize_step36.py

读 benchmarks/results/step36_{engine}_{case}_p{n}.json（每个引擎每个测点多次独立进程），
按测点给出：中位耗时、输出吞吐、进程间离散度。显存一栏只取我们自己引擎的
进程内数字；vLLM 的数字见 step36_mem_breakdown.py 的说明（父进程看不到子进程的
显存，原矩阵里的 `gpu_used_after_gib` 对 vLLM 无效）。
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
CASES = ["short_c1", "short_c8", "prefill_c1", "prefill_c8", "decode_c1", "decode_c8"]


def load(engine, case, tag="v2"):
    # tag 把两版数据分开：首版（无 tag）有两个配置错误，不能和新版混在一起。
    prefix = f"step36_{tag}_" if tag else "step36_"
    out = []
    for p in sorted(RESULTS.glob(f"{prefix}{engine}_{case}_p*.json")):
        out.append(json.loads(p.read_text()))
    return out


def main():
    tag = "v2"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    raw = {}
    header = f"{'测点':<12}{'引擎':<7}{'进程':>4}{'中位耗时':>11}{'吞吐 tok/s':>12}{'进程间离散':>11}{'工作量':>9}{'显存':>10}"
    print(header)
    print("-" * len(header.encode("gbk", "ignore")))
    for case in CASES:
        row = {}
        for engine in ("mine", "vllm"):
            runs = load(engine, case, tag)
            if not runs:
                print(f"{case:<12}{engine:<7}  （未运行）")
                continue
            meds = [r["median_s"] for r in runs]
            grand = statistics.median(meds)
            thr = [r["throughput_tok_s"] for r in runs]
            spread = (max(meds) - min(meds)) / grand
            tokens = runs[0]["output_tokens"]
            gpu = runs[0]["gpu_used_after_gib"]
            gpu_s = f"{gpu:.2f} GiB" + ("*" if engine == "vllm" else "")
            print(f"{case:<12}{engine:<7}{len(runs):>4}{grand:>10.4f}s"
                  f"{statistics.median(thr):>12.1f}{spread:>10.1%}{tokens:>9}{gpu_s:>10}")
            row[engine] = dict(median_s=grand, throughput=statistics.median(thr),
                               runs=len(runs), tokens=tokens, spread=spread,
                               kv=runs[0]["kv"], load_s=runs[0]["load_s"])
        if "mine" in row and "vllm" in row:
            ratio = row["mine"]["median_s"] / row["vllm"]["median_s"]
            print(f"{'':<12}{'→ 慢':<7}{ratio:>10.2f}x\n")
        raw[case] = row
    print("* vLLM 的显存值无效：父进程 mem_get_info 看不到 EngineCore 子进程的分配，"
          "该列只是 WSL 基线 1.312 GiB。修正后的数字见 step36_mem_breakdown.py。")
    (RESULTS / f"step36_{tag}_summary.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
