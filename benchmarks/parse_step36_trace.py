"""解析 profiler trace，回答「时间花在哪」：GPU 忙多久、发射多少次、哪些 kernel 占大头。

我们的引擎用 torch.profiler 的 key_averages；vLLM 的 trace 是 chrome trace JSON，
两者的字段不同，这里分开处理，输出同一套口径：

    wall       —— 这次负载的墙钟时间
    gpu_busy   —— GPU kernel 上实际花的时间（自耗时累加）
    busy_ratio —— gpu_busy / wall，低于 1 就说明 GPU 有空隙
    发射次数    —— kernel 发射了多少次（CPU 侧提交压力）
"""

import argparse
import json
import pathlib
from collections import Counter


def _read_json(path):
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def merge_union(intervals):
    """区间并集长度。多流并行时 kernel 会重叠，直接累加时长会高估 GPU 忙碌时间；
    正确做法是把时间区间求并。复核 §5 第 3 条要求这一点。"""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:          # 与当前区间不相交，收尾并起新的一段
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:                    # 相交或相接，合并
            cur_hi = max(cur_hi, hi)
    return total + (cur_hi - cur_lo)


def from_chrome_trace(path, wall=None):
    data = _read_json(path)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    by_name = Counter()
    calls = Counter()
    total = 0.0
    intervals = []
    for e in kernels:
        dur = e.get("dur", 0)
        total += dur
        name = e.get("name", "?")
        by_name[name] += dur
        calls[name] += 1
        ts = e.get("ts")
        if ts is not None:
            intervals.append((ts, ts + dur))
    union = merge_union(intervals)
    span = (max((hi for _, hi in intervals), default=0)
            - min((lo for lo, _ in intervals), default=0))
    return dict(source=str(path), gpu_kernels=len(kernels), distinct=len(by_name),
                sum_kernel_us=round(total, 1),
                gpu_busy_union_us=round(union, 1),
                overlap_ratio=round(total / union, 4) if union else None,
                gpu_span_us=round(span, 1),
                wall_s=wall, busy_over_wall=(round(union / 1e6 / wall, 3) if wall else None),
                top=[dict(name=n[:64], us=round(t, 1), calls=calls[n])
                     for n, t in by_name.most_common(12)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--wall", type=float, default=None, help="对应负载的墙钟秒数")
    args = ap.parse_args()
    r = from_chrome_trace(args.trace, args.wall)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r["wall_s"]:
        print(f"\n  GPU 忙（区间并集）{r['gpu_busy_union_us']/1e3:.2f} ms / 墙钟 "
              f"{r['wall_s']*1e3:.2f} ms = {r['busy_over_wall']*100:.1f}%")
    print(f"  kernel 执行 {r['gpu_kernels']} 次，{r['distinct']} 种；"
          f"时长累加/区间并集 = {r['overlap_ratio']}")
    print("  注意：kernel 执行次数 ≠ Python 发射次数（一次 Graph replay 会跑很多 kernel）；"
          "这是 GPU 侧执行记录，不是 CPU 侧提交记录。")


if __name__ == "__main__":
    main()
