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


def from_chrome_trace(path, wall=None):
    data = _read_json(path)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    by_name = Counter()
    total = 0.0
    for e in kernels:
        dur = e.get("dur", 0)
        total += dur
        by_name[e.get("name", "?")] += dur
    span = max((e["ts"] + e.get("dur", 0) for e in kernels), default=0) - \
           min((e["ts"] for e in kernels), default=0)
    return dict(source=str(path), gpu_kernels=len(kernels), distinct=len(by_name),
                gpu_busy_us=round(total, 1), gpu_span_us=round(span, 1),
                wall_s=wall, busy_over_wall=(round(total / 1e6 / wall, 3) if wall else None),
                top=[dict(name=n[:64], us=round(t, 1), calls=Counter(
                    e["name"] for e in kernels if e["name"] == n)[n])
                     for n, t in by_name.most_common(12)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--wall", type=float, default=None, help="对应负载的墙钟秒数")
    args = ap.parse_args()
    r = from_chrome_trace(args.trace, args.wall)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r["wall_s"]:
        print(f"\n  GPU 忙 {r['gpu_busy_us']/1e3:.2f} ms / 墙钟 {r['wall_s']*1e3:.2f} ms "
              f"= {r['busy_over_wall']*100:.1f}%")
    print(f"  kernel 发射 {r['gpu_kernels']} 次，{r['distinct']} 种")


if __name__ == "__main__":
    main()
