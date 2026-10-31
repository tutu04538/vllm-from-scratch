"""88 数值诊断：vLLM B=4 的 eager vs FULL Graph 首分歧 margin。

主矩阵已经显示 vLLM B=4 的 eager/Graph 输出在第 90 个 token 分叉；K=0 也分叉，
说明这不是 assistant/投机路径。这里用 `SamplingParams(logprobs=5)` 在同一位置取
top-1/top-2 logprob，记录首分歧位置和 margin。只跑 B=4 K=0，作为“无草稿控制项”；
不重新跑主矩阵，也不把结果当作严格等价。

用法：
  python benchmarks/check_step88_numerics.py
  python benchmarks/check_step88_numerics.py --output-dir benchmarks/results/step88/numerics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
for path in (ROOT, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from compare_step88_execution import (  # noqa: E402
    _load_prompt_ids, _tokenizer, _vllm_build, _vllm_rpc, _vllm_graph_mode,
)
from step88_common import make_case, token_diff  # noqa: E402
from step88_runtime import worker_snapshot  # noqa: E402

RESULT_DIR = ROOT / "benchmarks" / "results" / "step88" / "numerics"


def _top_logprobs(entry: dict[int, Any], k: int = 5) -> list[dict[str, Any]]:
    values = []
    for token_id, logprob in entry.items():
        values.append({
            "token_id": int(token_id),
            "logprob": float(logprob.logprob),
            "rank": int(getattr(logprob, "rank", 0)),
        })
    values.sort(key=lambda item: item["logprob"], reverse=True)
    return values[:k]


def _margin(entry: dict[int, Any]) -> dict[str, Any]:
    top = _top_logprobs(entry, k=5)
    if len(top) < 2:
        return {"top": top, "margin_logprob": None, "top1_prob": None}
    return {
        "top": top,
        "margin_logprob": top[0]["logprob"] - top[1]["logprob"],
        "top1_prob": __import__("math").exp(top[0]["logprob"]),
        "top2_prob": __import__("math").exp(top[1]["logprob"]),
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams

    tokenizer = _tokenizer()
    prompt = _load_prompt_ids(tokenizer, "short")
    llm, build_s = _vllm_build(case)
    try:
        snapshot = _vllm_rpc(llm, worker_snapshot)
        engine = llm.llm_engine
        params = SamplingParams(
            max_tokens=case["args"]["max_new_tokens"],
            temperature=0.0, ignore_eos=True, logprobs=5)
        for index in range(int(case["batch"])):
            engine.add_request(f"r{index}", list(prompt), params)
        final: dict[str, Any] = {}
        for _ in range(20000):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                if output.outputs:
                    final[output.request_id] = output.outputs[0]
        if len(final) != int(case["batch"]):
            raise RuntimeError(f"只拿到 {len(final)} 条输出")
        tokens = {request_id: list(out.token_ids)
                  for request_id, out in final.items()}
        logprobs = {
            request_id: [
                _top_logprobs(entry, k=5) for entry in (out.logprobs or [])
            ]
            for request_id, out in final.items()
        }
        base = list(tokens["r0"])
        diffs = {}
        for request_id, seq in tokens.items():
            if request_id == "r0":
                continue
            diffs[request_id] = token_diff(base, seq)
        first_diff = diffs.get("r1", {}).get("first_diff")
        margins = {}
        if first_diff is not None:
            index = min(int(first_diff), len(logprobs["r0"]) - 1)
            for request_id in ("r0", "r1"):
                entry = final[request_id].logprobs
                if entry is not None and index < len(entry):
                    margins[request_id] = {
                        "position": index,
                        "chosen_token": (tokens[request_id][index]
                                         if index < len(tokens[request_id])
                                         else None),
                        **_margin(entry[index]),
                    }
        return {
            "label": case["label"],
            "graph": bool(case["graph"]),
            "build_s": build_s,
            "actual_graph_mode": _vllm_graph_mode(snapshot),
            "graph_snapshot": (snapshot.get("graph") or {}).get("target"),
            "prompt_token_ids": prompt,
            "tokens": tokens,
            "first_diff": first_diff,
            "per_request_diff": diffs,
            "margins_at_first_diff": margins,
        }
    finally:
        del llm


def _dispatch(case: dict[str, Any]) -> dict[str, Any]:
    return _run(case)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_dispatch)
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    args = parser.parse_args(argv)
    cases = [
        make_case("vllm", 4, 0, graph, "short")
        for graph in (False, True)
    ]
    for case in cases:
        case["args"]["repeats"] = 1
    out_dir = Path(args.output_dir)
    matrix = run_matrix(
        [sys.executable, str(Path(__file__).resolve())], cases,
        timeout_s=args.timeout_s,
        env={"VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
             "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    rows = {record["case"]["graph"]: record["result"]
            for record in matrix["cases"] if record["status"] == "ok"}
    payload = {
        "ok": matrix["ok"],
        "metric_schema": "step88_numerics",
        "cases": rows,
        "failures": matrix["failures"],
        "note": ("B=4 K=0 vLLM eager/Graph 的固定输入 logprobs 诊断；"
                 "不是严格等价，也不进入主表"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    eager = rows.get(False)
    graph = rows.get(True)
    if eager and graph:
        print(json.dumps({
            "tokens_equal": eager["tokens"] == graph["tokens"],
            "first_diff": token_diff(
                eager["tokens"]["r0"], graph["tokens"]["r0"]),
            "eager_margin": eager.get("margins_at_first_diff", {}).get("r0"),
            "graph_margin": graph.get("margins_at_first_diff", {}).get("r0"),
            "output": str(out_dir / "summary.json"),
        }, ensure_ascii=False, indent=1))
    if not matrix["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
