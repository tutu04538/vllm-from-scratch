"""88 关纯逻辑：有限矩阵、配置身份、指标换算与汇总。

这个模块**不 import torch / minivllm / vllm**，父进程和 CPU 单测都能安全使用。
真正的 engine 只在 `compare_step88_execution.py` 的子进程里创建。

矩阵固定为：

    主表  2 引擎 × B{1,4} × K{0,2} × {eager, FULL decode Graph} = 16
    补充  2 引擎 × B=1 × K=2 × {eager, FULL graph} @ ~1024-token = 4

配对 case 的启动顺序交替（偶数对 mini 先、奇数对 vLLM 先），避免热状态只偏向一侧。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

TARGET = ROOT / "models" / "gemma-4-E2B-it"
ASSISTANT = ROOT / "models" / "gemma-4-E2B-it-assistant"
TARGET_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
ASSISTANT_REVISION = "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df"

SHORT_PROMPT = "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"
CONTEXT_PROMPT = (
    "下面是用于中等上下文公平对照的固定材料，请只根据材料回答。"
    "分页注意力把逻辑 token 位置映射到固定大小的物理块，"
    "这样不同请求可以共享块池，不需要为每条请求预留最大长度。"
    "投机解码先让草稿模型提出若干候选，再由目标模型一次验证。"
    "图执行把稳定 decode 阶段的 kernel 启动序列录下来，重放时减少 CPU 发起开销。"
) * 40
CONTEXT_TARGET_TOKENS = 1024

ENGINES = ("minivllm", "vllm")
BATCHES = (1, 4)
SPEC_TOKS = (0, 2)
GRAPH_MODES = (False, True)

DTYPE = "bfloat16"
MAX_MODEL_LEN = 2048
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 512
MAX_NUM_BATCHED_TOKENS = 128
MAX_NEW_TOKENS = 128
REPEATS = 3
GPU_MEMORY_UTILIZATION = 0.88
# 512 × 16 × 1 × (12×256 + 3×512) × K/V × BF16 = 144 MiB。
# 与 mini 的 512 个 block 是同一物理 KV 字节；vLLM 的 block 布局不同，
# 因此 block 数/可容纳 token 数会不同，报告分别记录，不强行抹平。
MINI_KV_CACHE_BYTES = 150_994_944
VLLM_KV_CACHE_BYTES = MINI_KV_CACHE_BYTES
VLLM_GRAPH_CAPTURE_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]
VLLM_GRAPH_MAX_CAPTURE_SIZE = 16


def case_label(engine: str, batch: int, k: int, graph: bool,
               workload: str = "short") -> str:
    mode = "graph" if graph else "eager"
    suffix = "" if workload == "short" else f"_{workload}"
    return f"{engine}_B{batch}_K{k}_{mode}{suffix}"


def pair_label(batch: int, k: int, graph: bool,
               workload: str = "short") -> str:
    mode = "graph" if graph else "eager"
    suffix = "" if workload == "short" else f"_{workload}"
    return f"B{batch}_K{k}_{mode}{suffix}"


def _base_args(workload: str) -> dict[str, Any]:
    return {
        "workload": workload,
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "block_size": BLOCK_SIZE,
        "blocks": NUM_GPU_BLOCKS,
        "budget": MAX_NUM_BATCHED_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repeats": REPEATS,
        "ignore_eos": True,
        "context_target_tokens": CONTEXT_TARGET_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "vllm_kv_cache_bytes": VLLM_KV_CACHE_BYTES,
        "mini_kv_cache_bytes": MINI_KV_CACHE_BYTES,
        "attention_backend": "triton",
        "vllm_graph_capture_sizes": list(VLLM_GRAPH_CAPTURE_SIZES),
        "vllm_graph_max_capture_size": VLLM_GRAPH_MAX_CAPTURE_SIZE,
    }


def make_case(engine: str, batch: int, k: int, graph: bool,
              workload: str = "short") -> dict[str, Any]:
    if engine not in ENGINES:
        raise ValueError(f"未知 engine={engine!r}")
    if batch not in BATCHES:
        raise ValueError(f"未知 batch={batch!r}")
    if k not in SPEC_TOKS:
        raise ValueError(f"未知 k={k!r}")
    label = case_label(engine, batch, k, graph, workload)
    return {
        "kind": "step88_perf",
        "label": label,
        "pair": pair_label(batch, k, graph, workload),
        "engine": engine,
        "batch": int(batch),
        "k": int(k),
        "graph": bool(graph),
        "workload": workload,
        "context_target_tokens": (CONTEXT_TARGET_TOKENS
                                  if workload == "context" else None),
        "args": _base_args(workload),
    }


def build_case_matrix(*, include_context: bool = True) -> list[dict[str, Any]]:
    """按配对交替顺序返回 16 个主 case（+ 可选 4 个上下文 case）。"""
    cases: list[dict[str, Any]] = []
    pair_index = 0
    for batch in BATCHES:
        for k in SPEC_TOKS:
            for graph in GRAPH_MODES:
                pair = [make_case(engine, batch, k, graph)
                        for engine in ENGINES]
                if pair_index % 2:
                    pair.reverse()
                for order, case in enumerate(pair):
                    case["pair_order"] = order
                    case["case_index"] = len(cases)
                    cases.append(case)
                pair_index += 1
    if include_context:
        for graph in GRAPH_MODES:
            pair = [make_case(engine, 1, 2, graph, "context")
                    for engine in ENGINES]
            if pair_index % 2:
                pair.reverse()
            for order, case in enumerate(pair):
                case["pair_order"] = order
                case["case_index"] = len(cases)
                cases.append(case)
            pair_index += 1
    return cases


def matrix_shape(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    main = [case for case in cases if case["workload"] == "short"]
    context = [case for case in cases if case["workload"] == "context"]
    return {
        "total": len(cases),
        "main": len(main),
        "context": len(context),
        "pairs": len({case["pair"] for case in cases}),
        "engines": sorted({case["engine"] for case in cases}),
        "batches": sorted({case["batch"] for case in cases}),
        "ks": sorted({case["k"] for case in cases}),
        "graphs": sorted({case["graph"] for case in cases}),
    }


def validate_case_matrix(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """检查 16+4 覆盖、配对完整、启动顺序交替和必需字段。

    返回 `{"ok", "errors", "shape"}`；测试和正式运行都调用它，避免“少跑一格还
    声称完成矩阵”。
    """
    errors: list[str] = []
    labels = [case.get("label") for case in cases]
    if len(labels) != len(set(labels)):
        errors.append("case label 不唯一")
    required = {
        "kind", "label", "pair", "engine", "batch", "k", "graph",
        "workload", "args",
    }
    for case in cases:
        missing = required - set(case)
        if missing:
            errors.append(f"{case.get('label')} 缺字段 {sorted(missing)}")
        args = case.get("args", {})
        for name in ("dtype", "max_model_len", "block_size", "blocks",
                     "budget", "max_new_tokens", "repeats",
                     "attention_backend", "context_target_tokens",
                     "gpu_memory_utilization", "vllm_kv_cache_bytes"):
            if name not in args:
                errors.append(f"{case.get('label')} args 缺 {name}")

    expected_main = {
        (engine, batch, k, graph)
        for engine in ENGINES
        for batch in BATCHES
        for k in SPEC_TOKS
        for graph in GRAPH_MODES
    }
    actual_main = {
        (case["engine"], case["batch"], case["k"], case["graph"])
        for case in cases if case["workload"] == "short"
    }
    if actual_main != expected_main:
        errors.append(
            f"主表不是 2×2×2×2：缺 {sorted(expected_main - actual_main)}，"
            f"多 {sorted(actual_main - expected_main)}")
    expected_context = {
        (engine, 1, 2, graph)
        for engine in ENGINES
        for graph in GRAPH_MODES
    }
    actual_context = {
        (case["engine"], case["batch"], case["k"], case["graph"])
        for case in cases if case["workload"] == "context"
    }
    if actual_context != expected_context:
        errors.append(
            f"上下文补充表不完整：缺 {sorted(expected_context - actual_context)}，"
            f"多 {sorted(actual_context - expected_context)}")
    if len(cases) != 20:
        errors.append(f"总 case 数应为 20，实际 {len(cases)}")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_pair[case["pair"]].append(case)
    for pair, members in by_pair.items():
        engines = {case["engine"] for case in members}
        if engines != set(ENGINES):
            errors.append(f"{pair} 不是 mini/vLLM 成对：{sorted(engines)}")
        if len(members) != 2:
            errors.append(f"{pair} 应有 2 个 case，实际 {len(members)}")
        orders = sorted(case["pair_order"] for case in members)
        if orders != [0, 1]:
            errors.append(f"{pair} 的 pair_order 不是 0/1：{orders}")
    pair_starts: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case["pair"] not in seen:
            seen.add(case["pair"])
            pair_starts.append(case["engine"])
    expected_starts = ["minivllm" if i % 2 == 0 else "vllm"
                       for i in range(len(pair_starts))]
    if pair_starts != expected_starts:
        errors.append(f"配对启动顺序未交替：{pair_starts}")
    return {"ok": not errors, "errors": errors,
            "shape": matrix_shape(cases)}


def identity_expectations(case: dict[str, Any]) -> dict[str, Any]:
    """两边必须逐项对上的配置；KV block 数/字节不在此列，单独记录。"""
    args = case["args"]
    return {
        "dtype": args["dtype"],
        "max_model_len": args["max_model_len"],
        "block_size": args["block_size"],
        "max_num_seqs": case["batch"],
        "max_num_batched_tokens": args["budget"],
        "max_new_tokens": args["max_new_tokens"],
        "enable_prefix_caching": False,
        "num_speculative_tokens": case["k"],
        "ignore_eos": True,
        "temperature": 0.0,
        "attention_backend": "TRITON_ATTN",
    }


def check_identity(case: dict[str, Any],
                   observed: dict[str, Any]) -> dict[str, Any]:
    """比较 worker 实际生效的配置，而不是只回显请求参数。"""
    expected = identity_expectations(case)
    mismatches: dict[str, Any] = {}
    for key, want in expected.items():
        if key not in observed:
            mismatches[key] = {"expected": want, "observed": "<missing>"}
            continue
        got = observed[key]
        if key == "attention_backend":
            normalized = str(got).upper().replace("-", "_")
            if normalized not in {str(want).upper(), "TRITON_ATTN"}:
                mismatches[key] = {"expected": want, "observed": got}
            continue
        if key == "enable_prefix_caching":
            if bool(got) is not bool(want):
                mismatches[key] = {"expected": want, "observed": got}
            continue
        if got != want:
            mismatches[key] = {"expected": want, "observed": got}
    return {"ok": not mismatches, "mismatches": mismatches,
            "expected": expected, "observed": observed}


def token_diff(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    first_diff = next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        None,
    )
    common = (len(left) if first_diff is None
              else first_diff)
    return {
        "exact": first_diff is None and len(left) == len(right),
        "first_diff": first_diff,
        "common_prefix_len": common,
        "left_len": len(left),
        "right_len": len(right),
    }


def ordered_outputs(row: dict[str, Any]) -> list[list[int]]:
    """把 worker 的 `tokens` 字典变成按 r0..rN 排好的 batch 输出。"""
    token_map = row.get("tokens") or {}
    if isinstance(token_map, list):
        return [list(item) for item in token_map]
    batch = int(row.get("batch", len(token_map) or 1))
    return [list(token_map.get(f"r{i}", [])) for i in range(batch)]


def compare_output_batches(left: Sequence[Sequence[int]],
                           right: Sequence[Sequence[int]]) -> dict[str, Any]:
    if len(left) != len(right):
        return {
            "exact": False,
            "reason": f"请求数不同：{len(left)} vs {len(right)}",
            "per_request": [],
        }
    per_request = [token_diff(a, b) for a, b in zip(left, right)]
    exact = all(item["exact"] for item in per_request)
    first = next((item["first_diff"] for item in per_request
                  if item["first_diff"] is not None), None)
    return {
        "exact": exact,
        "first_diff": first,
        "per_request": per_request,
        "left_sha256": _sequences_hash(left),
        "right_sha256": _sequences_hash(right),
    }


def _sequences_hash(sequences: Sequence[Sequence[int]]) -> str:
    payload = ";".join(",".join(map(str, seq)) for seq in sequences)
    return hashlib.sha256(payload.encode()).hexdigest()


def _row_index(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    index = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (row["engine"], int(row["batch"]), int(row["k"]),
               bool(row["graph"]), row.get("workload", "short"))
        index[key] = row
    return index


def _acceptance_exact(left: dict[str, Any] | None,
                      right: dict[str, Any] | None) -> dict[str, Any]:
    if not left or not right:
        return {"available": False, "reason": "一侧没有接受统计"}
    keys = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    values = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        values[key] = {"left": a, "right": b,
                       "same": a is not None and b is not None and a == b}
    return {
        "available": True,
        "same": all(item["same"] for item in values.values()),
        "values": values,
        "acceptance_rate": {
            "left": left.get("acceptance_rate"),
            "right": right.get("acceptance_rate"),
        },
    }


def build_identity_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """同引擎 eager/Graph 与跨引擎输出身份，独立于性能表审核。"""
    index = _row_index(rows)
    within: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    missing: list[str] = []
    for workload in ("short", "context"):
        batches = BATCHES if workload == "short" else (1,)
        ks = SPEC_TOKS if workload == "short" else (2,)
        for engine in ENGINES:
            for batch in batches:
                for k in ks:
                    eager = index.get((engine, batch, k, False, workload))
                    graph = index.get((engine, batch, k, True, workload))
                    label = f"{engine}_B{batch}_K{k}_{workload}"
                    if not eager or not graph:
                        missing.append(label + "_eager_or_graph")
                        continue
                    within.append({
                        "engine": engine,
                        "batch": batch,
                        "k": k,
                        "workload": workload,
                        "left": "eager",
                        "right": "graph",
                        "diff": compare_output_batches(
                            ordered_outputs(eager), ordered_outputs(graph)),
                        "acceptance": _acceptance_exact(
                            eager.get("acceptance"), graph.get("acceptance")),
                    })
        for batch in batches:
            for k in ks:
                for graph in GRAPH_MODES:
                    mini = index.get(("minivllm", batch, k, graph, workload))
                    vllm = index.get(("vllm", batch, k, graph, workload))
                    label = f"B{batch}_K{k}_{graph}_{workload}"
                    if not mini or not vllm:
                        missing.append(label + "_mini_or_vllm")
                        continue
                    cross.append({
                        "batch": batch,
                        "k": k,
                        "graph": graph,
                        "workload": workload,
                        "diff": compare_output_batches(
                            ordered_outputs(mini), ordered_outputs(vllm)),
                        "acceptance": _acceptance_exact(
                            mini.get("acceptance"), vllm.get("acceptance")),
                    })
    determinism = []
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        hashes = []
        for sample in row.get("samples", []):
            hashes.append(_sequences_hash(ordered_outputs(sample)))
        determinism.append({
            "engine": row["engine"],
            "batch": int(row["batch"]),
            "k": int(row["k"]),
            "graph": bool(row["graph"]),
            "workload": row.get("workload", "short"),
            "stable": len(set(hashes)) <= 1,
            "hashes": hashes,
        })
    mini = [item for item in within if item["engine"] == "minivllm"]
    vllm = [item for item in within if item["engine"] == "vllm"]
    vllm_boundary = [
        {"batch": item["batch"], "k": item["k"],
         "workload": item["workload"],
         "first_diff": item["diff"]["first_diff"]}
        for item in vllm if not item["diff"]["exact"]
    ]
    return {
        "within_engine": within,
        "cross_engine": cross,
        "missing": missing,
        "within_engine_exact": all(item["diff"]["exact"] for item in within),
        "within_engine_acceptance_same": all(
            item["acceptance"].get("same", False)
            for item in within if item["k"] == 2),
        "mini_within_engine_exact": all(
            item["diff"]["exact"] for item in mini),
        "mini_within_acceptance_same": all(
            item["acceptance"].get("same", False)
            for item in mini if item["k"] == 2),
        "vllm_within_engine_exact": all(
            item["diff"]["exact"] for item in vllm),
        "vllm_within_acceptance_same": all(
            item["acceptance"].get("same", False)
            for item in vllm if item["k"] == 2),
        # vLLM 0.28 在本机 B=4 的 FULL Graph 与 eager 出现 BF16 首分歧；
        # 只记录为 layout/numeric boundary，不冒充严格等价。调用方用
        # `_vllm_boundary_gate()` 判断它是否能由 K=0 控制项解释。
        "vllm_numeric_boundary": vllm_boundary,
        "cross_engine_exact": all(item["diff"]["exact"] for item in cross),
        # 跨引擎接受计数不是“必须相等”的验收门槛；相同是最强对照，不同则报告。
        "cross_engine_same_acceptance": all(
            item["acceptance"].get("same", False)
            for item in cross if item["k"] == 2),
        "run_to_run_deterministic": all(
            item["stable"] for item in determinism),
        "determinism": determinism,
    }


def _metric_name(metric: Any) -> str:
    name = str(getattr(metric, "name", ""))
    return name[:-6] if name.endswith("_total") else name


def spec_metrics_snapshot(metrics: Iterable[Any]) -> dict[str, Any]:
    """把 vLLM Prometheus 快照归一成和 mini 同名的累计统计。

    只依赖鸭子类型（Counter.value / Vector.values），不 import vllm。
    """
    counters = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
    }
    per_pos: list[int] = []
    for metric in metrics:
        name = _metric_name(metric)
        key = {
            "vllm:spec_decode_num_drafts": "num_drafts",
            "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
        }.get(name)
        if key is not None:
            value = getattr(metric, "value", None)
            if value is not None:
                counters[key] += int(value)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = getattr(metric, "values", None)
            if values is not None and len(values) > len(per_pos):
                per_pos = [int(value) for value in values]
    return {**counters,
            "num_accepted_tokens_per_pos": per_pos,
            "available": any(counters.values()) or bool(per_pos)}


def spec_metrics_delta(after: dict[str, Any],
                       before: dict[str, Any]) -> dict[str, Any]:
    keys = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    delta = {key: max(int(after.get(key, 0)) - int(before.get(key, 0)), 0)
             for key in keys}
    after_pos = after.get("num_accepted_tokens_per_pos") or []
    before_pos = before.get("num_accepted_tokens_per_pos") or []
    width = max(len(after_pos), len(before_pos))
    delta["num_accepted_tokens_per_pos"] = [
        max((after_pos[i] if i < len(after_pos) else 0)
            - (before_pos[i] if i < len(before_pos) else 0), 0)
        for i in range(width)
    ]
    delta["available"] = any(delta[key] for key in keys) or bool(
        delta["num_accepted_tokens_per_pos"])
    return delta


def acceptance_from_spec_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    num_drafts = int(metrics.get("num_drafts", 0))
    num_draft_tokens = int(metrics.get("num_draft_tokens", 0))
    num_accepted = int(metrics.get("num_accepted_tokens", 0))
    per_pos = [int(value) for value in
               metrics.get("num_accepted_tokens_per_pos", [])]
    # vLLM 只暴露 accepted_per_pos，不暴露 draft_per_pos；position 口径仍按
    # accepted[j] / num_drafts（和 mini 的 `position_acceptance` 一致）。
    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "acceptance_rate": (num_accepted / num_draft_tokens
                            if num_draft_tokens else None),
        "mean_acceptance_length": (1.0 + num_accepted / num_drafts
                                   if num_drafts else None),
        "accepted_per_pos": per_pos,
        "draft_per_pos": ([num_drafts] * len(per_pos) if num_drafts else []),
        "position_acceptance": ([value / num_drafts for value in per_pos]
                                if num_drafts else []),
        "position_acceptance_actual": ([value / num_drafts for value in per_pos]
                                       if num_drafts else []),
        "draft_per_pos_available": False,
        "source": "vllm_prometheus_delta",
    }


def _pair_index(rows: Iterable[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    index = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (int(row["batch"]), int(row["k"]), bool(row["graph"]),
               row.get("workload", "short"))
        index.setdefault(key, {})[row["engine"]] = row
    return index


def _median_value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if isinstance(value, dict):
        return value.get("median")
    return value


def build_perf_summary(rows: Sequence[dict[str, Any]],
                       failures: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """把每个配对的 mini/vLLM 中位数排成可审核表。"""
    index = _pair_index(rows)
    table: list[dict[str, Any]] = []
    missing: list[str] = []
    for workload in ("short", "context"):
        batches = BATCHES if workload == "short" else (1,)
        ks = SPEC_TOKS if workload == "short" else (2,)
        for batch in batches:
            for k in ks:
                for graph in GRAPH_MODES:
                    key = (batch, k, graph, workload)
                    pair = index.get(key, {})
                    label = pair_label(batch, k, graph, workload)
                    if "minivllm" not in pair or "vllm" not in pair:
                        missing.append(label)
                        continue
                    mini, vllm = pair["minivllm"], pair["vllm"]
                    mini_tps = _median_value(mini, "output_tokens_per_s")
                    vllm_tps = _median_value(vllm, "output_tokens_per_s")
                    ratio = (mini_tps / vllm_tps
                             if mini_tps and vllm_tps and vllm_tps > 0
                             else None)
                    table.append({
                        "pair": label,
                        "workload": workload,
                        "batch": batch,
                        "k": k,
                        "graph": graph,
                        "mini": _row_brief(mini),
                        "vllm": _row_brief(vllm),
                        "mini_over_vllm": ratio,
                        "same_engine_outputs": (
                            _within_engine_pair_exact(rows, batch, k, workload)),
                    })
    return {
        "ok": not missing and not failures,
        "table": table,
        "missing": missing,
        "failures": list(failures),
        "pairs": len(table),
    }


def _within_engine_pair_exact(rows: Sequence[dict[str, Any]], batch: int,
                              k: int, workload: str) -> dict[str, Any]:
    index = _row_index(rows)
    out = {}
    for engine in ENGINES:
        eager = index.get((engine, batch, k, False, workload))
        graph = index.get((engine, batch, k, True, workload))
        if eager and graph:
            out[engine] = compare_output_batches(
                ordered_outputs(eager), ordered_outputs(graph))["exact"]
    return out


def _row_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("label"),
        "tok_s": _median_value(row, "output_tokens_per_s"),
        "wall_s": _median_value(row, "wall_time_s"),
        "ttft_s": (row.get("ttft_s") or {}).get("mean")
        if isinstance(row.get("ttft_s"), dict) else row.get("ttft_s"),
        "tpot_s": (row.get("tpot_s") or {}).get("mean")
        if isinstance(row.get("tpot_s"), dict) else row.get("tpot_s"),
        "acceptance_rate": _median_value(row, "acceptance_rate"),
        "build_s": row.get("build_s"),
        "graph_capture_s": row.get("graph_capture_s"),
        "kv_bytes": (row.get("kv") or {}).get("bytes"),
        "graph": row.get("graph"),
        "identity_ok": (row.get("identity") or {}).get("ok"),
    }


def summarize_matrix_outcome(matrix: dict[str, Any]) -> dict[str, Any]:
    """合并 `run_matrix` 的失败记录与成功 row，确保失败不会变成 0 tok/s。"""
    records = matrix.get("cases", [])
    rows = [record["result"] for record in records
            if record.get("status") == "ok"]
    failures = list(matrix.get("failures", []))
    identity = build_identity_report(rows)
    perf = build_perf_summary(rows, failures)
    return {
        "ok": bool(matrix.get("ok")) and not identity["missing"],
        "rows": rows,
        "failures": failures,
        "identity": identity,
        "performance": perf,
        "case_status": [
            {"label": record.get("case", {}).get("label"),
             "status": record.get("status"),
             "error": record.get("error")}
            for record in records
        ],
    }


def model_manifest() -> dict[str, Any]:
    """父进程可读的模型身份；不加载 tensor。"""
    target_cfg = _read_json(TARGET / "config.json")
    assistant_cfg = _read_json(ASSISTANT / "config.json")
    return {
        "target": {
            "path": str(TARGET.relative_to(ROOT)),
            "revision": TARGET_REVISION,
            "architectures": target_cfg.get("architectures"),
            "model_type": target_cfg.get("model_type"),
            "config_sha256": _file_sha256(TARGET / "config.json"),
        },
        "assistant": {
            "path": str(ASSISTANT.relative_to(ROOT)),
            "revision": ASSISTANT_REVISION,
            "architectures": assistant_cfg.get("architectures"),
            "model_type": assistant_cfg.get("model_type"),
            "config_sha256": _file_sha256(ASSISTANT / "config.json"),
        },
        "tokenizer": str((TARGET / "tokenizer.json").relative_to(ROOT)),
        "tokenizer_sha256": _file_sha256(TARGET / "tokenizer.json"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json
    return json.loads(path.read_text())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_hash() -> dict[str, Any]:
    """包内 `*.py` 的稳定摘要；与 step81/87 的口径一致。"""
    root = ROOT / "minivllm"
    names = sorted(root.rglob("*.py"), key=lambda p: str(p.relative_to(root)))
    digest = hashlib.sha256("".join(
        f"{path.relative_to(root)}\0"
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        for path in names).encode()).hexdigest()
    return {"root": "minivllm/", "python_files": len(names),
            "sha256": digest}


def source_hashes() -> dict[str, str]:
    names = (
        "benchmarks/compare_step88_execution.py",
        "benchmarks/trace_step88_execution.py",
        "benchmarks/step88_common.py",
        "benchmarks/step88_runtime.py",
        "benchmarks/benchmark_runner.py",
        "benchmarks/spec_decode_metrics.py",
    )
    out = {}
    for name in names:
        path = ROOT / name
        if path.is_file():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def vllm_boundary_gate(report: dict[str, Any]) -> dict[str, Any]:
    """vLLM 的 eager/Graph 差异能否由“批布局/BF16”解释，而不是状态错误。

    验收规则：
    - B=1 必须逐 token 相同（没有批布局可辩解）；
    - B>1 的 K=2 差异必须同时存在同 batch/workload 的 K=0 控制差异；
    - 任一条件不满足 -> gate=False，交由报告/排查，不允许进入严格等价表。
    """
    failures: list[str] = []
    within = [item for item in report.get("within_engine", [])
              if item["engine"] == "vllm"]
    if report.get("vllm_within_engine_exact"):
        return {"ok": True, "failures": [], "reason": "vllm_eager_graph_exact"}
    for item in within:
        if item["batch"] == 1 and not item["diff"]["exact"]:
            failures.append(
                f"B=1 B{item['batch']} K{item['k']} {item['workload']} "
                f"first_diff={item['diff']['first_diff']}")
    boundary_keys = {(item["batch"], item["workload"]) for item in within
                      if not item["diff"]["exact"]}
    k0_keys = {(item["batch"], item["workload"]) for item in within
               if item["k"] == 0 and not item["diff"]["exact"]}
    for batch, workload in sorted(boundary_keys):
        if (batch, workload) not in k0_keys:
            failures.append(
                f"B={batch} {workload} 只有 K>0 分叉，缺少 K=0 控制")
    nondeterministic = not report.get("run_to_run_deterministic", True)
    return {
        "ok": not failures,
        "failures": failures,
        "reason": ("vllm_run_to_run_nondeterministic" if nondeterministic
                   else ("k0_control_present" if not failures
                         else "unexplained_vllm_boundary")),
        "boundaries": report.get("vllm_numeric_boundary", []),
    }
