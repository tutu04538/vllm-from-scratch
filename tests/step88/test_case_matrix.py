"""88：有限矩阵、配置身份与配对启动顺序（不需要 GPU）。"""

from __future__ import annotations

from benchmarks.step88_common import (
    BATCHES,
    ENGINES,
    GRAPH_MODES,
    SPEC_TOKS,
    build_case_matrix,
    build_identity_report,
    check_identity,
    matrix_shape,
    validate_case_matrix,
)


def test_main_matrix_is_2x2x2x2_and_context_is_2x2():
    cases = build_case_matrix()
    shape = matrix_shape(cases)
    assert shape == {
        "total": 20,
        "main": 16,
        "context": 4,
        "pairs": 10,
        "engines": ["minivllm", "vllm"],
        "batches": [1, 4],
        "ks": [0, 2],
        "graphs": [False, True],
    }
    validation = validate_case_matrix(cases)
    assert validation["ok"], validation["errors"]
    assert len({case["label"] for case in cases}) == 20
    assert len({case["pair"] for case in cases}) == 10


def test_matrix_has_exact_product_and_no_k4_prefix_or_dynamic():
    cases = build_case_matrix()
    main = {
        (case["engine"], case["batch"], case["k"], case["graph"])
        for case in cases if case["workload"] == "short"
    }
    assert main == {
        (engine, batch, k, graph)
        for engine in ENGINES for batch in BATCHES
        for k in SPEC_TOKS for graph in GRAPH_MODES
    }
    assert all(case["k"] in (0, 2) for case in cases)
    assert all(case["workload"] in {"short", "context"} for case in cases)
    context = [case for case in cases if case["workload"] == "context"]
    assert {(case["batch"], case["k"]) for case in context} == {(1, 2)}


def test_pair_order_alternates_and_members_are_same_config():
    cases = build_case_matrix()
    pairs = {}
    for case in cases:
        pairs.setdefault(case["pair"], []).append(case)
    starts = []
    for index, (pair, members) in enumerate(pairs.items()):
        assert {case["engine"] for case in members} == set(ENGINES)
        assert sorted(case["pair_order"] for case in members) == [0, 1]
        starts.append(members[0]["engine"])
        assert len({(case["batch"], case["k"], case["graph"],
                     case["workload"]) for case in members}) == 1
    assert starts == ["minivllm" if index % 2 == 0 else "vllm"
                      for index in range(len(starts))]


def test_identity_checks_effective_config_not_request_echo():
    case = build_case_matrix()[0]
    observed = {
        "dtype": "bfloat16",
        "max_model_len": 2048,
        "block_size": 16,
        "max_num_seqs": case["batch"],
        "max_num_batched_tokens": 128,
        "max_new_tokens": 128,
        "enable_prefix_caching": False,
        "num_speculative_tokens": 0,
        "ignore_eos": True,
        "temperature": 0.0,
        "attention_backend": "TRITON_ATTN",
    }
    assert check_identity(case, observed)["ok"]
    observed["max_num_batched_tokens"] = 8192
    result = check_identity(case, observed)
    assert not result["ok"]
    assert "max_num_batched_tokens" in result["mismatches"]
    observed["max_num_batched_tokens"] = 128
    observed["attention_backend"] = "FLASH_ATTN"
    assert "attention_backend" in check_identity(case, observed)["mismatches"]


def _row(engine, batch, k, graph, tokens):
    return {
        "status": "ok", "engine": engine, "batch": batch, "k": k,
        "graph": graph, "workload": "short", "tokens": {
            f"r{i}": list(seq) for i, seq in enumerate(tokens)},
        "acceptance": {
            "num_drafts": 2 if k else 0,
            "num_draft_tokens": 2 * k,
            "num_accepted_tokens": 1 if k else 0,
            "acceptance_rate": 0.5 if k else None,
        },
    }


def test_identity_report_separates_within_and_cross_engine():
    rows = [
        _row("minivllm", 1, 0, False, [[1, 2, 3]]),
        _row("minivllm", 1, 0, True, [[1, 2, 3]]),
        _row("vllm", 1, 0, False, [[1, 2, 9]]),
        _row("vllm", 1, 0, True, [[1, 2, 9]]),
        _row("minivllm", 1, 2, False, [[1, 2, 3]]),
        _row("minivllm", 1, 2, True, [[1, 2, 3]]),
        _row("vllm", 1, 2, False, [[1, 2, 3]]),
        _row("vllm", 1, 2, True, [[1, 2, 3]]),
    ]
    report = build_identity_report(rows)
    assert report["within_engine_exact"]
    assert report["within_engine_acceptance_same"]
    assert not report["cross_engine_exact"]
    cross_k0 = next(item for item in report["cross_engine"]
                    if item["k"] == 0 and not item["graph"])
    assert cross_k0["diff"]["exact"] is False
    assert cross_k0["diff"]["first_diff"] == 2
    assert report["missing"], "只有 B1 K0/K2，其他格子应记为 missing"
