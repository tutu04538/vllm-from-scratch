"""82/83 单场景 vLLM 0.28 对照：同一个真实 Gemma4 target/assistant。

83 关起每个配置在**独立子进程**里跑：target-only 与 K=2 不共享 vLLM engine / CUDA
allocator / prefix cache。父进程只负责编排和汇总，不 import vllm/torch。

WSL2 必须设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，否则 EngineCore 报 UVA is not available。
这份脚本只做 B=1 的框架间对照，不声称两边输出逐 token 等价。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_runner import (case_worker_main, run_matrix)  # noqa: E402

TARGET = ROOT / "models/gemma-4-E2B-it"
ASSISTANT = ROOT / "models/gemma-4-E2B-it-assistant"
PROMPT = "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"
MAX_NEW_TOKENS = 128


def _run_case(case: dict) -> dict:
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    extra = {}
    if case["spec_tokens"]:
        extra = dict(spec_method="mtp", spec_model=str(ASSISTANT),
                     spec_tokens=case["spec_tokens"])
    build_start = time.perf_counter()
    llm = LLM(model=str(TARGET), max_model_len=2048,
              gpu_memory_utilization=0.88, dtype="bfloat16",
              enforce_eager=True, disable_log_stats=True, **extra)
    build_s = time.perf_counter() - build_start
    try:
        # warmup；正式计时与 step82 保持一致，包含 prompt/prefill + decode。
        llm.generate([ids], SamplingParams(max_tokens=8, temperature=0.0,
                                          ignore_eos=True))
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = llm.generate([ids], SamplingParams(
            max_tokens=MAX_NEW_TOKENS, temperature=0.0, ignore_eos=True))
        elapsed = time.perf_counter() - start
        tokens = list(out[0].outputs[0].token_ids)
        return {
            "framework": "vllm",
            "version": vllm.__version__,
            "label": case["label"],
            "spec_tokens": int(case["spec_tokens"]),
            "build_s": build_s,
            "wall_s": elapsed,
            "tokens": len(tokens),
            "tok_s": len(tokens) / elapsed,
            "token_ids": tokens,
            "prompt_token_ids": ids,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prefix_caching": "vllm default (enabled)",
            "attention_backend": "vLLM auto (TRITON_ATTN/FlashInfer on this host)",
        }
    finally:
        # 子进程即将退出，仍显式断开，避免在同一进程里留下第二个 LLM 的引用。
        del llm


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        return case_worker_main(_run_case)

    cases = [
        {"kind": "vllm", "label": "target_only", "spec_tokens": 0},
        {"kind": "vllm", "label": "spec_k2", "spec_tokens": 2},
    ]
    matrix = run_matrix(
        [sys.executable, str(Path(__file__).resolve())], cases,
        env={"VLLM_WSL2_ENABLE_PIN_MEMORY": "1"})
    rows = {record["case"]["label"]: record["result"]
            for record in matrix["cases"] if record["status"] == "ok"}
    payload = {
        "ok": matrix["ok"],
        "metric_schema": "step83",
        "cases": rows,
        "failures": matrix["failures"],
    }
    target = rows.get("target_only")
    spec = rows.get("spec_k2")
    if target and spec and target["tok_s"] > 0:
        payload["framework_speedup_k2"] = spec["tok_s"] / target["tok_s"]
    out_path = ROOT / "benchmarks/results/step82/vllm_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    for label, row in rows.items():
        print(label, row["tok_s"], flush=True)
    if not matrix["ok"]:
        print(f"vLLM comparison failures: {len(matrix['failures'])}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
