"""82 单场景 vLLM 0.28 对照：同一个真实 Gemma4 target/assistant。

WSL2 必须设置 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，否则 EngineCore 报 UVA is not available。
这份脚本只做 B=1/K=2 的框架间对照，不声称两边输出逐 token 等价。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models/gemma-4-E2B-it"
ASSISTANT = ROOT / "models/gemma-4-E2B-it-assistant"
PROMPT = "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    result = {}
    for label, extra in (
        ("target_only", {}),
        ("spec_k2", dict(spec_method="mtp", spec_model=str(ASSISTANT),
                         spec_tokens=2)),
    ):
        llm = LLM(model=str(TARGET), max_model_len=2048,
                  gpu_memory_utilization=0.88, dtype="bfloat16",
                  enforce_eager=True, disable_log_stats=True, **extra)
        # warmup；下面这次计时包含 prompt 处理（vLLM 默认 prefix caching 开）
        llm.generate([ids], SamplingParams(max_tokens=8, temperature=0.0,
                                          ignore_eos=True))
        start = time.perf_counter()
        out = llm.generate([ids], SamplingParams(max_tokens=128, temperature=0.0,
                                                ignore_eos=True))
        elapsed = time.perf_counter() - start
        tokens = list(out[0].outputs[0].token_ids)
        result[label] = {"wall_s": elapsed, "tokens": len(tokens),
                         "tok_s": len(tokens) / elapsed, "token_ids": tokens}
        print(label, result[label]["tok_s"], flush=True)
        del llm
    out_path = ROOT / "benchmarks/results/step82/vllm_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
