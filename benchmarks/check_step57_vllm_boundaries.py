"""57F 对照 C：**在真实 vLLM 上验证两条边界**（issue/PR 练习的"确认 + 解释"形式）。

200 §C 要求：挑一个确定边界 → 写最小复现与预期不变量 → 指出本机源码分支 → 在真实目标版本运行
→ **若真实 vLLM 没问题，解释其代码如何避免；这同样是合格的学习结果**。

两条边界都是我这次真的踩过的坑：

  1. **被拒的草稿要把进度退回来**。我这边的第一版漏了这一步，请求会卡在
     `computed > num_tokens` 上（`docs/step57e_speculative.md` 记着）。在真实 vLLM 上跑 ngram
     投机，断言不变量"每轮结束时 `num_computed_tokens == num_tokens - 1`"，并指出
     vLLM 在哪一行做这件事。

  2. **投机验证路径上 `min_tokens` 也要屏蔽停止 token**。我这边的第一版漏了，草稿可以在
     min_tokens 之前把 EOS 送进提交序列。这里直接调用 vLLM 的
     `MinTokensLogitsProcessor.apply_with_spec_decode`（可隔离入口），看它确实按草稿行逐行屏蔽。

需要环境变量（WSL2 的 pinned memory / 同进程引擎），脚本会提示：

    VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 python ...
"""

import json
import os
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

FAIL = []
from minivllm.testing.tiny_models import tiny_qwen3_dir   # 测试模型现场生成（仓库不再放 fixtures）
TINY = tiny_qwen3_dir("tiny_gqa")


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1":
    print("（提示：本机 WSL2 需要 VLLM_WSL2_ENABLE_PIN_MEMORY=1 才能起 vLLM 引擎；"
          "没设的话这两条对照会被记为「未验收」）")


# ------------------------------------------------ 边界 1：被拒草稿的进度回退

def boundary_progress_rollback():
    from vllm import LLM, SamplingParams as VllmSamplingParams

    llm = LLM(model=TINY, dtype="float32", max_model_len=64, gpu_memory_utilization=0.2,
              enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False,
              speculative_config={"method": "ngram", "num_speculative_tokens": 3})
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler

    # 每轮结束（update_from_output 之后）检查一次不变量
    violations = []
    rounds = {"count": 0}
    original = scheduler.update_from_output

    def traced(scheduler_output, model_output):
        result = original(scheduler_output, model_output)
        rounds["count"] += 1
        for req_id, request in scheduler.requests.items():
            # 不变量：进度只该落在"最后一个已提交 token 还没算"这个位置
            if request.num_computed_tokens != request.num_tokens - 1:
                violations.append((rounds["count"], req_id, request.num_tokens,
                                   request.num_computed_tokens))
        return result

    scheduler.update_from_output = traced
    prompt = [1, 2, 3, 4, 1, 2, 3, 4]
    llm.generate([{"prompt_token_ids": prompt}],
                 VllmSamplingParams(max_tokens=12, temperature=0.0), use_tqdm=False)
    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return rounds["count"], violations


try:
    rounds, violations = boundary_progress_rollback()
    check("C1. 真实 vLLM（ngram 投机）：每轮结束时 num_computed_tokens == num_tokens - 1",
          rounds > 3 and not violations,
          f"跑了 {rounds} 轮，违例 {violations[:2] if violations else '无'}")
    print("      ↳ vLLM 在哪一行做的：`Scheduler.update_from_output()` 里")
    print("        num_rejected = len(scheduled_spec_decode_tokens) - (len(generated) - 1)")
    print("        request.num_computed_tokens -= num_rejected")
    print("        （`_update_after_schedule()` 是按「排了多少行」推进的，这里把被拒的补回来；")
    print("          对应我们 `core/sched/scheduler.py::update_from_output` 的同一段）")
except Exception as exc:                     # noqa: BLE001
    check("C1. 真实 vLLM 跑 ngram 投机（环境不支持则记录原因）", False, first_line(str(exc)))


# ------------------------------------------------ 边界 2：min_tokens 在验证路径上的屏蔽

def boundary_min_tokens_censor():
    """直接调用 vLLM 的 `MinTokensLogitsProcessor`（可隔离入口），看它按草稿行逐行屏蔽。

    `update_state()` 要的是它自己的 `BatchUpdate`；这里用鸭子类型的替身传入
    `(行号, 采样参数, prompt, 已提交输出)`，只为了让它把"还没到 min_tokens 的行"记下来。
    """
    from vllm.sampling_params import SamplingParams as VllmSamplingParams
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

    class FakeBatchUpdate:
        """`process_dict_updates` 只用到 `added` / `removed` / `moved` 三个字段，鸭子类型即可。"""

        removed = []
        moved = []
        # (行号, 采样参数, prompt token ids, 已提交输出)
        # vLLM 的 SamplingParams 没有 eos_token_id（那是模型/tokenizer 的属性）：
        # 这里用显式 stop token 4 当"停止 token"
        added = [(0, VllmSamplingParams(max_tokens=8, min_tokens=3, stop_token_ids=[4]), [], []),
                 (1, VllmSamplingParams(max_tokens=8, min_tokens=0, stop_token_ids=[4]), [], [])]

    processor = MinTokensLogitsProcessor(vllm_config=None, device=torch.device("cuda"),
                                         is_pin_memory=False)
    processor.update_state(FakeBatchUpdate())

    vocab = 8
    # 两条请求、K=[2,1] → 3 个草稿行；每行 logits 都是"eos=4 最高"
    logits = torch.zeros(3, vocab, device="cuda")
    logits[:, 4] = 9.0
    processed = processor.apply_with_spec_decode(logits.clone(), [2, 1])
    censored_rows = (processed[:, 4] == float("-inf")).tolist()
    return censored_rows


try:
    censored = boundary_min_tokens_censor()
    check("C2. 真实 vLLM：min_tokens 未到的请求，它的**每个草稿行**都被屏蔽停止 token",
          censored[0] and censored[1] and not censored[2],
          f"按草稿行（r0 的 2 行 + r1 的 1 行）屏蔽情况={censored}")
    print("      ↳ vLLM 在哪一行做的：`MinTokensLogitsProcessor.apply_with_spec_decode()`")
    print("        它按 num_draft_tokens 算出每个请求占的草稿行，只屏蔽 min_tokens 未到的那些行")
    print("        （对应我们 `spec_decode/rejection_sampler.py::forward` 里")
    print("          调 `sampler.apply_logits_processors(target_logits, target_metadata)` 的那一步）")
except Exception as exc:                     # noqa: BLE001
    check("C2. 真实 vLLM 的 min_tokens 屏蔽可隔离调用（环境不支持则记录原因）", False,
          first_line(str(exc)))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
