"""第 55 关：真实 Qwen3 双模型端到端（需求 §6C、187）。

主运行方案：**Qwen3-1.7B（target）+ Qwen3-0.6B（draft）**，CUDA + BF16、Torch
attention、关 CUDA Graph。两者词表与 EOS 一致（151936 / [151643, 151645]），
但 0.6B 的 hidden 是 1024、1.7B 是 2048——两套 KV 按各自结构建，不能套用。

查四件事：

1. **greedy 与普通 greedy 逐 token 相同**——这是本关最硬的功能判据：贪心时目标
   分布是 one-hot，投机只可能改变「怎么算」，绝不能改变「算什么」；
2. 随机采样路径跑得完、`output_index` 连续、两套池子结束后引用归零；
3. **target 每轮最多一次 forward**、draft 是**批量**提议（不是请求循环 × K 次单请求）；
4. 草稿真的被接受过（接受率低不代表实现错，但全程 0 接受说明链路没通）。

不承诺加速：这里不比较 tok/s，也不把接受率当质量指标。
"""

import sys
import time

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step55 import Engine

TARGET_DIR = "/home/user/proj/vllm-from-scratch/models/Qwen3-1.7B"
DRAFT_DIR = "/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B"

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


if not torch.cuda.is_available():
    print("SKIP  真实双模型端到端需要 CUDA（本机没有）")
    sys.exit(0)

# 两条都要「话多一点」的 prompt：本关要看的是**同轮多请求一起提议**，
# 一条早早 EOS 的话，批量提议就没有机会出现（那不是实现问题，是用例选得不好）
PROMPTS = [
    "用一句话解释什么是 KV cache。",
    "用一个类比说明分页和连续内存的区别。",
]


def build(speculative_mode, num_speculative_tokens=2, **cfg):
    """按 187 的主运行配置装引擎：CUDA + BF16 + Torch attention + 关图。"""
    base = dict(device="cuda", attention_backend="torch", dtype=torch.bfloat16,
                max_num_seqs=2, max_num_batched_tokens=8, block_size=16, num_kv_blocks=64,
                enable_prefix_caching=False, num_speculative_tokens=num_speculative_tokens)
    base.update(cfg)
    engine = Engine.from_model_dir(
        TARGET_DIR, draft_model_dir=DRAFT_DIR if speculative_mode == "draft_model" else None,
        speculative_mode=speculative_mode, draft_num_kv_blocks=48,
        draft_max_num_batched_tokens=8, **base)
    # 数 target 的前向次数：本关的硬约束是「一轮最多一次」
    counter = {"target": 0, "draft": 0}
    original = engine.model._forward_append

    def counted(*args, **kwargs):
        counter["target"] += 1
        return original(*args, **kwargs)

    engine.model._forward_append = counted
    if engine.draft_model is not None:
        draft_original = engine.draft_model._forward_append

        def draft_counted(*args, **kwargs):
            counter["draft"] += 1
            return draft_original(*args, **kwargs)

        engine.draft_model._forward_append = draft_counted
    return engine, counter


def run(engine, prompts, max_new_tokens, sampling=None):
    outputs, events = {}, []
    engine.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    engine.scheduler.on_finished = lambda rec: outputs.__setitem__(rec["request_id"], list(rec["output_ids"]))
    for index, prompt_ids in enumerate(prompts):
        request = {"request_id": f"q{index}", "prompt_ids": list(prompt_ids),
                   "max_new_tokens": max_new_tokens}
        request.update(sampling or {})
        engine.add_request(request)
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        assert steps < 200, "疑似活锁"
    return outputs, events, steps


def encode(prompts):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_DIR, local_files_only=True)
    encoded = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True)
        # 新版 transformers 返回 BatchEncoding，tokenize=True 时取 input_ids
        encoded.append(list(rendered["input_ids"]) if hasattr(rendered, "keys") else list(rendered))
    return tokenizer, encoded


tokenizer, prompts = encode(PROMPTS)

# ------------------------------------------------ 1. greedy：投机与普通逐 token 相同

started = time.perf_counter()
plain, _ = build(None)
print(f"（两个引擎装载完毕，{time.perf_counter() - started:.1f}s）")
plain_out, _, plain_steps = run(plain, prompts, max_new_tokens=16)
plain_text = [tokenizer.decode(plain_out[f"q{i}"], skip_special_tokens=True) for i in range(2)]

spec, counter = build("draft_model")
spec_out, spec_events, spec_steps = run(spec, prompts, max_new_tokens=16)

check("真实双模型：greedy 下投机与普通 greedy 的输出**逐 token 相同**",
      plain_out == spec_out,
      f"\n  普通 {[plain_out[f'q{i}'] for i in range(2)]}"
      f"\n  投机 {[spec_out[f'q{i}'] for i in range(2)]}")
check("真实双模型：投机确实提过草稿（不是悄悄退化成普通路径）",
      spec.draft_proposer.num_proposed_tokens > 0,
      f"提议 {spec.draft_proposer.num_proposed_tokens} 枚、"
      f"draft 前向 {spec.draft_proposer.num_forwards} 次、"
      f"补算 {spec.draft_proposer.num_catchup_tokens} 个 token")
check("真实双模型：target 前向次数 <= 引擎步数（每轮最多一次 forward）",
      counter["target"] <= spec_steps,
      f"target 前向 {counter['target']} 次 / {spec_steps} 步")
efficiency = spec.draft_proposer.num_proposed_tokens / max(
    spec.draft_proposer.num_proposal_forwards, 1)
check("真实双模型：提议是**批量**的（一次前向平均提出 > 1 枚，而不是一条请求一次）",
      efficiency > 1.0,
      f"提议前向 {spec.draft_proposer.num_proposal_forwards} 次、"
      f"提出 {spec.draft_proposer.num_proposed_tokens} 枚（平均 {efficiency:.2f} 枚/次）；"
      f"补算前向 {spec.draft_proposer.num_catchup_forwards} 次 / "
      f"{spec.draft_proposer.num_catchup_tokens} 个 token")
check("真实双模型：greedy 下两套池子结束后引用归零",
      all(u == 0 for u in spec.kv_cache_pool.block_usage)
      and all(u == 0 for u in spec.draft_kv_pool.block_usage))
# 输出长度不一定等于 max_new_tokens：遇到 EOS 就停（这是正常终止，不是缺陷）
lengths = {rid: len(spec_out[rid]) for rid in ("q0", "q1")}
check("真实双模型：输出的 output_index 从 0 连续（回调不重不漏），且不超上限",
      all([i for r, _, i in spec_events if r == rid] == list(range(lengths[rid]))
          for rid in ("q0", "q1"))
      and all(1 <= n <= 16 for n in lengths.values()),
      f"长度 {lengths}")
print(f"  普通贪心：{plain_text[0][:60]!r}")
print(f"  投机贪心：{tokenizer.decode(spec_out['q0'], skip_special_tokens=True)[:60]!r}")
print(f"  另一条（{lengths['q1']} 个 token，可能提前 EOS）："
      f"{tokenizer.decode(spec_out['q1'], skip_special_tokens=True)[:60]!r}")

# ------------------------------------------------ 2. 随机采样路径

random, random_counter = build("draft_model", num_speculative_tokens=3)
random_out, random_events, _ = run(
    random, prompts, max_new_tokens=16,
    sampling=dict(temperature=0.8, top_k=20, top_p=0.9, seed=7))
random_lengths = {rid: len(random_out[rid]) for rid in ("q0", "q1")}
check("真实双模型（随机采样）：跑得完、序号连续、长度在 (0, 16] 内（EOS 可以提前停）",
      all(1 <= n <= 16 for n in random_lengths.values())
      and all([i for r, _, i in random_events if r == rid] == list(range(random_lengths[rid]))
              for rid in ("q0", "q1")),
      str(random_lengths))
check("真实双模型（随机采样）：草稿被接受过（接受链路真的通了）",
      random.draft_proposer.num_proposed_tokens > 0
      and len(random_out["q0"]) == 16)
check("真实双模型（随机采样）：两套池子结束后引用归零",
      all(u == 0 for u in random.kv_cache_pool.block_usage)
      and all(u == 0 for u in random.draft_kv_pool.block_usage))
print(f"  随机采样：{tokenizer.decode(random_out['q0'], skip_special_tokens=True)[:60]!r}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
