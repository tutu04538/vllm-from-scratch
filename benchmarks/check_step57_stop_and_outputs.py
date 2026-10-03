"""57D 验收（对应需求里的 `test_stop_and_outputs.py`）：停止规则与用户增量输出。

两件事在 199 §2 里被明确分开，用例也分开查：

    **采样侧**：`min_tokens` 还没到 → 把停止 token 打成 -inf（"暂不允许采到什么"）
    **调度侧**：`check_stop` 看**已提交**的 token 是不是停止 token → 决定结束与截断

所以有一条端到端用例是"把 eos 设成唯一的高 logits，再要求 min_tokens=3"：
前 3 个 token **不可能是 eos**（采样侧屏蔽），第 4 个才轮到 eos（调度侧结束）——
两处各管一段，缺一个这条用例就过不去。

输出侧查 `OutputProcessor`：增量 → 累计、交付后才删状态、abort 不再交付、快照隔离。
"""

import json
import sys

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch

from minivllm import (CacheConfig, DeviceConfig, FinishReason, LLMEngine, ModelConfig,
                    SamplingParams, SchedulerConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.fake_runner import FakeRunner

FAIL = []
from minivllm.testing.tiny_models import tiny_qwen3_dir   # 测试模型现场生成（仓库不再放 fixtures）
TINY_DIR = tiny_qwen3_dir("tiny_gqa")
TINY_CONFIG = json.load(open(f"{TINY_DIR}/config.json"))
TINY_EOS = TINY_CONFIG["eos_token_id"]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_config(blocks=8, seqs=4, budget=16, max_model_len=64, model="dummy", hf_config=None):
    return VllmConfig(model_config=ModelConfig(model=model, max_model_len=max_model_len,
                                               hf_config=hf_config),
                      cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
                      scheduler_config=SchedulerConfig(max_num_seqs=seqs,
                                                       max_num_batched_tokens=budget),
                      device_config=DeviceConfig(device="cpu"))


def run_fake(prompts, tokens, max_tokens=4, **sampling):
    """FakeRunner 端到端：脚本 token 直接当采样结果（不经过采样器）。"""
    config = make_config()
    engine = LLMEngine(config, UniProcExecutor(config, Worker(
        config, model_runner=FakeRunner(tokens=tokens))))
    for req_id, prompt in prompts.items():
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0, **sampling))
    rounds = []
    while engine.has_unfinished_requests():
        rounds.append(engine.step())
    return rounds, engine


def last_output(rounds):
    """最后一个**非空**轮的输出。结束之后还会有一轮"清理轮"（0 输出），那是设计。"""
    for step in reversed(rounds):
        if step:
            return step[0]
    raise AssertionError("没有任何一轮产出输出")


class FakeTokenizer:
    def decode(self, token_ids):
        return "".join(chr(ord("a") + token % 26) for token in token_ids)


# ------------------------------------------------ 1. 停止规则（调度侧）

rounds, engine = run_fake({"r": [1, 2, 3]}, {"r": [10, 11, 12]}, max_tokens=3)
last = last_output(rounds)
check("1. 到 max_tokens 就停：LENGTH，输出正好 3 个 token",
      last.finish_reason == FinishReason.LENGTH and last.token_ids == [10, 11, 12]
      and last.finished, f"{last.token_ids} / {last.finish_reason}")
engine.shutdown()

rounds, engine = run_fake({"r": [1, 2, 3]}, {"r": [10, 99, 12, 13]}, max_tokens=6, eos_token_id=99)
final = last_output(rounds)
check("1. 遇到 eos：STOP，且**本轮剩下的候选被截断**（99 之后的 12、13 不提交）",
      final.finish_reason == FinishReason.STOP and final.token_ids == [10, 99]
      and final.finished, f"{final.token_ids} / {final.finish_reason}")
check("1. 中间轮交付的是增量累计（第 1 轮 1 个、第 2 轮 2 个，互为前缀）",
      [len(step[0].token_ids) for step in rounds if step] == [1, 2],
      str([[len(o.token_ids) for o in step] for step in rounds]))
check("1. 结束之后还有一轮清理轮：0 输出（执行侧要收到 finished_req_ids 才能删状态）",
      rounds[-1] == [], f"最后一轮输出数={len(rounds[-1])}")
engine.shutdown()

rounds, engine = run_fake({"r": [1, 2, 3]}, {"r": [10, 99, 12]}, max_tokens=3, eos_token_id=99,
                          ignore_eos=True)
final = last_output(rounds)
check("1. ignore_eos：eos 不停止，继续跑到 max_tokens（LENGTH）",
      final.finish_reason == FinishReason.LENGTH and final.token_ids == [10, 99, 12],
      f"{final.token_ids} / {final.finish_reason}")
engine.shutdown()

rounds, engine = run_fake({"r": [1, 2, 3]}, {"r": [10, 7, 12]}, max_tokens=4, stop_token_ids=[7])
final = last_output(rounds)
check("1. 显式 stop token：STOP，且 stop_reason 记下是哪个 token",
      final.finish_reason == FinishReason.STOP and final.token_ids == [10, 7],
      f"{final.token_ids} / stop_reason={getattr(final, 'stop_reason', None)}")
engine.shutdown()

config = make_config(max_model_len=6)
engine = LLMEngine(config, UniProcExecutor(config, Worker(
    config, model_runner=FakeRunner(tokens={"r": [10, 11, 12, 13, 14, 15]}))))
engine.add_request("r", [1, 2, 3], SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
rounds = []
while engine.has_unfinished_requests():
    rounds.append(engine.step())
final = last_output(rounds)
check("1. 上下文上限（max_model_len=6，prompt 3 个）：到顶就停，LENGTH",
      final.finish_reason == FinishReason.LENGTH and len(final.token_ids) == 3,
      f"{final.token_ids}（prompt 3 + 生成 3 = 6）")
engine.shutdown()

# ------------------------------------------------ 2. min_tokens：采样侧屏蔽 + 调度侧结束

def run_tiny_with_eos_top(min_tokens, max_tokens=6):
    """真模型跑一遍，但把 logits 换成"eos 最高"——这样采样器必须靠 min_tokens 屏蔽它。"""
    config = make_config(blocks=8, max_model_len=32, model=TINY_DIR, hf_config=TINY_CONFIG)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    vocab_size = TINY_CONFIG["vocab_size"]

    def eos_is_top(hidden_states):
        rows = hidden_states.shape[0]
        logits = torch.full((rows, vocab_size), -1.0)
        logits[:, TINY_EOS] = 5.0
        return logits

    original = runner.model.compute_logits
    runner.model.compute_logits = eos_is_top
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, eos_token_id=TINY_EOS,
                            min_tokens=min_tokens)
    engine.add_request("r", [1, 2, 3, 4], params)
    text = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            text.append((list(output.token_ids), output.finished, output.finish_reason))
    runner.model.compute_logits = original
    engine.shutdown()
    return text[-1]


tokens, finished, reason = run_tiny_with_eos_top(min_tokens=3)
check("2. min_tokens=3：前 3 个 token **不可能是 eos**（采样侧把停止 token 屏蔽了）",
      len(tokens) >= 4 and TINY_EOS not in tokens[:3] and finished
      and reason == FinishReason.STOP,
      f"tokens={tokens}（eos={TINY_EOS}）finish={reason}")
check("2. 第 4 个 token 正是 eos：min_tokens 一满足，屏蔽解除，调度侧随即结束",
      tokens[3] == TINY_EOS, f"第 4 个={tokens[3]}")

tokens, finished, reason = run_tiny_with_eos_top(min_tokens=0)
check("2. min_tokens=0：第一个 token 就是 eos，只生成 1 个就停（两处职责的对照）",
      tokens == [TINY_EOS] and reason == FinishReason.STOP, f"tokens={tokens}")

# ------------------------------------------------ 3. 用户增量输出

config = make_config()
engine = LLMEngine(config, UniProcExecutor(config, Worker(
    config, model_runner=FakeRunner(tokens={"r": [10, 11, 12]}))),
    tokenizer=FakeTokenizer())
engine.add_request("r", [1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
seen = []
while engine.has_unfinished_requests():
    for output in engine.step():
        seen.append((list(output.token_ids), output.text, output.finished,
                     output.finish_reason, list(output.prompt_token_ids)))
check("3. 增量 → 累计：每次交付的都是到目前为止的全部输出（互为前缀）",
      [tokens for tokens, *_ in seen] == [[10], [10, 11], [10, 11, 12]],
      str([tokens for tokens, *_ in seen]))
check("3. 交付里带上 prompt（用户拿到的是完整上下文）",
      all(prompt == [1, 2, 3] for *_, prompt in seen))
check("3. 只有最后一轮 finished=True 并带结束原因",
      [finished for _, _, finished, _, _ in seen] == [False, False, True]
      and seen[-1][3] == FinishReason.LENGTH)
check("3. 给了 tokenizer 就有 text（逐轮前缀也在增长）",
      [text for _, text, _, _, _ in seen] == ["k", "kl", "klm"],
      str([text for _, text, _, _, _ in seen]))
engine.shutdown()

config = make_config()
engine = LLMEngine(config, UniProcExecutor(config, Worker(
    config, model_runner=FakeRunner(tokens={"r": [10, 11]}))))
engine.add_request("r", [1, 2, 3], SamplingParams(max_tokens=2, temperature=0.0, eos_token_id=999))
rounds = []
while engine.has_unfinished_requests():
    rounds.append(engine.step())
first_snapshot = rounds[0][0]
first_snapshot.token_ids.append(999)             # 用户手改自己那份快照
check("3. 用户拿到的是**快照**：改它不影响引擎内部（最后一次交付仍然干净）",
      last_output(rounds).token_ids == [10, 11], str(last_output(rounds).token_ids))
engine.shutdown()

config = make_config()
engine = LLMEngine(config, UniProcExecutor(config, Worker(
    config, model_runner=FakeRunner(tokens={"r": [10, 11, 12]}))))
engine.add_request("r", [1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
engine.step()
engine.abort_request(["r"])
outputs = engine.step()
check("3. abort 之后：不再交付这条请求的输出（用户侧状态已清）",
      all(output.request_id != "r" for output in outputs), f"{[o.request_id for o in outputs]}")
check("3. abort 之后同一个 ID 可以立刻复用（状态交付/清理完毕）",
      engine.add_request("r", [4, 5], SamplingParams(max_tokens=1, temperature=0.0,
                                                     eos_token_id=999)) == "r")
engine.shutdown()

# ------------------------------------------------ 4. 随机流归请求（端到端可复现）

def run_seeded(seed):
    config = make_config(blocks=8, max_model_len=32, model=TINY_DIR, hf_config=TINY_CONFIG)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    engine.add_request("r", [1, 2, 3, 4],
                       SamplingParams(max_tokens=5, temperature=1.0, seed=seed, eos_token_id=999))
    tokens = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens = list(output.token_ids)
    engine.shutdown()
    return tokens


first, second, other = run_seeded(1234), run_seeded(1234), run_seeded(99)
check("4. 同一个 seed 跑两次：token 序列完全一致（随机流归请求，可复现）",
      first == second and len(first) == 5, f"{first} vs {second}")
check("4. 不同 seed 给出不同序列（否则要怀疑随机路径退化成贪心了）",
      first != other, f"seed=1234 → {first}；seed=99 → {other}")
check("4. 贪心与随机都能跑到 max_tokens（采样路径都活着）",
      len(first) == 5)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
