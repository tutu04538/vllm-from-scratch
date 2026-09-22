"""第 44 关 §16 第 2 步：未计算 token 的新定义 + legacy 等价回归。

两件事：
  1. 手工把一条请求的 cache 清零（等价于被抢占），重放 prompt + 旧 output，
     只能产生一个**新** token —— 不是把旧 output 又生成一遍。
  2. step43 与 step44(preemption_mode=None) 在同一批随机配置上逐 token 相同。

CPU / FP32 / greedy，完全确定。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step43 import Engine as Engine43
from step44 import Engine as Engine44

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make(engine_cls, *, block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16,
         enable_prefix_caching=False, **kw):
    torch.manual_seed(1234)
    return engine_cls(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                      block_size=block_size, num_kv_blocks=num_kv_blocks,
                      device="cpu", enable_prefix_caching=enable_prefix_caching,
                      **DIMS, **kw)


def run(engine, requests, limit=400):
    for r in requests:
        engine.add_request(r)
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        if steps > limit:
            raise RuntimeError(f"超过 {limit} 步仍未排空")
    return {rec["request_id"]: rec["output_ids"] for rec in engine.scheduler.step_done
            if "error" not in rec}


# ---------------------------------------------------------------- 1. 重放语义

PROMPT = [1, 5, 9, 3, 7, 11]
ref = run(make(Engine44), [{"request_id": "A", "prompt_ids": PROMPT, "max_new_tokens": 4}])
OUT = ref["A"]
check("独占运行拿到 4 个 token", len(OUT) == 4, f"{OUT}")

# 被抢占的等价形式：历史 = prompt + 已经生成的前 k 个 token，KV 清空重放
for k in range(len(OUT)):
    history = PROMPT + OUT[:k]
    got = run(make(Engine44), [{"request_id": "R", "prompt_ids": history, "max_new_tokens": 1}])
    check(f"重放 prompt+前 {k} 个 output -> 恰好 1 个新 token",
          got["R"] == [OUT[k]], f"得到 {got['R']}，期望 [{OUT[k]}]")

# 换一个不会退化成重复 token 的模型确认不是「凑巧」
torch.manual_seed(99)
big = Engine44(max_num_seqs=1, max_num_batched_tokens=64, block_size=4, num_kv_blocks=64,
               device="cpu", enable_prefix_caching=False, eos_token_ids=[96],
               vocab_size=97, d_model=32, max_seq_len=64, num_q_heads=4, num_kv_heads=2,
               num_layers=3, intermediate_size=64, head_dim=8)
ref2 = run(big, [{"request_id": "B", "prompt_ids": PROMPT + [40, 41], "max_new_tokens": 6}])["B"]
check("第二组独占运行的输出不是常量", len(set(ref2)) > 1, f"{ref2}")
torch.manual_seed(99)
big2 = Engine44(max_num_seqs=1, max_num_batched_tokens=64, block_size=4, num_kv_blocks=64,
                device="cpu", enable_prefix_caching=False, eos_token_ids=[96],
                vocab_size=97, d_model=32, max_seq_len=64, num_q_heads=4, num_kv_heads=2,
                num_layers=3, intermediate_size=64, head_dim=8)
hist = PROMPT + [40, 41] + ref2[:3]
got2 = run(big2, [{"request_id": "B2", "prompt_ids": hist, "max_new_tokens": 1}])["B2"]
check("第二组重放同样只多出 1 个 token", got2 == [ref2[3]], f"{got2} vs [{ref2[3]}]")

# ------------------------------------------------- 2. 未计算 token 数的定义

torch.manual_seed(7)
LONG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
e = make(Engine44, max_num_batched_tokens=8)
e.add_request({"request_id": "C", "prompt_ids": LONG, "max_new_tokens": 3})
e.step()                                   # 分块 prefill：预算 8，prompt 12，第一块算 8 个
seq = e.scheduler.running[0]
check("分块 prefill 中间块：cache.length 只推进到预算", seq.cache.length == 8, str(seq.cache.length))
check("中间块没有 output（不采样）", seq.output_ids == [], str(seq.output_ids))
check("未计算数 = 剩下的 prompt token", seq.num_uncomputed_tokens == 4, str(seq.num_uncomputed_tokens))
check("prefill_len 旧名等价", seq.prefill_len == seq.num_uncomputed_tokens)
e.step()                                   # 第二块算完剩下的 4 个 prompt token 并采样
seq = e.scheduler.running[0]
check("prompt 算完后 all_token_ids = prompt + output",
      seq.all_token_ids == LONG + seq.output_ids, str(seq.output_ids))
check("新采样的 token 还没进 KV，未计算数为 1", seq.num_uncomputed_tokens == 1,
      f"cache.length={seq.cache.length} output={seq.output_ids}")
check("这一状态被认成 ready（等价于旧 decode）", seq.is_ready_for_next_token)

# ------------------------------------------------------------- 3. legacy 等价

CASES = [
    dict(block_size=4, num_kv_blocks=32, max_num_seqs=2, max_num_batched_tokens=16),
    dict(block_size=4, num_kv_blocks=8, max_num_seqs=2, max_num_batched_tokens=16),
    dict(block_size=2, num_kv_blocks=16, max_num_seqs=3, max_num_batched_tokens=8),
    dict(block_size=8, num_kv_blocks=4, max_num_seqs=1, max_num_batched_tokens=4),
    dict(block_size=1, num_kv_blocks=64, max_num_seqs=4, max_num_batched_tokens=32),
]
REQS = [
    {"request_id": "p1", "prompt_ids": [1, 2, 3], "max_new_tokens": 5},
    {"request_id": "p2", "prompt_ids": [4, 5, 6, 7, 8, 9, 10], "max_new_tokens": 3},
    {"request_id": "p3", "prompt_ids": [11], "max_new_tokens": 6},
    {"request_id": "p4", "prompt_ids": [1, 2, 3], "max_new_tokens": 4},   # 前缀重复
    {"request_id": "z", "prompt_ids": [2], "max_new_tokens": 0},          # 零预算
]

for i, cfg in enumerate(CASES):
    for pc in (False, True):
        a = run(make(Engine43, enable_prefix_caching=pc, **cfg), REQS)
        b = run(make(Engine44, enable_prefix_caching=pc, **cfg), REQS)
        check(f"legacy 等价 case{i} prefix_cache={pc}", a == b,
              "" if a == b else f"\n  step43={a}\n  step44={b}")

# on_token 在 legacy 下也保持一致
def collect(engine_cls, **kw):
    events = []
    eng = make(engine_cls, **kw)
    eng.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    run(eng, REQS)
    return events

ea = collect(Engine43)
eb = collect(Engine44)
check("legacy 下 on_token 事件序列一致", ea == eb, "" if ea == eb else f"\n  {ea}\n  {eb}")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
