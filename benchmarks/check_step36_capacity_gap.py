"""有界复现：队首请求超出容量时，调度器不报错也不推进。

场景来自文档里的复现：
    池子只有 2 块；队首 A 要预留 5 块；后面的 B 只需要 1 块。
    连续跑三次 step()：既没有模型计算，也没有错误返回。

这里只跑到固定步数就停，不等它「永远结束」。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import step35 as m

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "step30_qwen3" / "tiny_gqa"

def run(tag, num_kv_blocks, requests, max_steps=3):
    e = m.Engine.from_model_dir(str(FIX), device="cuda", attention_backend="torch",
                                max_num_seqs=4, max_num_batched_tokens=8,
                                num_kv_blocks=num_kv_blocks, block_size=4,
                                enable_prefix_caching=False)
    for r in requests:
        e.add_request(dict(r))
    rows = []
    for k in range(max_steps):
        e.step()
        used = sum(1 for u in e.kv_cache_pool.block_usage if u > 0)
        rows.append(dict(step=k + 1, running=len(e.scheduler.running),
                         waiting=len(e.scheduler.waiting), blocks_in_use=used,
                         scheduled=len(e.scheduler.scheduled_items)))
    print(f"  --- {tag}（池子 {num_kv_blocks} 块）---")
    for r in rows:
        print(f"      step {r['step']}: running={r['running']} waiting={r['waiting']} "
              f"用块={r['blocks_in_use']} 本轮计划={r['scheduled']}")
    stuck = (rows[-1]["running"] == 0 and rows[-1]["blocks_in_use"] == 0
             and rows[-1]["waiting"] > 0 and rows[-1]["scheduled"] == 0)
    print(f"      队首卡住、无进展 = {stuck}   还有未完成请求 = {e.has_unfinished_requests()}")
    return stuck, rows

print("=== A. 复现：队首要 5 块但池子只有 2 块 ===")
A = {"request_id": "A", "prompt_ids": [1, 5, 3, 9, 2, 7, 4, 0, 6, 10], "max_new_tokens": 10}
B = {"request_id": "B", "prompt_ids": [2], "max_new_tokens": 1}
stuck, _ = run("A 要 5 块、B 只要 1 块，池子 2 块", 2, [A, B])

print()
print("=== B. 对照：容量够时正常推进 ===")
run("同样两个请求，池子 32 块", 32, [A, B])

print()
print("=== C. 对照：小请求单独在队列里时，池子 2 块够用 ===")
# 单独放小请求，证明「池子 2 块 + 小请求」本身是可行的；
# 于是 A 里 B 进不来，只能是因为队首 A 把它挡住了，而不是 B 自己不可行。
small = {"request_id": "S", "prompt_ids": [2], "max_new_tokens": 1}
run("只有小请求 S（1 块）", 2, [small])

print()
print("=== D. 单个请求本身就超容量（无解配置）===")
huge = {"request_id": "H", "prompt_ids": [1] * 40, "max_new_tokens": 40}
run("一个请求要 20 块，池子 2 块", 2, [huge], max_steps=2)

print()
print("=== E. 容量检查：请求自身的预留需求 vs 池子总量 ===")
import math
pool_blocks = 2
for name, req in (("A", A), ("huge", huge)):
    need = math.ceil((len(req["prompt_ids"]) + req["max_new_tokens"] - 1) / 4)
    print(f"  {name:5} prompt={len(req['prompt_ids']):>2} 预算={req['max_new_tokens']:>2} "
          f"-> 需要 {need} 块；池子 {pool_blocks} 块 -> {'永远不可能满足' if need > pool_blocks else '可以满足'}")
print("  -> 现在的实现里，这类请求只会让接纳循环直接 break，没有报错、也没有拒绝")
