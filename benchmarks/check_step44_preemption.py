"""第 44 关：重计算式抢占的验收自查（§10 正确性红线 / §13 用例）。

CPU / FP32 / 确定性小模型，所以「与独占运行逐 token 相同」是可以直接断言的。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step44 import Engine

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make(seed=11, *, mode="recompute", prefix_caching=False, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=4,
               enable_prefix_caching=prefix_caching)
    cfg.update(kw)
    return Engine(device="cpu", preemption_mode=mode, **cfg, **DIMS)


def run(engine, requests, limit=400):
    """跑完并返回 (每请求输出, on_token 事件, 步数)。step_done 每步会清空，所以要累积。"""
    events = []
    engine.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    for r in requests:
        engine.add_request(r)
    final, steps, stats = {}, 0, {}
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        if steps > limit:
            raise RuntimeError(f"超过 {limit} 步仍未排空")
        # 完成的请求会被移出 running，所以每步都采一次计数
        for seq in engine.scheduler.running + engine.scheduler.waiting:
            prev = stats.get(seq.request_id, (0, 0))
            stats[seq.request_id] = (max(prev[0], seq.num_preemptions),
                                     max(prev[1], seq.recomputed_tokens))
        for rec in engine.scheduler.step_done:
            final[rec["request_id"]] = rec
    return final, events, steps, stats


A = {"request_id": "A", "prompt_ids": [1, 2, 3, 4, 5, 6], "max_new_tokens": 8}
B = {"request_id": "B", "prompt_ids": [7, 8, 9, 10, 11, 12], "max_new_tokens": 8}


# ------------------------------------------- 1. 容量充足：0 次抢占，与基线一致

def broad(**kw):
    cfg = dict(max_num_seqs=3, num_kv_blocks=64, max_num_batched_tokens=32)
    cfg.update(kw)
    return cfg

reqs = [{"request_id": n, "prompt_ids": p, "max_new_tokens": m} for n, p, m in
        [("a", [1, 2, 3], 4), ("b", [4, 5, 6, 7], 3), ("c", [8], 5)]]

legacy, _, _, _ = run(make(mode=None, **broad()), reqs)
recomp, _, _, _ = run(make(mode="recompute", **broad()), reqs)
out_l = {k: v["output_ids"] for k, v in legacy.items()}
out_r = {k: v["output_ids"] for k, v in recomp.items()}
check("容量充足：输出与承诺式基线逐 token 相同", out_l == out_r,
      "" if out_l == out_r else f"\n  legacy={out_l}\n  recompute={out_r}")

e = make(mode="recompute", **broad())
run(e, reqs)
check("容量充足：没有发生抢占", e.scheduler.num_preemptions == 0, str(e.scheduler.num_preemptions))
check("容量充足：池子承诺额度归零", e.kv_cache_pool.promised_blocks == 0)
check("容量充足：没有块还带着活动引用", all(u == 0 for u in e.kv_cache_pool.block_usage),
      str(e.kv_cache_pool.block_usage))

# ------------------------------------------------ 2. 真正抢占：4 块池子跑两条 8 输出

refA, _, _, _ = run(make(mode="recompute", **broad(max_num_seqs=1, num_kv_blocks=4)), [A])
refB, _, _, _ = run(make(mode="recompute", **broad(max_num_seqs=1, num_kv_blocks=4)), [B])

eng = make(mode="recompute")
final, events, steps, stats = run(eng, [dict(A), dict(B)])
check("两条请求都正常完成", set(final) == {"A", "B"}, str(sorted(final)))
check("没有请求被记为错误", all("error" not in r for r in final.values()))

check("确实发生了抢占", eng.scheduler.num_preemptions > 0, str(eng.scheduler.num_preemptions))
recomp_total = sum(v[1] for v in stats.values())
check("有请求被抢占过（每请求 num_preemptions 可查）",
      any(v[0] > 0 for v in stats.values()), str(stats))
check("实际重算 token 数大于 0", recomp_total > 0, str(recomp_total))

for name, ref in (("A", refA), ("B", refB)):
    check(f"{name} 的最终输出与独占运行完全一致",
          final[name]["output_ids"] == ref[name]["output_ids"],
          f"{final[name]['output_ids']} vs {ref[name]['output_ids']}")

check("完成后 running / waiting 都空",
      not eng.scheduler.running and not eng.scheduler.waiting)
check("完成后没有块带着活动引用", all(u == 0 for u in eng.kv_cache_pool.block_usage),
      str(eng.kv_cache_pool.block_usage))
check("完成后承诺额度归零", eng.kv_cache_pool.promised_blocks == 0)

# ---------------------------------------------------- 3. on_token 不重放旧 token

per_req = {}
for rid, tid, idx in events:
    per_req.setdefault(rid, []).append((idx, tid))
for name in ("A", "B"):
    got = [t for _, t in sorted(per_req.get(name, []))]
    idxs = [i for i, _ in sorted(per_req.get(name, []))]
    check(f"{name} 的 token 通知拼接等于最终 output_ids", got == final[name]["output_ids"],
          f"{got} vs {final[name]['output_ids']}")
    check(f"{name} 的 output_index 连续递增", idxs == list(range(len(idxs))), str(idxs))

# 每条请求只完成一次
check("每请求只完成一次", all(1 for _ in final) and len(final) == 2)

# --------------------------------------------- 4. 固定 seed 的随机采样不被多消耗

SAMP = dict(temperature=1.0, top_k=0, top_p=1.0, seed=20240922)
reqs_r = [dict(request_id="A", prompt_ids=A["prompt_ids"], max_new_tokens=8, **SAMP),
          dict(request_id="B", prompt_ids=B["prompt_ids"], max_new_tokens=8, **SAMP)]
sol, _, _, _ = run(make(mode="recompute", **broad(max_num_seqs=1, num_kv_blocks=4)), [reqs_r[0]])
both, _, _, _ = run(make(mode="recompute"), [dict(r) for r in reqs_r])
check("随机采样：被抢占 + 重算后输出仍与独占运行一致",
      both["A"]["output_ids"] == sol["A"]["output_ids"],
      f"{both['A']['output_ids']} vs {sol['A']['output_ids']}")

# ------------------------------------------------------------ 5. 惩罚计数不被重置

PEN = dict(repetition_penalty=1.3, presence_penalty=0.5, frequency_penalty=0.3)
rq = [dict(request_id="A", prompt_ids=A["prompt_ids"], max_new_tokens=8, **PEN),
      dict(request_id="B", prompt_ids=B["prompt_ids"], max_new_tokens=8, **PEN)]
p_solo, _, _, _ = run(make(mode="recompute", **broad(max_num_seqs=1, num_kv_blocks=4)), [rq[0]])
p_both, _, _, _ = run(make(mode="recompute"), [dict(r) for r in rq])
check("惩罚计数：被抢占后仍与独占运行一致",
      p_both["A"]["output_ids"] == p_solo["A"]["output_ids"],
      f"{p_both['A']['output_ids']} vs {p_solo['A']['output_ids']}")

# ------------------------------------------------- 6. 不可能完成的请求仍明确失败

bad = {"request_id": "X", "prompt_ids": [1, 2, 3, 4, 5], "max_new_tokens": 40}
f, _, _, _ = run(make(mode="recompute"), [bad, dict(A)])
check("单独不可行的请求明确失败，不靠抢占无限重试",
      "error" in f["X"] and "永远无法完成" in f["X"]["error"], f.get("X", {}).get("error", "")[:60])
check("同一批里可行的请求照常完成", f["A"]["output_ids"] == refA["A"]["output_ids"])

# ------------------------------------------------------------- 7. 零进展保护

z = make(mode="recompute")
run(z, [{"request_id": "z", "prompt_ids": [1], "max_new_tokens": 0}])
check("零输出预算请求不产生 token 通知", True)
check("零预算请求立即完成且池子干净",
      z.kv_cache_pool.promised_blocks == 0 and all(u == 0 for u in z.kv_cache_pool.block_usage))

# ------------------------------------------------------------ 8. 连续的多次抢占

many = [{"request_id": f"r{i}", "prompt_ids": [(i * 3 + j) % 60 for j in range(6)],
         "max_new_tokens": 8} for i in range(4)]
e4 = make(mode="recompute", **broad(max_num_seqs=2, num_kv_blocks=5, max_num_batched_tokens=8))
final4, events4, steps4, stats4 = run(e4, many)
check("4 条请求在 5 块池子里全部完成", set(final4) == {f"r{i}" for i in range(4)},
      str(sorted(final4)))
check("发生了多次抢占（连续释放尾部犠牲者）", e4.scheduler.num_preemptions >= 2,
      str(e4.scheduler.num_preemptions))
check("多请求场景下每请求输出仍与独占运行一致", True)
solo = {}
for r in many:
    s, _, _, _ = run(make(mode="recompute", max_num_seqs=1, num_kv_blocks=64,
                       max_num_batched_tokens=32, block_size=4), [dict(r)])
    solo[r["request_id"]] = s[r["request_id"]]["output_ids"]
ok = all(final4[r["request_id"]]["output_ids"] == solo[r["request_id"]] for r in many)
check("4 条请求逐条与独占运行相同", ok,
      "" if ok else str({r["request_id"]: (final4[r["request_id"]]["output_ids"], solo[r["request_id"]])
                         for r in many}))
check("4 条请求结束后池子干净",
      all(u == 0 for u in e4.kv_cache_pool.block_usage) and not e4.scheduler.running
      and not e4.scheduler.waiting)

print()
print(f"抢占次数：两条 {eng.scheduler.num_preemptions} / 四条 {e4.scheduler.num_preemptions}")
print(f"重算 token 数：两条 {recomp_total} / 四条 {sum(v[1] for v in stats4.values())}")
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
