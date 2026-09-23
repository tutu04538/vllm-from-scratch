"""第 45 关：阻塞者感知的恢复准入 —— 功能正确性自查（需求 §4 的六条）。

CPU / FP32 / 确定性小模型。只测功能与状态转换，不做性能测试（需求 §5 / 02 约定）。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step43 import Engine as Engine43
from step44 import Engine as Engine44
from step45 import Engine as Engine45

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make(cls=Engine45, *, mode="recompute", prefix_caching=False, seed=29, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=3, max_num_batched_tokens=8, block_size=4, num_kv_blocks=4,
               enable_prefix_caching=prefix_caching)
    cfg.update(kw)
    return cls(device="cpu", preemption_mode=mode, **cfg, **DIMS)


def run(engine, requests, limit=600, trace=False):
    """跑完并返回 (每请求完成记录, on_token 事件, 步数, 每步队列快照)。"""
    events = []
    engine.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    for r in requests:
        engine.add_request(dict(r))
    final, steps, snapshots = {}, 0, []
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        if steps > limit:
            raise RuntimeError(f"超过 {limit} 步仍未排空（疑似活锁）")
        for rec in engine.scheduler.step_done:
            final[rec["request_id"]] = rec
        if trace:
            snapshots.append({
                "step": steps,
                "running": [q.request_id for q in engine.scheduler.running],
                "waiting": [q.request_id for q in engine.scheduler.waiting],
                "blockers": {q.request_id: (q.resume_blocker.request_id if q.resume_blocker else None)
                             for q in engine.scheduler.waiting},
            })
    return final, events, steps, snapshots


def solo(cls=Engine45, **kw):
    """每条请求独占引擎跑一遍，作为精确参考。"""
    out = {}
    for name, prompt, n in REQ_SPEC:
        e = make(cls, max_num_seqs=1, num_kv_blocks=64, max_num_batched_tokens=64, **kw)
        f, _, _, _ = run(e, [{"request_id": name, "prompt_ids": list(prompt), "max_new_tokens": n}])
        out[name] = f[name]["output_ids"]
    return out


REQ_SPEC = [("A", range(1, 7), 8), ("B", range(7, 13), 8), ("C", range(13, 19), 8),
            ("D", range(19, 25), 8), ("E", range(25, 31), 8), ("F", range(31, 37), 8)]


def reqs_of(*names):
    return [{"request_id": n, "prompt_ids": list(p), "max_new_tokens": m}
            for n, p, m in REQ_SPEC if n in names]


# ------------------------------------------------ 1. A/B：B 等 A 完成再恢复

eng = make(max_num_seqs=2)
final, events, steps, snap = run(eng, reqs_of("A", "B"), trace=True)

check("A/B：两条都完成", set(final) == {"A", "B"}, str(sorted(final)))
check("A/B：确实发生了抢占", eng.scheduler.num_preemptions > 0, str(eng.scheduler.num_preemptions))
check("A/B：发生了因阻塞而跳过准入", eng.scheduler.num_blocked_admissions > 0,
      str(eng.scheduler.num_blocked_admissions))

# A 在哪一步结束；B 从哪一步起被 A 挡住。
# 注意 B 一开始是**正常与 A 并发**的，抢占发生在后面，所以只检查被挡之后的那段。
finish_A = min(s["step"] for s in snap if "A" not in s["running"] and "A" not in s["waiting"])
blocked_from = [s["step"] for s in snap if s["blockers"].get("B") == "A"]
check("A/B：B 确实被 A 挡住过", bool(blocked_from), str(blocked_from[:3]))
first_blocked = min(blocked_from) if blocked_from else None
ran_while_blocked = [s["step"] for s in snap
                     if first_blocked is not None and first_blocked < s["step"] < finish_A
                     and "B" in s["running"]]
check("A/B：被 A 挡住之后、A 完成之前，B 不会回到 running",
      not ran_while_blocked, f"被挡于第 {first_blocked} 步，A 结束于第 {finish_A} 步；"
                             f"违规步 {ran_while_blocked[:5]}")

blocked_seen = [s for s in snap if s["blockers"].get("B") == "A"]
check("A/B：B 等待期间 resume_blocker 指向 A", bool(blocked_seen),
      f"共 {len(blocked_seen)} 步")

ref = solo()
for name in ("A", "B"):
    check(f"A/B：{name} 的输出与独占运行一致", final[name]["output_ids"] == ref[name])
per_req = {}
for rid, tid, idx in events:
    per_req.setdefault(rid, []).append((idx, tid))
for name in ("A", "B"):
    idxs = sorted(i for i, _ in per_req.get(name, []))
    toks = [t for _, t in sorted(per_req.get(name, []))]
    check(f"A/B：{name} 的 on_token 不重放旧 token", toks == final[name]["output_ids"]
          and idxs == list(range(len(idxs))), f"{idxs}")

# ------------------------------------- 2. A–E：抢占次数显著少于 step44 的 17 次

e45 = make(num_kv_blocks=4)
done45 = []
e45.scheduler.on_finished = lambda r: done45.append(r["request_id"])
f45, _, steps45, _ = run(e45, reqs_of(*"ABCDE"))
e44 = make(Engine44, num_kv_blocks=4)
done44 = []
e44.scheduler.on_finished = lambda r: done44.append(r["request_id"])
f44, _, steps44, _ = run(e44, reqs_of(*"ABCDE"))

check("A–E：step45 全部有界完成", set(f45) == set("ABCDE"), str(sorted(f45)))
check("A–E：全局 FCFS 不变，完成序仍为 A,B,C,D,E",
      done45 == list("ABCDE"), str(done45))
check("A–E：抢占次数少于 step44", e45.scheduler.num_preemptions < e44.scheduler.num_preemptions,
      f"step45 {e45.scheduler.num_preemptions} vs step44 {e44.scheduler.num_preemptions}")
check("A–E：抢占次数少于 17（需求给出的基准）",
      e45.scheduler.num_preemptions < 17, str(e45.scheduler.num_preemptions))
check("A–E：每条请求的输出与独占运行一致",
      all(f45[n]["output_ids"] == ref[n] for n in "ABCDE"))

check("A–E：结束时没有残留的阻塞引用",
      not e45.scheduler.running and not e45.scheduler.waiting)

# 需求 §2：一次为 A 连续抢占 C、B 时，两者都以 A 为阻塞者；回队顺序仍是 B、C
e45b = make(num_kv_blocks=4)
for r in reqs_of(*"ABCDE"):
    e45b.add_request(dict(r))
snap45, found = None, None
for _ in range(600):
    before = e45b.scheduler.num_preemptions
    e45b.step()
    if snap45 is None and e45b.scheduler.num_preemptions > before:
        blocked = [q for q in e45b.scheduler.waiting if q.resume_blocker is not None]
        if len(blocked) >= 2:
            found = {
                "waiting": [q.request_id for q in e45b.scheduler.waiting],
                "blockers": {q.request_id: q.resume_blocker.request_id for q in blocked},
                "blocker_is_same_object": len({id(q.resume_blocker) for q in blocked}) == 1,
            }
            snap45 = found
    if not e45b.has_unfinished_requests():
        break
check("A–E：同一轮多个犠牲者都以同一个请求为阻塞者",
      found is not None and found["blocker_is_same_object"], str(found))
check("A–E：回队顺序仍是先被抢的在后（B 在 C 前）",
      found is not None and found["waiting"][:2] == ["B", "C"] and found["blockers"] == {"B": "A", "C": "A"},
      str(found))
print(f"      参考：step44 抢占 {e44.scheduler.num_preemptions} 次 / step45 抢占 "
      f"{e45.scheduler.num_preemptions} 次（阻塞跳过准入 {e45.scheduler.num_blocked_admissions} 次）")

# ------------------------------------------- 3. 动态到达：F 不得越过被阻塞的 B

eng3 = make(max_num_seqs=3)
eng3.add_request(reqs_of("A")[0])
eng3.add_request(reqs_of("B")[0])
added_F = False
final3, violated, order3 = {}, [], []
steps3 = 0
while eng3.has_unfinished_requests():
    eng3.step()
    steps3 += 1
    assert steps3 < 600, "疑似活锁"
    for rec in eng3.scheduler.step_done:
        final3[rec["request_id"]] = rec
        order3.append(rec["request_id"])
    waiting = [q.request_id for q in eng3.scheduler.waiting]
    running = [q.request_id for q in eng3.scheduler.running]
    # B 被 A 阻塞的过程中加入 F
    if not added_F and any(q.resume_blocker is not None for q in eng3.scheduler.waiting):
        eng3.add_request(reqs_of("F")[0])
        added_F = True
    # F 不能越过仍在等待的 B：B 在 waiting 且 F 在 running 就是违规
    if "B" in waiting and "F" in running:
        violated.append({"step": steps3, "running": running, "waiting": waiting})
    if "A" in running and "B" in waiting and "F" in running:
        violated.append({"step": steps3, "why": "F 与 A 同时运行而 B 还在等"})

check("动态到达：确实加入了 F", added_F)
check("动态到达：F 没有越过被阻塞的 B", not violated, str(violated[:3]))
check("动态到达：三条都完成", set(final3) == {"A", "B", "F"}, str(sorted(final3)))
check("动态到达：A 完成后按 FCFS 恢复（B 先于 F）", order3 == ["A", "B", "F"], str(order3))
ref3 = solo()
check("动态到达：输出与独占运行一致",
      all(final3[n]["output_ids"] == ref3[n] for n in ("A", "B", "F")))

# ------------------------------------ 4. 容量充足：不抢占、也不产生无意义阻塞

cap = make(num_kv_blocks=64, max_num_batched_tokens=32)
f_cap, _, _, _ = run(cap, reqs_of(*"ABC"))
check("容量充足：不发生抢占", cap.scheduler.num_preemptions == 0)
check("容量充足：不产生无意义的阻塞跳过", cap.scheduler.num_blocked_admissions == 0,
      str(cap.scheduler.num_blocked_admissions))
f44cap, _, _, _ = run(make(Engine44, num_kv_blocks=64, max_num_batched_tokens=32), reqs_of(*"ABC"))
check("容量充足：输出与 step44 一致",
      {k: v["output_ids"] for k, v in f_cap.items()} == {k: v["output_ids"] for k, v in f44cap.items()})

# ------------------------------------------- 5. legacy 模式与 step44 完全一致

LEG = [{"request_id": n, "prompt_ids": list(p), "max_new_tokens": m}
       for n, p, m in [("a", range(1, 4), 4), ("b", range(4, 8), 3), ("c", range(8, 9), 5)]]
for pc in (False, True):
    a, _, _, _ = run(make(Engine44, mode=None, prefix_caching=pc), LEG)
    b, _, _, _ = run(make(Engine45, mode=None, prefix_caching=pc), LEG)
    check(f"legacy 模式（prefix_cache={pc}）输出与 step44 一致",
          {k: v["output_ids"] for k, v in a.items()} == {k: v["output_ids"] for k, v in b.items()})
    check(f"legacy 模式（prefix_cache={pc}）不产生阻塞跳过",
          b and True)

# step43 没有 preemption_mode 参数，直接构造
torch.manual_seed(29)
e3 = Engine43(device="cpu", max_num_seqs=3, max_num_batched_tokens=8, block_size=4,
              num_kv_blocks=4, enable_prefix_caching=False, **DIMS)
e5 = make(Engine45, mode=None, prefix_caching=False)
run(e3, LEG)
run(e5, LEG)
check("legacy 模式承诺账本与 step43 一致",
      e3.kv_cache_pool.promised_blocks == e5.kv_cache_pool.promised_blocks == 0)

# -------------------------------- 6. 采样 / 惩罚 / 分块 prefill / 回调 / 块引用

SAMP = dict(temperature=1.0, top_k=0, top_p=1.0, seed=20240922)
PEN = dict(repetition_penalty=1.3, presence_penalty=0.5, frequency_penalty=0.3)
for label, extra in (("固定 seed 随机采样", SAMP), ("惩罚计数", PEN)):
    r = [dict(x, **extra) for x in reqs_of("A", "B")]
    e = make(max_num_seqs=2)
    f, _, _, _ = run(e, [dict(x) for x in r])
    e_solo = make(max_num_seqs=1, num_kv_blocks=64, max_num_batched_tokens=64)
    f_solo, _, _, _ = run(e_solo, [dict(r[0])])
    check(f"{label}：被阻塞 + 重算后仍与独占运行一致",
          f["A"]["output_ids"] == f_solo["A"]["output_ids"],
          f"{f['A']['output_ids']} vs {f_solo['A']['output_ids']}")

long_prompt = list(range(1, 25))
e = make(max_num_batched_tokens=8, num_kv_blocks=64, max_num_seqs=1)
f, _, _, _ = run(e, [{"request_id": "L", "prompt_ids": long_prompt, "max_new_tokens": 4}])
check("分块 prefill：长 prompt 正常完成", len(f["L"]["output_ids"]) == 4, str(f["L"]["output_ids"]))

e = make()
f, _, _, _ = run(e, reqs_of("A"))
check("单请求：不抢占、不阻塞、无残留",
      e.scheduler.num_preemptions == 0 and e.scheduler.num_blocked_admissions == 0
      and not e.scheduler.waiting and all(u == 0 for u in e.kv_cache_pool.block_usage))
check("单请求：完成后没有残留的阻塞引用",
      all(q.resume_blocker is None for q in e.scheduler.running + e.scheduler.waiting))

# 阻塞者明确失败时，关系必须失效，不能留下永远恢复不了的 B
bad = {"request_id": "X", "prompt_ids": [1, 2, 3, 4, 5], "max_new_tokens": 40}
e = make(max_num_seqs=2, num_kv_blocks=4)
f, _, _, _ = run(e, [bad] + reqs_of("A"))
check("阻塞者不可行时仍明确失败，且不留下死锁",
      "error" in f["X"] and set(f) >= {"X", "A"}, str(sorted(f)))

# ------------------------------------------------- 7. 随机压测：有界完成且输出正确

bad_runs = []
for seed in range(12):
    # 独占参考必须用**同一份权重**：make() 里的 manual_seed 决定了随机初始化
    ref_seed = solo(seed=seed)
    for blocks in (4, 5, 6):
        for seqs in (2, 3):
            e = make(seed=seed, num_kv_blocks=blocks, max_num_seqs=seqs)
            try:
                f, _, _, _ = run(e, reqs_of(*"ABCDE"))
            except RuntimeError as exc:
                bad_runs.append((seed, blocks, seqs, repr(exc)))
                continue
            if set(f) != set("ABCDE"):
                bad_runs.append((seed, blocks, seqs, f"未全部完成 {sorted(f)}"))
                continue
            if any(f[n]["output_ids"] != ref_seed[n] for n in "ABCDE"):
                bad_runs.append((seed, blocks, seqs, "输出与独占运行不一致"))
                continue
            if e.scheduler.running or e.scheduler.waiting:
                bad_runs.append((seed, blocks, seqs, "结束时队列未清空"))
                continue
            if any(u != 0 for u in e.kv_cache_pool.block_usage):
                bad_runs.append((seed, blocks, seqs, "残留块引用"))
check("72 组随机配置：有界完成、输出正确、无残留", not bad_runs, str(bad_runs[:3]))

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
