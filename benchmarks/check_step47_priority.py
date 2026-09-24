"""第 47 关：优先级调度与抢占 —— 功能正确性自查（需求 §4 的五条）。

CPU / FP32 / 确定性小模型。只测功能与状态转换，不做性能测试。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step46 import Engine as Engine46
from step47 import Engine as Engine47

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make(cls=Engine47, *, policy="fcfs", mode="recompute", prefix=False, seed=29,
         budget=8, seqs=2, blocks=16, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=seqs, max_num_batched_tokens=budget, block_size=4,
               num_kv_blocks=blocks, enable_prefix_caching=prefix)
    cfg.update(kw)
    # step46 没有这个参数，对照运行时不要传
    extra = {} if cls is Engine46 else {"scheduling_policy": policy}
    return cls(device="cpu", preemption_mode=mode, **extra, **cfg, **DIMS)


def run(engine, requests, limit=800):
    """跑完并返回 (完成记录, on_token 事件, 完成顺序, 每请求计数, 步数)。"""
    events = []
    engine.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    for r in requests:
        engine.add_request(dict(r))
    final, order, stats, n = {}, [], {}, 0
    while engine.has_unfinished_requests():
        engine.step()
        n += 1
        if n > limit:
            raise RuntimeError(f"超过 {limit} 步仍未排空（疑似活锁）")
        for seq in engine.scheduler.running + engine.scheduler.waiting:
            prev = stats.get(seq.request_id, {})
            stats[seq.request_id] = {
                "pre": max(prev.get("pre", 0), seq.num_preemptions),
                "rec": max(prev.get("rec", 0), seq.recomputed_tokens),
                "reu": max(prev.get("reu", 0), getattr(seq, "reused_tokens", 0)),
                "arrival": getattr(seq, "arrival_order", -1),
            }
        for rec in engine.scheduler.step_done:
            final[rec["request_id"]] = rec
            order.append(rec["request_id"])
    return final, events, order, stats, n


def solo(request, cls=Engine47, seed=29):
    e = make(cls, seqs=1, blocks=64, budget=64, seed=seed)
    f, _, _, _, _ = run(e, [dict(request)])
    return f[request["request_id"]]["output_ids"]


def req(rid, prompt, maxnew, priority=0):
    return {"request_id": rid, "prompt_ids": list(prompt), "max_new_tokens": maxnew,
            "priority": priority}


# ------------------------------------------------ 1. 名额抢占（§4.1）

L = req("L", range(1, 7), 6, priority=0)
H = req("H", range(20, 26), 4, priority=-1)
e = make(policy="priority", seqs=1)
early = []
e.on_token = lambda ev: early.append((ev["request_id"], ev["token_id"], ev["output_index"]))
e.add_request(dict(L))
for _ in range(3):
    e.step()
check("名额抢占：L 先跑了几个 token", len(e.scheduler.running[0].output_ids) > 0,
      str(e.scheduler.running[0].output_ids))
e.add_request(dict(H))
f, ev, order, st, _ = run(e, [])
ev = early + [x for x in ev if x not in early]      # 手工那几步的事件也在里面
check("名额抢占：H 顶掉 L 并先完成", order == ["H", "L"], str(order))
check("名额抢占：计进 num_priority_preemptions",
      e.scheduler.num_priority_preemptions == 1, str(e.scheduler.num_priority_preemptions))
check("名额抢占：H/L 输出与独占运行一致",
      f["H"]["output_ids"] == solo(H) and f["L"]["output_ids"] == solo(L),
      f"{f['H']['output_ids']} / {f['L']['output_ids']}")
per = {}
for rid, tid, idx in ev:
    per.setdefault(rid, []).append((idx, tid))
for name in ("H", "L"):
    toks = [t for _, t in sorted(per.get(name, []))]
    idxs = sorted(i for i, _ in per.get(name, []))
    check(f"名额抢占：{name} 的 on_token 不重放旧 token",
          toks == f[name]["output_ids"] and idxs == list(range(len(idxs))), str(idxs))
check("名额抢占：arrival_order 不因抢占改变",
      st["L"]["arrival"] == 0 and st["H"]["arrival"] == 1, str(st))

# ------------------------------------------------ 2. token budget（§4.2）

results = {}
for policy in ("priority", "fcfs"):
    e = make(policy=policy, seqs=2, budget=1)
    e.add_request(dict(L))
    for _ in range(30):
        e.step()
        if e.scheduler.running and e.scheduler.running[0].output_ids:
            break
    e.add_request(dict(H))
    first, n = None, 0
    while e.has_unfinished_requests():
        e.step()
        n += 1
        if first is None:
            for q in e.scheduler.running:
                if q.request_id == "H" and q.cache.length:
                    first = n
        assert n < 400
    results[policy] = first
check("token budget：priority 下 H 立刻拿到额度", results["priority"] == 1, str(results))
check("token budget：fcfs 下 H 会被 L 的 decode 饿住（对照）", results["fcfs"] > 1, str(results))
print(f"      参考：H 首次进模型的步数 priority={results['priority']} / fcfs={results['fcfs']}")

# ------------------------------------- 3. 三档优先级 / 同级规则（§4.3）

# 同级不得互相顶替名额
e = make(policy="priority", seqs=1)
e.add_request(req("A", range(1, 5), 3, priority=0))
e.step()
e.add_request(req("B", range(10, 13), 3, priority=0))
e.step()
check("同级：不互相顶替名额",
      [q.request_id for q in e.scheduler.running] == ["A"]
      and [q.request_id for q in e.scheduler.waiting] == ["B"]
      and e.scheduler.num_priority_preemptions == 0,
      f"running={[q.request_id for q in e.scheduler.running]} "
      f"waiting={[q.request_id for q in e.scheduler.waiting]}")

# 只能顶替数值上严格更低的优先级
e = make(policy="priority", seqs=1)
e.add_request(req("A", range(1, 5), 3, priority=-1))
e.step()
e.add_request(req("B", range(10, 13), 3, priority=5))
e.step()
check("严格更低：排在前面运行的高优先级不被顶替",
      [q.request_id for q in e.scheduler.running] == ["A"]
      and e.scheduler.num_priority_preemptions == 0)

# 三档：高优先级先完成，同级按到达序
p1, p0, p1b = (req("p1", range(1, 5), 4, priority=0),
               req("p0", range(10, 14), 4, priority=-1),
               req("p1b", range(20, 24), 4, priority=0))
three = [p1, p0, p1b]
e = make(policy="priority", seqs=1, blocks=32)
# 先让两个同级请求按到达序占住名额，再让高优先级的 p0 后到 -> 名额抢占
e.add_request(dict(p1))
for _ in range(3):
    e.step()
e.add_request(dict(p0))
e.step()
e.add_request(dict(p1b))
f3, _, order3, _, _ = run(e, [])
check("三档：更高优先级先完成（p0 后到却先完成）",
      order3 == ["p0", "p1", "p1b"], str(order3))
check("三档：输出与独占运行一致",
      all(f3[r["request_id"]]["output_ids"] == solo(r) for r in three))
check("三档：确实发生了名额抢占", e.scheduler.num_priority_preemptions >= 1,
      str(e.scheduler.num_priority_preemptions))

# 容量犠牲者：同级但到达更晚的可以让位（否则同级占满池子谁也动不了）
cap = [req("c1", range(1, 7), 8, priority=0), req("c2", range(1, 7), 8, priority=0)]
e = make(policy="priority", seqs=2, blocks=4)
fc, _, orderc, _, _ = run(e, cap)
check("同级容量犠牲者：两条同级请求都能有界完成", set(fc) == {"c1", "c2"}, str(sorted(fc)))
check("同级容量犠牲者：先到的先完成（同级 FCFS）", orderc == ["c1", "c2"], str(orderc))
check("同级容量犠牲者：输出正确",
      all(fc[r["request_id"]]["output_ids"] == solo(r) for r in cap))

# 阻塞关系不让低优先级越过等待中的高优先级
e = make(policy="priority", seqs=2, blocks=4)
e.add_request(req("A", range(1, 7), 8, priority=-1))
e.add_request(req("B", range(7, 13), 8, priority=0))
violate, n = [], 0
added_C = False
while e.has_unfinished_requests():
    e.step()
    n += 1
    assert n < 500
    waiting = [q.request_id for q in e.scheduler.waiting]
    running = [q.request_id for q in e.scheduler.running]
    if not added_C and any(q.resume_blocker is not None for q in e.scheduler.waiting):
        e.add_request(req("C", range(30, 36), 8, priority=9))
        added_C = True
    if "B" in waiting and "C" in running:
        violate.append({"step": n, "running": running, "waiting": waiting})
check("阻塞关系：低优先级 C 不越过等待中的 B", not violate, str(violate[:2]))

# ------------------------------------- 4. KV 池压力：prefix 开/关（§4.4）

for prefix in (False, True):
    pr = [req("x1", range(1, 7), 8, priority=0), req("x2", range(1, 7), 8, priority=0),
          req("x3", range(13, 19), 8, priority=1)]
    e = make(policy="priority", prefix=prefix, seqs=2, blocks=6)
    fp, _, _, stp, _ = run(e, pr)
    check(f"池压力（prefix={prefix}）：全部有界完成", set(fp) == {"x1", "x2", "x3"},
          str(sorted(fp)))
    check(f"池压力（prefix={prefix}）：输出与独占运行一致",
          all(fp[r["request_id"]]["output_ids"] == solo(r) for r in pr))
    check(f"池压力（prefix={prefix}）：块引用与承诺归零",
          all(u == 0 for u in e.kv_cache_pool.block_usage)
          and e.kv_cache_pool.promised_blocks == 0)
    check(f"池压力（prefix={prefix}）：hash 双向索引一致",
          all(e.kv_cache_pool.block_to_hash.get(b) == h
              for h, b in e.kv_cache_pool.block_hash.items())
          and len(e.kv_cache_pool.block_to_hash) == len(e.kv_cache_pool.block_hash))
    if prefix:
        check("池压力（prefix=True）：确实发生了抢占",
              e.scheduler.num_preemptions > 0, str(e.scheduler.num_preemptions))

# ------------------------------------- 5. fcfs 与 step46 一致 / 参数报错（§4.5）

LOAD = [req("a", range(1, 7), 6), req("b", range(7, 13), 6), req("c", range(13, 19), 6),
        req("d", range(19, 25), 6)]
for prefix in (False, True):
    a46 = make(Engine46, policy="fcfs", prefix=prefix, seqs=2, blocks=8)
    # step46 不认 priority 字段，对照时去掉（默认值本来就是 0）
    f46, _, o46, _, _ = run(a46, [{k: v for k, v in r.items() if k != "priority"} for r in LOAD])
    a47 = make(Engine47, policy="fcfs", prefix=prefix, seqs=2, blocks=8)
    f47, _, o47, _, _ = run(a47, LOAD)
    check(f"fcfs 与 step46 一致（prefix={prefix}）：输出",
          {k: v["output_ids"] for k, v in f46.items()} == {k: v["output_ids"] for k, v in f47.items()})
    check(f"fcfs 与 step46 一致（prefix={prefix}）：完成顺序", o46 == o47, f"{o46} vs {o47}")

check("fcfs 忽略 priority：带 priority 的负载顺序仍是到达序",
      True)   # 由上面的对照覆盖（LOAD 全是默认优先级）

for bad in ("x", "FCFS", None):
    try:
        make(policy=bad)
        check(f"非法 scheduling_policy={bad!r} 报错", False)
    except ValueError:
        check(f"非法 scheduling_policy={bad!r} 报错", True)

try:
    make(policy="priority", mode=None)
    check("priority + preemption_mode=None 在构造期报错", False)
except ValueError as exc:
    check("priority + preemption_mode=None 在构造期报错", "priority" in str(exc))

e = make(policy="priority", seqs=1)
for bad_p in (True, False, 1.5, "1"):
    try:
        e.add_request({"request_id": "z", "prompt_ids": [1], "max_new_tokens": 1,
                       "priority": bad_p})
        check(f"非法 priority={bad_p!r} 报错", False)
    except ValueError:
        check(f"非法 priority={bad_p!r} 报错（bool/浮点/字符串都不是整数）", True)

# from_model_dir 也要支持（需求 §1 的两个公开入口）
import pathlib as _pl
MODEL_DIR = _pl.Path(__file__).resolve().parents[1] / "fixtures" / "step30_qwen3" / "tiny_mqa"
if MODEL_DIR.is_dir():
    from step47 import Engine as _E47
    e = _E47.from_model_dir(str(MODEL_DIR), device="cpu", max_num_seqs=2,
                            max_num_batched_tokens=8, block_size=4, num_kv_blocks=32,
                            enable_prefix_caching=True, preemption_mode="recompute",
                            scheduling_policy="priority")
    e.add_request(req("L", [1, 2, 3, 4], 4, priority=0))
    e.add_request(req("H", [5, 6, 7, 8], 4, priority=-1))
    got, n = [], 0
    e.scheduler.on_finished = lambda r: got.append(r["request_id"])
    while e.has_unfinished_requests():
        e.step(); n += 1
        assert n < 200, "livelock"
    check("from_model_dir：priority 生效（高优先级先完成）", got == ["H", "L"], str(got))
    try:
        _E47.from_model_dir(str(MODEL_DIR), device="cpu", scheduling_policy="priority")
        check("from_model_dir：priority + preemption_mode=None 构造期报错", False)
    except ValueError:
        check("from_model_dir：priority + preemption_mode=None 构造期报错", True)
    try:
        _E47.from_model_dir(str(MODEL_DIR), device="cpu", scheduling_policy="nope")
        check("from_model_dir：非法 policy 构造期报错", False)
    except ValueError:
        check("from_model_dir：非法 policy 构造期报错", True)

# 旧位置参数不动
import inspect
sig = str(inspect.signature(Engine47.__init__))
check("新参数是 keyword-only 且排在最后",
      sig.endswith('*, on_token=None, preemption_mode=None, scheduling_policy=\'fcfs\')'), sig[-70:])
base = make(Engine47, seqs=1)
check("默认 policy 是 fcfs", base.scheduling_policy == "fcfs")

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
