"""第 46 关：抢占恢复与前缀缓存整合 —— 功能正确性自查（需求 §4 的八条）。

CPU / FP32 / 确定性小模型。只测功能与状态转换，不做性能测试。
"""

import sys

import torch

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

from step45 import Engine as Engine45
from step46.cache import InfeasibleRequest, SequenceConfig
from step46 import Engine as Engine46

FAIL = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def make(cls=Engine46, *, mode="recompute", prefix=True, seed=29, **kw):
    torch.manual_seed(seed)
    cfg = dict(max_num_seqs=3, max_num_batched_tokens=8, block_size=4, num_kv_blocks=4,
               enable_prefix_caching=prefix)
    cfg.update(kw)
    return cls(device="cpu", preemption_mode=mode, **cfg, **DIMS)


def run(engine, requests, limit=800):
    """跑完并返回 (完成记录, 每请求计数, 步数)。计数在运行中采样，完成后对象会被移出队列。"""
    events = []
    engine.on_token = lambda ev: events.append((ev["request_id"], ev["token_id"], ev["output_index"]))
    for r in requests:
        engine.add_request(dict(r))
    final, stats, steps = {}, {}, 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        if steps > limit:
            raise RuntimeError(f"超过 {limit} 步仍未排空（疑似活锁）")
        for seq in engine.scheduler.running + engine.scheduler.waiting:
            prev = stats.get(seq.request_id, {})
            stats[seq.request_id] = {
                "preemptions": max(prev.get("preemptions", 0), seq.num_preemptions),
                "recomputed": max(prev.get("recomputed", 0), seq.recomputed_tokens),
                # step45 还没有这个字段（本关新增），对照运行时按 0 处理
                "reused": max(prev.get("reused", 0), getattr(seq, "reused_tokens", 0)),
                "max_cache_len": max(prev.get("max_cache_len", 0), seq.cache.length),
                "restarts": prev.get("restarts", 0) + (1 if seq.cache.length == 0 else 0),
            }
        for rec in engine.scheduler.step_done:
            final[rec["request_id"]] = rec
    return final, stats, events, steps


SPEC = [("A", range(1, 7), 8), ("B", range(7, 13), 8), ("C", range(13, 19), 8),
        ("D", range(19, 25), 8), ("E", range(25, 31), 8)]


def reqs_of(*names):
    return [{"request_id": n, "prompt_ids": list(p), "max_new_tokens": m}
            for n, p, m in SPEC if n in names]


def solo(cls=Engine46, **kw):
    out = {}
    for name, prompt, n in SPEC:
        e = make(cls, max_num_seqs=1, num_kv_blocks=64, max_num_batched_tokens=64, **kw)
        f, _, _, _ = run(e, [{"request_id": name, "prompt_ids": list(prompt), "max_new_tokens": n}])
        out[name] = f[name]["output_ids"]
    return out


def solo_req(request, cls=Engine46, seed=29):
    e = make(cls, seed=seed, max_num_seqs=1, num_kv_blocks=64, max_num_batched_tokens=64)
    f, _, _, _ = run(e, [dict(request)])
    return f[request["request_id"]]["output_ids"]


REF = solo()

# ------------------------------------------------ 1. 配置组合放开

ok_combos = True
try:
    e = make(mode="recompute", prefix=True)
    e.add_request(reqs_of("A")[0])
    run(e, [])
except Exception as exc:                                   # noqa: BLE001
    ok_combos = False
    detail = repr(exc)
check("放开配置：recompute + prefix cache 可以同时开启", ok_combos,
      "" if ok_combos else detail)
check("放开配置：Enable 的组合没有被偷偷关掉", e.enable_prefix_caching
      and e.preemption_mode == "recompute")

# 两个公开入口都要支持新组合（需求 §2.A）
import pathlib as _pl
MODEL_DIR = _pl.Path(__file__).resolve().parents[1] / "fixtures" / "step30_qwen3" / "tiny_mqa"
if MODEL_DIR.is_dir():
    from step46 import Engine as _E46
    outs = {}
    for mode, prefix in ((("recompute", True)), ("recompute", False), (None, True)):
        e = _E46.from_model_dir(str(MODEL_DIR), device="cpu", max_num_seqs=2,
                                max_num_batched_tokens=8, block_size=4, num_kv_blocks=16,
                                enable_prefix_caching=prefix, preemption_mode=mode)
        e.add_request({"request_id": "A", "prompt_ids": [1, 2, 3, 4, 5, 6], "max_new_tokens": 4})
        e.add_request({"request_id": "B", "prompt_ids": [1, 2, 3, 4, 5, 7], "max_new_tokens": 4})
        got, n = {}, 0
        while e.has_unfinished_requests():
            e.step(); n += 1
            assert n < 200, "livelock"
            for rec in e.scheduler.step_done:
                got[rec["request_id"]] = rec["output_ids"]
        outs[(mode, prefix)] = got
    check("from_model_dir：recompute + prefix 开可以正常构造并跑完",
          outs.get(("recompute", True)) is not None, str(outs.get(("recompute", True))))
    check("from_model_dir：三种组合输出一致",
          outs.get(("recompute", True)) == outs.get(("recompute", False)) == outs.get((None, True)),
          str(outs))
    try:
        _E46.from_model_dir(str(MODEL_DIR), device="cpu", preemption_mode="bad")
        check("from_model_dir：非法模式仍在构造期报 ValueError", False)
    except ValueError:
        check("from_model_dir：非法模式仍在构造期报 ValueError", True)

for bad in ("recompute!", "x"):
    try:
        make(mode=bad, prefix=True)
        check(f"非法 preemption_mode={bad!r} 仍报错", False)
    except ValueError:
        check(f"非法 preemption_mode={bad!r} 仍报错", True)

# ------------------------------------- 2 & 3. 正命中恢复 / 生成 token 的块

# block_size=4、prompt 长 3：y0 真正进过模型后，含 y0 的块才可命中
e = make(num_kv_blocks=32, max_num_batched_tokens=8)
e.add_request({"request_id": "P", "prompt_ids": [1, 2, 3], "max_new_tokens": 20})
e.step()
s = e.scheduler.running[0]
check("生成 token：prompt 算完、刚采样出 y0 时不发布（已知 4 / 已计算 3）",
      len(s.all_token_ids) == 4 and s.cache.length == 3 and len(s.block_hashes) == 0,
      f"已知 {len(s.all_token_ids)} / 已计算 {s.cache.length} / 已发布 {len(s.block_hashes)}")
e.step()
s = e.scheduler.running[0]
check("生成 token：y0 真进过模型后，含 y0 的块被发布",
      s.cache.length == 4 and len(s.block_hashes) == 1,
      f"已计算 {s.cache.length} / 已发布 {len(s.block_hashes)}")
hist = list(s.all_token_ids)
mb, _ = e.kv_cache_pool.find_matched_prefix_blocks(hist, max_tokens=len(hist) - 1)
check("生成 token：命中长度超过 prompt 长度",
      len(mb) * 4 > len(s.prompt_ids), f"命中 {len(mb)*4} token，prompt {len(s.prompt_ids)}")

# 正命中恢复：真实抢占 + 块没被淘汰
POS = dict(num_kv_blocks=8, max_num_seqs=2)
POS_REQS = [{"request_id": "A", "prompt_ids": list(range(1, 7)), "max_new_tokens": 12},
            {"request_id": "B", "prompt_ids": list(range(20, 26)), "max_new_tokens": 12}]
ref_pos = {r["request_id"]: solo_req(r) for r in POS_REQS}
e_on = make(prefix=True, **POS)
f_on, st_on, _, _ = run(e_on, [dict(r) for r in POS_REQS])
e_off = make(prefix=False, **POS)
f_off, st_off, _, _ = run(e_off, [dict(r) for r in POS_REQS])

check("正命中：确实发生了抢占", e_on.scheduler.num_preemptions > 0,
      str(e_on.scheduler.num_preemptions))
reused = {k: v["reused"] for k, v in st_on.items()}
check("正命中：有请求真的从缓存复用了完整块", any(v > 0 for v in reused.values()), str(reused))
rec_on = sum(v["recomputed"] for v in st_on.values())
rec_off = sum(v["recomputed"] for v in st_off.values())
check("正命中：开启 prefix 后实际重算 token 少于关闭时", rec_on < rec_off,
      f"prefix 开 {rec_on} vs 关 {rec_off}")
check("正命中：输出仍与独占运行一致",
      all(f_on[n]["output_ids"] == ref_pos[n] for n in ("A", "B")),
      str({n: (f_on[n]["output_ids"], ref_pos[n]) for n in ("A", "B")
           if f_on[n]["output_ids"] != ref_pos[n]}))
check("正命中：结束时块引用与承诺都归零",
      all(u == 0 for u in e_on.kv_cache_pool.block_usage)
      and e_on.kv_cache_pool.promised_blocks == 0)
print(f"      正命中参考：prefix 开 重算 {rec_on} / 复用 {sum(reused.values())}；"
      f"prefix 关 重算 {rec_off}")

# 命中但随后失败的块不能计进 reused_tokens（需求 §3）
e = make(num_kv_blocks=32, max_num_batched_tokens=8, max_num_seqs=1)
e.add_request({"request_id": "W", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8], "max_new_tokens": 2})
run(e, [])
# 前 8 个 token 能命中缓存，但整条请求最坏需要 ceil((128+40-1)/4)=42 块 > 池子的 32
hit_seq = SequenceConfig("probe", [1, 2, 3, 4, 5, 6, 7, 8] + [0] * 120, 40, 4)
try:
    e.kv_cache_pool.allocate_block(hit_seq)
    check("命中但不可行：确实抛了 InfeasibleRequest", False, "没有抛")
except InfeasibleRequest:
    check("命中但不可行：确实抛了 InfeasibleRequest", True)
check("命中但不可行：reused_tokens 仍是 0（计划命中不算数）",
      hit_seq.reused_tokens == 0, str(hit_seq.reused_tokens))
check("命中但不可行：块表与 cache.length 没有被部分改动",
      hit_seq.cache.length == 0 and not hit_seq.cache.block_table,
      f"length={hit_seq.cache.length} table={hit_seq.cache.block_table}")

# ------------------------------------------------------ 4. 边界

# 历史长度恰好是整块倍数：命中后仍要留 token 进模型
bs = 4
for hist_len, label in ((8, "恰好两块"), (7, "差一个 token"), (9, "两块多一个")):
    ids = list(range(1, hist_len + 1))
    e = make(num_kv_blocks=32, max_num_batched_tokens=8, max_num_seqs=1)
    # 先让一条同前缀的请求把这些块算完并发布
    e.add_request({"request_id": "W", "prompt_ids": ids[:max(4, hist_len - 1)],
                   "max_new_tokens": 4})
    run(e, [])
    # 准入的调用方式：上限 len-1，留一个 token 出 logits
    mb, _ = e.kv_cache_pool.find_matched_prefix_blocks(ids, max_tokens=len(ids) - 1)
    hit = len(mb) * bs
    check(f"边界（{label}，历史 {hist_len}）：命中 {hit} token 且严格小于历史长度",
          hit < hist_len and hit % bs == 0, f"hit={hit} len={hist_len}")

# 多块命中后再 miss：命中必须从块 0 起连续
e = make(num_kv_blocks=32)
e.add_request({"request_id": "W", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8], "max_new_tokens": 2})
run(e, [])
hist8 = [1, 2, 3, 4, 5, 6, 7, 8]
cap, _ = e.kv_cache_pool.find_matched_prefix_blocks(hist8, max_tokens=len(hist8) - 1)
check("边界：准入口径（留 1 个 token）同前缀命中 1 块 = 4 token", len(cap) * 4 == 4,
      f"命中 {len(cap)*4}")
full, _ = e.kv_cache_pool.find_matched_prefix_blocks(hist8)
check("边界：查询原语默认不设上限，两整块都返回 8 token",
      len(full) * 4 == 8, f"命中 {len(full)*4}")
mb2, _ = e.kv_cache_pool.find_matched_prefix_blocks([1, 2, 3, 4, 99, 99, 99, 99])
check("边界：第二块内容不同则第一块之后立刻停", len(mb2) * 4 == 4, f"命中 {len(mb2)*4}")

# ------------------------------------------- 5. 共享与淘汰

e = make(num_kv_blocks=32, max_num_seqs=2)
run(e, [{"request_id": "S1", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8], "max_new_tokens": 2}])
phys = e.kv_cache_pool.block_hash.copy()
e.add_request({"request_id": "S2", "prompt_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "max_new_tokens": 2})
e.step()
s2 = e.scheduler.running[0]
check("共享：后到的请求复用了同一批物理块",
      set(s2.cache.block_table) & set(phys.values()) != set(), str(s2.cache.block_table))
run(e, [])
check("共享：两条都完成后引用计数归零",
      all(u == 0 for u in e.kv_cache_pool.block_usage), str(e.kv_cache_pool.block_usage))

# LRU 淘汰后 hash 不再指向已复用的物理块
e = make(num_kv_blocks=3)
e.add_request({"request_id": "T", "prompt_ids": [1, 2, 3, 4], "max_new_tokens": 2})
run(e, [])
cached_blocks = set(e.kv_cache_pool.block_hash.values())
check("淘汰前：缓存里有被登记的块", bool(cached_blocks))
for i in range(3):
    e.add_request({"request_id": f"F{i}", "prompt_ids": [10 * i + j for j in range(1, 7)],
                   "max_new_tokens": 3})
run(e, [])
check("淘汰后：block_hash / block_to_hash 双向一致",
      all(e.kv_cache_pool.block_to_hash.get(b) == h
          for h, b in e.kv_cache_pool.block_hash.items())
      and len(e.kv_cache_pool.block_to_hash) == len(e.kv_cache_pool.block_hash))
check("淘汰后：不再有 hash 指向已被别的请求占用的物理块",
      all(e.kv_cache_pool.block_usage[b] == 0 for b in e.kv_cache_pool.block_hash.values()),
      str({b: e.kv_cache_pool.block_usage[b] for b in e.kv_cache_pool.block_hash.values()}))

# ------------------------------------------- 6. 容量不足与多请求压力

bad = {"request_id": "X", "prompt_ids": [1, 2, 3, 4, 5], "max_new_tokens": 40}
e = make(prefix=True, num_kv_blocks=4)
f, _, _, _ = run(e, [bad, reqs_of("A")[0]])
check("容量不足：单请求最坏需求超池仍明确失败",
      "error" in f["X"] and "永远无法完成" in f["X"]["error"], str(f.get("X", {}).get("error"))[:50])
check("容量不足：命中块不能让它蒙混过关（可行性按最坏逻辑块数）",
      "永远无法完成" in f["X"]["error"])

bad_runs = []
for seed in range(8):
    ref_seed = solo(seed=seed)
    for blocks in (5, 6, 8):
        for seqs in (2, 3):
            e = make(seed=seed, prefix=True, num_kv_blocks=blocks, max_num_seqs=seqs)
            try:
                f, st, _, _ = run(e, reqs_of(*"ABCDE"))
            except RuntimeError as exc:
                bad_runs.append((seed, blocks, seqs, repr(exc)))
                continue
            if set(f) != set("ABCDE"):
                bad_runs.append((seed, blocks, seqs, f"未全部完成 {sorted(f)}"))
                continue
            if any(f[n]["output_ids"] != ref_seed[n] for n in "ABCDE"):
                bad_runs.append((seed, blocks, seqs, "输出与独占运行不一致"))
                continue
            if any(u != 0 for u in e.kv_cache_pool.block_usage):
                bad_runs.append((seed, blocks, seqs, "残留块引用"))
                continue
            if e.kv_cache_pool.promised_blocks != 0:
                bad_runs.append((seed, blocks, seqs, "承诺额度泄漏"))
check("多请求压力（48 组）：有界完成、输出正确、无负引用、无承诺泄漏",
      not bad_runs, str(bad_runs[:3]))

# --------------------------- 7. 采样 / 惩罚 / 分块 prefill / 回调一致性

SAMP = dict(temperature=1.0, top_k=0, top_p=1.0, seed=20240922)
PEN = dict(repetition_penalty=1.3, presence_penalty=0.5, frequency_penalty=0.3)
for label, extra in (("固定 seed 随机采样", SAMP), ("惩罚计数", PEN)):
    r = [dict(x, **extra) for x in reqs_of("A", "B")]
    e = make(prefix=True, num_kv_blocks=8, max_num_seqs=2)
    f, _, ev, _ = run(e, [dict(x) for x in r])
    ref = {x["request_id"]: solo_req(x) for x in r}
    check(f"{label}：抢占 + 复用后仍与独占运行一致",
          f["A"]["output_ids"] == ref["A"], f"{f['A']['output_ids']} vs {ref['A']}")
    per = {}
    for rid, tid, idx in ev:
        per.setdefault(rid, []).append((idx, tid))
    check(f"{label}：on_token 拼接等于最终输出且索引连续",
          all([t for _, t in sorted(per[n])] == f[n]["output_ids"]
              and sorted(i for i, _ in per[n]) == list(range(len(per[n]))) for n in ("A", "B")))

e = make(prefix=True, max_num_batched_tokens=8, num_kv_blocks=64, max_num_seqs=1)
long_ids = list(range(1, 25))
f, _, _, _ = run(e, [{"request_id": "L", "prompt_ids": long_ids, "max_new_tokens": 4}])
check("分块 prefill：长 prompt 正常完成", len(f["L"]["output_ids"]) == 4, str(f["L"]["output_ids"]))

done_count = {}
e = make(prefix=True, num_kv_blocks=8, max_num_seqs=2)
e.scheduler.on_finished = lambda r: done_count.__setitem__(r["request_id"],
                                                           done_count.get(r["request_id"], 0) + 1)
f, _, _, _ = run(e, reqs_of("A", "B"))
check("on_finished 每请求只发一次", set(done_count.values()) == {1}, str(done_count))

# --------------------------- 7.5 阻塞者规则保留（step45 的策略不被本关削弱）

for prefix in (True, False):
    e = make(prefix=prefix, num_kv_blocks=4, max_num_seqs=3)
    done = []
    e.scheduler.on_finished = lambda r: done.append(r["request_id"])
    f, _, _, _ = run(e, reqs_of(*"ABCDE"))
    check(f"阻塞者规则（prefix={prefix}）：抢占次数仍远少于 17",
          e.scheduler.num_preemptions < 17, str(e.scheduler.num_preemptions))
    check(f"阻塞者规则（prefix={prefix}）：确实发生过阻塞跳过准入",
          e.scheduler.num_blocked_admissions > 0, str(e.scheduler.num_blocked_admissions))
    check(f"阻塞者规则（prefix={prefix}）：完成序仍是 A,B,C,D,E",
          done == list("ABCDE"), str(done))

# --------------------------- 8. 旧配置与 step45 一致

for label, mode, prefix in (("None + prefix 开", None, True),
                            ("None + prefix 关", None, False),
                            ("recompute + prefix 关", "recompute", False)):
    a, _, _, _ = run(make(Engine45, mode=mode, prefix=prefix, num_kv_blocks=8), reqs_of("A", "B"))
    b, _, _, _ = run(make(Engine46, mode=mode, prefix=prefix, num_kv_blocks=8), reqs_of("A", "B"))
    check(f"旧配置（{label}）输出与 step45 一致",
          {k: v["output_ids"] for k, v in a.items()} == {k: v["output_ids"] for k, v in b.items()})

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
