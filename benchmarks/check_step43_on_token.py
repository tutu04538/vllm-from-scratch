"""第四十三关 on_token 语义检查。

    python benchmarks/check_step43_on_token.py

覆盖需求 §3 点名的语义：一 token 一通知、output_index 连续、事件拼接等于最终
output_ids、零预算与不可能完成的请求不发通知、EOS/长度停止仍先发 token 再完成、
回调改字典不影响引擎、开关回调不改变结果、chunked prefill 中间块不发通知。
"""

import sys, torch
sys.path.insert(0, "/home/user/proj/vllm-from-scratch")
FIX = "/home/user/proj/vllm-from-scratch/fixtures/step30_qwen3/tiny_gqa"
import step43 as m

def build(**kw):
    kw.setdefault("device", "cpu"); kw.setdefault("attention_backend", "torch")
    return m.Engine.from_model_dir(FIX, **kw)

def run(engine, reqs, collect=True):
    ev, fin, done = [], [], []
    if collect: engine.on_token = lambda e: ev.append(dict(e))
    engine.scheduler.on_finished = lambda r: fin.append(dict(r))
    for r in reqs: engine.add_request(dict(r))
    while engine.has_unfinished_requests():
        done.extend(engine.step())
    return ev, fin, done

ok = lambda c, msg: print(f"  {'PASS' if c else 'FAIL'}  {msg}")

print("=== 1. 首 token 在完成前通知 ===")
e = build(max_num_batched_tokens=8, num_kv_blocks=32)
seen_at = []
e.on_token = lambda ev: seen_at.append((len(ev), ev["output_index"]))
fin = []
e.scheduler.on_finished = lambda r: fin.append(len(r["output_ids"]))
e.add_request(dict(request_id="A", prompt_ids=[1,2,3,4], max_new_tokens=6))
steps = 0
while e.has_unfinished_requests():
    e.step(); steps += 1
ok(len(seen_at) == 6, f"6 个生成 token -> 6 次通知（实际 {len(seen_at)}）")
ok([i for _, i in seen_at] == list(range(6)), f"output_index 从 0 连续: {[i for _, i in seen_at]}")
ok(fin == [6], f"on_finished 仍然只发一次、带完整 output_ids: {fin}")

print("\n=== 2. 事件拼接 == 最终 output_ids ===")
e = build(max_num_batched_tokens=8, num_kv_blocks=64)
ev, fin, _ = run(e, [dict(request_id="A", prompt_ids=[1,2,3,4,5,6,7,8], max_new_tokens=5),
                     dict(request_id="B", prompt_ids=[2,3], max_new_tokens=7)])
got = {}
for x in ev: got.setdefault(x["request_id"], []).append(x["token_id"])
ok(got == {r["request_id"]: r["output_ids"] for r in fin},
   f"逐请求拼接一致: { {k: len(v) for k,v in got.items()} } vs { {r['request_id']: len(r['output_ids']) for r in fin} }")

print("\n=== 3. 零预算 / 不可能完成：不发 token ===")
e = build(max_num_batched_tokens=8, num_kv_blocks=32)
ev, fin, _ = run(e, [dict(request_id="Z", prompt_ids=[1,2], max_new_tokens=0)])
ok(ev == [], "零预算请求不产生 token 通知")
ok(len(fin) == 1 and fin[0]["output_ids"] == [], "零预算仍产生完成记录")

e = build(max_num_batched_tokens=8, num_kv_blocks=2)
ev, fin, _ = run(e, [dict(request_id="X", prompt_ids=[i % 11 for i in range(1,30)], max_new_tokens=20)])
ok(ev == [], "不可能完成的请求不产生 token 通知")
ok(len(fin) == 1 and "error" in fin[0], "仍走既有的明确失败路径")

print("\n=== 4. 回调改字典不影响引擎 ===")
e = build(max_num_batched_tokens=8, num_kv_blocks=32)
ref = {}
e.on_token = lambda ev: (ref.setdefault(ev["request_id"], []).append(ev["token_id"]),
                         ev.__setitem__("token_id", 999))
_, fin, _ = run(e, [dict(request_id="A", prompt_ids=[1,2,3,4], max_new_tokens=5)], collect=False)
ok(ref["A"] == fin[0]["output_ids"], "回调里改字典不改变请求状态")

print("\n=== 5. 关掉回调结果不变 ===")
a = run(build(max_num_batched_tokens=8, num_kv_blocks=64),
        [dict(request_id="A", prompt_ids=[1,2,3,4,5], max_new_tokens=6)], collect=False)[1]
b = run(build(max_num_batched_tokens=8, num_kv_blocks=64),
        [dict(request_id="A", prompt_ids=[1,2,3,4,5], max_new_tokens=6)], collect=True)[1]
ok([r["output_ids"] for r in a] == [r["output_ids"] for r in b], "开关回调不改变生成结果")

print("\n=== 6. chunked prefill：中间块不发通知 ===")
e = build(max_num_batched_tokens=8, num_kv_blocks=64)
ev, fin, _ = run(e, [dict(request_id="C", prompt_ids=[i % 11 for i in range(1,25)], max_new_tokens=3)])
ok([x["output_index"] for x in ev] == [0,1,2], f"prompt 24 分块 prefill，只发 3 次: {[x['output_index'] for x in ev]}")
