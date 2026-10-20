"""79 关验收探针：基于置信度与成本曲线的自适应验证。

跑法：`python benchmarks/check_step79_adaptive.py`（退出码非 0 = 失败）。

分段（对应需求 079 §3/§4）：

    A 成本表     graph 内阶梯、graph 外插值、尾部线性外推、单调/非负裁剪
    B 预算       人工置信度/成本表下手算 full / 0 / intermediate / tie；
                 还要验证 bonus 只数真正会采样的请求、预算受 logits chunk 上限约束
    C 布局       CPU 均分上界 vs GPU 当前前缀；预算 0 的精确 offsets；
                 batch 重排后仍按 req_id 取预算；budget 只消费一次
    D 能力检查   query-lens mismatch、AttentionCGSupport.ALWAYS 两条失败路径
    E 端到端     固定枚数 vs 满预算逐 token 相同；预算 0 只留 bonus 行；
                 中间预算的真实 offsets 是连续前缀；概率草稿裁剪后仍走拒绝采样
    F 真实测量   capture_model() 里的 GPU event profiling 生成成本表；
                 关闭/开启自适应各跑一段，比较提议数、验证数与每 decode 步耗时

⚠️ 本机**没有真实 DSpark 权重**（只有 tiny target + tiny DSpark），所以这里验的是
"策略 / 布局 / 接线 / 真实 GPU 计时"，不是真实模型的加速比（需求 079 §4 允许
人工曲线只做功能测试；F 段的计时是真实 event，但模型是 tiny）。
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "step79"))
sys.path.insert(0, str(ROOT / "tests" / "step77"))
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from adaptive_helpers import (DEVICE, ManagerFixture, RecordBatch,  # noqa: E402
                              make_adaptive_engine, run_adaptive)
from minivllm.attention.backend import AttentionCGSupport  # noqa: E402
from minivllm.worker.gpu.attn_utils import AttentionCGSupportInfo  # noqa: E402
from minivllm.worker.gpu.spec_decode.adaptive_verification import (  # noqa: E402
    _assign_draft_token_budget,
    build_cost_tables_from_curves,
    maybe_create_adaptive_verification_manager,
)

PASSED, FAILED = 0, []
CUDA = torch.cuda.is_available()
PROMPTS = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7])]


def check(name, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"PASS  {name}  {detail}")
    else:
        FAILED.append(name)
        print(f"FAIL  {name}  {detail}")


def _set_full(runner):
    runner.adaptive_verification.set_cost_curves(
        [(1, 0.0), (4, 0.0)], [(n, 1.0) for n in range(65)])


def _set_zero(runner):
    runner.adaptive_verification.set_cost_curves(
        [(1, 100.0), (4, 100.0)],
        [(n, 1000.0 * n) for n in range(65)])


def _set_mid(runner):
    runner.adaptive_verification.set_cost_curves(
        [(1, 10.0), (4, 10.0)], [(0, 1.0), (5, 1.0), (64, 1000.0)])


def _decode_rows(rec):
    return [r for r in rec.rows if sum(r["scheduled_drafts"]) > 0]


def _run_timed(engine, runner, *, max_tokens=8):
    """跑完一组 prompt；返回 (decode 步耗时, 提议数, 验证数, 输出)。"""
    from minivllm import SamplingParams

    rec = RecordBatch(runner)
    for rid, prompt in PROMPTS:
        engine.add_request(rid, list(prompt), SamplingParams(
            max_tokens=max_tokens, temperature=0.0, eos_token_id=99))
    times, proposed, verified = [], 0, 0
    outputs = {}
    try:
        for _ in range(200):
            if not engine.has_unfinished_requests():
                break
            before = len(rec.rows)
            start = time.perf_counter()
            outs = engine.step()
            elapsed = time.perf_counter() - start
            new_rows = rec.rows[before:]
            if any(sum(r["scheduled_drafts"]) > 0 for r in new_rows):
                times.append(elapsed)
                for row in new_rows:
                    proposed += sum(row["scheduled_drafts"])
                    verified += row["num_draft_tokens"]
            for out in outs:
                outputs[out.request_id] = list(out.token_ids)
    finally:
        rec.stop()
    return times, proposed, verified, outputs


def main():
    print("== A. 成本表 ==")
    draft, verify = build_cost_tables_from_curves(
        [(1, 1.0), (4, 4.0)], [(1, 2.0), (4, 5.0)], 4, 4, 0)
    check("A1 graph 内按补齐档位取成本", draft.tolist() == [1, 1, 4, 4, 4]
          and verify.tolist() == [2, 2, 5, 5, 5], f"draft={draft.tolist()}")
    _, verify = build_cost_tables_from_curves(
        [(1, 1.0)], [(1, 1.0), (4, 10.0), (8, 14.0), (12, 20.0)],
        4, 12, 4)
    check("A2 graph 外只在实测点之间插值",
          verify[8:].tolist() == [14.0, 15.5, 17.0, 18.5, 20.0],
          f"{verify[5:].tolist()}")
    _, verify = build_cost_tables_from_curves(
        [(1, 1.0)], [(1, 1.0), (4, 4.0)], 4, 8, 0)
    check("A3 尾部按最后斜率外推", verify[5:].tolist() == [5.0, 6.0, 7.0, 8.0],
          f"{verify[5:].tolist()}")

    print("== B. CPU 预算 ==")
    fx = ManagerFixture()
    fx.manager.set_cost_curves([(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    fx.select()
    check("B1 满预算", fx.manager._batch_budget[2] == 6,
          f"budget={fx.manager._batch_budget[2]}")

    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 1.0), (4, 1.0)], [(n, 1000.0 * n) for n in range(65)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    fx.select()
    check("B2 零预算", fx.manager._batch_budget[2] == 0,
          f"budget={fx.manager._batch_budget[2]}")

    fx = ManagerFixture()
    fx.manager.set_cost_curves(
        [(1, 10.0), (4, 10.0)], [(0, 1.0), (5, 1.0), (64, 1000.0)])
    fx.set_stale({"a": [0.9, 0.8, 0.7], "b": [0.5, 0.4, 0.3]})
    fx.select()
    check("B3 中间预算（手算 b=3）", fx.manager._batch_budget[2] == 3,
          f"budget={fx.manager._batch_budget[2]}")

    fx = ManagerFixture()
    fx.manager.set_cost_curves([(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]})
    fx.select()
    check("B4 平局取更小预算", fx.manager._batch_budget[2] == 0,
          f"budget={fx.manager._batch_budget[2]}")

    fx = ManagerFixture(prefill_len=10)
    fx.state.num_computed_tokens_np[fx.slot("a")] = 0
    fx.state.num_computed_tokens_np[fx.slot("b")] = 10
    fx.manager.set_cost_curves([(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.select()
    check("B5 prefill 请求不计 bonus", fx.manager.last_num_sampling_requests == 1,
          f"sampling={fx.manager.last_num_sampling_requests}")

    fx = ManagerFixture(k=4, max_total_logits=5)
    fx.manager.set_cost_curves([(1, 1.0), (4, 1.0)], [(0, 1.0), (64, 1.0)])
    fx.set_stale({"a": [1.0] * 4, "b": [1.0] * 4})
    fx.select(tokens={"a": 5, "b": 5},
              drafts={"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})
    check("B6 预算受 logits chunk 上限约束", fx.manager._batch_budget[2] == 3,
          f"budget={fx.manager._batch_budget[2]}")

    print("== C. GPU 分配与布局 ==")
    conf = torch.tensor([[0.9, 0.1, 0.1], [0.8, 0.7, 0.6]],
                        dtype=torch.float32, device=DEVICE)
    caps = torch.tensor([3, 3], dtype=torch.int32, device=DEVICE)
    _assign_draft_token_budget(conf, torch.tensor([0, 1], device=DEVICE), caps, 3, 3)
    check("C1 全局 top-k 给连续前缀", caps.tolist() == [1, 2], f"{caps.tolist()}")

    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 0)
    tokens, cu = fx.manager.compact_batch(
        np.array([3, 3], dtype=np.int32), np.array([4, 4], dtype=np.int32),
        np.array([0, 4, 8], dtype=np.int32))
    check("C2 预算 0 的 CPU offsets 精确", tokens.tolist() == [1, 1]
          and cu.tolist() == [0, 1, 2], f"tokens={tokens.tolist()} cu={cu.tolist()}")

    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 3)
    tokens, cu = fx.manager.compact_batch(
        np.array([3, 3], dtype=np.int32), np.array([4, 4], dtype=np.int32),
        np.array([0, 4, 8], dtype=np.int32))
    check("C3 中间预算 CPU 均分、offsets 保留上界",
          tokens.tolist() == [3, 2] and cu.tolist() == [0, 4, 8],
          f"tokens={tokens.tolist()} cu={cu.tolist()}")

    fx = ManagerFixture()
    fx.manager._batch_budget = ({"a": 3, "b": 3}, {"a": 1, "b": 1}, 3)
    fx.manager._confidence_probs[fx.slot("a")] = torch.tensor(
        [0.9, 0.8, 0.7], device=DEVICE)
    fx.manager._confidence_probs[fx.slot("b")] = torch.tensor(
        [0.1, 0.1, 0.1], device=DEVICE)
    idx = torch.tensor([fx.slot("b"), fx.slot("a")], dtype=torch.int32, device=DEVICE)
    cu, _, total = fx.manager.reallocate_drafts(["b", "a"], idx)
    check("C4 重排后按 req_id 取预算", total == 3 and cu.tolist() == [0, 1, 5],
          f"cu={cu.tolist()}")

    print("== D. 能力检查 ==")

    class BadBackend:
        __name__ = "BAD"
        @staticmethod
        def supports_device_cpu_query_lens_mismatch():
            return False

    fx = ManagerFixture()
    try:
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=[[type("G", (), {"backend": BadBackend})()]],
            attn_cg_support=AttentionCGSupportInfo(AttentionCGSupport.ALWAYS, "BAD"),
            req_states=fx.state, query_start_loc=fx.query_start_loc,
            num_bonus_tokens=1, max_total_logits=64)
        ok = False
    except ValueError as exc:
        ok = "does not support" in str(exc)
    check("D1 拒绝依赖 CPU query 长度的后端", ok)

    class GoodBackend:
        __name__ = "GOOD"
        @staticmethod
        def supports_device_cpu_query_lens_mismatch():
            return True

    fx = ManagerFixture()
    try:
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=[[type("G", (), {"backend": GoodBackend})()]],
            attn_cg_support=AttentionCGSupportInfo(
                AttentionCGSupport.UNIFORM_BATCH, "GOOD"),
            req_states=fx.state, query_start_loc=fx.query_start_loc,
            num_bonus_tokens=1, max_total_logits=64)
        ok = False
    except ValueError as exc:
        ok = "ALWAYS" in str(exc)
    check("D2 默认要求 ALWAYS（target 图存在时）", ok)

    if not CUDA:
        print("没有 CUDA：跳过 E/F 段")
        return _finish()

    print("== E. 端到端 ==")
    off_engine, _, _ = make_adaptive_engine(adaptive=False)
    off_out = run_adaptive(off_engine, PROMPTS, max_tokens=4)

    on_engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_full(runner)
    rec = RecordBatch(runner)
    try:
        on_out = run_adaptive(on_engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    check("E1 满预算 == 固定枚数（greedy 逐 token）", on_out == off_out,
          f"{on_out} vs {off_out}")
    check("E2 满预算真的验证全部草稿",
          all(r["num_draft_tokens"] == sum(r["scheduled_drafts"])
              for r in _decode_rows(rec)), "")

    zero_engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_zero(runner)
    rec = RecordBatch(runner)
    try:
        zero_out = run_adaptive(zero_engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    zrows = _decode_rows(rec)[1:]  # 第一步 stale=1.0，见需求/测试注释
    check("E3 预算 0 只留 bonus 行",
          bool(zrows) and all(r["num_draft_tokens"] == 0
                              and r["cu_num_logits"] == list(range(r["num_reqs"] + 1))
                              for r in zrows),
          f"rows={len(zrows)} out={zero_out}")

    mid_engine, _, runner = make_adaptive_engine(adaptive=True)
    _set_mid(runner)
    rec = RecordBatch(runner)
    try:
        mid_out = run_adaptive(mid_engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    mrows = [r for r in rec.rows if r["num_draft_tokens"] == 3]
    caps_ok = True
    for row in mrows:
        caps = []
        for i in range(row["num_reqs"]):
            cap = row["cu_num_logits"][i + 1] - row["cu_num_logits"][i] - 1
            caps.append(cap)
            caps_ok &= (0 <= cap <= row["per_req_drafts"][i]
                        and row["query_start_loc"][i + 1]
                        - row["query_start_loc"][i] == cap + 1)
        caps_ok &= sum(caps) == 3
    check("E4 中间预算是 GPU 连续前缀（总和==预算）",
          bool(mrows) and caps_ok, f"rows={len(mrows)} out={mid_out}")

    prob_engine, _, runner = make_adaptive_engine(
        adaptive=True, draft_sample_method="probabilistic")
    _set_mid(runner)
    rec = RecordBatch(runner)
    try:
        prob_out = run_adaptive(prob_engine, PROMPTS, max_tokens=4)
    finally:
        rec.stop()
    check("E5 概率草稿裁剪后仍走拒绝采样",
          bool(prob_out["a"]) and any(r["num_draft_tokens"] == 3 for r in rec.rows),
          f"out={prob_out}")

    print("== F. 真实 profiling 与关闭/开启对比 ==")
    prof_engine, _, prof_runner = make_adaptive_engine(adaptive=True)
    manager = prof_runner.adaptive_verification
    draft_curve, verify_curve = manager.cost_curves
    check("F1 真实 GPU event profiling 生成非空曲线",
          bool(draft_curve) and bool(verify_curve)
          and all(v >= 0 for _, v in draft_curve)
          and all(v > 0 for _, v in verify_curve),
          f"draft={draft_curve} verify={verify_curve}")
    prof_engine.shutdown()

    off_engine, _, off_runner = make_adaptive_engine(adaptive=False)
    off_times, off_proposed, off_verified, off_out = _run_timed(
        off_engine, off_runner, max_tokens=8)
    off_engine.shutdown()

    on_engine, _, on_runner = make_adaptive_engine(adaptive=True)
    on_times, on_proposed, on_verified, on_out = _run_timed(
        on_engine, on_runner, max_tokens=8)
    on_curves = on_runner.adaptive_verification.cost_curves
    on_budgets = Counter()
    # 用最后一轮解码预算的分布做报告（管理器的预算来自真实 stale 置信度）。
    on_budgets[on_runner.adaptive_verification.last_draft_budget] += 1
    on_engine.shutdown()

    def mean_ms(ts):
        return round(1000.0 * sum(ts) / len(ts), 4) if ts else None

    check("F2 关闭/开启都跑完且确实发生提议/验证",
          off_proposed > 0 and on_proposed > 0
          and off_verified > 0 and on_verified > 0
          and bool(off_out) and bool(on_out),
          f"proposed {off_proposed}->{on_proposed}, verified "
          f"{off_verified}->{on_verified}")
    check("F3 每 decode 步耗时已测（真 event/wall，不承诺加速）",
          mean_ms(off_times) is not None and mean_ms(on_times) is not None,
          f"off={mean_ms(off_times)} ms, on={mean_ms(on_times)} ms")

    result = {
        "model": "tiny_gqa target + tiny_dspark (confidence head)",
        "note": "本机没有真实 DSpark 权重；计时是真实 GPU event/wall，但模型是 tiny，"
                "不给出加速比结论。",
        "device": str(DEVICE),
        "k": 3,
        "profile_context_len": int(os.environ.get(
            "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN", "0")),
        "cost_curves": {"draft": draft_curve, "verify": verify_curve},
        "off": {"decode_steps": len(off_times), "proposed": off_proposed,
                "verified": off_verified, "mean_decode_ms": mean_ms(off_times)},
        "on": {"decode_steps": len(on_times), "proposed": on_proposed,
               "verified": on_verified, "mean_decode_ms": mean_ms(on_times),
               "budget_counter": dict(on_budgets),
               "cost_curves": on_curves},
    }
    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "step79_cost_curves.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"成本曲线/对比数据写入 {out_path}")
    return _finish()


def _finish():
    if FAILED:
        print(f"失败 {len(FAILED)} 项：" + "、".join(FAILED))
        raise SystemExit(1)
    print(f"全部通过（{PASSED} 项）")


if __name__ == "__main__":
    main()
