"""83-D：固定中文 prompt 的第一分歧诊断（target + assistant 数值快照）。

目的不是声称“两边应该完全相等”，而是回答：

    在第一个输出分歧之前，两边看到的是同一段 token；
    到底是 target 的验证前向/最终投影先出现了不同候选，
    还是 target 候选相同、assistant/拒绝采样路径把输出带偏了？

每个配置仍然在独立子进程里跑。诊断钩子只在**本脚本**安装，正式基准不启用。
快照保存到 `benchmarks/results/step83/trace_k{0,2}.pt`（结果目录不跟踪）。

运行：
  python benchmarks/check_step83_numerics.py --max-new-tokens 64
  python benchmarks/check_step83_numerics.py --fp32-final-control

本机可以跑完整 FP32 对照，但必须用较小的 KV 池（`--blocks 64`）；默认 blocks=512
时 BF16 已经占去大部分显存。实测 FP32 隔离运行成功，详见 docs/step83_alignment.md。
`--fp32-final-control` 则是更窄的控制项：只把同一份 BF16 hidden 的最终投影换成
FP32，用来区分“最后一层 GEMM 舍入”与“更早 hidden 已经不同”。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_runner import case_worker_main, run_matrix  # noqa: E402
from check_step82_real_spec import (TARGET, _build_engine, _load_tokenizer,  # noqa: E402
                                    _prompt_ids, _sampling_params)

PROMPT = "请用三句话解释什么是 KV cache，以及为什么分页能省显存。"


def _to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(x) for x in value)
    if isinstance(value, list):
        return [_to_cpu(x) for x in value]
    return value


def _topk(logits, k: int = 20) -> dict:
    """在 GPU 上算 top-k 与 margin，避免把整份词表 logits 搬到 CPU。"""
    import torch

    values = logits.float()
    top_values, top_ids = torch.topk(values, k=min(k, values.shape[-1]), dim=-1)
    probs = torch.softmax(values, dim=-1)
    log_probs = torch.log_softmax(values, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    margin = (top_values[..., 0] - top_values[..., 1]
              if top_values.shape[-1] > 1
              else torch.zeros_like(top_values[..., 0]))
    return {
        "top_ids": top_ids.to(torch.int64).cpu(),
        "top_values": top_values.cpu(),
        "margin": margin.cpu(),
        "entropy": entropy.cpu(),
    }


class DiagnosticCapture:
    """只在诊断脚本里使用的 hook 集合；不改生产代码。"""

    def __init__(self, engine, runner):
        import torch

        self.torch = torch
        self.engine = engine
        self.runner = runner
        self.model = runner.model
        self.target_steps = []
        self.target_step = -1
        self.current_input_batch = None
        self.current_record = None
        self.current_logits_indices = None
        self.layer_hooks = []
        self.assistant_forwards = []
        self.assistant_samples = []
        self.assistant_proposals = []
        self._current_assistant = None
        self._current_proposal = None
        self._install()

    def _install(self):
        import torch

        runner = self.runner
        model = self.model
        capture = self

        original_prepare = runner.prepare_inputs

        def prepare_inputs(*args, **kwargs):
            input_batch = original_prepare(*args, **kwargs)
            capture.target_step += 1
            capture.current_input_batch = input_batch
            return input_batch

        runner.prepare_inputs = prepare_inputs

        original_forward = model.forward

        def forward(*args, **kwargs):
            input_batch = capture.current_input_batch
            if input_batch is None:
                return original_forward(*args, **kwargs)
            record = {
                "step": capture.target_step,
                "num_tokens": int(input_batch.num_tokens),
                "input_ids": input_batch.input_ids[:input_batch.num_tokens]
                .to(torch.int64).cpu(),
                "positions": input_batch.positions[:input_batch.num_tokens]
                .to(torch.int64).cpu(),
                "logits_indices": input_batch.logits_indices.to(torch.int64).cpu(),
                "logits_positions": input_batch.positions[input_batch.logits_indices]
                .to(torch.int64).cpu(),
                "layer_hidden": {},
            }
            capture.target_steps.append(record)
            capture.current_record = record
            capture.current_logits_indices = input_batch.logits_indices
            return original_forward(*args, **kwargs)

        model.forward = forward

        # 逐层 hidden：只取本轮会产生 logits 的行（decode 是 1+K 行，prefill 是最后一行）。
        for name, module in model.named_modules():
            if type(module).__name__ != "Gemma4DecoderLayer":
                continue

            def hook(module, inputs, output, layer_name=name):
                if capture.current_record is None:
                    return
                hidden = output[0] if isinstance(output, tuple) else output
                indices = capture.current_logits_indices
                if indices is None or hidden.ndim != 2:
                    return
                indices = indices.to(hidden.device)
                capture.current_record["layer_hidden"][layer_name] = (
                    hidden.index_select(0, indices).detach().float().cpu())

            capture.layer_hooks.append(module.register_forward_hook(hook))

        original_compute_logits = model.compute_logits

        def compute_logits(hidden_states, *args, **kwargs):
            logits = original_compute_logits(hidden_states, *args, **kwargs)
            record = capture.current_record
            if record is not None:
                record["final_hidden"] = hidden_states.detach().float().cpu()
                record["logits_top"] = _topk(logits)
            return logits

        model.compute_logits = compute_logits

        # assistant：输入、采样 hidden、草稿和拒绝计数。
        speculator = getattr(runner, "speculator", None)
        if speculator is None:
            return
        original_run_model = speculator._run_model

        def run_model(num_tokens, *args, **kwargs):
            record = {
                "target_step": capture.target_step,
                "draft_step": int(speculator.current_draft_step.item()),
                "num_tokens": int(num_tokens),
                "input_ids": speculator.input_buffers.input_ids[:num_tokens]
                .to(torch.int64).cpu(),
                "positions": speculator.input_buffers.positions[:num_tokens]
                .to(torch.int64).cpu(),
                "input_hidden": speculator.hidden_states[:num_tokens]
                .detach().float().cpu(),
                "rejected": (list(capture._current_proposal["num_rejected"])
                             if capture._current_proposal is not None else None),
            }
            capture.assistant_forwards.append(record)
            capture._current_assistant = record
            return original_run_model(num_tokens, *args, **kwargs)

        speculator._run_model = run_model

        original_sample_draft = speculator.sample_draft

        def sample_draft(hidden_states, positions, idx_mapping, *args, **kwargs):
            result = original_sample_draft(
                hidden_states, positions, idx_mapping, *args, **kwargs)
            record = {
                "target_step": capture.target_step,
                "draft_step": int(speculator.current_draft_step.item()),
                "positions": positions.detach().to(torch.int64).cpu(),
                "hidden": hidden_states.detach().float().cpu(),
                "sampled": result.detach().to(torch.int64).cpu(),
                "rejected": (list(capture._current_proposal["num_rejected"])
                             if capture._current_proposal is not None else None),
            }
            capture.assistant_samples.append(record)
            return result

        speculator.sample_draft = sample_draft

        original_propose = speculator.propose

        def propose(input_batch, attn_metadata, slot_mappings, *args, **kwargs):
            num_rejected = args[3] if len(args) > 3 else kwargs.get("num_rejected")
            num_sampled = args[2] if len(args) > 2 else kwargs.get("num_sampled")
            proposal = {
                "target_step": capture.target_step,
                "num_rejected": (num_rejected.detach().to(torch.int64).cpu().tolist()
                                 if num_rejected is not None else None),
                "num_sampled": (num_sampled.detach().to(torch.int64).cpu().tolist()
                                 if num_sampled is not None else None),
                "draft_tokens": None,
            }
            capture.assistant_proposals.append(proposal)
            capture._current_proposal = proposal
            result = original_propose(
                input_batch, attn_metadata, slot_mappings, *args, **kwargs)
            proposal["draft_tokens"] = result.detach().to(torch.int64).cpu()
            capture._current_proposal = None
            return result

        speculator.propose = propose

    def close(self):
        for handle in self.layer_hooks:
            handle.remove()
        self.layer_hooks.clear()


def _fp32_final_logits(model, hidden_states):
    """同一份 BF16 hidden 上的 FP32 最终投影（不是完整 FP32 前向）。"""
    import torch
    import torch.nn.functional as F

    hidden = hidden_states.to(model.lm_head.weight.device, dtype=torch.float32)
    weight = model.lm_head.weight.to(torch.float32)
    logits = F.linear(hidden, weight)
    processor = model.logits_processor
    logits = logits[..., :processor.org_vocab_size]
    if processor.soft_cap is not None:
        logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
    if processor.scale != 1.0:
        logits = logits * processor.scale
    return logits


def _build_diagnostic_hooks(engine):
    runner = (engine.engine_core.engine_core.model_executor
              .driver_worker.model_runner)
    return DiagnosticCapture(engine, runner)


def _run_diagnostic(case: dict) -> dict:
    import torch

    args = argparse.Namespace(**case["args"])
    args.graph = False
    args.prefix = False
    args.max_new_tokens = int(case["max_new_tokens"])
    k = int(case["k"])
    output_path = Path(case["trace_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer()
    prompt = _prompt_ids(tokenizer, PROMPT)
    engine, runner, build_s = _build_engine(args, k=k, batch=1, graph=False)
    capture = _build_diagnostic_hooks(engine)
    trace = {
        "k": k,
        "dtype": args.dtype,
        "prompt_token_ids": prompt,
        "max_new_tokens": args.max_new_tokens,
        "build_s": build_s,
        "target_steps": capture.target_steps,
        "assistant_forwards": capture.assistant_forwards,
        "assistant_samples": capture.assistant_samples,
        "assistant_proposals": capture.assistant_proposals,
        "source_layers": {},
        "tokens": [],
    }
    speculator = getattr(runner, "speculator", None)
    try:
        engine.add_request(
            "diag", prompt,
            _sampling_params(argparse.Namespace(**{
                **vars(args), "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True})))
        final = []
        for _ in range(20000):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final = list(out.token_ids)
        if speculator is not None:
            for name, module in getattr(
                    speculator, "draft_layer_modules", {}).items():
                trace["source_layers"][name] = getattr(
                    module, "kv_sharing_target_layer_name", None)
    finally:
        capture.close()
        engine.shutdown()
    trace["tokens"] = final
    torch.save(trace, output_path)
    return {
        "k": k,
        "tokens": final,
        "trace_path": str(output_path),
        "num_target_steps": len(trace["target_steps"]),
        "num_assistant_forwards": len(trace["assistant_forwards"]),
        "num_assistant_samples": len(trace["assistant_samples"]),
        "num_assistant_proposals": len(trace["assistant_proposals"]),
        "build_s": build_s,
    }


def _load_trace(path: str):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _layer_number(name: str) -> int:
    try:
        return int(name.rsplit(".", 1)[-1])
    except ValueError:
        return 10**9


def _index_target(trace: dict) -> dict:
    """绝对 position -> {top1, top_values, margin, hidden, layer_hidden, step}。"""
    index = {}
    for record in trace["target_steps"]:
        positions = record["logits_positions"].tolist()
        hidden = record.get("final_hidden")
        layer_hidden = record.get("layer_hidden", {})
        top = record.get("logits_top")
        input_positions = record["positions"].tolist()
        input_ids = record["input_ids"].tolist()
        for row, position in enumerate(positions):
            position = int(position)
            input_token = next(
                (int(input_ids[i]) for i, value in enumerate(input_positions)
                 if int(value) == position), None)
            index[position] = {
                "step": record["step"],
                "input_token": input_token,
                "top_ids": top["top_ids"][row] if top is not None else None,
                "top_values": top["top_values"][row] if top is not None else None,
                "margin": float(top["margin"][row]) if top is not None else None,
                "hidden": hidden[row] if hidden is not None else None,
                "layer_hidden": {
                    name: values[row] for name, values in layer_hidden.items()
                },
            }
    return index


def _common_target_summary(index0: dict, index2: dict, last_position: int) -> dict:
    """共同 history 上两边 target 的 top1/hidden 一致性（不要求输出已分叉）。"""
    positions = sorted(set(index0) & set(index2))
    positions = [p for p in positions if p <= last_position]
    top1_same = 0
    max_hidden = 0.0
    max_layer = None
    max_layer_rel = 0.0
    for position in positions:
        left, right = index0[position], index2[position]
        if (left["top_ids"] is not None and right["top_ids"] is not None
                and int(left["top_ids"][0]) == int(right["top_ids"][0])):
            top1_same += 1
        if left["hidden"] is not None and right["hidden"] is not None:
            a, b = left["hidden"].float(), right["hidden"].float()
            max_hidden = max(max_hidden,
                             float((a - b).norm()) / max(float(a.norm()), 1e-12))
        for name in set(left["layer_hidden"]) & set(right["layer_hidden"]):
            a = left["layer_hidden"][name].float()
            b = right["layer_hidden"][name].float()
            relative = float((a - b).norm()) / max(float(a.norm()), 1e-12)
            if relative > max_layer_rel:
                max_layer_rel, max_layer = relative, name
    return {
        "num_positions": len(positions),
        "top1_same": top1_same,
        "top1_agreement": (top1_same / len(positions) if positions else None),
        "max_final_hidden_relative_l2": max_hidden,
        "max_layer_hidden_relative_l2": max_layer_rel,
        "max_layer_hidden_relative_l2_layer": max_layer,
    }


def _compare_traces(trace0: dict, trace2: dict, *, fp32_control: bool = False) -> dict:
    tokens0, tokens2 = trace0["tokens"], trace2["tokens"]
    first_diff = next((i for i, (a, b) in enumerate(zip(tokens0, tokens2))
                       if a != b), None)
    result = {
        "tokens_k0": tokens0,
        "tokens_k2": tokens2,
        "first_diff": first_diff,
        "common_prefix_len": (first_diff if first_diff is not None
                              else min(len(tokens0), len(tokens2))),
        "attribution": None,
        "reason": None,
    }
    if first_diff is None:
        index0, index2 = _index_target(trace0), _index_target(trace2)
        prompt_len = len(trace0.get("prompt_token_ids", []))
        result["common_target"] = _common_target_summary(
            index0, index2, prompt_len + min(len(tokens0), len(tokens2)) - 1)
        result["attribution"] = "no_divergence"
        result["reason"] = "两个配置在本次 token 预算内输出完全相同"
        return result
    if first_diff == 0:
        result["attribution"] = "insufficient_prefix"
        result["reason"] = "第 0 个 token 就不同，没有可用的共同历史"
        return result

    index0, index2 = _index_target(trace0), _index_target(trace2)
    # 输出 token 的 index 是**生成长度**，而 target 记录的是 request 内的绝对 position：
    # 绝对 position = prompt 长度 + 生成长度。第 first_diff 个生成 token 由前一个绝对
    # position（prompt_len + first_diff - 1）的 logits 选出。
    prompt_len = len(trace0.get("prompt_token_ids", []))
    predict_position = prompt_len + first_diff - 1
    result["common_target"] = _common_target_summary(
        index0, index2, prompt_len + first_diff - 1)
    left = index0.get(predict_position)
    right = index2.get(predict_position)
    if left is None or right is None or left["top_ids"] is None or right["top_ids"] is None:
        result["attribution"] = "insufficient_trace"
        result["reason"] = (
            f"共同历史位置 {predict_position} 缺少 target logits："
            f"k0={left is not None}, k2={right is not None}")
        return result

    top1_left, top1_right = int(left["top_ids"][0]), int(right["top_ids"][0])
    result["predict_position"] = predict_position
    result["teacher_forcing"] = {
        "prompt_len": prompt_len,
        "common_output_tokens": first_diff,
        "common_history_sha256": __import__("hashlib").sha256(
            ",".join(map(str, trace0["prompt_token_ids"]
                         + tokens0[:first_diff])).encode()).hexdigest(),
        "input_token_at_predict_position": {
            "k0": left["input_token"], "k2": right["input_token"],
            "same": left["input_token"] == right["input_token"],
        },
    }
    result["target_top1"] = {"k0": top1_left, "k2": top1_right,
                             "same": top1_left == top1_right}
    result["target_margin"] = {"k0": left["margin"], "k2": right["margin"]}
    result["target_top_values"] = {
        "k0": left["top_values"][:5].tolist(),
        "k2": right["top_values"][:5].tolist(),
    }
    result["committed_token"] = {"k0": tokens0[first_diff],
                                 "k2": tokens2[first_diff]}
    # 逐层 hidden：用相对 L2 定位“从哪一层开始不一样”。
    layer_diffs = {}
    common_layers = sorted(set(left["layer_hidden"]) & set(right["layer_hidden"]),
                           key=_layer_number)
    for name in common_layers:
        a = left["layer_hidden"][name].float()
        b = right["layer_hidden"][name].float()
        denom = max(float(a.norm()), 1e-12)
        layer_diffs[name] = {
            "relative_l2": float((a - b).norm()) / denom,
            "max_abs": float((a - b).abs().max()),
        }
    result["layer_diffs"] = layer_diffs
    result["first_layer_relative_l2_gt_1e-3"] = next(
        (name for name in common_layers
         if layer_diffs[name]["relative_l2"] > 1e-3), None)
    if left["hidden"] is not None and right["hidden"] is not None:
        a, b = left["hidden"].float(), right["hidden"].float()
        result["final_hidden_diff"] = {
            "relative_l2": float((a - b).norm()) / max(float(a.norm()), 1e-12),
            "max_abs": float((a - b).abs().max()),
        }

    if top1_left != top1_right:
        margin = min(abs(left["margin"]), abs(right["margin"]))
        result["attribution"] = ("target_logits_differ_near_tie" if margin < 0.5
                                 else "target_logits_differ")
        result["reason"] = (
            f"共同历史位置 {predict_position} 的 target top1 已经不同："
            f"{top1_left} vs {top1_right}；margin 较小值为 {margin:.6f}"
            + ("（接近 BF16 可分辨边界）" if margin < 0.5 else ""))
        # 同一份 BF16 hidden 上的 FP32 最终投影：只排除/确认“最后一层 GEMM 的
        # BF16 舍入”，不把结果说成完整 FP32 前向。
        if fp32_control:
            # 完整 FP32 前向不在这里重跑；由 --fp32-final-control 的独立子进程
            # 从保存的 BF16 hidden 重算最终投影。
            result["fp32_final_control_available"] = True
    else:
        result["attribution"] = "assistant_or_rejection_path"
        result["reason"] = (
            f"target top1 相同（{top1_left}），但提交 token 不同："
            f"{tokens0[first_diff]} vs {tokens2[first_diff]}；"
            f"问题更可能在 assistant/拒绝采样/提交路径")

    # assistant 首轮与首次拒绝后的输入。
    def first_assistant(trace, *, post_rejection=False):
        proposals = trace.get("assistant_proposals", [])
        rejected_steps = {
            p["target_step"] for p in proposals
            if p.get("num_rejected") and any(v > 0 for v in p["num_rejected"])
        }
        forwards = {(r["target_step"], r["draft_step"]): r
                    for r in trace.get("assistant_forwards", [])}
        samples = trace.get("assistant_samples", [])
        for record in samples:
            if not post_rejection or record["target_step"] in rejected_steps:
                forward = forwards.get(
                    (record["target_step"], record["draft_step"]), {})
                return {
                    "target_step": record["target_step"],
                    "draft_step": record["draft_step"],
                    "positions": record["positions"].tolist(),
                    "input_ids": (forward.get("input_ids").tolist()
                                  if forward.get("input_ids") is not None else None),
                    "input_hidden_norm": (
                        float(forward["input_hidden"].float().norm())
                        if forward.get("input_hidden") is not None else None),
                    "sampled": record["sampled"].tolist(),
                    "hidden_norm": float(record["hidden"].float().norm()),
                    "rejected": record.get("rejected"),
                }
        return None

    result["assistant_first"] = first_assistant(trace2)
    result["assistant_first_post_rejection"] = first_assistant(
        trace2, post_rejection=True)
    result["assistant_proposals"] = [
        {"target_step": p["target_step"], "num_rejected": p["num_rejected"],
         "draft_tokens": (p["draft_tokens"].tolist()
                          if p.get("draft_tokens") is not None else None)}
        for p in trace2.get("assistant_proposals", [])
        if p["target_step"] <= predict_position + 1
    ][-8:]
    result["source_layers"] = trace2.get("source_layers", {})
    return result


def _fp32_at_position(path: str, position: int):
    rows = json.loads(Path(path).read_text())
    chosen = None
    for row in rows["rows"]:
        if position in row["positions"]:
            index = row["positions"].index(position)
            chosen = {
                "step": row["step"],
                "top_ids": row["top_ids"][index][:5],
                "top_values": row["top_values"][index][:5],
                "margin": row["margin"][index],
            }
    return chosen


def _run_fp32_final_control(trace_path: str, output_path: str,
                            args: dict) -> dict:
    """用同一份 trace 的 BF16 hidden 重算 FP32 最终投影。

    这个控制项只回答“最后 lm_head 的 BF16 舍入是否把 top1 翻了”，不等于完整 FP32
    前向。它需要模型权重，所以单独在子进程里跑；正常诊断不启用。
    """
    import torch
    import torch.nn.functional as F

    trace = _load_trace(trace_path)
    engine, runner, _ = _build_engine(
        argparse.Namespace(**{**args, "graph": False, "prefix": False}),
        k=int(trace["k"]), batch=1, graph=False)
    try:
        model = runner.model
        weight = model.lm_head.weight.to(torch.float32)
        processor = model.logits_processor
        rows = []
        for record in trace["target_steps"]:
            hidden = record.get("final_hidden")
            if hidden is None:
                continue
            logits = F.linear(hidden.to(weight.device, torch.float32), weight)
            logits = logits[..., :processor.org_vocab_size]
            if processor.soft_cap is not None:
                logits = torch.tanh(logits / processor.soft_cap) * processor.soft_cap
            if processor.scale != 1.0:
                logits = logits * processor.scale
            top = _topk(logits)
            rows.append({
                "step": record["step"],
                "positions": record["logits_positions"].tolist(),
                "top_ids": top["top_ids"].tolist(),
                "top_values": top["top_values"].tolist(),
                "margin": top["margin"].tolist(),
            })
        result = {"k": trace["k"], "rows": rows,
                  "note": "FP32 final projection from BF16 hidden; not full FP32 forward"}
        Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=1))
        return {"output_path": output_path, "rows": len(rows)}
    finally:
        engine.shutdown()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--case-json" in argv:
        case = None
        for i, token in enumerate(argv):
            if token == "--case-json":
                case = json.loads(Path(argv[i + 1]).read_text())
                break
        if case is None:
            raise SystemExit("missing case json")
        if case.get("kind") == "fp32_final":
            return case_worker_main(
                lambda case: _run_fp32_final_control(
                    case["trace_path"], case["fp32_output_path"], case["args"]))
        return case_worker_main(_run_diagnostic)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=512)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-backend", default="torch")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--fp32-final-control", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks/results/step83"))
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_args = vars(args).copy()
    cases = [
        {"kind": "numerics", "label": f"k{k}", "k": k,
         "max_new_tokens": args.max_new_tokens,
         "trace_path": str(out_dir / f"trace_k{k}.pt"),
         "args": base_args}
        for k in (0, 2)
    ]
    matrix = run_matrix([sys.executable, str(Path(__file__).resolve())], cases)
    if not matrix["ok"]:
        print(json.dumps({"ok": False, "failures": matrix["failures"]},
                         ensure_ascii=False, indent=1), file=sys.stderr)
        return 1
    traces = {record["case"]["k"]: _load_trace(record["result"]["trace_path"])
              for record in matrix["cases"] if record["status"] == "ok"}
    first = _compare_traces(traces[0], traces[2])
    comparison = {"ok": True, "mode": "diagnostic", "comparison": first,
                  "runs": {record["case"]["label"]: record["result"]
                           for record in matrix["cases"]}}
    if args.fp32_final_control:
        fp32_cases = [
            {"kind": "fp32_final", "label": f"fp32_k{k}", "k": k,
             "trace_path": str(out_dir / f"trace_k{k}.pt"),
             "fp32_output_path": str(out_dir / f"fp32_top_k{k}.json"),
             "args": base_args}
            for k in (0, 2)
        ]
        fp32_matrix = run_matrix(
            [sys.executable, str(Path(__file__).resolve())], fp32_cases)
        comparison["fp32_final_control"] = {
            "ok": fp32_matrix["ok"],
            "cases": fp32_matrix["cases"],
            "note": "只重算最终投影，不改变 BF16 前向；full FP32 前向未跑",
        }
        if not fp32_matrix["ok"]:
            comparison["ok"] = False
        else:
            position = first.get("predict_position")
            if position is not None:
                fp32_rows = {}
                for record in fp32_matrix["cases"]:
                    k = record["case"]["k"]
                    fp32_rows[k] = _fp32_at_position(
                        record["case"]["fp32_output_path"], position)
                comparison["fp32_final_projection"] = {
                    "position": position,
                    "k0": fp32_rows.get(0),
                    "k2": fp32_rows.get(2),
                    "top1_same": (
                        fp32_rows.get(0) is not None
                        and fp32_rows.get(2) is not None
                        and fp32_rows[0]["top_ids"][0] == fp32_rows[2]["top_ids"][0]),
                    "note": ("同一份 BF16 hidden 上只把最终投影换成 FP32；"
                             "hidden 本身已经不同，因此这不是完整 FP32 前向对照"),
                }
    out_path = out_dir / "numerics_summary.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=1))
    print(json.dumps({
        "first_diff": first["first_diff"],
        "attribution": first["attribution"],
        "reason": first["reason"],
        "target_top1": first.get("target_top1"),
        "target_margin": first.get("target_margin"),
        "first_layer_relative_l2_gt_1e-3": first.get(
            "first_layer_relative_l2_gt_1e-3"),
        "summary": str(out_path),
    }, ensure_ascii=False, indent=1))
    return 0 if comparison["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
