"""83-D：诊断比较器的 position 对齐与 teacher-forcing 断言（纯 CPU）。"""

from __future__ import annotations

import torch

from benchmarks.check_step83_numerics import _compare_traces


def _top(top_ids, margins):
    return {
        "top_ids": torch.tensor(top_ids, dtype=torch.int64),
        "top_values": torch.tensor([[10.0] * len(row) for row in top_ids]),
        "margin": torch.tensor(margins, dtype=torch.float32),
        "entropy": torch.tensor([1.0] * len(top_ids)),
    }


def _trace(tokens, prompt, position, input_token, top_id, margin):
    top = _top([[top_id, 7]], [margin])
    return {
        "tokens": tokens,
        "prompt_token_ids": prompt,
        "target_steps": [{
            "step": 0,
            "input_ids": torch.tensor([input_token], dtype=torch.int64),
            "positions": torch.tensor([position], dtype=torch.int64),
            "logits_positions": torch.tensor([position], dtype=torch.int64),
            "final_hidden": torch.ones(1, 4),
            "layer_hidden": {"model.layers.0": torch.ones(1, 4),
                             "model.layers.1": torch.ones(1, 4)},
            "logits_top": top,
        }],
        "assistant_samples": [],
        "assistant_forwards": [],
        "assistant_proposals": [],
    }


def test_generation_index_is_converted_to_absolute_position():
    prompt = [5, 6]
    # 第 2 个生成 token（index=2）由绝对 position 2+2-1=3 的 logits 选出。
    left = _trace([10, 11, 12], prompt, 3, 11, 12, 0.5)
    right = _trace([10, 11, 99], prompt, 3, 11, 99, 0.1)
    result = _compare_traces(left, right)
    assert result["first_diff"] == 2
    assert result["predict_position"] == 3
    assert result["teacher_forcing"]["input_token_at_predict_position"] == {
        "k0": 11, "k2": 11, "same": True}
    assert result["target_top1"] == {"k0": 12, "k2": 99, "same": False}
    assert result["attribution"] == "target_logits_differ_near_tie"
    assert result["first_layer_relative_l2_gt_1e-3"] is None


def test_no_divergence_and_zero_prefix_are_explicit():
    same = _compare_traces(
        _trace([10, 11], [5], 2, 10, 11, 1.0),
        _trace([10, 11], [5], 2, 10, 11, 1.0),
    )
    assert same["attribution"] == "no_divergence"
    zero = _compare_traces(
        _trace([10], [5], 0, 0, 10, 1.0),
        _trace([99], [5], 0, 0, 99, 1.0),
    )
    assert zero["attribution"] == "insufficient_prefix"


def test_missing_logits_is_not_silently_treated_as_equal():
    left = _trace([10, 11, 12], [5, 6], 3, 11, 12, 0.5)
    right = _trace([10, 11, 99], [5, 6], 3, 11, 99, 0.1)
    right["target_steps"][0]["logits_top"] = None
    result = _compare_traces(left, right)
    assert result["attribution"] == "insufficient_trace"
