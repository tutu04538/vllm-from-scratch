"""88：logprobs margin 诊断的排序与差值（不加载 vLLM）。"""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.check_step88_numerics import _margin


def test_margin_sorts_logprobs_and_uses_top2():
    entry = {
        11: SimpleNamespace(logprob=-2.0, rank=2),
        7: SimpleNamespace(logprob=-0.5, rank=1),
        9: SimpleNamespace(logprob=-3.0, rank=3),
    }
    result = _margin(entry)
    assert [item["token_id"] for item in result["top"]] == [7, 11, 9]
    assert result["margin_logprob"] == 1.5
    assert result["top1_prob"] > result["top2_prob"] > 0
