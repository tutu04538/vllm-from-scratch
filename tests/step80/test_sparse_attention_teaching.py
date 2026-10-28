"""教学版 learned top-k 稀疏注意力的行为测试（无真实权重）。"""

from __future__ import annotations

import torch

from minivllm.testing.sparse_attention import SparseAttentionConfig, SparseLM


def _config(top_k: int = 4) -> SparseAttentionConfig:
    return SparseAttentionConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_layers=2, num_heads=4, num_kv_heads=2, head_dim=8,
        max_seq_len=16, sparse_top_k=top_k, index_heads=2, index_dim=8,
    )


def _model(top_k: int = 4) -> SparseLM:
    torch.manual_seed(0)
    model = SparseLM(_config(top_k))
    model.eval()
    return model


def test_topk_at_least_sequence_matches_dense():
    model = _model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    dense = model(ids, top_k=None)
    sparse = model(ids, top_k=8)
    torch.testing.assert_close(dense, sparse, rtol=0, atol=1e-6)


def test_selected_keys_are_causal_and_bounded():
    model = _model(top_k=2)
    ids = torch.arange(1, 9).unsqueeze(0)
    _, selected_layers = model(ids, top_k=2, return_selected=True)
    assert len(selected_layers) == 2
    for selected in selected_layers:
        assert selected.shape == (1, 8, 8)
        # 每条 query 只能看自己及其左侧，且 top-k 数量不超过 min(k, i+1)
        for i in range(8):
            row = selected[0, i]
            assert not row[i + 1:].any()
            assert int(row.sum()) == min(2, i + 1)


def test_sparse_selection_changes_attention_output():
    model = _model(top_k=1)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    dense_logits = model(ids, top_k=None)
    sparse_logits = model(ids, top_k=1)
    assert torch.isfinite(sparse_logits).all()
    assert not torch.allclose(dense_logits, sparse_logits)


def test_generate_runs_with_random_weights():
    model = _model(top_k=2)
    ids = torch.tensor([[1, 2, 3]])
    first = model.generate(ids, max_new_tokens=4, top_k=2)
    second = model.generate(ids, max_new_tokens=4, top_k=2)
    assert first.shape == (1, 7)
    assert torch.equal(first, second)
    assert torch.isfinite(model(first, top_k=2)).all()


def test_padding_keys_are_never_selected():
    model = _model(top_k=2)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    attention_mask = torch.tensor([[True] * 5 + [False] * 3])
    _, selected_layers = model(
        ids, attention_mask=attention_mask, top_k=2, return_selected=True)
    for selected in selected_layers:
        # position 5/6/7 是 padding，任何 query 都不能选它们
        assert not selected[0, :, 5:].any()
