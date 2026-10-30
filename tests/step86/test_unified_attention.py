"""086：Triton unified paged attention 的 kernel 差分。"""

from __future__ import annotations

import pytest
import torch

from spec86_helpers import requires_cuda
from minivllm.attention.backends.torch_sdpa import TorchAttentionImpl
from minivllm.attention.metadata import AttentionMetadata
from minivllm.attention.ops.triton_unified_attention import (
    can_use_3d,
    make_3d_workspace,
    unified_attention,
)


def _make_case(*, q_lens, seq_lens, heads=4, head_size=16, kv_heads=1,
               block_size=4, dtype=torch.float32, window=None, causal=True,
               cache_pad=0, q_pad=0, out_pad=0, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    num_tokens = sum(q_lens)
    num_reqs = len(q_lens)
    max_blocks = 3
    seq_max = max(seq_lens)
    assert seq_max <= max_blocks * block_size
    num_blocks = num_reqs * max_blocks

    # 每页数据独立；cache_pad 让 block 维变成非连续 stride。
    k_storage = torch.randn(
        2, num_blocks, block_size + cache_pad, kv_heads, head_size,
        device=device, dtype=dtype)
    kv = k_storage[:, :, :block_size]

    block_table = torch.arange(
        num_reqs * max_blocks, device=device, dtype=torch.int32
    ).view(num_reqs, max_blocks)

    qsl = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
                       device=device, dtype=torch.int32)
    seq = torch.tensor(seq_lens, device=device, dtype=torch.int32)

    q_storage = torch.randn(
        num_tokens, heads * head_size + q_pad, device=device, dtype=dtype)
    q = q_storage[:, : heads * head_size].unflatten(
        -1, (heads, head_size))
    out_storage = torch.zeros(
        num_tokens, heads * head_size + out_pad, device=device, dtype=dtype)
    out = out_storage[:, : heads * head_size].view(
        num_tokens, heads, head_size)

    meta = AttentionMetadata(
        query_start_loc=qsl,
        seq_lens=seq,
        block_table=block_table,
        slot_mapping=torch.zeros(num_tokens, device=device, dtype=torch.int64),
        block_size=block_size,
        num_reqs=num_reqs,
        max_query_len=max(q_lens),
        sliding_window=window,
        causal=causal,
    )
    return {
        "q": q, "q_ref": q_storage[:, : heads * head_size],
        "kv": kv, "out": out, "out_ref": out_storage,
        "qsl": qsl, "seq": seq, "block_table": block_table,
        "meta": meta, "q_lens": q_lens,
    }


def _run(case, *, window=None, workspace=None, max_q=None,
         threshold=None, segments=None, three_d=False):
    kw = {}
    if three_d:
        kw = dict(
            seq_threshold_3D=threshold or len(case["seq"]),
            num_par_softmax_segments=segments or 16,
            softmax_segm_output=workspace[0],
            softmax_segm_max=workspace[1],
            softmax_segm_expsum=workspace[2],
        )
    unified_attention(
        case["q"], case["kv"][0], case["kv"][1], case["out"],
        case["qsl"], case["seq"], case["block_table"],
        softmax_scale=1.0,
        causal=case["meta"].causal,
        window_size=(-1, -1) if window is None else (window - 1, 0),
        max_seqlen_q=(max_q if max_q is not None
                      else max(case["q_lens"])),
        **kw,
    )
    return case["out"]


def _reference(case):
    q = case["q_ref"].reshape(case["q"].shape[0], -1)
    return TorchAttentionImpl(
        case["q"].shape[1], case["q"].shape[2], 1.0,
        case["kv"].shape[3],
    )._forward_generic(q, case["kv"], case["meta"]).view_as(case["q"])


def _assert_close(got, want, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
    elif dtype == torch.float16:
        torch.testing.assert_close(got, want, rtol=5e-3, atol=5e-3)
    else:
        torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_fp32_2d_prefill_decode_matches_torch():
    case = _make_case(q_lens=(1, 3, 5), seq_lens=(4, 7, 9))
    got = _run(case)
    _assert_close(got, _reference(case), torch.float32)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,head_size", [(8, 256), (4, 512), (7, 16)])
def test_low_precision_real_head_dims_and_gqa(dtype, heads, head_size):
    case = _make_case(
        q_lens=(1, 2, 5), seq_lens=(6, 7, 9), heads=heads,
        head_size=head_size, dtype=dtype, block_size=4)
    got = _run(case)
    _assert_close(got, _reference(case), dtype)


@requires_cuda
@pytest.mark.parametrize("q_lens", [(1,), (3,), (5,)])
def test_swa_q1_q3_q5(q_lens):
    case = _make_case(
        q_lens=q_lens, seq_lens=(9,), window=4)
    got = _run(case, window=4)
    want = _reference(case)
    _assert_close(got, want, torch.float32)


@requires_cuda
def test_swa_left_boundary_and_batch_mix():
    # 请求 1：q=5, seq=9, window=4 -> 第 0 个 query 只看 key 5；
    # 请求 2：q=1, seq=2 -> 只看 0..1。
    case = _make_case(
        q_lens=(5, 1), seq_lens=(9, 3), window=4)
    got = _run(case, window=4)
    _assert_close(got, _reference(case), torch.float32)


@requires_cuda
def test_packed_qkv_and_noncontiguous_cache_stride():
    case = _make_case(
        q_lens=(2, 3), seq_lens=(6, 8), q_pad=11, out_pad=7,
        cache_pad=3)
    got = _run(case)
    want = _reference(case)
    _assert_close(got, want, torch.float32)
    # 输出只应写进 out 的切片，后面的 padding 保持哨兵值。
    assert (case["out_ref"][:, -7:] == 0).all()


@requires_cuda
def test_non_causal_matches_torch():
    case = _make_case(
        q_lens=(2, 3), seq_lens=(6, 8), causal=False)
    got = _run(case)
    _assert_close(got, _reference(case), torch.float32)


@requires_cuda
def test_empty_request_and_tail_rows_are_finite():
    case = _make_case(q_lens=(3, 0), seq_lens=(5, 0))
    got = _run(case)
    want = _reference(case)
    _assert_close(got[:3], want[:3], torch.float32)
    assert torch.isfinite(got).all()


@requires_cuda
def test_3d_segmented_matches_2d_and_torch():
    case = _make_case(
        q_lens=(1, 1, 1, 1), seq_lens=(5, 6, 9, 3),
        heads=8, head_size=256, kv_heads=1, block_size=4,
        dtype=torch.bfloat16)
    ws = make_3d_workspace(
        "cuda", 8, 256, len(case["seq"]), 16)
    got_3d = _run(case, workspace=ws, max_q=1, threshold=len(case["seq"]),
                  segments=16, three_d=True).clone()
    got_2d = _run(case, max_q=1).clone()
    _assert_close(got_3d, _reference(case), torch.bfloat16)
    # 3D 与 2D 的在线 softmax 顺序不同，低精度下只要求接近。
    _assert_close(got_3d, got_2d, torch.bfloat16)


@requires_cuda
def test_3d_empty_sequence_is_safe():
    case = _make_case(
        q_lens=(1, 1, 1), seq_lens=(0, 1, 5),
        heads=4, head_size=16, block_size=4, dtype=torch.float32)
    ws = make_3d_workspace("cuda", 4, 16, 3, 16)
    got = _run(case, workspace=ws, max_q=1, threshold=3, segments=16,
               three_d=True)
    assert torch.isfinite(got).all()
    _assert_close(got, _reference(case), torch.float32)


def test_3d_selection_rejects_multi_row_k2():
    # K=2 target 有 3 行 query，必须 2D。
    assert can_use_3d(
        max_seqlen_q=1, num_seqs=1, seq_threshold_3D=128,
        num_par_softmax_segments=16) is True
    assert can_use_3d(
        max_seqlen_q=3, num_seqs=1, seq_threshold_3D=128,
        num_par_softmax_segments=16) is False
    assert can_use_3d(
        max_seqlen_q=1, num_seqs=129, seq_threshold_3D=128,
        num_par_softmax_segments=16) is False
    assert can_use_3d(
        max_seqlen_q=1, num_seqs=1, seq_threshold_3D=None,
        num_par_softmax_segments=16) is False
