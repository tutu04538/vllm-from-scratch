"""Stride specialization regression: exact legacy arithmetic + real K/V async.

The legacy variant reverses only this patch's annotations/launch policy in a
temporary module. It is a regression reference, not an independent algorithm;
step86 supplies the independent upstream/Torch numerical comparisons.
"""
import importlib.util
import inspect
import sys

import pytest
import torch

from minivllm.attention.ops import triton_unified_attention as production


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    source = inspect.getsource(production)
    source = source.replace("from ...triton_utils import tl, triton",
                            "import triton\nimport triton.language as tl")
    source = source.replace("stride_k_cache_3: tl.constexpr", "stride_k_cache_3: tl.int64")
    source = source.replace("stride_v_cache_3: tl.constexpr", "stride_v_cache_3: tl.int64")
    source = source.replace(
        "if not is_3d and not preserve_tiling and head_size <= 256:",
        "if False:")
    source = source.replace(
        "64 if preserve_tiling and head_size <= 256 else 32",
        "64 if head_size <= 256 else 32")
    path = tmp_path_factory.mktemp("legacy_attention") / "step88_legacy_attention.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def case(head, dtype, stride, window, context=129):
    torch.manual_seed(88)
    batch, qlen, heads, block = 2, 3, 8, 16
    blocks = (context + qlen + block - 1) // block
    # Non-contiguous page stride and optionally non-unit inner stride.
    storage = torch.randn(2, batch * blocks, block + 2, 1, head * stride,
                          device="cuda", dtype=dtype)
    k, v = storage[:, :, :block, :, ::stride].unbind(0)
    q = torch.randn(batch * qlen, heads, head, device="cuda", dtype=dtype)
    table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int32).view(batch, blocks)
    seq = torch.full((batch,), context + qlen, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, (batch + 1) * qlen, qlen, device="cuda", dtype=torch.int32)
    return (q, k, v, table, seq, cu, window)


def run(module, data):
    q, k, v, table, seq, cu, window = data
    out = torch.empty_like(q)
    module.unified_attention(q, k, v, out, cu, seq, table,
                            softmax_scale=1.0, causal=True,
                            window_size=window, max_seqlen_q=3)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("head,dtype,stride,window", [
    (256, torch.bfloat16, 1, (-1, -1)),
    (512, torch.bfloat16, 1, (-1, -1)),
    (256, torch.bfloat16, 1, (63, 0)),
    (512, torch.bfloat16, 1, (63, 0)),
    (64, torch.bfloat16, 2, (-1, -1)),
    (16, torch.float32, 1, (-1, -1)),
    (32, torch.float32, 2, (63, 0)),
    (64, torch.float32, 1, (-1, -1)),
    (256, torch.float16, 1, (-1, -1)),
])
def test_bitwise_legacy_output(legacy, head, dtype, stride, window):
    data = case(head, dtype, stride, window)
    expected = run(legacy, data)
    actual = run(production, data)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_actual_kv_loads_are_async(legacy, monkeypatch):
    data = case(256, torch.bfloat16, 1, (-1, -1), context=1024)
    captured = {}
    for name, module in [("legacy", legacy), ("optimized", production)]:
        original = module.kernel_unified_attention.run
        def record(*args, _original=original, _name=name, **kwargs):
            compiled = _original(*args, **kwargs)
            captured[_name] = compiled
            return compiled
        with monkeypatch.context() as patch:
            patch.setattr(module.kernel_unified_attention, "run", record)
            run(module, data)
    old, new = captured["legacy"], captured["optimized"]
    # BF16 source pointers prove these are data loads, not just int block tables.
    copies = [line for line in new.asm["ttgir"].splitlines()
              if "async_copy_global_to_local" in line and "!tt.ptr<bf16>" in line]
    assert len(copies) >= 2
    assert "cp.async" in new.asm["ptx"]
    assert "async_copy_global_to_local" not in old.asm["ttgir"]
    # The dynamic stride operands disappear entirely from the TTIR signature.
    signature = next(line for line in new.asm["ttir"].splitlines() if "tt.func public" in line)
    assert '%stride_k_cache_3:' not in signature
    assert '%stride_v_cache_3:' not in signature


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("normalized", [True, False])
def test_swa_prefill_empty_subtile_history(legacy, heads, normalized):
    """TILE64 can start with an all-masked 32-key subtile; keep its FP history.

    q position 574 has first visible key 63. TILE64 starts at 0 (empty first
    half); TILE32 starts at 32. For a negative first score, clamping empty-row
    M to zero changes the BF16 P rounding history. Unit-normalized Q/K expose
    this, while large random dot products often conceal it.
    """
    torch.manual_seed(88)
    context, qlen, head, block = 512, 128, 256, 16
    seq_len = context + qlen
    q = torch.randn(qlen, heads, head, device="cuda", dtype=torch.bfloat16) / 16
    k = torch.randn(seq_len // block, block, 1, head,
                    device="cuda", dtype=torch.bfloat16) / 16
    if not normalized:
        q.fill_(0.1)
        k.fill_(-0.1)
    v = torch.randn_like(k)
    table = torch.arange(seq_len // block, device="cuda", dtype=torch.int32)[None]
    seq = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, qlen], device="cuda", dtype=torch.int32)
    outputs = []
    for module in (legacy, production):
        out = torch.empty_like(q)
        module.unified_attention(q, k, v, out, cu, seq, table,
                                 softmax_scale=1.0, causal=True,
                                 window_size=(511, 0), max_seqlen_q=qlen)
        outputs.append(out)
    assert torch.isfinite(outputs[1]).all()
    assert torch.equal(*outputs), (outputs[0] - outputs[1]).abs().max().item()
