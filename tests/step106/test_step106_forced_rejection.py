"""强制每个接受前缀，比较真实 GPU 拒绝采样器与冻结上游。"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


path = Path(__file__).resolve().parents[1] / 'step73' / 'test_v2_rejection_diff.py'
spec = importlib.util.spec_from_file_location('step106_rejection_reference', path)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)


@pytest.mark.parametrize('k,r', [(k, r) for k in (1, 2, 4) for r in range(k + 1)])
def test_forced_prefix_matches_frozen(k, r):
    assert torch.cuda.is_available(), '106 强制拒绝需要实际 CUDA，不能计作 skip'
    inputs = reference._uniform_case(
        drafts_per_req=[k], slots=[0], temperature_by_slot={0: 0.0},
        seed_by_slot={0: 106}, use_fp64=False,
        reject_steps={(0, r)} if r < k else set())
    mini = inputs.call(reference.mv_rs)
    upstream = inputs.call(reference.up_rs)
    reference.assert_same_outputs(inputs, mini, upstream)
    assert mini[1].tolist() == [r + 1]
