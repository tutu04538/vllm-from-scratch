"""W4 人工预期、边界与隔离上游差分。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import step118_w4_reference as ref


def test_hand_calculation_and_dot_product():
    q = [0, 1, 7, 8, 15, 2, 3, 4]
    assert ref.pack_standard_word(q) == 0x432F8710
    assert ref.unpack_standard_word(0x432F8710) == q
    logical = torch.tensor(q, dtype=torch.int32).reshape(8, 1)
    values = ref.dequantize(logical, torch.tensor([[0.25]]), torch.tensor([[8]]), group_size=8)
    assert values[:, 0].tolist() == [-2, -1.75, -0.25, 0, 1.75, -1.5, -1.25, -1]
    x = torch.tensor([1., 2., 0., -1., 1., 0., 0., 2.])
    assert torch.dot(x, values[:, 0]).item() == -5.75


def test_nibble_sign_and_extremes():
    assert ref.pack_standard_word([15] * 8) == -1
    assert ref.unpack_standard_word(-1) == [15] * 8
    assert ref.pack_standard_word([0] * 7 + [8]) == -(2**31)
    assert ref.unpack_standard_word(-(2**31)) == [0] * 7 + [8]
    for start in (0, 8):
        q = list(range(start, start + 8))
        assert ref.unpack_standard_word(ref.pack_standard_word(q)) == q


def test_awq_physical_order_has_independent_expected_values():
    raw = torch.tensor([[0x76543210]], dtype=torch.int32)
    assert ref.unpack_standard_cols(raw).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert ref.unpack_awq_cols(raw).tolist() == [[0, 4, 1, 5, 2, 6, 3, 7]]
    assert not torch.equal(ref.unpack_awq_cols(raw), ref.unpack_standard_cols(raw))


def test_group_boundaries_and_padding_values_do_not_leak():
    q = torch.full((256, 16), 9, dtype=torch.int32)
    z = torch.full((2, 16), 8, dtype=torch.int32)
    scale = torch.stack([torch.full((16,), .25), torch.full((16,), 2.)])
    out = ref.dequantize(q, scale, z, group_size=128, valid_k=255, valid_n=9)
    assert out.shape == (255, 9)
    assert out[127, 0] == .25 and out[128, 0] == 2.
    q[255, :] = 15
    q[:, 9:] = 0
    assert torch.equal(out, ref.dequantize(q, scale, z, group_size=128, valid_k=255, valid_n=9))
    z[0, 0] = 0; z[1, 0] = 15
    out = ref.dequantize(q, scale, z, group_size=128)
    assert out[127, 0] == 2.25 and out[128, 0] == -12.


def test_awq_zero_has_no_unconditional_plus_one():
    # 直接构造 AWQ 零点字：全 0 / 全 15 不依赖编码器。
    packed_z = torch.tensor([[0, -1]], dtype=torch.int32)
    zero = ref.unpack_awq_cols(packed_z)
    assert zero.tolist() == [[0]*8 + [15]*8]
    q = torch.zeros((8, 16), dtype=torch.int32)
    scale = torch.full((1, 16), .25)
    expected = ref.dequantize(q, scale, zero, group_size=8)
    wrong = (q.float() - (zero + 1).float()) * .25
    assert expected[0, 0] == 0 and wrong[0, 0] == -.25
    # GPTQ 文件编码在这里不做 +1 或转换；实际路径需要单独对照 127。


def test_awq_frozen_upstream_differential(tmp_path):
    # 不是由本测试 oracle 打包出 AWQ：直接固定物理位模式。
    rows = [[((i * 0x13579b + j * 0x2468ac) & 0xffffffff) - 2**32
             if ((i * 0x13579b + j * 0x2468ac) & 0xffffffff) >= 2**31
             else ((i * 0x13579b + j * 0x2468ac) & 0xffffffff)
             for j in range(2)] for i in range(256)]
    rows[0] = [-1, -(2**31)]
    zero = [[0, -1], [0x432F8710, -(2**31)]]
    logical = torch.tensor([[((i * 7 + j * 3) % 16) for j in range(16)] for i in range(256)], dtype=torch.int32)
    input_path, output_path = tmp_path / 'input.json', tmp_path / 'output.json'
    input_path.write_text(json.dumps({'cases': [{'name': 'synthetic', 'qweight': rows,
                                                  'qzeros': zero, 'logical': logical.tolist()}]}))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(Path(__file__).with_name('step118_frozen_probe.py')),
                    '--input', str(input_path), '--output', str(output_path)],
                   env=env, check=True, timeout=120, capture_output=True, text=True)
    actual = json.loads(output_path.read_text())['cases'][0]
    qw, qz, q, z = ref.awq_to_standard(torch.tensor(rows, dtype=torch.int32), torch.tensor(zero, dtype=torch.int32))
    assert torch.equal(qw, torch.tensor(actual['qweight'], dtype=torch.int32))
    assert torch.equal(qz, torch.tensor(actual['qzeros'], dtype=torch.int32))
    assert torch.equal(q, torch.tensor(actual['q'], dtype=torch.int32))
    assert torch.equal(z, torch.tensor(actual['z'], dtype=torch.int32))
    assert torch.equal(q, ref.unpack_standard_rows(qw))
    assert torch.equal(z.T, ref.unpack_standard_rows(qz))
    assert torch.equal(ref.pack_standard_rows(logical), torch.tensor(actual['standard_rows'], dtype=torch.int32))
    assert torch.equal(ref.pack_standard_cols(logical), torch.tensor(actual['standard_cols'], dtype=torch.int32))
    awq_packed = torch.tensor(actual['awq_packed'], dtype=torch.int32)
    assert torch.equal(ref.unpack_awq_cols(awq_packed), logical)
    assert actual['qweight_shape'] == [32, 16] and actual['qzeros_shape'] == [2, 2]
    assert actual['qweight_stride'] == [16, 1] and actual['qzeros_stride'] == [2, 1]
