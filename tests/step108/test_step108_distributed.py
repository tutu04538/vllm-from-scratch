"""Real process/GPU checks for forced EAGLE3 accept/reject and rank consistency."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from minivllm.testing.step108_worker import assert_rank_consistent

PROBE = Path(__file__).with_name('step108_tiny_probe.py')
LIFECYCLE = Path(__file__).with_name('step108_lifecycle_probe.py')


def run_probe(tmp_path, tp, k, batch, *, accepted=None, mixed=False, probabilistic=False,
              detect_divergence=False, eos=False):
    assert torch.cuda.device_count() >= tp, f'requires {tp} GPUs'
    output = tmp_path / f'tp{tp}_k{k}_b{batch}_r{accepted}_mixed{mixed}.json'
    cmd = [sys.executable, str(PROBE), '--tp', str(tp), '--k', str(k),
           '--batch', str(batch), '--output', str(output)]
    if accepted is not None:
        cmd += ['--accepted', str(accepted)]
    if mixed:
        cmd += ['--mixed']
    if probabilistic:
        cmd += ['--probabilistic']
    if detect_divergence:
        cmd += ['--detect-divergence']
    if eos:
        cmd += ['--eos']
    proc = subprocess.run(cmd, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                          text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    return json.loads(output.read_text())


@pytest.mark.parametrize('batch', [1, 2, 4])
@pytest.mark.parametrize('k,accepted', [(k, r) for k in (1, 2, 4) for r in range(k+1)])
def test_forced_acceptance_tp2(tmp_path, batch, k, accepted):
    result = run_probe(tmp_path, 2, k, batch, accepted=accepted)
    assert result['accepted'] == accepted


@pytest.mark.parametrize('tp,k', [(2, 2), (4, 1), (4, 2), (4, 4)])
def test_per_request_mixed_acceptance(tmp_path, tp, k):
    run_probe(tmp_path, tp, k, 2, mixed=True)


@pytest.mark.parametrize('tp', [2, 4])
def test_probability_mapping_on_every_rank(tmp_path, tp):
    run_probe(tmp_path, tp, 2, 2, probabilistic=True)


def test_detects_actual_worker_seed_and_length_divergence(tmp_path):
    run_probe(tmp_path, 2, 2, 2, mixed=True, detect_divergence=True)


def test_eos_truncation_tp2(tmp_path):
    run_probe(tmp_path, 2, 2, 2, eos=True)


@pytest.mark.parametrize('mode,prefix', [('preempt', False), ('preempt', True),
                                         ('cancel', False), ('cancel', True), ('failure', False)])
def test_speculative_lifecycle_tp2(tmp_path, mode, prefix):
    assert torch.cuda.device_count() >= 2
    output = tmp_path/f'{mode}.json'
    cmd = [sys.executable, str(LIFECYCLE), '--tp', '2', '--mode', mode, '--output', str(output)]
    if prefix:
        cmd += ['--prefix']
    proc = subprocess.run(cmd, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                          text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    assert json.loads(output.read_text())['mode'] == mode


@pytest.mark.parametrize('tp', [2, 4])
def test_padding_poison_and_cross_block_write_on_every_rank(tmp_path, tp):
    assert torch.cuda.device_count() >= tp
    output = tmp_path / f'padding_tp{tp}.json'
    proc = subprocess.run([sys.executable, str(LIFECYCLE), '--tp', str(tp),
                           '--mode', 'padding', '--output', str(output)],
                          env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                          text=True, capture_output=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    assert len(json.loads(output.read_text())['ranks']) == tp


def test_rank_diagnostic_rejects_seed_and_length_divergence():
    original = {'rank': 0, 'state': {'a': {'length': 7}},
                'trace': [{'kind': 'sample', 'seeds': [123]}]}
    other = copy.deepcopy(original); other['rank'] = 1
    assert_rank_consistent([original, other])
    other['state']['a']['length'] = 8
    with pytest.raises(AssertionError, match='request state'):
        assert_rank_consistent([original, other])
    other = copy.deepcopy(original); other['rank'] = 1
    other['trace'][0]['seeds'][0] += 1
    with pytest.raises(AssertionError, match='field seeds'):
        assert_rank_consistent([original, other])
