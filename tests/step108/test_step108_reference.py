"""Real frozen V2 all-rank trace on a shared head-size-32 tiny fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name('step108_reference_probe.py')
COMPARE = Path(__file__).with_name('step108_compare_reference.py')


@pytest.mark.parametrize('tp', [2, 4])
@pytest.mark.parametrize('k', [1, 2, 4])
def test_all_rank_frozen_trace(tmp_path, tp, k):
    assert torch.cuda.device_count() >= tp
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TRITON_F32_DEFAULT='ieee')
    fixture, ref, repeat, mini = (tmp_path/name for name in ('fixture', 'ref', 'repeat', 'mini'))
    contract = tmp_path/'contract.json'
    def run(args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
    common = ['--tp', tp, '--k', k, '--fixture', fixture]
    run([PROBE, '--implementation', 'frozen', *common, '--output', ref])
    run([PROBE, '--implementation', 'frozen', *common, '--output', repeat])
    run([COMPARE, '--reference', ref, '--other', repeat, '--contract', contract,
         '--register', '--output', tmp_path/'repeat.json'])
    run([PROBE, '--implementation', 'mini', *common, '--output', mini])
    run([COMPARE, '--reference', ref, '--other', mini, '--contract', contract,
         '--output', tmp_path/'comparison.json'])
    result = json.loads((tmp_path/'comparison.json').read_text())
    assert result['status'] == 'passed'
    for name in ('propose/target_valid_kv', 'propose/draft_valid_kv',
                 'draft_layer/written_kv', 'combine_hidden_states/inputs/0', 'compute_logits/output'):
        assert result['metrics'][name]['count'] >= tp


def test_forced_accept_then_reject_matches_frozen_on_each_rank(tmp_path):
    assert torch.cuda.device_count() >= 2
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TRITON_F32_DEFAULT='ieee')
    fixture, natural, ref, repeat, mini = (tmp_path/name for name in
                                           ('fixture', 'natural', 'ref', 'repeat', 'mini'))
    contract = tmp_path/'contract.json'
    def run(args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + '\n' + result.stderr
    common = ['--tp', 2, '--k', 2, '--fixture', fixture]
    run([PROBE, '--implementation', 'frozen', *common, '--output', natural])
    tokens = json.loads((natural/'summary.json').read_text())['tokens']
    forced = [tokens[1], (tokens[2] + 1) % 11]
    common += ['--force-first', json.dumps(forced)]
    run([PROBE, '--implementation', 'frozen', *common, '--output', ref])
    run([PROBE, '--implementation', 'frozen', *common, '--output', repeat])
    run([COMPARE, '--reference', ref, '--other', repeat, '--contract', contract,
         '--register', '--output', tmp_path/'repeat.json'])
    run([PROBE, '--implementation', 'mini', *common, '--output', mini])
    run([COMPARE, '--reference', ref, '--other', mini, '--contract', contract,
         '--output', tmp_path/'comparison.json'])
    for rank in range(2):
        events = torch.load(ref/f'rank{rank}.pt', weights_only=True)
        proposals = [e for e in events if e['event'] == 'propose']
        assert proposals[1]['num_sampled'].tolist() == [2]
        assert proposals[1]['num_rejected'].tolist() == [1]
