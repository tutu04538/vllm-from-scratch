"""Real multi-rank tiny EAGLE3 graph capture and replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

PROBE = Path(__file__).with_name('step109_tiny_probe.py')


@pytest.mark.parametrize('tp,k', [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)])
def test_tiny_graph_captures_target_and_draft(tmp_path, tp, k):
    assert torch.cuda.device_count() >= tp
    path = tmp_path/f'tp{tp}_k{k}.json'
    proc = subprocess.run([sys.executable, str(PROBE), '--tp', str(tp), '--k', str(k),
                           '--batch', '2', '--mixed', '--graph', '--output', str(path)],
                          capture_output=True, text=True, timeout=240,
                          env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                   TRITON_F32_DEFAULT='ieee'))
    assert proc.returncode == 0, proc.stdout + '\n' + proc.stderr
    payload = json.loads(path.read_text())
    assert len(payload['graph_stats']) == tp
    for rank in payload['graph_stats']:
        for name in ('target','prefill') + (('decode',) if k > 1 else ()):
            assert rank[name]['captured'] > 0 and rank[name]['replays'] > 0


def test_rank_failure_during_draft_capture_reaps_workers(tmp_path):
    probe=Path(__file__).with_name('step109_capture_failure.py')
    path=tmp_path/'failure.json'
    proc=subprocess.run([sys.executable,str(probe),'--output',str(path)],capture_output=True,text=True,timeout=300,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    assert json.loads(path.read_text())['workers_reaped']


def test_same_bucket_request_reuse_and_engine_recreation(tmp_path):
    probe=Path(__file__).with_name('step109_reuse_probe.py')
    path=tmp_path/'reuse.json'
    proc=subprocess.run([sys.executable,str(probe),'--output',str(path)],capture_output=True,text=True,timeout=360,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_F32_DEFAULT='ieee'))
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    assert json.loads(path.read_text())['status']=='passed'


def test_rank_failure_during_graph_proposal_does_not_commit(tmp_path):
    probe=Path(__file__).with_name('step109_propose_failure.py')
    path=tmp_path/'propose_failure.json'
    proc=subprocess.run([sys.executable,str(probe),'--tp','2','--mode','failure','--output',str(path)],
                        capture_output=True,text=True,timeout=300,
                        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr
    assert json.loads(path.read_text())['mode']=='failure'


def test_frozen_speculator_sources_match_installed_vllm():
    import hashlib
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    reference=root/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料'
    manifest=json.loads((reference/'源码清单.json').read_text())
    hashes={item['path']:item['sha256'] for item in manifest['files']}
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    for name in ('v1/worker/gpu/spec_decode/autoregressive/speculator.py',
                 'v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py',
                 'v1/worker/gpu/model_runner.py'):
        expected=hashes[name]
        assert hashlib.sha256((reference/'vllm_0_28_0'/name).read_bytes()).hexdigest()==expected
        assert hashlib.sha256((installed/name).read_bytes()).hexdigest()==expected
