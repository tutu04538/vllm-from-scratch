"""Real production TP Worker lifecycle and independent slot/KV invariants."""
from collections import Counter
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'tests/step102/step102_probe.py'


def run(tp, mode, prefix, path):
    cmd = [sys.executable, str(PROBE), '--tp', str(tp), '--mode', mode, '--output', str(path)]
    if prefix:
        cmd.append('--prefix')
    p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'},
                       capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stdout + p.stderr
    trace_dir = os.environ.get('STEP102_TRACE_DIR')
    if trace_dir:
        dest = Path(trace_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest / f'{mode}_tp{tp}_prefix{int(prefix)}.pt')
    return torch.load(path, weights_only=True)


def verify_trace(result):
    for row in result['rounds']:
        ranks = row['ranks']
        base = ranks[0]
        assert all(x['requests'] == base['requests'] and x['trace_count'] == base['trace_count']
                   and x['trace'] == base['trace'] for x in ranks)
        # Independent arithmetic oracle: request/block/position determines its physical KV slot.
        trace = base['trace']
        if trace:
            expected = []
            for i, req in enumerate(trace['requests']):
                begin, end = trace['query_start'][i:i+2]
                assert end - begin == trace['counts'][i]
                positions = trace['positions'][begin:end]
                assert positions == list(range(trace['computed'][i], trace['computed'][i] + end - begin))
                table = trace['block_tables'][0][i]
                expected.extend(table[pos // 4] * 4 + pos % 4 for pos in positions)
            assert trace['slot_mapping'][0] == expected
        # Reference counts follow the authoritative live request block tables, not resident slots.
        counts = Counter(block for req in row['scheduler'].values() for block in req['blocks'][0])
        assert row['refcounts'] == [counts[i] for i in range(result['capacity'])]
        assert row['free'] == sum(n == 0 for n in row['refcounts'])
        for req in row['trace']['preempted']:
            assert req not in base['requests'], (req, row['trace'])
        for req, authoritative in row['scheduler'].items():
            if req in row['trace']['scheduled']:
                resident = base['requests'][req]
                assert resident['computed'] == authoritative['computed']
                assert resident['blocks'] == list(authoritative['blocks'][0])
                assert resident['tokens'] == authoritative['tokens']
    assert not result['rounds'][-1]['ranks'][0]['requests']
    assert not any(result['rounds'][-1]['refcounts'])
    assert result['rounds'][-1]['free'] == result['capacity']


def compare_valid_kv(reference, actual):
    assert len(reference['rounds']) == len(actual['rounds'])
    for expected, row in zip(reference['rounds'], actual['rounds']):
        assert row['outputs'] == expected['outputs']
        ref_rank = expected['ranks'][0]
        for req, state in row['scheduler'].items():
            # Same token history is a prerequisite for a KV numerical comparison;
            # scheduler correctness is checked by the independent invariants above.
            ref_state = expected['scheduler'][req]
            assert state['tokens'] == ref_state['tokens'] and state['computed'] == ref_state['computed']
            blocks, ref_blocks = state['blocks'][0], ref_state['blocks'][0]
            for pos in range(state['computed']):
                block, ref_block, offset = blocks[pos // 4], ref_blocks[pos // 4], pos % 4
                for rank, got in enumerate(row['ranks']):
                    # Tiny has two KV heads. For TP4 the two adjacent ranks replicate each head.
                    head = rank // max(1, actual['tp'] // 2)
                    for layer, tensor in ref_rank['kv'].items():
                        torch.testing.assert_close(got['kv'][layer][:,block,offset],
                            tensor[:,ref_block,offset,head:head+1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('mode', ['prefix', 'preempt'])
@pytest.mark.parametrize('prefix', [False, True])
def test_prefix_and_preemption_tp2_tp4(tmp_path, mode, prefix):
    assert torch.cuda.device_count() >= 4
    reference = run(1, mode, prefix, tmp_path/'tp1.pt')
    verify_trace(reference)
    for tp in (2, 4):
        actual = run(tp, mode, prefix, tmp_path/f'tp{tp}.pt')
        verify_trace(actual)
        compare_valid_kv(reference, actual)
        hits = [row['trace']['hits'] for row in actual['rounds']]
        if prefix:
            assert any(x.get('b', 0) >= 4 for x in hits)
        else:
            assert not any(hits)
        if mode == 'preempt':
            assert max(r['preemptions'] for r in actual['rounds']) > 0
            assert any(r['trace']['preempted'] for r in actual['rounds'])
            assert any(r['preemptions'] and 'b' in r['trace']['scheduled']
                       for r in actual['rounds'])


@pytest.mark.parametrize('tp', [2, 4])
@pytest.mark.parametrize('mode', ['cancel_before', 'cancel_after'])
@pytest.mark.parametrize('prefix', [False, True])
def test_cancel_and_slot_reuse(tmp_path, tp, mode, prefix):
    assert torch.cuda.device_count() >= tp
    result = run(tp, mode, prefix, tmp_path/'cancel.pt')
    verify_trace(result)
    first = result['rounds'][0]['ranks'][0]['requests']['a']
    reuse = next(row for row in result['rounds'] if 'c' in row['ranks'][0]['requests'])
    c = reuse['ranks'][0]['requests']['c']
    assert c['slot'] == first['slot'] and c['seed'] == 31
    assert c['tokens'][:5] == [10,9,8,7,6]
    assert 'a' not in reuse['ranks'][0]['requests']
    if mode == 'cancel_after':
        row = next(row for row in result['rounds'] if row['label'] == 'cancel-after-execute')
        assert row['outputs'] == {} and 'a' not in row['scheduler']


def test_failed_rank_never_commits(tmp_path):
    assert torch.cuda.device_count() >= 2
    result = run(2, 'fail', False, tmp_path/'fail.pt')
    assert result['before'] == result['after'] == []
    assert 'step102 injected rank 1 sample failure' in result['failure']


def test_heterogeneous_rank_capacity_and_rejection(tmp_path):
    assert torch.cuda.device_count() >= 4
    path = tmp_path/'capacity.pt'
    probe = ROOT/'tests/step098/step098_probe.py'
    base = [sys.executable, str(probe), '--tp', '4', '--output', str(path)]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    ok = subprocess.run(base+['--caps','8,7,6,8'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert ok.returncode == 0,ok.stdout+ok.stderr
    assert torch.load(path,weights_only=True)['capacity']==6
    path.unlink()
    bad = subprocess.run(base+['--caps','8,3,6,8'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert bad.returncode != 0 and 'rank 1 的 KV 容量 3 块' in bad.stderr
    assert not path.exists()


def test_frozen_scheduler_preemption_protocol():
    import hashlib
    from types import SimpleNamespace as N
    from vllm.v1.core.sched.scheduler import Scheduler as UpScheduler
    from vllm.v1.request import RequestStatus as UpStatus
    from minivllm.core.sched.scheduler import Scheduler
    from minivllm.request import RequestStatus
    from pathlib import Path
    import inspect
    frozen=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/v1/core/sched/scheduler.py'
    assert hashlib.sha256(Path(inspect.getfile(UpScheduler)).read_bytes()).digest()==hashlib.sha256(frozen.read_bytes()).digest()
    class Waiting:
        def __init__(self): self.items=[]
        def prepend_request(self, req): self.items.insert(0,req.request_id)
    up_req=N(request_id='b',status=UpStatus.RUNNING,num_computed_tokens=9,spec_token_ids=[3],
             num_in_flight_tokens=0,drop_stale_output=False,num_stale_output_tokens=0,
             num_output_placeholders=0,num_preemptions=0)
    up=N(_free_request_blocks=lambda request:None,encoder_cache_manager=N(free=lambda request:None),
         _inflight_prefills=N(discard=lambda request:None),waiting=Waiting(),reset_preempted_req_ids=set(),log_stats=False)
    UpScheduler._preempt_request(up,up_req,0.0)
    req=N(request_id='b',status=RequestStatus.RUNNING,num_computed_tokens=9,spec_token_ids=[3],
          num_in_flight_tokens=0,num_output_placeholders=0,num_preemptions=0)
    actual=N(kv_cache_manager=N(free=lambda request:None),waiting=Waiting(),
             reset_preempted_req_ids=set(),num_preemptions=0)
    Scheduler._preempt_request(actual,req)
    assert actual.reset_preempted_req_ids == up.reset_preempted_req_ids == {'b'}
    assert actual.waiting.items == up.waiting.items == ['b']
    assert req.num_computed_tokens == up_req.num_computed_tokens == 0
    assert req.num_preemptions == up_req.num_preemptions == 1
    assert req.spec_token_ids == up_req.spec_token_ids == []
    assert req.status == RequestStatus.PREEMPTED and up_req.status == UpStatus.PREEMPTED


@pytest.mark.parametrize('tp', [2, 4])
def test_prefix_hit_matches_cold_output(tmp_path, tp):
    assert torch.cuda.device_count() >= tp
    cold = run(tp, 'prefix', False, tmp_path/'cold.pt')
    warm = run(tp, 'prefix', True, tmp_path/'warm.pt')
    def final_b(result):
        return [row['outputs']['b'] for row in result['rounds'] if 'b' in row['outputs']][-1]
    assert final_b(cold) == final_b(warm)
    assert not any(row['trace']['hits'] for row in cold['rounds'])
    assert any(row['trace']['hits'].get('b') == 8 for row in warm['rounds'])
