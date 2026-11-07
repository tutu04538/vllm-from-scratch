"""Tiny Qwen TP=1/2/4 real processes, compared on valid KV slots."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
PROBE=ROOT/'tests/step098/step098_probe.py'


def run(tp, path, caps=None, expect=True):
    cmd=[sys.executable,str(PROBE),'--tp',str(tp),'--output',str(path)]
    if caps: cmd += ['--caps',caps]
    p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=120)
    if expect:
        assert p.returncode==0, p.stdout+p.stderr
        return torch.load(path,weights_only=True)
    assert p.returncode != 0 and 'rank 1 的 KV 容量 3 块' in p.stderr
    assert not path.exists()


@pytest.mark.parametrize('tp,caps', [(2,'8,6'),(4,'8,7,6,8')])
def test_real_rank_cache(tmp_path,tp,caps):
    assert torch.cuda.device_count() >= tp, '需要足够的真实 GPU'
    reference=run(1,tmp_path/'ref.pt')
    actual=run(tp,tmp_path/'tp.pt',caps)
    assert actual['capacity']==reference['capacity']==6
    assert actual['block_ids']==reference['block_ids']
    assert all(trace == reference['inputs'][0] for trace in actual['inputs'])
    assert [x['positions'] for x in actual['inputs'][0]] == [list(range(17)), [17], list(range(17))]
    ids=actual['block_ids']
    per_block=2*16*1*16*4*2
    assert actual['budgets']==[int(n)*per_block for n in caps.split(',')]
    for step, trace in enumerate(actual['inputs'][0]):
        expected_slots=[ids[step][0][pos//16]*16+pos%16 for pos in trace['positions']]
        assert trace['slot_mapping'][0] == expected_slots
    assert len(ids[0][0])==2 and set(ids[0][0])==set(ids[2][0])
    errors=[]
    reference_values=[]
    worst={"abs": -1.0}
    for step, length in enumerate((17,18,17)):
        for name, ref in reference['snapshots'][step][0].items():
            first_block=ids[step][0][0]
            assert not torch.equal(ref[:,first_block,0,0], ref[:,first_block,0,1])
            for rank in range(tp):
                got=actual['snapshots'][step][rank][name]
                assert got.shape==(2,6,16,1,16)
                head=rank//(tp//2)
                for pos in range(length):
                    block=ids[step][0][pos//16]; slot=pos%16
                    expected=ref[:,block,slot,head:head+1]
                    torch.testing.assert_close(got[:,block,slot],expected,rtol=1e-5,atol=1e-5)
                    diff=(got[:,block,slot]-expected).flatten()
                    errors.append(diff)
                    reference_values.append(expected.flatten())
                    idx=int(diff.abs().argmax())
                    if abs(diff[idx].item()) > worst['abs']:
                        worst={'abs':abs(diff[idx].item()), 'step':step, 'layer':name, 'rank':rank, 'position':pos, 'element':idx, 'actual':got[:,block,slot].flatten()[idx].item(), 'reference':expected.flatten()[idx].item()}
        if tp==4:
            for name in actual['snapshots'][step][0]:
                for rank in (0,2):
                    for pos in range(length):
                        block=ids[step][0][pos//16];slot=pos%16
                        assert torch.equal(actual['snapshots'][step][rank][name][:,block,slot],actual['snapshots'][step][rank+1][name][:,block,slot])

    errors=torch.cat(errors); reference_values=torch.cat(reference_values)
    print(f'TP{tp} max_abs={errors.abs().max().item():.9g} relative_l2={(errors.norm()/reference_values.norm()).item():.9g} worst={worst}')


def test_rank_budget_failure(tmp_path):
    assert torch.cuda.device_count() >= 2, '需要两张真实 GPU'
    run(2,tmp_path/'fail.pt','8,3',expect=False)
