"""TP rank sampling, mirror ownership and frozen Gumbel differential."""
import hashlib
import random
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROBE=ROOT/'tests/step100/step100_probe.py'


def run(tp,mode,tmp_path):
    out=tmp_path/f'{mode}{tp}.json'
    cmd=[sys.executable,str(PROBE),'--tp',str(tp),'--mode',mode,'--output',str(out)]
    p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
    assert p.returncode==0,p.stdout+p.stderr
    trace_dir=os.environ.get('STEP100_TRACE_DIR')
    if trace_dir:
        Path(trace_dir).mkdir(parents=True,exist_ok=True)
        shutil.copyfile(out,Path(trace_dir)/out.name)
    return json.loads(out.read_text())


def assert_rounds(value):
    previous={}
    for record in value['rounds']:
        ranks=record['ranks']; ref=ranks[0]
        assert all(rank['states']==ref['states'] for rank in ranks)
        assert all(len(rank['trace'])==len(ref['trace']) for rank in ranks)
        for rank in ranks[1:]:
            for actual,expected in zip(rank['trace'],ref['trace']):
                for key in ('req_ids','slots','positions','expanded_slots','seeds','sampled','num_sampled','num_rejected','logprobs'):
                    assert actual[key]==expected[key],key
                torch.testing.assert_close(torch.tensor(actual['logits']),torch.tensor(expected['logits']),rtol=1e-5,atol=1e-5)
        for req,tokens in record['outputs'].items():
            if req in previous:
                assert len(tokens)==len(previous[req])+1
                assert tokens[:-1]==previous[req]
            else:assert len(tokens)==1
            previous[req]=tokens
        for req,tokens in record['scheduler'].items():
            if req in previous:assert tokens==previous[req]


@pytest.mark.parametrize('tp',[2,4])
@pytest.mark.parametrize('mode',['greedy','random','preempt'])
def test_real_tp_sampling_and_mirrors(tp,mode,tmp_path):
    assert torch.cuda.device_count()>=tp, '需要真实 GPU'
    value=run(tp,mode,tmp_path);assert_rounds(value)
    rounds=value['rounds']
    if mode!='preempt':
        first=rounds[0]['ranks'][0]['trace'][-1];second=rounds[1]['ranks'][0]['trace'][-1]
        third=rounds[2]['ranks'][0]['trace'][-1]
        assert first['req_ids']==['a','b'] and first['seeds']==[17,23]
        assert second['req_ids']==['b','a'] and second['seeds']==[23,17]
        assert third['req_ids']==['b','c'] and third['seeds'][0]==23
        assert isinstance(third['seeds'][1],int) and third['seeds'][1] not in (17,23)
        assert first['slots'][0]==third['slots'][1]
    else:
        assert max(x['preemptions'] for x in rounds)>0
        first=rounds[0]['ranks'][0]['trace'][-1]
        assert first['req_ids']==['a','b'] and first['seeds'][0]==17
        assert first['seeds'][1] not in (17,23)  # B uses seed=None
        after=[x for x in rounds if x['preemptions']>0]
        assert any('b' in r['ranks'][0]['trace'][-1]['req_ids'] and
                   len(r['scheduler'].get('b',[]))>16 for r in after)


def test_model_seed_reset_and_logit_bias_rejection():
    from minivllm import ModelConfig, VllmConfig, SamplingParams
    from minivllm.worker.worker import Worker
    class Runner:
        def capture_model(self):
            np.random.random(37)
            torch.rand(17)
    config=VllmConfig(model_config=ModelConfig('unused',seed=123))
    worker=Worker(config,Runner(),rank=0,device_index=0)
    worker.compile_or_warm_up_model()
    got=np.random.randint(-(2**63),2**63-1)
    py_value=random.random()
    torch_value=torch.rand(4)
    np.random.seed(123); random.seed(123); torch.manual_seed(123)
    assert got==np.random.randint(-(2**63),2**63-1)
    assert py_value==random.random()
    assert torch.equal(torch_value,torch.rand(4))
    with pytest.raises(NotImplementedError,match='logit_bias'):
        SamplingParams(logit_bias={1:2.0})
    with pytest.raises(ValueError,match='model seed'):
        ModelConfig('unused',seed=-1)


def test_frozen_gumbel_and_distribution():
    import vllm.v1.worker.gpu.sample.gumbel as upstream
    from minivllm.worker.gpu.sample.gumbel import gumbel_sample
    frozen=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0/v1/worker/gpu/sample/gumbel.py'
    assert hashlib.sha256(Path(upstream.__file__).read_bytes()).digest()==hashlib.sha256(frozen.read_bytes()).digest()
    assert torch.cuda.is_available(), '需要真实 GPU'
    n=8192;device='cuda';rows=torch.arange(n,device=device,dtype=torch.int64)
    logits=torch.tensor([0.,0.5,1.0],device=device).expand(n,-1).contiguous()
    temps=torch.ones(n,device=device);seeds=rows+17;pos=torch.full((n,),7,dtype=torch.int64,device=device)
    ours=gumbel_sample(logits,rows,temps,seeds,pos,apply_temperature=False)
    expected=upstream.gumbel_sample(logits,rows,temps,seeds,pos,apply_temperature=False)
    assert torch.equal(ours,expected)
    probs=torch.bincount(ours,minlength=3).float()/n
    target=torch.softmax(logits[0],-1)
    assert torch.max(torch.abs(probs-target)).item()<0.03
    print('empirical',probs.tolist(),'expected',target.tolist())
