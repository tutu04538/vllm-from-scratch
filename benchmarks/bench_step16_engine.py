"""第十五/十六关同负载对照：初始化、复用 Engine、完整生命周期；CPU 单线程。
不改学生源码。模块内调试 print 临时替换为 no-op，补丁建立不计时。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
from unittest.mock import patch
import torch
from torch.utils.benchmark import Timer

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
FIXTURES={'step15':'step15_engine_contract_20260915T153003.110249Z.json',
          'step16':'step16_engine_contract_20260915T152918.255633Z.json'}
REQUESTS=[dict(request_id='A',prompt_ids=[0],max_new_tokens=3),
          dict(request_id='B',prompt_ids=[1,2,3],max_new_tokens=2),
          dict(request_id='C',prompt_ids=[0,1,0],max_new_tokens=4)]
TOKENS={'A':[2,2,0],'B':[3,0],'C':[0,2,2,2]}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def parameter_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--module',choices=['step15','step16'],required=True)
    parser.add_argument('--capacity',type=int,choices=[1,2,3],required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    module=importlib.import_module(args.module)
    source=Path(module.__file__);source_hash=digest(source)
    accepted=json.loads((RECORDS/FIXTURES[args.module]).read_text())
    assert accepted['status']=='passed' and accepted['source_sha256']==source_hash
    fixture=next(c for c in accepted['cases'] if c['name']==f'random_capacity_{args.capacity}')
    expected_calls=[{'kind':c['kind'],'ids':c['ids']} for c in fixture['calls']]
    expected_samples=[c['tokens'] for c in fixture['samples']]
    expected=[dict(request_id=k,output_ids=TOKENS[k]) for k in ('ABC' if args.capacity==1 else 'BAC')]
    torch.manual_seed(0);rng=torch.get_rng_state().clone()
    canonical=module.TinyCausalLM()
    weight_hash=parameter_hash(canonical)
    assert weight_hash=='0f521e5990a19f9a5fbf4865358eacc2ea23afff9b3bfe11b55fb6bfae408ff7'

    def initialize():
        torch.set_rng_state(rng)
        return module.Engine(max_num_seqs=args.capacity)

    def generate(engine):
        for req in REQUESTS:engine.add_request(req)
        results=[]
        while engine.has_unfinished_requests():results.extend(engine.step())
        return results

    def lifecycle():return generate(initialize())

    def check(engine):
        assert not engine.has_unfinished_requests()
        assert parameter_hash(engine.model)==weight_hash
        original=copy.deepcopy(REQUESTS);calls=[];samples=[];occupancy=[];steps=[]
        prefill,decode,sample=engine.model.forward_prefill,engine.model.forward_decode,engine.sampler.sample
        def observe(kind,fn,ids,*a,**kw):
            calls.append(dict(kind=kind,ids=ids.tolist()))
            return fn(ids,*a,**kw)
        def sample_checked(logits):
            assert not logits.requires_grad and torch.is_inference_mode_enabled()
            ids=sample(logits);samples.append(ids.tolist())
            live=[s for s in engine.running if s.cache is not None]
            if args.module=='step16':
                pool=engine.kv_cache_pool
                reserved=sum(len(s.cache.block_table) for s in live)
                used_blocks=sum((s.cache.length+pool.block_size-1)//pool.block_size for s in live)
                valid_tokens=sum(s.cache.length for s in live)
                occupancy.append(dict(running=[s.request_id for s in live],reserved_blocks=reserved,
                    valid_blocks=used_blocks,reserved_but_wholly_unused_blocks=reserved-used_blocks,
                    valid_tokens=valid_tokens,reserved_unused_token_slots=reserved*pool.block_size-valid_tokens,
                    free_blocks=pool.num_kv_blocks-reserved,
                    persistent_kv_bytes=sum(t.numel()*t.element_size() for t in (pool.k_cache,pool.v_cache))))
            else:
                valid_tokens=sum(s.cache['length'] for s in live)
                occupancy.append(dict(running=[s.request_id for s in live],valid_tokens=valid_tokens,
                    persistent_kv_bytes=sum(s.cache[k].numel()*s.cache[k].element_size() for s in live for k in ('k','v'))))
            return ids
        with patch.object(engine.model,'forward_prefill',side_effect=lambda ids,*a,**kw:observe('prefill',prefill,ids,*a,**kw)),\
             patch.object(engine.model,'forward_decode',side_effect=lambda ids,*a,**kw:observe('decode',decode,ids,*a,**kw)),\
             patch.object(engine.sampler,'sample',side_effect=sample_checked):
            for req in REQUESTS:engine.add_request(req)
            results=[]
            while engine.has_unfinished_requests():
                assert len(steps)<16
                finished=engine.step();results.extend(finished);steps.append(copy.deepcopy(finished))
        assert results==expected and calls==expected_calls and samples==expected_samples
        assert REQUESTS==original and parameter_hash(engine.model)==weight_hash
        assert not engine.waiting and not engine.running
        if args.module=='step16':assert not any(engine.kv_cache_pool.block_usage)
        return dict(calls=calls,samples=samples,steps=len(steps),new_tokens=9,
            model_calls=len(calls),processed_token_positions=sum(len(c['ids'])*len(c['ids'][0]) for c in calls),
            occupancy_before_each_sample=occupancy,peak_persistent_kv_bytes=max(o['persistent_kv_bytes'] for o in occupancy))

    with patch.object(module,'print',lambda *a,**kw:None,create=True):
        reused=initialize()
        pre=check(reused)
        assert lifecycle()==expected
        scopes={
            'initialize':(initialize,'RNG restore + model/Engine init + pool init (step16); no generation; temporary result disposal included'),
            'steady_engine':(lambda:generate(reused),'existing model/Engine; submit + schedule + KV writes/gather + compute + sampling + output + release; no init/RNG restore'),
            'lifecycle':(lifecycle,'RNG restore + initialization + complete request processing + result/Engine disposal'),
        }
        timing={}
        for name,(fn,desc) in scopes.items():
            measure=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(min_run_time=0.5)
            timing[name]=dict(median_us=measure.median*1e6,iqr_us=measure.iqr*1e6,
                iqr_over_median=measure.iqr/measure.median,high_variance=measure.iqr/measure.median>0.1,
                number_per_run=measure.number_per_run,raw_times=measure.raw_times,scope=desc)
        post=check(reused)
        assert pre==post and lifecycle()==expected
    assert digest(source)==source_hash
    report=dict(module=args.module,capacity=args.capacity,source_sha256=source_hash,
        script_sha256=digest(Path(__file__)),fixture=FIXTURES[args.module],torch=torch.__version__,
        device='cpu',dtype='float32',threads=1,seed=0,parameter_hash=weight_hash,
        block_size=4 if args.module=='step16' else None,num_kv_blocks=8 if args.module=='step16' else None,
        requests=REQUESTS,outputs=expected,validation=pre,timing=timing)
    out=PROJECT/'benchmarks/results'/f"step16_engine_{args.module}_cap{args.capacity}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}_{os.getpid()}.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.module,args.capacity,{k:round(v['median_us'],3) for k,v in timing.items()},out)

if __name__=='__main__':main()
