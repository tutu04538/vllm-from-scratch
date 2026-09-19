"""第十七关：旧版/大预算/预算4，对照静态三请求与decode中长prompt到达。
CPU float32 单线程；计时外做独立数值检查，计时内保留全部调度/gather/计算。
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
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(p) for p in PROJECT.glob('step[0-9][0-9]') if p.is_dir()]+[str(PROJECT),str(RECORDS/'tools')]
from verify_step08_contract import reference
FIXTURES={'step16':'step16_engine_contract_20260916T071055.671479Z.json',
          'step17':'step17_engine_contract_20260916T070904.705990Z.json'}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def parameter_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode());h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['step16','step17_large','step17_chunk4'],required=True)
    parser.add_argument('--workload',choices=['static','arrival'],required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    name='step16' if args.mode=='step16' else 'step17'
    mod=importlib.import_module(name);source=Path(mod.__file__);sha=digest(source)
    accepted=json.loads((RECORDS/FIXTURES[name]).read_text())
    assert accepted['status']=='passed' and accepted['source_sha256']==sha
    budget=None if name=='step16' else (32 if args.mode=='step17_large' else 4)
    static=args.workload=='static';cap=3 if static else 2
    def req(r,p,n):return dict(request_id=r,prompt_ids=p,max_new_tokens=n)
    requests=([req('A',[0],3),req('B',[1,2,3],2),req('C',[0,1,0],4)] if static else
              [req('A',[0],4),req('B',[i%5 for i in range(9)],1)])
    tokens=({'A':[2,2,0],'B':[3,0],'C':[0,2,2,2]} if static else {'A':[2,2,0,2],'B':[0]})
    chunked=args.mode=='step17_chunk4'
    order='BAC' if static else ('AB' if chunked else 'BA')
    expected=[dict(request_id=k,output_ids=tokens[k]) for k in order]
    expected_output_steps=({'A':[1,2,3],'B':[1,2],'C':([3,4,5,6] if chunked else [1,2,3,4])} if static else
                           {'A':[1,2,3,4],'B':([4] if chunked else [2])})
    torch.manual_seed(0);rng=torch.get_rng_state().clone()
    weights=parameter_hash(mod.TinyCausalLM())
    assert weights=='0f521e5990a19f9a5fbf4865358eacc2ea23afff9b3bfe11b55fb6bfae408ff7'
    def initialize():
        torch.set_rng_state(rng)
        kw=dict(max_num_seqs=cap)
        if budget is not None:kw['max_num_batched_tokens']=budget
        return mod.Engine(**kw)
    def generate(e):
        results=[]
        if static:
            for r in requests:e.add_request(r)
        else:
            e.add_request(requests[0]);results.extend(e.step());e.add_request(requests[1])
        while e.has_unfinished_requests():results.extend(e.step())
        return results
    def lifecycle():return generate(initialize())
    def check(e):
        assert not e.has_unfinished_requests() and not any(e.kv_cache_pool.block_usage)
        assert parameter_hash(e.model)==weights
        before=copy.deepcopy(requests);records=[];calls=[];samples=[];output_steps={k:[] for k in tokens}
        step_index=0;held={};results=[]
        pre,dec,sample=e.model.forward_prefill,e.model.forward_decode,e.sampler.sample
        def observe(kind,fn,ids,*a,**kw):
            lengths=kw.get('prefill_lengths',kw.get('prompt_lengths',[1]*ids.shape[0]))
            calls.append(dict(step=step_index,kind=kind,ids=ids.tolist(),real_lengths=list(lengths),
                              true_tokens=sum(lengths),padded_positions=ids.numel()))
            assert torch.is_inference_mode_enabled() and not e.model.training
            return fn(ids,*a,**kw)
        def sampling(logits):
            ready=[s for s in e.running if s.cache.length>=len(s.prompt_ids)]
            want=torch.stack([reference(e.model,torch.tensor([s.prompt_ids+s.output_ids]))[0,-1] for s in ready])
            torch.testing.assert_close(logits.double(),want,atol=1e-5,rtol=1e-5)
            out=sample(logits)
            for s in ready:output_steps[s.request_id].append(step_index)
            pool=e.kv_cache_pool
            reserved=sum(len(s.cache.block_table) for s in e.running)
            valid_blocks=sum((s.cache.length+pool.block_size-1)//pool.block_size for s in e.running)
            samples.append(dict(step=step_index,ready=[s.request_id for s in ready],tokens=out.tolist(),
                reserved_blocks=reserved,wholly_unused_reserved_blocks=reserved-valid_blocks))
            return out
        with patch.object(e.model,'forward_prefill',side_effect=lambda ids,*a,**kw:observe('prefill',pre,ids,*a,**kw)),\
             patch.object(e.model,'forward_decode',side_effect=lambda ids,*a,**kw:observe('decode',dec,ids,*a,**kw)),\
             patch.object(e.sampler,'sample',side_effect=sampling):
            for r in (requests if static else requests[:1]):e.add_request(r)
            while e.has_unfinished_requests():
                step_index+=1;assert step_index<20
                if not static and step_index==2:e.add_request(requests[1])
                for s in e.waiting+e.running:held[s.request_id]=s
                finished=e.step();results.extend(finished)
                these=[c for c in calls if c['step']==step_index]
                true=sum(c['true_tokens'] for c in these)
                if budget is not None:assert true<=budget
                records.append(dict(step=step_index,true_tokens=true,
                    padded_positions=sum(c['padded_positions'] for c in these),model_calls=len(these),
                    outputs={k:list(s.output_ids) for k,s in held.items()},finished=copy.deepcopy(finished)))
        assert results==expected and output_steps==expected_output_steps
        assert not any(e.kv_cache_pool.block_usage) and not e.waiting and not e.running
        assert requests==before and parameter_hash(e.model)==weights
        return dict(steps=records,calls=calls,samples=samples,output_steps=output_steps,
                    total_steps=step_index,total_model_calls=len(calls),new_tokens=sum(map(len,tokens.values())),
                    total_true_tokens=sum(c['true_tokens'] for c in calls),
                    total_padded_positions=sum(c['padded_positions'] for c in calls))
    with patch.object(mod,'print',lambda *a,**kw:None,create=True):
        e=initialize();pre=check(e)
        assert lifecycle()==expected
        scopes={'initialize':initialize,'steady_engine':lambda:generate(e),'lifecycle':lifecycle}
        timing={}
        for scope,fn in scopes.items():
            t=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(min_run_time=0.5)
            timing[scope]=dict(median_us=t.median*1e6,iqr_us=t.iqr*1e6,iqr_over_median=t.iqr/t.median,
                high_variance=t.iqr/t.median>0.1,number_per_run=t.number_per_run,raw_times=t.raw_times)
        assert check(e)==pre and lifecycle()==expected
    assert digest(source)==sha
    report=dict(mode=args.mode,module=name,workload=args.workload,capacity=cap,token_budget=budget,
        source_sha256=sha,script_sha256=digest(Path(__file__)),fixture=FIXTURES[name],
        device='cpu',dtype='float32',threads=1,torch=torch.__version__,seed=0,parameter_hash=weights,
        block_size=4,num_kv_blocks=8,requests=requests,outputs=expected,validation=pre,timing=timing,
        scopes={'initialize':'RNG restore, model/Engine/pool creation, temporary result disposal; no generation',
                'steady_engine':'reused initialized Engine, submit/schedule/padding/gather/compute/sample/results/release; no RNG restore or model init',
                'lifecycle':'RNG restore, Engine init, all request processing and result/Engine disposal'},
        notes='CPU wall time; module debug print patched to noop outside timers. Step indices are not TTFT or inter-token latency measurements. Input/cache validation outside timers.')
    path=PROJECT/'benchmarks/results'/f"step17_{args.workload}_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}_{os.getpid()}.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.workload,args.mode,{k:round(v['median_us'],3) for k,v in timing.items()},path)
if __name__=='__main__':main()
