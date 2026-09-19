"""step19/20 paired comparison: unequal prefill, mixed arrival, warm mixed arrival.
Same weights, request/step policy and cache start state; CPU float32 single thread.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,importlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from torch.utils.benchmark import Timer
PROJECT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(p) for p in PROJECT.glob('step[0-9][0-9]') if p.is_dir()]+[str(PROJECT),str(RECORDS/'tools')]
from verify_step08_contract import reference

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weights(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['step19','step20'],required=True)
    parser.add_argument('--workload',choices=['unequal_prefill','mixed','warm_mixed'],required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    m=importlib.import_module(args.mode);source=Path(m.__file__);digest=sha(source)
    helper=importlib.import_module(args.mode+'_test_helpers')
    fixtures={}
    for kind in ['prefix','model','disabled_engine']:
        f=sorted(RECORDS.glob(args.mode+'_'+kind+'_contract_*.json'))[-1];d=json.loads(f.read_text())
        assert d['status']=='passed' and d['source_sha256']==digest
        fixtures[kind]=str(f)
    unequal=args.workload=='unequal_prefill';warm=args.workload=='warm_mixed';budget=10 if unequal else 5
    def req(r,p,n):return dict(request_id=r,prompt_ids=p,max_new_tokens=n)
    requests=([req('A',[0,1,2,3,0,1,2],3),req('B',[1,2],2),req('C',[0],4)] if unequal else
              [req('A',[0,1,2],4),req('B',([0,1,2,3] if warm else [])+[1,0,3],1),req('C',[2],1)])
    original=copy.deepcopy(requests)
    torch.manual_seed(0);rng=torch.get_rng_state().clone()
    def initialize():
        torch.set_rng_state(rng)
        return m.Engine(max_num_seqs=3,max_num_batched_tokens=budget,num_kv_blocks=24,enable_prefix_caching=True)
    e=initialize();p=e.kv_cache_pool;weight_hash=weights(e.model)
    assert weight_hash=='0f521e5990a19f9a5fbf4865358eacc2ea23afff9b3bfe11b55fb6bfae408ff7'
    expected={}
    with torch.inference_mode():
        for r in requests:
            out=[]
            for _ in range(r['max_new_tokens']):
                token=reference(e.model,torch.tensor([r['prompt_ids']+out]))[0,-1].argmax().item()
                out.append(token)
                if token==4:break
            expected[r['request_id']]=out
    prime_trace=[]
    if warm:
        e.add_request(req('seed',[0,1,2,3],1))
        while e.has_unfinished_requests():prime_trace.append(helper.step_trace(e))
    assert not any(p.block_usage)
    state=(dict(p.block_hash),dict(p.block_to_hash),list(p.block_usage),list(p.block_last_used),p.lru_seq)
    retained={b:(p.k_cache[b].clone(),p.v_cache[b].clone()) for b in p.block_to_hash}
    def restore():
        p.block_hash.clear();p.block_hash.update(state[0]);p.block_to_hash.clear();p.block_to_hash.update(state[1])
        p.block_usage[:]=state[2];p.block_last_used[:]=state[3];p.lru_seq=state[4]
    def generate():
        restore();results=[]
        if unequal:
            for r in requests:e.add_request(r)
        else:
            e.add_request(requests[0]);results.extend(e.step())
            for r in requests[1:]:e.add_request(r)
        while e.has_unfinished_requests():results.extend(e.step())
        return results
    def check():
        assert not e.has_unfinished_requests() and not any(p.block_usage)
        restore();trace=[];results=[];out_steps={r['request_id']:[] for r in requests}
        for r in (requests if unequal else requests[:1]):e.add_request(r)
        while e.has_unfinished_requests():
            if not unequal and len(trace)==1:
                for r in requests[1:]:e.add_request(r)
            assert len(trace)<40
            row=helper.step_trace(e);trace.append(row);results.extend(row['result'])
            for sample in row['samples']:
                for rid in sample['requests']:out_steps[rid].append(len(trace))
            assert sum(sum(c['counts']) for c in row['calls'])<=budget
            assert len(row['calls'])<=1
        assert {r['request_id']:r['output_ids'] for r in results}==expected
        assert len(results)==len(requests) and requests==original and not any(p.block_usage)
        assert weights(e.model)==weight_hash
        for b,(k,v) in retained.items():assert torch.equal(k,p.k_cache[b]) and torch.equal(v,p.v_cache[b])
        calls=[c for row in trace for c in row['calls']]
        true=sum(sum(c['counts']) for c in calls)
        projected=sum(len(c['inputs']) for c in calls)
        new=sum(map(len,expected.values()))
        hit=4 if warm else 0
        assert true==sum(len(r['prompt_ids'])+len(expected[r['request_id']])-1 for r in requests)-hit
        assert projected==true
        return dict(trace=trace,results=results,output_steps=out_steps,total_steps=len(trace),total_model_calls=len(calls),
                    true_tokens=true,projected_positions=projected,hit_tokens=hit,new_tokens=new,
                    initial_cached_blocks=len(state[0]),final_cached_blocks=len(p.block_hash))
    before=check();assert generate()==before['results']
    timing={}
    for name,fn in [('initialize',initialize),('workload_with_metadata_reset',generate)]:
        t=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(min_run_time=0.5)
        timing[name]=dict(median_us=t.median*1e6,iqr_us=t.iqr*1e6,iqr_over_median=t.iqr/t.median,
                         high_variance=t.iqr/t.median>0.1,number_per_run=t.number_per_run,raw_times=t.raw_times)
    assert check()==before and sha(source)==digest
    report=dict(mode=args.mode,workload=args.workload,source_sha256=digest,script_sha256=sha(Path(__file__)),fixtures=fixtures,
                device='cpu',dtype='float32',threads=1,torch=torch.__version__,seed=0,parameter_hash=weight_hash,
                capacity=3,block_size=4,num_kv_blocks=24,token_budget=budget,requests=requests,validation=before,timing=timing,
                prime_trace=prime_trace,scope='metadata restore + submit/schedule/model/sample/hash/result/release for whole workload; excludes Engine initialization and warm-cache seed creation; K/V seed blocks stay unchanged, no pool copying in timer',
                notes='Every iteration starts with identical cold/warm prefix metadata. CPU wall time, no GPU/TTFT/ITL claims. No eviction needed in timing workload.')
    path=PROJECT/'benchmarks/results'/('step20_compare_'+args.workload+'_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.workload,args.mode,{k:round(v['median_us'],3) for k,v in timing.items()},'true',before['true_tokens'],'projected',before['projected_positions'],'calls',before['total_model_calls'],path)
if __name__=='__main__':main()
