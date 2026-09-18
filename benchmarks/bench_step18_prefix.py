"""step18 同一目标请求，四种 prefix cache 状态。CPU/float32/单线程。
准备缓存不计时；每次计时内恢复池的元数据快照，防止 cold/miss 被重复调用变成 warm。
KV Tensor 不在计时中清零：cold 无有效历史，warm/miss 的种子块不会被目标覆盖。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from torch.utils.benchmark import Timer
PROJECT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(PROJECT),str(RECORDS/'tools')]
import step18 as m
from verify_step08_contract import reference
from verify_step18_without_lru_initial import step_trace

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weights(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['disabled','cold','warm','miss'],required=True)
    args=parser.parse_args();torch.set_num_threads(1);source=Path(m.__file__);digest=sha(source)
    fixtures={}
    for pattern in ['step18_prefix_contract_*.json','step18_model_contract_*.json','step18_disabled_engine_contract_*.json']:
        f=sorted(RECORDS.glob(pattern))[-1];d=json.loads(f.read_text())
        assert d['status']=='passed' and d['source_sha256']==digest
        fixtures[pattern]=str(f)
    torch.manual_seed(0);rng=torch.get_rng_state().clone()
    def initialize():
        torch.set_rng_state(rng)
        return m.Engine(max_num_seqs=1,max_num_batched_tokens=4,num_kv_blocks=16,enable_prefix_caching=args.mode!='disabled')
    e=initialize();p=e.kv_cache_pool;weight_hash=weights(e.model)
    assert weight_hash=='0f521e5990a19f9a5fbf4865358eacc2ea23afff9b3bfe11b55fb6bfae408ff7'
    prompt=[i%5 for i in range(26)]
    target=dict(request_id='target',prompt_ids=prompt,max_new_tokens=4);original=copy.deepcopy(target)
    expected_tokens=[]
    with torch.inference_mode():
        for _ in range(4):
            token=reference(e.model,torch.tensor([prompt+expected_tokens]))[0,-1].argmax().item()
            expected_tokens.append(token)
            if token==4:break
    expected=[dict(request_id='target',output_ids=expected_tokens)]
    prime_trace=[]
    if args.mode in ['warm','miss']:
        seed=prompt[:24] if args.mode=='warm' else [(i+1)%5 for i in range(24)]
        e.add_request(dict(request_id='seed',prompt_ids=seed,max_new_tokens=1))
        while e.has_unfinished_requests():
            assert len(prime_trace)<32
            prime_trace.append(step_trace(e))
    assert not any(p.block_usage)
    state=(dict(p.block_hash),dict(p.block_to_hash),list(p.block_usage),list(p.block_last_used),p.lru_seq)
    retained={b:(p.k_cache[b].clone(),p.v_cache[b].clone()) for b in p.block_to_hash}
    def restore():
        p.block_hash.clear();p.block_hash.update(state[0])
        p.block_to_hash.clear();p.block_to_hash.update(state[1])
        p.block_usage[:]=state[2];p.block_last_used[:]=state[3];p.lru_seq=state[4]
    def generate():
        restore();e.add_request(target);results=[]
        while e.has_unfinished_requests():results.extend(e.step())
        return results
    def check():
        assert not e.has_unfinished_requests() and not any(p.block_usage)
        restore();e.add_request(target);trace=[];results=[]
        while e.has_unfinished_requests():
            assert len(trace)<32
            row=step_trace(e);trace.append(row);results.extend(row['result'])
        assert results==expected and target==original and not any(p.block_usage)
        assert weights(e.model)==weight_hash
        for b,(k,v) in retained.items():assert torch.equal(k,p.k_cache[b]) and torch.equal(v,p.v_cache[b])
        calls=[c for row in trace for c in row['calls']]
        hit=24 if args.mode=='warm' else 0
        actual=sum(sum(c['counts']) for c in calls)
        assert actual==len(prompt)-hit+len(expected_tokens)-1
        assert len(state[0])==(6 if args.mode in ['warm','miss'] else 0)
        return dict(trace=trace,hit_tokens=hit,actual_input_tokens=actual,padded_positions=sum(len(c['inputs'])*len(c['inputs'][0]) for c in calls),
                    model_calls=len(calls),steps=len(trace),new_tokens=len(expected_tokens),outputs=results,
                    initial_cached_blocks=len(state[0]),final_cached_blocks=len(p.block_hash),final_active_refs=list(p.block_usage))
    before=check();assert generate()==expected
    timing={}
    for name,fn in [('initialize',initialize),('request_with_metadata_reset',generate)]:
        t=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(min_run_time=0.5)
        timing[name]=dict(median_us=t.median*1e6,iqr_us=t.iqr*1e6,iqr_over_median=t.iqr/t.median,
                         high_variance=t.iqr/t.median>0.1,number_per_run=t.number_per_run,raw_times=t.raw_times)
    assert check()==before and sha(source)==digest
    report=dict(mode=args.mode,source_sha256=digest,script_sha256=sha(Path(__file__)),fixtures=fixtures,
                device='cpu',dtype='float32',threads=1,torch=torch.__version__,seed=0,parameter_hash=weight_hash,
                capacity=1,block_size=4,num_kv_blocks=16,token_budget=4,target=target,validation=before,timing=timing,
                prime_trace=prime_trace,scope='request_with_metadata_reset includes dictionary/list state restoration + add/schedule/padding/hash/KV/model/sample/results/release; excludes model init and seed-cache population; seed K/V not copied in timer; warm/miss are reset to identical initial cache state each iteration',
                notes='No eviction needed in these timing workloads. LRU correctness tested separately. Not GPU/TTFT/ITL or full cold-start service latency.')
    path=PROJECT/'benchmarks/results'/('step18_prefix_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.mode,{k:round(v['median_us'],3) for k,v in timing.items()},'input_tokens',before['actual_input_tokens'],'calls',before['model_calls'],path)
if __name__=='__main__':main()
