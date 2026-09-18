"""Three timing boundaries for the same GPU/weights; CPU records are not baselines."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,copy,hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
PROJECT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(PROJECT),str(RECORDS/'tools')]
from step22_gpu_helpers import m,attention_fixture,observe_step


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(xs,scope):
    ys=sorted(xs);q=statistics.quantiles(ys,n=4,method='inclusive');med=statistics.median(ys)
    return dict(median_us=med,iqr_us=q[2]-q[0],iqr_over_median=(q[2]-q[0])/med,
                high_variance=(q[2]-q[0])/med>0.1,samples_us=xs,scope=scope)


def wall_samples(fn,n=31):
    batch=5
    for _ in range(16):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(n):
        torch.cuda.synchronize();start=time.perf_counter_ns()
        for _ in range(batch):fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000/batch)
    result=stats(samples,'Synchronized CPU wall clock, 31 groups of 5 calls, per-call mean within each group; 16 warmups. Includes preparation inside fn, final synchronize amortized over 5 calls; not an individual-request latency distribution.')
    result['calls_per_sample']=batch
    return result


def graph_samples(fn):
    # Batch calls INSIDE one captured graph to amortize the host launch gap at Events.
    for _ in range(16):fn()
    torch.cuda.synchronize()
    pilot=torch.cuda.CUDAGraph()
    with torch.cuda.graph(pilot):initial=fn()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();pilot.replay();b.record();b.synchronize()
    estimate=max(a.elapsed_time(b)*1000,1.)
    batch=max(2,min(256,int(5000/estimate)))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):captured=fn()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(31):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)*1000/batch)
    result=stats(samples,'Pre-resident inputs, calls batched INSIDE measurement-only CUDA Graph; CUDA Events around replay divided by captured-call count, reducing launch-gap/timer overhead. 31 groups after 16 warmups. Excludes metadata creation/H2D/compilation/capture; residual graph scheduling overhead remains. Same method for both backends, NOT production Engine latency.')
    result['calls_per_sample']=batch
    return result,captured


def attention(args):
    settings={'small':dict(),
              'decode16':dict(size=16,dim=32,lengths=(128,)*16,counts=(1,)*16),
              'mixed8':dict(size=16,dim=32,lengths=(128,65,33,17,128,65,33,17),counts=(1,5,3,1,1,5,3,1))}
    f=attention_fixture(**settings[args.workload]);q,k,v,bt,lens,owners,pos=f['args']
    model=m.TinyCausalLM(d_model=f['dim'],device='cuda',attention_backend=args.backend).eval()
    pool=m.KVCachePool(f['size'],k.shape[0],f['dim'],torch.device('cuda'))
    pool.k_cache.copy_(k);pool.v_cache.copy_(v)
    caches=[m.CacheConfig(block_table=t.copy(),length=n) for t,n in zip(f['tables'],f['lengths'])]
    offsets=[0]
    for n in f['counts']:offsets.append(offsets[-1]+n)
    def preloaded():
        if args.backend=='triton':return m.paged_attention(*f['args'])
        return torch.cat([model.block_attention(q[a:b],c,pool,pos[a:b]) for c,a,b in zip(caches,offsets[:-1],offsets[1:])])
    def with_metadata():
        if args.backend=='triton':return model._triton_attention(q,caches,f['counts'],pool)
        return torch.cat([model.block_attention(q[a:b],c,pool,torch.arange(c.length-(b-a),c.length,device='cuda'))
                          for c,a,b in zip(caches,offsets[:-1],offsets[1:])])
    for fn in [preloaded,with_metadata]:torch.testing.assert_close(fn().cpu().double(),f['want'],atol=2e-5,rtol=2e-5)
    input_hash=hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in f['args'])).hexdigest()
    resident,captured=graph_samples(preloaded)
    torch.testing.assert_close(captured.cpu().double(),f['want'],atol=2e-5,rtol=2e-5)
    timings=dict(preloaded_gpu_graph=resident,metadata_and_attention=wall_samples(with_metadata))
    for fn in [preloaded,with_metadata]:torch.testing.assert_close(fn().cpu().double(),f['want'],atol=2e-5,rtol=2e-5)
    assert input_hash==hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in f['args'])).hexdigest()
    torch.testing.assert_close(pool.k_cache,k,atol=0,rtol=0,equal_nan=True)
    torch.testing.assert_close(pool.v_cache,v,atol=0,rtol=0,equal_nan=True)
    return dict(input_hash=input_hash,shape=dict(lengths=f['lengths'],counts=f['counts'],dim=f['dim'],block_size=f['size']),timing=timings)


def engine(args):
    warm=args.workload=='warm_mixed'
    torch.manual_seed(0)
    base=m.TinyCausalLM(device='cpu');state=copy.deepcopy(base.state_dict())
    e=m.Engine(device='cuda',attention_backend=args.backend,max_num_seqs=3,max_num_batched_tokens=5,num_kv_blocks=24)
    e.model.load_state_dict(state);p=e.kv_cache_pool
    def req(r,p,n):return dict(request_id=r,prompt_ids=p,max_new_tokens=n)
    requests=[req('A',[0,1,2],4),req('B',([0,1,2,3] if warm else [])+[1,0,3],1),req('C',[2],1)]
    original=copy.deepcopy(requests)
    if warm:
        e.add_request(req('seed',[0,1,2,3],1))
        while e.has_unfinished_requests():observe_step(e,state)
    initial=(dict(p.block_hash),dict(p.block_to_hash),list(p.block_usage),list(p.block_last_used),p.lru_seq)
    retained={b:(p.k_cache[b].clone(),p.v_cache[b].clone()) for b in p.block_to_hash}
    def restore():
        p.block_hash.clear();p.block_hash.update(initial[0]);p.block_to_hash.clear();p.block_to_hash.update(initial[1])
        p.block_usage[:]=initial[2];p.block_last_used[:]=initial[3];p.lru_seq=initial[4]
    def generate(observe=False):
        restore();results=[];trace=[]
        def step():
            if observe:
                r=observe_step(e,state);trace.append(r);results.extend(r['results'])
            else:results.extend(e.step())
        e.add_request(requests[0]);step()
        for r in requests[1:]:e.add_request(r)
        while e.has_unfinished_requests():step()
        return (results,trace) if observe else results
    before=generate(True)
    for _ in range(2):assert generate()==before[0]
    timing=dict(engine_with_metadata_reset=wall_samples(generate))
    after=generate(True)
    assert after==before and requests==original and not any(p.block_usage)
    for b,(k,v) in retained.items():assert torch.equal(k,p.k_cache[b]) and torch.equal(v,p.v_cache[b])
    assert all(torch.equal(value.cpu(),state[n]) for n,value in e.model.state_dict().items())
    return dict(weight_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in state.values())).hexdigest(),requests=requests,
                validation=dict(results=before[0],steps=len(before[1]),true_tokens=sum(sum(c['counts']) for r in before[1] for c in r['calls']),
                                calls=sum(len(r['calls']) for r in before[1]),hit_tokens=4 if warm else 0,trace=before[1]),timing=timing,
                scope='Reused Engine, prefix metadata reset + submission/scheduling/forward/sampling/hash/results/release + final CUDA synchronize. Engine/model initialization and warm seed excluded; no CUDA Graph.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['attention','engine'],required=True)
    parser.add_argument('--backend',choices=['torch','triton'],required=True);parser.add_argument('--workload',required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    source=Path(m.__file__);digest=sha(source)
    fixtures={}
    for kind in ['gpu','model','prefix','disabled_engine','slots','attention']:
        path=sorted(RECORDS.glob('step22_'+kind+'_contract_*.json'))[-1];d=json.loads(path.read_text())
        assert d['status']=='passed' and d['source_sha256']==digest;fixtures[kind]=str(path)
    with torch.inference_mode():result=(attention if args.kind=='attention' else engine)(args)
    assert sha(source)==digest
    report=dict(measurement_revision='batched_v2',kind=args.kind,backend=args.backend,workload=args.workload,source_sha256=digest,script_sha256=sha(Path(__file__)),
                helper_sha256=sha(RECORDS/'tools/step22_gpu_helpers.py'),fixtures=fixtures,device=torch.cuda.get_device_name(0),
                torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,dtype='float32',threads=1,tf32=False,**result)
    path=PROJECT/'benchmarks/results'/('step22_gpu_'+args.kind+'_'+args.workload+'_'+args.backend+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.kind,args.workload,args.backend,{k:round(v['median_us'],3) for k,v in result['timing'].items()},path,flush=True)


if __name__=='__main__':main()
