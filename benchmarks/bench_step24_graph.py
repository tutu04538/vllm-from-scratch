"""Same step24 GPU computation with graph off/on: cold capture and steady scopes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,copy,hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
PROJECT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(PROJECT),str(RECORDS/'tools')]
from step24_gpu_helpers import m,reference,observe_step


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_hash(state):return hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in state.values())).hexdigest()
def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter_ns();result=fn();torch.cuda.synchronize()
    return (time.perf_counter_ns()-t)/1000,result
def measure(fn):
    for _ in range(16):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(31):
        def group():
            for _ in range(5):fn()
        elapsed,_=timed(group);samples.append(elapsed/5)
    med=statistics.median(samples);q=statistics.quantiles(samples,n=4,method='inclusive')
    return dict(median_us=med,iqr_us=q[2]-q[0],iqr_over_median=(q[2]-q[0])/med,high_variance=(q[2]-q[0])/med>0.1,
                samples_us=samples,calls_per_sample=5,scope='Synchronized CPU wall clock, 31 groups of 5 calls averaged per call, 16 warmups. No measurement-only graph; graph=on uses the actual student graph path.')


def make_engine(graph,**kwargs):
    torch.manual_seed(0);cpu=m.TinyCausalLM(device='cpu');state=copy.deepcopy(cpu.state_dict())
    e=m.Engine(device='cuda',attention_backend='triton',use_cuda_graph=graph,max_num_seqs=3,max_num_batched_tokens=5,num_kv_blocks=24,**kwargs)
    e.model.load_state_dict(state);return e,state


def forward(args):
    e,state=make_engine(args.graph);model=e.model;pool=e.kv_cache_pool
    histories,chunks=([list(range(3)),[],[]],[[0],[1,0,3],[2]]) if args.workload=='mixed5' else ([[i%5 for i in range(31)]],[[2]])
    counts=list(map(len,chunks));n=sum(counts);caches=[]
    for i,h in enumerate(histories):
        table=list(range(i*8,(i+1)*8))[::-1];caches.append(m.CacheConfig(block_table=table,length=len(h)))
        for b in table:pool.block_usage[b]=1
        if h:
            _,kv=reference(state,h)
            for storage,values in zip((pool.k_cache,pool.v_cache),kv):
                for pos in range(len(h)):storage[table[pos//4],pos%4]=values[pos].float().cuda()
    ids=torch.tensor([t for chunk in chunks for t in chunk]);expected=torch.cat([reference(state,h+c)[0][len(h):] for h,c in zip(histories,chunks)])
    old=[len(h) for h in histories]
    model._prepare_inputs(ids,counts,caches,pool)
    # Pre-warm CUDA library and Triton compilation without creating any graph.
    for _ in range(5):model.gpu_forward(n,pool)
    torch.cuda.synchronize()
    cold_capture=None
    if args.graph:
        def capture_first():
            g=model._capture_graph(n,pool);g.replay();return model.graph_outputs[n]
        cold_capture,out=timed(capture_first)
    else:out=model.gpu_forward(n,pool)
    torch.testing.assert_close(out.cpu().double(),expected,atol=2e-5,rtol=2e-5)
    saved_pool=(pool.k_cache.clone(),pool.v_cache.clone())
    pointers=[x.data_ptr() for x in (model.input_buffer,model.position_buffer,model.slot_buffer,pool.k_cache,pool.v_cache,model.attention_metadata.gpu_buffer)]
    graphs=dict(model.graphs)
    def prepared():
        if args.graph:
            model.graphs[n].replay();return model.graph_outputs[n]
        return model.gpu_forward(n,pool)
    def full_forward():
        # Recompute exactly the same logical batch each iteration; reset included equally.
        for cache,length in zip(caches,old):cache.length=length
        return model._forward_append(ids,counts,caches,pool)
    timings=dict(prepared_gpu_forward=measure(prepared),forward_with_prepare_and_length_reset=measure(full_forward))
    for fn in [prepared,full_forward]:torch.testing.assert_close(fn().cpu().double(),expected,atol=2e-5,rtol=2e-5)
    assert [c.length for c in caches]==[l+c for l,c in zip(old,counts)] and model.graphs==graphs
    assert pointers==[x.data_ptr() for x in (model.input_buffer,model.position_buffer,model.slot_buffer,pool.k_cache,pool.v_cache,model.attention_metadata.gpu_buffer)]
    assert all(torch.equal(a,b) for a,b in zip(saved_pool,(pool.k_cache,pool.v_cache)))
    assert state_hash(model.state_dict())==state_hash(state)
    return dict(weight_hash=state_hash(state),histories=histories,chunks=chunks,counts=counts,timing=timings,
                cold_capture_and_first_replay_us=cold_capture,
                cold_scope='One student _capture_graph (3 side-stream warmups + capture) and first replay, synchronized; pure GPU computation/Triton already prewarmed. Excludes preparation and first JIT/library setup. One sample/process, not latency quantiles.',
                validation=dict(max_abs_error=(full_forward().cpu().double()-expected).abs().max().item(),graph_keys=sorted(model.graphs),fixed_addresses=True,pool_unchanged=True),
                scopes=dict(prepared_gpu_forward='CPU submit through GPU completion for embedding/QKV/KV write/attention/lm_head. No metadata/input preparation. Actual Engine graph or eager function, NOT a separate measurement capture.',
                            forward_with_prepare_and_length_reset='CPU cache length reset + actual _forward_append including all preparation/upload + GPU forward + final synchronization. Excludes scheduler/sample/Engine initialization/capture.'))


def engine(args):
    e,state=make_engine(args.graph);p=e.kv_cache_pool;warm=args.workload=='warm_mixed'
    def req(r,p,n):return dict(request_id=r,prompt_ids=p,max_new_tokens=n)
    requests=[req('A',[0,1,2],4),req('B',([0,1,2,3] if warm else [])+[1,0,3],1),req('C',[2],1)]
    originals=copy.deepcopy(requests)
    if warm:
        e.add_request(req('seed',[0,1,2,3],1))
        while e.has_unfinished_requests():observe_step(e,state)
    initial=(dict(p.block_hash),dict(p.block_to_hash),list(p.block_usage),list(p.block_last_used),p.lru_seq)
    retained={b:(p.k_cache[b].clone(),p.v_cache[b].clone()) for b in p.block_to_hash}
    def generate(observe=False):
        p.block_hash.clear();p.block_hash.update(initial[0]);p.block_to_hash.clear();p.block_to_hash.update(initial[1])
        p.block_usage[:]=initial[2];p.block_last_used[:]=initial[3];p.lru_seq=initial[4]
        results=[];trace=[]
        def step():
            if observe:
                row=observe_step(e,state);trace.append(row);results.extend(row['results'])
            else:results.extend(e.step())
        e.add_request(requests[0]);step()
        for r in requests[1:]:e.add_request(r)
        while e.has_unfinished_requests():step()
        return (results,trace) if observe else results
    keys_before=sorted(e.model.graphs)
    first_us,first=timed(generate)
    # All target N variants now exist in graph mode; observer does not create/capture graphs in timing.
    before=generate(True);assert before[0]==first
    graph_objects=dict(e.model.graphs)
    timings=dict(engine_with_metadata_reset=measure(generate))
    after=generate(True)
    assert before==after and requests==originals and not any(p.block_usage) and graph_objects==e.model.graphs
    for b,(k,v) in retained.items():assert torch.equal(k,p.k_cache[b]) and torch.equal(v,p.v_cache[b])
    assert state_hash(e.model.state_dict())==state_hash(state)
    return dict(weight_hash=state_hash(state),requests=requests,timing=timings,first_target_workload_us=first_us,
                first_target_scope='First target workload in this process; includes missing graph captures and any still-cold JIT/library setup, metadata reset/submission/generation. Warm seed is excluded and may have already created graph N=4. Not isolated capture cost.',
                graph_keys_before_target=keys_before,graph_keys_after_target=sorted(e.model.graphs),
                validation=dict(results=before[0],steps=len(before[1]),true_tokens=sum(sum(c['counts']) for row in before[1] for c in row['calls']),
                                calls=sum(len(row['calls']) for row in before[1]),hit_tokens=4 if warm else 0,trace=before[1]),
                scope='Steady reused Engine, prefix metadata reset + submission/scheduling/forward/sampling/hash/results/release + synchronization. Excludes initialization, warm seed, first compile/capture.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['forward','engine'],required=True)
    parser.add_argument('--mode',choices=['eager','graph'],required=True);parser.add_argument('--workload',required=True)
    args=parser.parse_args();args.graph=args.mode=='graph';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    source=Path(m.__file__);digest=sha(source);fixtures={}
    for kind in ['model','prefix','disabled_engine','slots','attention','gpu','metadata','graph']:
        path=sorted(RECORDS.glob('step24_'+kind+'_contract_*.json'))[-1];record=json.loads(path.read_text())
        assert record['status']=='passed' and record['source_sha256']==digest;fixtures[kind]=str(path)
    with torch.inference_mode():result=(forward if args.kind=='forward' else engine)(args)
    assert sha(source)==digest
    report=dict(kind=args.kind,mode=args.mode,workload=args.workload,source_sha256=digest,script_sha256=sha(Path(__file__)),
                helper_sha256=sha(RECORDS/'tools/step24_gpu_helpers.py'),fixtures=fixtures,device=torch.cuda.get_device_name(0),
                torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,dtype='float32',tf32=False,threads=1,**result)
    path=PROJECT/'benchmarks/results'/('step24_graph_'+args.kind+'_'+args.workload+'_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(args.kind,args.workload,args.mode,{k:round(v['median_us'],3) for k,v in result['timing'].items()},path,flush=True)


if __name__=='__main__':main()
