"""Diagnostic timings for valid FP32 directories; malformed-config acceptance reported separately."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,copy,gc,hashlib,json,statistics,sys,tempfile,time
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
ROOT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path.insert(0,str(RECORDS/'tools'))
from step29_test_helpers import m,engine_for,state_for,reference

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(samples):
    med=statistics.median(samples);q=statistics.quantiles(samples,n=4,method='inclusive');iqr=q[2]-q[0]
    return dict(median_us=med,iqr_over_median=iqr/med,high_variance=iqr/med>0.1,samples_us=samples)
def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter_ns();v=fn();torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)/1000,v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['torch','eager','graph'],required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sha=digest(Path(m.__file__));core=sorted(RECORDS.glob('step29_contract_*.json'))[-1]
    cr=json.loads(core.read_text());assert cr['source_sha256']==sha and cr['status']=='passed'
    torch.manual_seed(17);seed_engine=engine_for(4,2,'cpu',enable_prefix_caching=False,head_dim=16,use_qk_norm=True);state=copy.deepcopy(seed_engine.model.state_dict());del seed_engine
    for k,v in state.items():
        if k.endswith('q_norm.weight'):v.copy_(torch.linspace(.6,1.4,16))
        elif k.endswith('k_norm.weight'):v.copy_(torch.linspace(1.3,.7,16))
    reqs=[dict(request_id='A',prompt_ids=[0,1,2,3,0,1,2],max_new_tokens=4),dict(request_id='B',prompt_ids=[0,1,2,3,2],max_new_tokens=2),dict(request_id='C',prompt_ids=[1],max_new_tokens=3)]
    expected={}
    for r in reqs:
        ids=[]
        for _ in range(r['max_new_tokens']):
            tok=reference(state,r['prompt_ids']+ids,4,2)[0][-1].argmax().item();ids.append(tok)
            if tok==4:break
        expected[r['request_id']]=ids
    options=dict(device='cuda',attention_backend='torch' if args.mode=='torch' else 'triton',use_cuda_graph=args.mode=='graph',max_num_seqs=3,max_num_batched_tokens=5,num_kv_blocks=24,block_size=4,enable_prefix_caching=False)
    with tempfile.TemporaryDirectory(prefix='step29_bench_') as tmp,torch.inference_mode():
        directory=Path(tmp)/'model';src=engine_for(4,2,'cpu',enable_prefix_caching=False,head_dim=16,use_qk_norm=True);src.model.load_state_dict(state);m.save_model(src.model,directory);del src
        def load():return m.Engine.from_model_dir(directory,**options)
        first_load,e=timed(load);assert not e.model.graphs
        def generate():
            results=[];steps=0
            for r in reqs:e.add_request(copy.deepcopy(r))
            while e.has_unfinished_requests():
                results.extend(e.step());steps+=1
                assert steps<100
            return results,steps
        first_run,(out,steps)=timed(generate)
        assert {r['request_id']:r['output_ids'] for r in out}==expected
        graphs=dict(e.model.graphs);p=e.kv_cache_pool;ptrs=[p.k_cache.data_ptr(),p.v_cache.data_ptr()]
        save_samples=[];load_samples=[]
        for _ in range(11):
            dt,_=timed(lambda:m.save_model(e.model,directory));save_samples.append(dt)
            dt,other=timed(load);load_samples.append(dt);assert not other.model.graphs;del other;gc.collect()
        for _ in range(16):generate()
        steady=[]
        for _ in range(31):
            dt,_=timed(lambda:[generate() for _ in range(5)]);steady.append(dt/5)
        out2,steps2=generate();assert out2==out and steps2==steps
        assert graphs==e.model.graphs and ptrs==[p.k_cache.data_ptr(),p.v_cache.data_ptr()]
        assert not any(p.block_usage) and not p.block_hash and not p.block_to_hash
        assert all(torch.equal(v.cpu(),state[k]) for k,v in e.model.state_dict().items())
        data=dict(source_sha256=sha,script_sha256=digest(Path(__file__)),core_report=str(core),status='valid_directory_diagnostic_not_full_stage_acceptance',mode=args.mode,device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,dtype='float32',tf32=False,threads=1,model=e.model.model_config(),options=options,requests=reqs,results=out,steps=steps,graph_keys=sorted(graphs),bytes_on_disk=sum(f.stat().st_size for f in directory.iterdir()),first_load_us=first_load,first_workload_us=first_run,save=stats(save_samples),load=stats(load_samples),steady=stats(steady),scope='CUDA context initialized before timings. Fresh temporary local directory and OS page cache left warm; no fsync, no forced cache eviction. First workload includes first forward/library setup and per-N graph capture; existing on-disk Triton cache retained. Save includes GPU to CPU copy plus JSON/safetensors writes. Load includes config read/model init/weights/Engine+KV buffers, excludes graph capture. Save/load: 11 individual samples, deletion+gc excluded. Steady: prefix off, reused loaded Engine, 16 warmups then 31 groups of 5 complete workloads; submission, scheduling, forward, sampling, release and synchronization included. Independent head_dim=16 and non-unit Q/K norm; not the same model as stage28; request deepcopy inside generation is benchmark overhead. Invalid config validation is NOT certified by these timings.')
    assert sha==digest(Path(m.__file__))
    path=ROOT/'benchmarks/results'/f"step29_io_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(path);print(args.mode,{k:round(data[k]['median_us'],3) for k in ['save','load','steady']},'first_workload_us',round(first_run,3))
if __name__=='__main__':main()
