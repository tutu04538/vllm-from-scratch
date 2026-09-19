"""External Qwen3 directory: load, first full workload, warm full Engine. No official forward in timings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,copy,gc,hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
ROOT=Path(__file__).resolve().parents[1];RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path.insert(0,str(RECORDS/'tools'))
from step30_source import source_digest,source_files
from step30_test_helpers import m,reference

def hashes(d):return {str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file()}
def stats(x):
    med=statistics.median(x);q=statistics.quantiles(x,n=4,method='inclusive');ratio=(q[2]-q[0])/med
    return dict(median_us=med,iqr_over_median=ratio,high_variance=ratio>.1,samples_us=x)
def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter_ns();v=fn();torch.cuda.synchronize();return (time.perf_counter_ns()-start)/1000,v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['torch','eager','graph']);ap.add_argument('--directory',type=Path,default=ROOT/'fixtures/step30_qwen3/tiny_gqa');a=ap.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sha=source_digest();report=sorted(RECORDS.glob('step30_external_contract_*.json'))[-1];contract=json.loads(report.read_text());assert contract['source_sha256']==sha and contract['passed']==contract['total']
    before=hashes(a.directory)
    cpu=m.Engine.from_model_dir(a.directory,device='cpu');c=cpu.model.model_config();state=copy.deepcopy(cpu.model.state_dict());del cpu
    reqs=[dict(request_id='A',prompt_ids=[0,1,2,3,0,1,2],max_new_tokens=4),dict(request_id='B',prompt_ids=[0,1,2,3,2],max_new_tokens=2),dict(request_id='C',prompt_ids=[1],max_new_tokens=3)]
    expected={}
    for r in reqs:
        ids=[]
        for _ in range(r['max_new_tokens']):
            tok=int(reference(state,r['prompt_ids']+ids,c['num_q_heads'],c['num_kv_heads'],c['rms_norm_eps'],c['rope_theta'])[0][-1].argmax());ids.append(tok)
            if tok==4:break
        expected[r['request_id']]=ids
    options=dict(device='cuda',attention_backend='torch' if a.mode=='torch' else 'triton',use_cuda_graph=a.mode=='graph',max_num_seqs=3,max_num_batched_tokens=5,num_kv_blocks=24,block_size=4,enable_prefix_caching=False)
    with torch.inference_mode():
        def load():return m.Engine.from_model_dir(a.directory,**options)
        first_load,e=timed(load);assert not e.model.graphs
        def generate():
            results=[];steps=0
            for r in reqs:e.add_request(copy.deepcopy(r))
            while e.has_unfinished_requests():
                results.extend(e.step());steps+=1;assert steps<100
            return results,steps
        first_run,(out,steps)=timed(generate);assert {r['request_id']:r['output_ids'] for r in out}==expected
        p=e.kv_cache_pool;ptrs=[p.k_cache.data_ptr(),p.v_cache.data_ptr()];graphs=dict(e.model.graphs)
        loads=[]
        for _ in range(11):
            dt,other=timed(load);loads.append(dt);assert not other.model.graphs;del other;gc.collect()
        for _ in range(16):generate()
        samples=[]
        for _ in range(31):
            dt,_=timed(lambda:[generate() for _ in range(5)]);samples.append(dt/5)
        out2,steps2=generate();assert out2==out and steps2==steps
        assert not any(p.block_usage) and not p.block_hash and not p.block_to_hash
        assert ptrs==[p.k_cache.data_ptr(),p.v_cache.data_ptr()] and graphs==e.model.graphs
        assert all(torch.equal(v.cpu(),state[k]) for k,v in e.model.state_dict().items())
    assert before==hashes(a.directory) and sha==source_digest()
    data=dict(source_sha256=sha,source_files=source_files(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract_report=str(report),mode=a.mode,directory=str(a.directory),input_files=before,model=c,options=options,device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,dtype='float32',tf32=False,threads=1,requests=reqs,results=out,steps=steps,graph_keys=sorted(graphs),first_load_us=first_load,first_workload_us=first_run,load=stats(loads),steady=stats(samples),scope='Context initialized before timings; checkpoint was read by CPU verification beforehand, so OS page cache is warm. Triton disk cache retained. First GPU workload includes first library work and Graph capture. Eleven individual repeat loads include external config/name adaptation, model init, weights, runtime+KV, but exclude Graph capture and deletion/gc. Steady:16 warmups,31 groups of5 full workloads, prefix off; submission through release and synchronization included. Different weights/intermediate size than step29; not a cross-stage speedup experiment.')
    p=ROOT/'benchmarks/results'/f"step30_external_{a.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(p);print(a.mode,data['load']['median_us'],data['first_workload_us'],data['steady']['median_us'])
if __name__=='__main__':main()
