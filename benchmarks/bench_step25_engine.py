"""Benchmark of the accepted packed Engine; obsolete model(input_ids) entry explicitly retired."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,copy,hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch,triton
ROOT=Path(__file__).resolve().parents[1]
RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(p) for p in ROOT.glob('step[0-9][0-9]') if p.is_dir()]+[str(ROOT),str(RECORDS/'tools')]
from step25_test_helpers import m,engine_for,state_for,reference

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_hash(state):return hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in state.values())).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['torch','eager','graph'],required=True);args=a.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    digest=sha(Path(m.__file__))
    report_path=sorted(RECORDS.glob('step25_contract_*.json'))[-1];report=json.loads(report_path.read_text())
    assert report['source_sha256']==digest
    assert report['status']=='passed' and all(c['passed'] for c in report['cases'])
    e=engine_for(4,2,args.mode,enable_prefix_caching=False);state=state_for(4,2);e.model.load_state_dict(state)
    reqs=[dict(request_id='A',prompt_ids=[0,1,2,3,0,1,2],max_new_tokens=4),dict(request_id='B',prompt_ids=[0,1,2,3,2],max_new_tokens=2),dict(request_id='C',prompt_ids=[1],max_new_tokens=3)]
    original=copy.deepcopy(reqs);expected={}
    for r in reqs:
        out=[]
        for _ in range(r['max_new_tokens']):
            token=reference(state,r['prompt_ids']+out,4,2)[0][-1].argmax().item();out.append(token)
            if token==4:break
        expected[r['request_id']]=out
    def generate():
        result=[];steps=0
        for r in reqs:e.add_request(r)
        while e.has_unfinished_requests():
            result.extend(e.step());steps+=1
            if steps>100:raise AssertionError('no progress')
        return result,steps
    with torch.inference_mode():
        result,steps=generate();assert {r['request_id']:r['output_ids'] for r in result}==expected
        p=e.kv_cache_pool
        assert not any(p.block_usage) and not p.block_hash
        ptrs=[p.k_cache.data_ptr(),p.v_cache.data_ptr()];graphs=dict(e.model.graphs)
        for _ in range(16):generate()
        torch.cuda.synchronize();samples=[]
        for _ in range(31):
            torch.cuda.synchronize();start=time.perf_counter_ns()
            for _ in range(5):generate()
            torch.cuda.synchronize();samples.append((time.perf_counter_ns()-start)/1000/5)
        result2,steps2=generate();assert result2==result and steps2==steps
        assert not any(p.block_usage) and not p.block_hash and p.block_to_hash=={}
        assert ptrs==[p.k_cache.data_ptr(),p.v_cache.data_ptr()] and graphs==e.model.graphs
        assert state_hash(state)==state_hash(e.model.state_dict()) and reqs==original
    med=statistics.median(samples);quart=statistics.quantiles(samples,n=4,method='inclusive');iqr=quart[2]-quart[0]
    assert digest==sha(Path(m.__file__))
    data=dict(source_sha256=digest,script_sha256=sha(Path(__file__)),helper_sha256=sha(RECORDS/'tools/step25_test_helpers.py'),contract_report=str(report_path),mode=args.mode,
        status='verified_engine_retired_legacy_entry',device=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,dtype='float32',tf32=False,threads=1,
        model=dict(d_model=32,num_q_heads=4,num_kv_heads=2,head_dim=8),weight_hash=state_hash(state),requests=reqs,results=result,steps=steps,graph_keys=sorted(graphs),
        kv_pool_shape=list(p.k_cache.shape),kv_bytes=2*p.k_cache.numel()*p.k_cache.element_size(),
        timing=dict(median_us=med,iqr_us=iqr,iqr_over_median=iqr/med,high_variance=iqr/med>0.1,samples_us=samples),
        scope='Reused Engine, prefix caching disabled, same weights/requests. Submission + scheduling + packed forward + sampling + release + results + synchronization. 16 warmups, 31 groups of five complete workloads. Excludes model/Engine init, first JIT/library setup and graph capture. The obsolete model(input_ids) entry is explicitly unsupported; only the accepted packed Engine is measured.')
    path=ROOT/'benchmarks/results'/('step25_engine_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json');path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(path);print(f'{args.mode}: {med:.3f} us; IQR/median={iqr/med:.3f}')
if __name__=='__main__':main()
