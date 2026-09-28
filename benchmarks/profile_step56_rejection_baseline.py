"""定向基线：单次 categorical kernel 与已构造 p/q 后的 verify_batch，不是模型吞吐。"""
from pathlib import Path
from types import SimpleNamespace
from statistics import median
import json
import sys
import time
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step56 import rejection_triton as rt
from step56.rejection import BatchedRejectionSampler, RejectionItem
from step56.rejection_rng import PHILOX_ROUNDS
from step56.sampling import SamplingParams, TorchSampler


def measure_gpu(fn, iterations=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    runs=[]
    for _ in range(3):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations): fn()
        end.record();end.synchronize()
        runs.append(start.elapsed_time(end)/iterations)
    return runs


def measure_wall(fn, iterations=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    runs=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(iterations): fn()
        torch.cuda.synchronize()
        runs.append((time.perf_counter()-start)*1000/iterations)
    return runs


def main():
    torch.set_num_threads(1)
    records=[]
    for vocab in (257,151936):
        for size in (1,8,16):
            weights=torch.full((size,vocab),1/vocab,device='cuda')
            lo=torch.full((size,),11,dtype=torch.int64,device='cuda')
            hi=torch.zeros_like(lo);counter=torch.zeros_like(lo)
            tokens=torch.empty(size,dtype=torch.int64,device='cuda')
            errors=torch.empty(size,dtype=torch.int32,device='cuda')
            def kernel():
                rt.sample_token_kernel[(size,)](weights,lo,hi,counter,tokens,errors,
                                                vocab,BLOCK_V=rt.BLOCK_V,ROUNDS=PHILOX_ROUNDS)
            params=SamplingParams(temperature=0.8,seed=7)
            backend=BatchedRejectionSampler('triton',set(),TorchSampler(),device='cuda')
            batch=[]
            for i in range(size):
                seq=SimpleNamespace(rejection_seed=11+i,rejection_rng_counter=0,
                                    sampling_params=params,request_id=str(i))
                batch.append(RejectionItem(plan={'request':seq},mode='distribution',draft_ids=[0,1],
                                            remaining_outputs=3,row_probs=[weights[i]]*3,
                                            draft_probs=[weights[i]]*2))
            gpu=measure_gpu(kernel)
            wall=measure_wall(lambda:backend.verify_batch(batch))
            record=dict(batch=size,vocab=vocab,tile_size=rt.BLOCK_V,
                        tiles=(vocab+rt.BLOCK_V-1)//rt.BLOCK_V,
                        categorical_gpu_ms=median(gpu),categorical_repetitions_ms=gpu,
                        verify_wall_ms=median(wall),verify_repetitions_ms=wall)
            records.append(record)
            print(json.dumps(record),flush=True)
    # 单次完整 verify 的 CPU profiler：区分 H2D 元数据拷贝与 GPU 标量读取。
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        backend.verify_batch(batch)
        torch.cuda.synchronize()
    events={e.key:e.count for e in prof.key_averages()}
    picked={k:v for k,v in events.items() if 'cudaMemcpy' in k or 'Synchronize' in k
            or k in ('aten::_local_scalar_dense','aten::item','aten::_to_copy')}
    uploads=[]
    original_tensor=torch.tensor
    def observe_tensor(data,*args,**kwargs):
        result=original_tensor(data,*args,**kwargs)
        if result.is_cuda and (not isinstance(data,torch.Tensor) or data.device.type=='cpu'):
            uploads.append(dict(shape=list(result.shape),dtype=str(result.dtype),
                                bytes=result.numel()*result.element_size()))
        return result
    with patch.object(torch,'tensor',observe_tensor):
        backend.verify_batch(batch)
    output=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,records=records,
                last_batch_verify_cpu_events=picked,
                last_batch_host_metadata_cuda_tensor_creations=uploads,
                note='prebuilt FP32 p/q; K=2, p=q; verify excludes prepare/materialize/commit; no model')
    path=Path('/tmp/step56_final_review/rejection_baseline.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    print('CPU_EVENTS',json.dumps(picked),flush=True)
    print('HOST_METADATA_CUDA_TENSORS',len(uploads),'nonempty=',sum(x['bytes']>0 for x in uploads),
          'bytes=',sum(x['bytes'] for x in uploads),flush=True)


if __name__=='__main__':
    main()
