"""Single RMSNorm microbenchmark; eager host submission vs captured repeated device work.
Each report includes raw samples. Not a full Engine speedup estimate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
TOOLS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/tools');sys.path.insert(0,str(TOOLS))
from step33_source import source_digest
from step33_test_helpers import m

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;sha=source_digest();results=[]
 for dt in [torch.float32,torch.bfloat16]:
  for shape in [(3,1024),(48,128),(16,1024)]:
   torch.manual_seed(33);x=torch.randn(shape,device='cuda',dtype=dt);w=torch.linspace(.3,1.7,shape[-1],device='cuda',dtype=dt)
   for backend in ['torch','triton']:
    norm=m.RMSNorm(shape[-1],1e-6,backend).to(device='cuda',dtype=dt);norm.weight.data.copy_(w)
    with torch.inference_mode():
     expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).to(dt)
     actual=norm(x);torch.testing.assert_close(actual.float(),expected.float(),atol=.0005 if dt==torch.bfloat16 else 2e-5,rtol=.008 if dt==torch.bfloat16 else 2e-5)
     for _ in range(20):norm(x)
     eager=[];device=[]
     for _ in range(7):
      torch.cuda.synchronize();t=time.perf_counter()
      for _ in range(200):norm(x)
      torch.cuda.synchronize();eager.append((time.perf_counter()-t)*1e6/200)
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for _ in range(10):norm(x)
     torch.cuda.current_stream().wait_stream(stream)
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):
      for _ in range(32):out=norm(x)
     for _ in range(5):g.replay()
     torch.cuda.synchronize()
     for _ in range(7):
      start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
      for _ in range(50):g.replay()
      end.record();end.synchronize();device.append(start.elapsed_time(end)*1000/(50*32))
     results.append(dict(dtype=str(dt),shape=shape,backend=backend,eager_us=statistics.median(eager),captured_device_us=statistics.median(device),eager_samples_us=eager,device_samples_us=device))
     del g,out
 assert sha==source_digest()
 report=dict(source_sha256=sha,torch=torch.__version__,device=torch.cuda.get_device_name(),results=results,scope='One process,7 groups per case. Eager:200 calls timed by host wall clock with synchronize only at boundaries. Captured:32 calls per graph,50 replays/group using CUDA events, divided by1600. Includes launch/dependency effects of captured graph, not claimed pure instruction time; not Engine latency.')
 p=Path(__file__).resolve().parent/'results'/f"step33_norm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";p.write_text(json.dumps(report,indent=2)+'\n');print(p)
 for r in results:print(r['dtype'],r['shape'],r['backend'],round(r['eager_us'],3),round(r['captured_device_us'],3))
if __name__=='__main__':main()
