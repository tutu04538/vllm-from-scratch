"""Independent 103 checks; the real test creates its own reference in a temp dir."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]


def command(args):
 proc=subprocess.run([sys.executable,*args],cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                     capture_output=True,text=True,timeout=600)
 assert proc.returncode==0,proc.stdout+proc.stderr


def test_workload_is_stable_and_batch_requests_distinct():
 from benchmarks.step103_workload import prompts
 for model in ('8b','14b'):
  x=prompts(model,4)
  assert list(map(len,x))==[128,120,112,104]
  assert len({tuple(item[:16]) for item in x})==4
  assert prompts(model,1)[0]==x[0]


def test_reference_kv_layout_reconstruction():
 # Frozen Triton cache packs [block, head, position, K/V * head_dim].
 # After selecting token locations, convert to mini's [token, K/V, head, dim].
 heads,dim=3,128
 expected=torch.arange(2*heads*dim).reshape(2,heads,dim)
 packed=expected.permute(1,0,2).reshape(heads,2*dim)
 restored=packed.reshape(heads,2,dim).permute(1,0,2)
 assert torch.equal(restored,expected)


def test_real_8b_tp1_32_token_generation_without_prior_103_benchmark(tmp_path):
 assert torch.cuda.device_count()>=1
 ref=tmp_path/'reference';mini=tmp_path/'mini'
 common=['--model','8b','--tp','1','--batch','1']
 # Parent pytest may retain a CUDA context from earlier tests. Fixed KV capacity
 # stays at 192 blocks on both sides; this lowers only the startup reservation.
 command(['benchmarks/step103_frozen_probe.py',*common,'--gpu-memory-utilization','0.78','--output',str(ref)])
 command(['benchmarks/step103_mini_probe.py',*common,'--output',str(mini)])
 a=json.loads((mini/'summary.json').read_text());b=json.loads((ref/'summary.json').read_text())
 assert a['kv_num_blocks_requested']==b['kv_num_blocks_requested']==192
 assert b['gpu_memory_utilization']==0.78
 assert a['prompts']==b['prompts'] and len(a['prompts'][0])==128
 assert len(a['runs'][0]['outputs'][0]['tokens'])==32
 assert a['runs'][0]['outputs'][0]['tokens']==b['runs'][0]['outputs'][0]['tokens']
 actual=torch.load(mini/'run0_rank0.pt',weights_only=True);expected=torch.load(ref/'run0_rank0.pt',weights_only=True)
 assert len(actual)==len(expected)==4
 assert [r['position'] for r in actual]==[127,128,129,130]
 assert [r['position'] for r in expected]==[127,128,129,130]
 contract_path=ROOT/'benchmarks/results/step090/numerics_contract.json'
 assert hashlib.sha256(contract_path.read_bytes()).hexdigest()=='29bcbf64f1212cf100845387ff46ddcd43981b3d9fd4487c044a455ec28d0e7c'
 limits=json.loads(contract_path.read_text())['models']['8b']['thresholds']
 for key,threshold in [('kv','kv'),('logits','raw_logits')]:
  x=actual[0][key].float();y=expected[0][key].float();assert x.shape==y.shape
  delta=x-y;rel=float(delta.norm()/y.norm())
  assert float(delta.abs().max())<=limits[threshold]['max_abs']
  assert rel<=limits[threshold]['relative_l2']


def test_bf16_rms_norm_matches_frozen_cuda_and_replays_graph():
 import vllm._custom_ops  # 注册上游算子，仅用于差分测试
 from minivllm.layers.ops.layernorm import rms_norm
 assert torch.cuda.is_available()
 generator=torch.Generator(device='cuda').manual_seed(103)
 for dim in (24,128,4096,5120):
  base=torch.randn((4,2*dim),dtype=torch.bfloat16,device='cuda',generator=generator)
  x=base[:,:dim]  # 带外层 stride 的输入
  residual=torch.randn((4,dim),dtype=x.dtype,device=x.device,generator=generator)
  weight=torch.randn((dim,),dtype=x.dtype,device=x.device,generator=generator)
  expected=torch.empty(x.shape,dtype=x.dtype,device=x.device)
  torch.ops._C.rms_norm(expected,x,weight,1e-6)
  assert torch.equal(rms_norm(x,weight,1e-6),expected)
  up=x.contiguous();up_res=residual.clone()
  torch.ops._C.fused_add_rms_norm(up,up_res,weight,1e-6)
  actual,actual_res=rms_norm(x,weight,1e-6,residual)
  assert torch.equal(actual,up) and torch.equal(actual_res,up_res)
 # 正常先热身，再捕获；graph replay 检查数据变化后仍调用本地 CUDA kernel。
 x=torch.randn((2,128),dtype=torch.bfloat16,device='cuda',generator=generator)
 r=torch.randn_like(x);w=torch.randn((128,),dtype=x.dtype,device=x.device,generator=generator)
 for _ in range(2):rms_norm(x,w,1e-6,r)
 torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out,out_r=rms_norm(x,w,1e-6,r)
 x.mul_(2);graph.replay();torch.cuda.synchronize()
 up=x.clone();up_r=r.clone();torch.ops._C.fused_add_rms_norm(up,up_r,w,1e-6)
 assert torch.equal(out,up) and torch.equal(out_r,up_r)


def test_bf16_cuda_extension_build_is_serialized(tmp_path):
 import concurrent.futures
 assert torch.cuda.is_available()
 # 两个新进程共享空缓存目录，同时首次使用；不能互相覆盖半成品 .so。
 code=('import torch; from minivllm.layers.ops.layernorm import rms_norm; '
       'x=torch.ones((1,128),device="cuda",dtype=torch.bfloat16); '
       'w=torch.ones(128,device="cuda",dtype=torch.bfloat16); '
       'y=rms_norm(x,w,1e-6); torch.cuda.synchronize(); assert torch.isfinite(y).all()')
 env={**os.environ,'TMPDIR':str(tmp_path),'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 def invoke():
  return subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(lambda _:invoke(),range(2)))
 for result in results:assert result.returncode==0,result.stdout+result.stderr
 assert len(list(tmp_path.rglob('*.so')))==1
