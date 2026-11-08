"""101 production multi-process Qwen3, four-layer per-step local tensor checks."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2];PROBE=ROOT/'tests/step101/step101_probe.py'


def run(tp,path,pause=False):
 cmd=[sys.executable,str(PROBE),'--model','tiny','--tp',str(tp),'--output',str(path)]
 if pause:cmd+=['--pause-layer','2']
 p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
 assert p.returncode==0,p.stdout+p.stderr
 return json.loads((path/'summary.json').read_text())


def test_real_production_tiny_layerwise_prefill_decode(tmp_path):
 assert torch.cuda.device_count()>=2
 one=tmp_path/'tp1';two=tmp_path/'tp2';paused=tmp_path/'tp2_paused';a=run(1,one);b=run(2,two);c=run(2,paused,pause=True)
 assert b['output_tokens']==c['output_tokens']
 for step in range(4):
  for rank in range(2):
   baseline=torch.load(two/f'step{step}_rank{rank}.pt',map_location='cpu',weights_only=True)
   delayed=torch.load(paused/f'step{step}_rank{rank}.pt',map_location='cpu',weights_only=True)
   assert baseline.keys()==delayed.keys()
   for key in baseline:assert torch.equal(baseline[key],delayed[key]),(step,rank,key)
 assert a['output_tokens']==b['output_tokens'] and len(a['output_tokens'])==4
 for step in range(4):
  ref=torch.load(one/f'step{step}_rank0.pt',map_location='cpu',weights_only=True)
  for rank in range(2):
   got=torch.load(two/f'step{step}_rank{rank}.pt',map_location='cpu',weights_only=True)
   assert torch.equal(got['input_ids'],ref['input_ids']) and torch.equal(got['positions'],ref['positions'])
   for layer in range(4):
    prefix=f'layer{layer}_'
    for name in ('input_norm','post_norm','attention','mlp','layer','layer_residual','post_norm_residual'):
     if prefix+name in ref:
      torch.testing.assert_close(got[prefix+name],ref[prefix+name],atol=1e-5,rtol=1e-5)
    q,k,v=ref[prefix+'qkv'].split([32,16,16],dim=-1)
    expected=torch.cat((q[...,rank*16:(rank+1)*16],k[...,rank*8:(rank+1)*8],v[...,rank*8:(rank+1)*8]),dim=-1)
    torch.testing.assert_close(got[prefix+'qkv'],expected,atol=1e-5,rtol=1e-5)
    expected=ref[prefix+'local_attn'][...,rank*16:(rank+1)*16]
    torch.testing.assert_close(got[prefix+'local_attn'],expected,atol=1e-5,rtol=1e-5)
   for name in ('hidden','logits'):
    torch.testing.assert_close(got[name],ref[name],atol=1e-5,rtol=1e-5)


def test_real_model_architecture_guard():
 from types import SimpleNamespace as N
 from minivllm.config import ModelConfig,CUDAGraphMode
 from minivllm.testing.tiny_models import tiny_qwen3_config
 from minivllm.worker.worker import Worker
 def make(**changes):
  hf=tiny_qwen3_config('tiny_gqa');hf.update(changes.pop('hf',{}))
  config=N(parallel_config=N(world_size=2,tensor_parallel_size=2,pipeline_parallel_size=changes.pop('pp',1),data_parallel_size=1),
    model_config=ModelConfig('test',hf_config=hf,max_model_len=64),use_v2_model_runner=True,
    speculative_config=changes.pop('spec',None),compilation_config=N(cudagraph_mode=changes.pop('graph',CUDAGraphMode.NONE)),
    device_config=N(device='cuda'),attention_backend=changes.pop('backend','triton'),
    cache_config=N(enable_prefix_caching=changes.pop('prefix',False)))
  return Worker(config,rank=0,device_index=0)
 make()._validate_tp_model()
 make(prefix=True)._validate_tp_model()  # 102 验证并开放 Qwen3 TP 前缀缓存
 for worker in (make(pp=2),make(spec=object()),make(graph=CUDAGraphMode.FULL),make(hf={'model_type':'qwen3_moe'}),make(hf={'sliding_window':128}),make(hf={'quantization_config':{'quant_method':'fp8'}}),make(hf={'intermediate_size':47}),make(backend='torch')):
  with pytest.raises((ValueError,NotImplementedError)):worker._validate_tp_model()


def test_frozen_vllm_ieee_all_layers_and_decode(tmp_path):
 assert torch.cuda.device_count()>=2
 mini=tmp_path/'mini';frozen=tmp_path/'frozen'
 cmd=[sys.executable,str(PROBE),'--model','tiny','--head-dim','32','--tp','2','--output',str(mini)]
 env={**os.environ,'TRITON_F32_DEFAULT':'ieee','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 assert p.returncode==0,p.stdout+p.stderr
 cmd=[sys.executable,str(ROOT/'benchmarks/check_step101_frozen_tiny.py'),'--model-dir',str(mini/'tiny'),'--tp','2','--backend','TRITON_ATTN','--output',str(frozen)]
 env.update(VLLM_USE_V2_MODEL_RUNNER='1',VLLM_ALLOW_INSECURE_SERIALIZATION='1')
 p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
 assert p.returncode==0,p.stdout+p.stderr
 assert json.loads((mini/'summary.json').read_text())['output_tokens']==json.loads((frozen/'summary.json').read_text())['output_tokens']
 worst=(0.0,None)
 for step in range(4):
  for rank in range(2):
   got=torch.load(mini/f'step{step}_rank{rank}.pt',map_location='cpu',weights_only=True)
   ref=torch.load(frozen/f'step{step}_rank{rank}.pt',map_location='cpu',weights_only=True)
   names=['hidden','logits']
   for layer in range(4):
    pre=f'layer{layer}_'
    names.extend(pre+n for n in ('input_norm','qkv','qkv_input','local_attn','local_attn_arg0','local_attn_arg1','local_attn_arg2','attention','post_norm','post_norm_residual','mlp','layer','layer_residual'))
   for name in names:
    assert name in got and name in ref,(step,rank,name)
    x,y=got[name],ref[name]
    assert x.shape==y.shape,(step,rank,name,x.shape,y.shape)
    torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5,msg=lambda msg:f'{step=} {rank=} {name=} {msg}')
    delta=float((x.float()-y.float()).abs().max())
    if delta>worst[0]:worst=(delta,(step,rank,name))
 print('frozen_ieee_worst',worst)
