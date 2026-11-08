"""Capture full prefill activations in the first two decoder layers."""
from pathlib import Path


def install(worker):
 import torch
 runner=worker.model_runner;model=runner.model;model._prefill103={};model._prefill103_on=False
 original=runner.prepare_attn
 def prepare(batch):
  result=original(batch);model._prefill103_on=bool(batch.req_ids and (batch.req_ids[0]=='0' or batch.req_ids[0].startswith('0-')) and batch.num_tokens==128 and int(batch.positions[0])==0)
  return result
 runner.prepare_attn=prepare
 def hook(name):
  def capture(module,args,output):
   if not model._prefill103_on:return
   vals=output if isinstance(output,tuple) else (output,)
   model._prefill103[name]=[x.detach().cpu().clone() if isinstance(x,torch.Tensor) else None for x in vals]
  return capture
 for i,layer in enumerate(model.model.layers[:2]):
  mods=[('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),('q_norm',layer.self_attn.q_norm),('k_norm',layer.self_attn.k_norm),('attn',layer.self_attn.attn),('o_proj',layer.self_attn.o_proj),('post_norm',layer.post_attention_layernorm),('gate_up',layer.mlp.gate_up_proj),('silu',layer.mlp.act_fn),('down_proj',layer.mlp.down_proj),('layer',layer)]
  for name,mod in mods:mod.register_forward_hook(hook(f'{i}:{name}'))
 return True


def flush(worker,directory):
 import torch
 import torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0;path=Path(directory)/f'rank{rank}.pt'
 torch.save(worker.model_runner.model._prefill103,path);return {'rank':rank,'path':str(path)}
