"""Capture request 0 at its first four sampled positions within a mixed batch."""
from pathlib import Path


def install(worker):
 import torch
 runner=worker.model_runner;model=runner.model;model._batch103={};model._batch103_sel=None
 original=runner.prepare_attn
 def prepare(batch):
  result=original(batch);model._batch103_sel=None
  for i,req in enumerate(batch.req_ids):
   if req=='0' or req.startswith('0-'):
    a,b=map(int,batch.query_start_loc_np[i:i+2]);positions=batch.positions[a:b].detach().cpu().tolist()
    for j,pos in enumerate(positions):
     if 127<=pos<=130:model._batch103_sel=(a+j,pos)
  return result
 runner.prepare_attn=prepare
 def hook(name):
  def capture(module,args,output):
   if model._batch103_sel is None:return
   row,pos=model._batch103_sel; tensors=output if isinstance(output,tuple) else (output,)
   saved=[]
   for x in tensors:
    if isinstance(x,torch.Tensor) and x.ndim>=2 and row<x.shape[0]:saved.append(x.detach()[row:row+1].cpu().clone())
    else:saved.append(None)
   model._batch103.setdefault(name,{})[pos]=saved
  return capture
 for i,layer in enumerate(model.model.layers):
  mods=[('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),('attn',layer.self_attn.attn),('o_proj',layer.self_attn.o_proj),('post_norm',layer.post_attention_layernorm),('gate_up',layer.mlp.gate_up_proj),('silu',layer.mlp.act_fn),('down_proj',layer.mlp.down_proj),('layer',layer)]
  for name,mod in mods:mod.register_forward_hook(hook(f'{i}:{name}'))
 return True


def flush(worker,directory):
 import torch
 import torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0
 path=Path(directory)/f'rank{rank}.pt';torch.save(worker.model_runner.model._batch103,path)
 return {'rank':rank,'path':str(path)}
