"""Test-only first-layer and decoder-layer tensor traces for 103 diagnosis."""
from pathlib import Path


def install(worker):
 import torch
 runner=worker.model_runner;model=runner.model
 model._step103_diag={};model._step103_diag_handles=[]
 def hook(name):
  def capture(module,args,output):
   rows=model._step103_diag.setdefault(name,[])
   if len(rows)>=4:return
   tensors=output if isinstance(output,tuple) else (output,)
   saved=[]
   for value in tensors:
    if isinstance(value,torch.Tensor):
     saved.append(value.detach()[-1:].cpu().clone())
    else:saved.append(None)
   rows.append(saved)
  return capture
 for i,layer in enumerate(model.model.layers):
  modules=[('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),
           ('q_norm',layer.self_attn.q_norm),('k_norm',layer.self_attn.k_norm),
           ('rope',layer.self_attn.rotary_emb),('attn',layer.self_attn.attn),
           ('o_proj',layer.self_attn.o_proj),('post_norm',layer.post_attention_layernorm),
           ('gate_up',layer.mlp.gate_up_proj),('silu',layer.mlp.act_fn),
           ('down_proj',layer.mlp.down_proj),('mlp',layer.mlp),('layer',layer)]
  for name,module in modules:model._step103_diag_handles.append(module.register_forward_hook(hook(f'{i}:{name}')))
 return len(model._step103_diag_handles)


def flush(worker,directory):
 import torch
 import torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0
 path=Path(directory)/f'rank{rank}.pt'
 torch.save(worker.model_runner.model._step103_diag,path)
 return {'rank':rank,'path':str(path)}
