"""Frozen vLLM tiny model hooks, test-only."""
from pathlib import Path

def install(model):
 import torch
 model._step101_data={};model._step101_steps=[];model._step101_handles=[]
 def save(name,with_input=False):
  def hook(module,args,out):
   value=out[0] if isinstance(out,tuple) else out
   if isinstance(value,torch.Tensor):model._step101_data[name]=value.detach().cpu().clone()
   if with_input and args and isinstance(args[0],torch.Tensor):model._step101_data[name+'_input']=args[0].detach().cpu().clone()
   if name.endswith('local_attn') and len(args)>=3:
    for i in range(3):model._step101_data[name+f'_arg{i}']=args[i].detach().cpu().clone()
   if isinstance(out,tuple) and len(out)>1 and isinstance(out[1],torch.Tensor):model._step101_data[name+'_residual']=out[1].detach().cpu().clone()
  return hook
 def save_logits(module,args,out):
  save('logits')(module,args,out);model._step101_steps.append(model._step101_data);model._step101_data={}
 model._step101_handles.append(model.register_forward_hook(save('hidden')))
 model._step101_handles.append(model.logits_processor.register_forward_hook(save_logits))
 for idx,layer in enumerate(model.model.layers):
  prefix=f'layer{idx}_'
  for name,module in (('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),('local_attn',layer.self_attn.attn),('attention',layer.self_attn),('post_norm',layer.post_attention_layernorm),('mlp',layer.mlp),('layer',layer)):
   model._step101_handles.append(module.register_forward_hook(save(prefix+name,True)))
 return len(model._step101_handles)

def flush(model,directory):
 import torch,torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0
 result=[]
 for i,record in enumerate(model._step101_steps):
  path=Path(directory)/f'step{i}_rank{rank}.pt';torch.save(record,path);result.append({'path':str(path),'keys':sorted(record)})
 return {'rank':rank,'steps':result}
