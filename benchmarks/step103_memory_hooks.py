"""Test-only per-rank model, KV, attention workspace and allocator statistics."""

def snapshot(worker,upstream):
 import torch
 import torch.distributed as dist
 runner=worker.model_runner;model=runner.model;dev=torch.device(worker.device)
 params=sum(p.numel()*p.element_size() for p in model.parameters())
 caches=runner.kv_caches.values() if isinstance(runner.kv_caches,dict) else runner.kv_caches
 kv=sum(t.numel()*t.element_size() for t in caches)
 workspaces={};seen=set()
 def add(label,t):
  if not isinstance(t,torch.Tensor):return
  storage=t.untyped_storage();ptr=storage.data_ptr()
  if ptr in seen:return
  seen.add(ptr);workspaces[label]=storage.nbytes()
 if upstream:
  for i,groups in enumerate(runner.attn_groups):
   for j,group in enumerate(groups):
    for k,builder in enumerate(group.metadata_builders):
     for name in ('softmax_segm_output','softmax_segm_max','softmax_segm_expsum'):
      add(f'attn_group{i}_{j}_builder{k}_{name}',getattr(builder,name,None))
  from vllm.v1.worker.workspace import is_workspace_manager_initialized,current_workspace_manager
  if is_workspace_manager_initialized():
   for i,tensor in enumerate(current_workspace_manager()._current_workspaces):add(f'worker_workspace_{i}',tensor)
 else:
  pool=runner.attn_metadata_builder.workspace_pool
  for i,(_key,tensors) in enumerate(pool._workspaces.items()):
   for j,tensor in enumerate(tensors):add(f'attn_workspace_{i}_{j}',tensor)
 rank=dist.get_rank() if dist.is_initialized() else 0
 return {'rank':rank,'uuid':str(torch.cuda.get_device_properties(dev).uuid),
         'parameter_bytes':params,'kv_tensor_bytes':kv,'known_workspace_bytes':sum(workspaces.values()),
         'workspace_storage':workspaces,'allocated_bytes':torch.cuda.memory_allocated(dev),
         'reserved_bytes':torch.cuda.memory_reserved(dev),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(dev),'peak_reserved_bytes':torch.cuda.max_memory_reserved(dev)}
