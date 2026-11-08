"""Test-only forced-token replay; production scheduler and runner remain unchanged."""
from pathlib import Path


def install(worker, histories):
 runner=worker.model_runner
 runner._step103_forced_seen={};runner._step103_forced_inputs=[]
 original_prepare=runner.prepare_attn;original_sample=runner.sample
 def prepare(batch):
  tables,slots=original_prepare(batch)
  ids=batch.input_ids[:batch.num_tokens].detach().cpu().tolist()
  positions=batch.positions[:batch.num_tokens].detach().cpu().tolist()
  mapping=slots[0,:batch.num_tokens].detach().cpu().tolist()
  blocks=tables[0][:batch.num_reqs].detach().cpu().tolist()
  starts=batch.query_start_loc_np.tolist()
  for row,req in enumerate(batch.req_ids):
   start,end=starts[row:row+2]
   runner._step103_forced_inputs.append({'request_id':req,'input_ids':ids[start:end],
       'positions':positions[start:end],'slot_mapping':mapping[start:end],'block_table':blocks[row]})
  return tables,slots
 def sample(hidden,batch,grammar):
  sampled,num,rejected=original_sample(hidden,batch,grammar)
  counts=num.detach().cpu().tolist()
  for row,req in enumerate(batch.req_ids):
   if counts[row]:
    request_index=int(req.split('-',1)[0])
    seen=runner._step103_forced_seen.get(req,0)
    sampled.sampled_token_ids[row,0]=histories[request_index][seen]
    runner._step103_forced_seen[req]=seen+1
  return sampled,num,rejected
 runner.prepare_attn=prepare;runner.sample=sample
 return True


def flush(worker,directory):
 import torch
 import torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0
 path=Path(directory)/f'forced_inputs_rank{rank}.pt';torch.save(worker.model_runner._step103_forced_inputs,path)
 return {'rank':rank,'path':str(path),'forced_counts':worker.model_runner._step103_forced_seen}
