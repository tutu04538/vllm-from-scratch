"""Capture per-token raw logits and FP32 LM-head replay for diagnostic only."""
from pathlib import Path


def install(worker):
 import torch
 runner=worker.model_runner;model=runner.model
 runner._step103_token_diag=[]
 original=model.compute_logits
 def compute(hidden):
  logits=original(hidden)
  weights=model.lm_head.weight
  index=model.lm_head.shard_indices
  ids=logits[0].float().topk(2).indices.tolist()
  for token in (31935,71304):
   if token not in ids:ids.append(token)
  fp32={}
  for token in ids:
   if index.org_vocab_start_index<=token<index.org_vocab_end_index:
    local=token-index.org_vocab_start_index
    fp32[str(token)]=float(torch.dot(hidden[0].float(),weights[local].float()).item())
  runner._step103_token_diag.append({'logits':logits.detach().cpu().clone(),
                                    'fp32_head_scores':fp32})
  return logits
 model.compute_logits=compute
 return True


def flush(worker,directory):
 import torch
 import torch.distributed as dist
 rank=dist.get_rank() if dist.is_initialized() else 0
 path=Path(directory)/f'rank{rank}.pt';torch.save(worker.model_runner._step103_token_diag,path)
 return {'rank':rank,'path':str(path)}
