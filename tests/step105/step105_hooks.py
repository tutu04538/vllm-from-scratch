"""Test-only capture of V2 runtime tensors; no instrumentation during CUDA capture."""
from pathlib import Path


def install(worker, poison_padding=False):
    import torch
    runner=worker.model_runner;model=runner.model
    runner._step105_records=[];runner._step105_slots=None;runner._step105_tables=None;runner._step105_logits=None
    prepare=runner.prepare_attn;compute=model.compute_logits;sample=runner.sample
    def prepare_hook(batch):
        tables,slots=prepare(batch)
        if poison_padding and batch.num_tokens_after_padding>batch.num_tokens:
            runner.input_buffers.input_ids[batch.num_tokens:batch.num_tokens_after_padding].fill_(17)
            runner.input_buffers.positions[batch.num_tokens:batch.num_tokens_after_padding].fill_(13)
        runner._step105_slots=slots[0,:batch.num_tokens_after_padding].detach().cpu().clone()
        runner._step105_tables=tables[0].detach().clone()
        return tables,slots
    def logits_hook(hidden):
        result=compute(hidden);runner._step105_logits=result.detach().cpu().clone();return result
    def sample_hook(hidden,batch,grammar):
        rows=hidden[batch.logits_indices].detach().cpu().clone()
        sampled,num,rejected=sample(hidden,batch,grammar)
        positions=batch.positions[batch.logits_indices].detach().cpu().tolist()
        counts=num.detach().cpu().tolist();sampled_ids=sampled.sampled_token_ids.detach().cpu().tolist()
        tables=runner._step105_tables;records=[]
        for row,req in enumerate(batch.req_ids):
            if not counts[row]:continue
            pos=int(positions[row]);length=pos+1
            kv=[]
            for layer in model.model.layers:
                cache=layer.self_attn.attn.kv_cache
                block_size=cache.shape[2]
                logical=torch.arange(length,device=cache.device,dtype=torch.long)
                blocks=tables[row,torch.div(logical,block_size,rounding_mode='floor')].long()
                offsets=logical%block_size
                kv.append(cache[:,blocks,offsets,:,:].permute(1,0,2,3).detach().cpu().clone())
            records.append({'request_id':req,'resident_slot':runner.req_states.req_id_to_index[req],
                            'physical_blocks':tables[row,:((length+block_size-1)//block_size)].detach().cpu().tolist(),
                            'position':pos,'hidden':rows[row],
                            'logits':runner._step105_logits[row], 'sampled':sampled_ids[row][0],'kv':kv})
        runner._step105_records.append({'rows':records,'slots':runner._step105_slots,
            'num_tokens':batch.num_tokens,'num_tokens_after_padding':batch.num_tokens_after_padding,
            'num_reqs':batch.num_reqs})
        return sampled,num,rejected
    runner.prepare_attn=prepare_hook;model.compute_logits=logits_hook;runner.sample=sample_hook
    return True


def flush(worker,directory):
    import torch
    import torch.distributed as dist
    rank=dist.get_rank() if dist.is_initialized() else 0
    path=Path(directory)/f'trace_rank{rank}.pt';torch.save(worker.model_runner._step105_records,path)
    return {'rank':rank,'path':str(path)}
