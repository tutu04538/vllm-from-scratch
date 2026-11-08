"""103 test-only V2 Runner hooks. No diagnostic D2H in production."""
from pathlib import Path


def install(worker, upstream):
    import torch
    import torch.distributed as dist
    runner=worker.model_runner
    model=runner.model
    runner._step103_records=[]
    runner._step103_counts={}
    runner._step103_seen={}
    runner._step103_sample_calls=0
    runner._step103_tables=None
    runner._step103_logits=None
    original_prepare=runner.prepare_attn
    original_logits=model.compute_logits
    original_sample=runner.sample

    def prepare(batch):
        tables,slots=original_prepare(batch)
        runner._step103_tables=tables[0].detach().clone()
        return tables,slots

    def logits(hidden):
        value=original_logits(hidden)
        runner._step103_logits=value.detach().cpu().clone()
        return value

    def sample(hidden,batch,grammar):
        hidden_rows=hidden[batch.logits_indices].detach().cpu().clone()
        result=original_sample(hidden,batch,grammar)
        sampled,num_sampled,_=result
        counts=num_sampled.detach().cpu().tolist()
        runner._step103_sample_calls += 1
        positions=batch.positions[batch.logits_indices].detach().cpu().tolist()
        sampled_ids=sampled.sampled_token_ids.detach().cpu().tolist()
        tables=runner._step103_tables
        cache=model.model.layers[0].self_attn.attn.kv_cache
        # vLLM 0.28 Triton packs [blocks, heads, block_size, 2*dim]; mini
        # uses [2, blocks, block_size, heads, dim]. Both expose the same data.
        block_size=cache.shape[2]
        for row,req in enumerate(batch.req_ids):
            if counts[row]:
                seen=runner._step103_seen.setdefault(req,{'count':0,'first_sample_call':runner._step103_sample_calls})
                seen['count']+=int(counts[row])
                seen['last_sample_call']=runner._step103_sample_calls
            if not counts[row] or runner._step103_counts.get(req,0)>=4:
                continue
            pos=int(positions[row]); logical=torch.arange(pos+1,device=cache.device,dtype=torch.long)
            blocks=tables[row,torch.div(logical,block_size,rounding_mode='floor')].long()
            offsets=logical % block_size
            if upstream:
                kv=cache[blocks,:,offsets,:].reshape(pos+1,-1,2,128).permute(0,2,1,3)
            else:
                kv=cache[:,blocks,offsets,:,:].permute(1,0,2,3)
            record={'request_id':req,'position':pos,'sequence_length':pos+1,
                    'hidden':hidden_rows[row], 'logits':runner._step103_logits[row],
                    'kv':kv.detach().cpu().clone(), 'sampled':int(sampled_ids[row][0])}
            runner._step103_records.append(record)
            runner._step103_counts[req]=runner._step103_counts.get(req,0)+1
        return result

    runner.prepare_attn=prepare
    model.compute_logits=logits
    runner.sample=sample
    dev=torch.device(worker.device)
    params=sum(p.numel()*p.element_size() for p in model.parameters())
    caches=runner.kv_caches.values() if isinstance(runner.kv_caches,dict) else runner.kv_caches
    # KV tensors are counted by logical tensor bytes; storage can be shared.
    kv_bytes=sum(t.numel()*t.element_size() for t in caches)
    rank=dist.get_rank() if dist.is_initialized() else 0
    return {'rank':rank,'device':str(dev),'uuid':str(torch.cuda.get_device_properties(dev).uuid),
            'parameter_bytes':params,'kv_tensor_bytes':kv_bytes,
            'allocated_bytes':torch.cuda.memory_allocated(dev),'reserved_bytes':torch.cuda.memory_reserved(dev),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(dev),'peak_reserved_bytes':torch.cuda.max_memory_reserved(dev)}


def flush(worker,directory,run_id):
    import torch
    import torch.distributed as dist
    runner=worker.model_runner
    rank=dist.get_rank() if dist.is_initialized() else 0
    path=Path(directory)/f'{run_id}_rank{rank}.pt'
    torch.save(runner._step103_records,path)
    seen=runner._step103_seen
    runner._step103_records=[];runner._step103_counts={};runner._step103_seen={};runner._step103_sample_calls=0
    dev=torch.device(worker.device)
    return {'rank':rank,'path':str(path),'sampling_progress':seen,'peak_allocated_bytes':torch.cuda.max_memory_allocated(dev),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(dev),'allocated_bytes':torch.cuda.memory_allocated(dev),
            'reserved_bytes':torch.cuda.memory_reserved(dev)}
