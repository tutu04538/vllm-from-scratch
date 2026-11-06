"""090 冻结参考逐 rank 采集；仅用于测试，不进入生产路径。"""
from pathlib import Path

def install(model):
    import torch
    import torch.distributed as dist
    import os
    import resource
    model._step090_records=[]
    model._step090_current={}
    def save(name, get_input=False):
        def hook(module, args, output):
            value=output[0] if isinstance(output,tuple) else output
            if isinstance(value,torch.Tensor):
                model._step090_current[name]=value.detach().cpu().clone()
            if get_input and args and isinstance(args[0],torch.Tensor):
                model._step090_current[name+'_input']=args[0].detach().cpu().clone()
        return hook
    layer=model.model.layers[0]
    def save_model_input(module, args, kwargs):
        ids=kwargs.get('input_ids', args[0] if args else None)
        positions=kwargs.get('positions', args[1] if len(args)>1 else None)
        if ids is None or positions is None:
            raise ValueError('参考 prefill 缺少 input_ids 或 positions')
        model._step090_current['input_ids']=ids.detach().cpu().clone()
        model._step090_current['positions']=positions.detach().cpu().clone()
    def save_kv(module, args, output):
        from vllm.forward_context import get_forward_context
        context=get_forward_context()
        slots=context.slot_mapping[module.layer_name]
        metadata=context.attn_metadata[module.layer_name]
        cache=context.no_compile_layers[module.layer_name].kv_cache
        if cache.ndim != 4:
            raise ValueError(f'意外 KV 布局: {tuple(cache.shape)}')
        slots=slots[:args[0].shape[0]].to(torch.long)
        if torch.any(slots < 0):
            raise ValueError('参考 prefill 含无效 slot，不能读取 padding')
        block_size=cache.shape[2]
        selected=cache[torch.div(slots,block_size,rounding_mode='floor'),:,slots%block_size,:]
        model._step090_current['kv']=selected.detach().cpu().clone()
        model._step090_current['slot_mapping']=slots.detach().cpu().clone()
        model._step090_current['block_table']=metadata.block_table.detach().cpu().clone()
        model._step090_current['seq_lens']=metadata.seq_lens.detach().cpu().clone()
    model._step090_handles=[layer.self_attn.qkv_proj.register_forward_hook(save('qkv',True)),
        layer.self_attn.attn.register_forward_hook(save_kv),
        layer.self_attn.register_forward_hook(save('attention')),
        model.register_forward_pre_hook(save_model_input,with_kwargs=True),
        model.register_forward_hook(save('hidden')),
        model.logits_processor.register_forward_hook(save('logits'))]
    model._step090_rank=dist.get_rank() if dist.is_initialized() else 0
    device=torch.cuda.current_device()
    return {'rank':model._step090_rank,'local_rank':int(os.environ.get('LOCAL_RANK',device)),
            'device':device,'device_uuid':str(getattr(torch.cuda.get_device_properties(device),'uuid','unavailable')),
            'device_name':torch.cuda.get_device_name(device),
            'cpu_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
            'qkv_weight_shape':list(layer.self_attn.qkv_proj.weight.shape)}


def flush(model, directory, iteration):
    import torch
    rank=model._step090_rank
    values=model._step090_current
    required={'qkv','qkv_input','attention','hidden','logits','kv','slot_mapping',
              'input_ids','positions','block_table','seq_lens'}
    if not required.issubset(values):
        raise RuntimeError(f'rank {rank} 缺少记录: {sorted(required-set(values))}')
    path=Path(directory)/f'iter{iteration}_rank{rank}.pt'
    torch.save(values,path)
    model._step090_current={}
    return {'rank':rank,'path':str(path),'shapes':{k:list(v.shape) for k,v in values.items()}}


def local_fp32(model, directory):
    import torch
    rank=model._step090_rank
    records=torch.load(Path(directory)/f'iter0_rank{rank}.pt',map_location='cpu',weights_only=True)
    layer=model.model.layers[0].self_attn.qkv_proj
    inp=records['qkv_input'].to(layer.weight.device,dtype=torch.float32)
    weight=layer.weight.detach().to(dtype=torch.float32)
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try: expected=inp @ weight.T
    finally: torch.backends.cuda.matmul.allow_tf32=old
    actual=records['qkv'].to(device=expected.device,dtype=torch.float32)
    err=actual-expected
    flat_index=int(err.abs().argmax().item())
    row,col=divmod(flat_index,actual.shape[-1])
    stats={'rank':rank,'max_abs':err.abs().max().item(),
           'relative_l2':(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(expected)).item(),
           'max_error_index':[row,col], 'observed_value':actual[row,col].item(),
           'reference_value':expected[row,col].item(),
           'shape':list(actual.shape)}
    return stats
