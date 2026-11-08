"""101 test-only per-rank instrumentation. Production Worker owns the model."""
from pathlib import Path


def manifest(worker, directory):
    import os,resource,torch
    from minivllm.layers.linear import QKVParallelLinear,MergedColumnParallelLinear,RowParallelLinear,ColumnParallelLinear,VocabParallelEmbedding
    runner=worker.model_runner;model=runner.model;rank=worker.rank or 0
    torch.cuda.synchronize(worker.device)
    modules=dict(model.named_modules());items=[]
    for name,param in model.named_parameters():
        module=modules[name.rsplit('.',1)[0]];field=name.rsplit('.',1)[1];source=[]
        if isinstance(module,QKVParallelLinear):
            for shard in ('q','k','v'):
                _,size=module._shard_range(shard);sr=rank if shard=='q' else rank//module.num_kv_head_replicas
                source.append({'checkpoint':name.replace('qkv_proj',shard+'_proj'),'axis':0,'start':sr*size,'end':(sr+1)*size,'packed_shard':shard})
        elif isinstance(module,MergedColumnParallelLinear):
            for idx,label in enumerate(('gate','up')):
                size=module.output_sizes[idx]//module.tp_size
                source.append({'checkpoint':name.replace('gate_up_proj',label+'_proj'),'axis':0,'start':rank*size,'end':(rank+1)*size,'packed_shard':idx})
        elif isinstance(module,RowParallelLinear) and field=='weight':
            size=module.input_size_per_partition;source=[{'checkpoint':name,'axis':1,'start':rank*size,'end':(rank+1)*size}]
        elif isinstance(module,ColumnParallelLinear) and field=='weight':
            size=module.output_size_per_partition;source=[{'checkpoint':name,'axis':0,'start':rank*size,'end':(rank+1)*size}]
        elif isinstance(module,VocabParallelEmbedding):
            idx=module.shard_indices;source=[{'checkpoint':name,'axis':0,'start':idx.org_vocab_start_index,'end':idx.org_vocab_end_index}]
        else:source=[{'checkpoint':name,'axis':None,'start':None,'end':None}]
        items.append({'name':name,'shape':list(param.shape),'dtype':str(param.dtype),'bytes':param.numel()*param.element_size(),'device':str(param.device),'sources':source})
    path=Path(directory)/f'manifest_rank{rank}.json';path.parent.mkdir(parents=True,exist_ok=True)
    import json
    data={'rank':rank,'pid':os.getpid(),'device':worker.device,'uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
          'parameters':items,'parameter_bytes':sum(x['bytes'] for x in items),
          'allocated':torch.cuda.memory_allocated(worker.device),'reserved':torch.cuda.memory_reserved(worker.device),
          'peak_allocated':torch.cuda.max_memory_allocated(worker.device),'peak_reserved':torch.cuda.max_memory_reserved(worker.device),
          'cpu_max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    path.write_text(json.dumps(data,indent=2)+'\n');return {'rank':rank,'path':str(path),'bytes':data['parameter_bytes'],'peak_allocated':data['peak_allocated'],'peak_reserved':data['peak_reserved']}


def install(worker, all_layers=False, pause_layer=None):
    import torch
    model=worker.model_runner.model;model._step101_current={};model._step101_handles=[]
    def save(name,with_input=False):
        def hook(module,args,out):
            value=out[0] if isinstance(out,tuple) else out
            if isinstance(value,torch.Tensor):model._step101_current[name]=value.detach().cpu().clone()
            if with_input and args and isinstance(args[0],torch.Tensor):model._step101_current[name+'_input']=args[0].detach().cpu().clone()
            if name.endswith('local_attn') and len(args)>=3:
                for i in range(3):model._step101_current[name+f'_arg{i}']=args[i].detach().cpu().clone()
            if isinstance(out,tuple) and len(out)>1 and isinstance(out[1],torch.Tensor):model._step101_current[name+'_residual']=out[1].detach().cpu().clone()
        return hook
    def model_input(module,args,kwargs):
        for key in ('input_ids','positions'):
            if key in kwargs:model._step101_current[key]=kwargs[key].detach().cpu().clone()
    model._step101_handles.append(model.register_forward_pre_hook(model_input,with_kwargs=True))
    model._step101_handles.append(model.register_forward_hook(save('hidden')))
    model._step101_handles.append(model.logits_processor.register_forward_hook(save('logits')))
    layers=model.model.layers if all_layers else model.model.layers[:1]
    for idx,layer in enumerate(layers):
        if idx == pause_layer:
            import time
            model._step101_handles.append(layer.register_forward_pre_hook(lambda module,args: time.sleep(0.05)))
        prefix=f'layer{idx}_'
        for label,mod in (('input_norm',layer.input_layernorm),('qkv',layer.self_attn.qkv_proj),('local_attn',layer.self_attn.attn),('attention',layer.self_attn),('post_norm',layer.post_attention_layernorm),('mlp',layer.mlp),('layer',layer)):
            model._step101_handles.append(mod.register_forward_hook(save(prefix+label,True)))
    return worker.rank


def flush(worker,directory,step):
    import torch
    model=worker.model_runner.model;path=Path(directory)/f'step{step}_rank{worker.rank or 0}.pt'
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(model._step101_current,path)
    keys=sorted(model._step101_current);model._step101_current={};return {'rank':worker.rank or 0,'path':str(path),'keys':keys}
