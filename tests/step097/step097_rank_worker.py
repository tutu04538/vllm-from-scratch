"""097 real rank tests; frozen vLLM layers and real Qwen layer-0 slices."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from safetensors import safe_open

MODELS={'8b':('Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218'),
        '14b':('Qwen3-14B','40c069824f4251a91eefaf281ebe4c544efd3e18')}


def qkv_tiny(rank,world,device,ref):
    from minivllm.layers.linear import QKVParallelLinear
    cases=[]
    hidden,dim,heads=6,2,8
    inp=torch.tensor([[1.,2.,3.,4.,5.,6.]],device=device)
    for kv in (1,2,8):
        source={'q':torch.arange(heads*dim*hidden,dtype=torch.float32).reshape(-1,hidden),
                'k':1000+torch.arange(kv*dim*hidden,dtype=torch.float32).reshape(-1,hidden),
                'v':2000+torch.arange(kv*dim*hidden,dtype=torch.float32).reshape(-1,hidden)}
        biases={k:v[:,0].clone() for k,v in source.items()}
        for packed in (False,True):
            for bias in (False,True):
                with torch.device(device):
                    ours=QKVParallelLinear(hidden,dim,heads,kv,bias=bias)
                    oracle=ref.QKVParallelLinear(hidden,dim,heads,kv,bias=bias)
                with torch.no_grad():
                    for par,refpar,weights in [(ours.weight,oracle.weight,source)]+([(ours.bias,oracle.bias,biases)] if bias else []):
                        if packed:
                            data=torch.cat([weights[k] for k in ('q','k','v')])
                            ours.weight_loader(par,data)
                            oracle.weight_loader(refpar,data)
                        else:
                            for k in ('v','q','k'):
                                ours.weight_loader(par,weights[k],k)
                                oracle.weight_loader(refpar,weights[k],k)
                ours.validate_loaded_shards({'q','k','v'})
                assert torch.equal(ours.weight,oracle.weight)
                if bias: assert torch.equal(ours.bias,oracle.bias)
                qn=heads//world;kn=max(1,kv//world);rep=max(1,world//kv)
                qstart=rank*qn*dim;kstart=(rank//rep)*kn*dim
                expected=torch.cat([source['q'][qstart:qstart+qn*dim],source['k'][kstart:kstart+kn*dim],source['v'][kstart:kstart+kn*dim]])
                assert torch.equal(ours.weight.cpu(),expected)
                assert ours.weight.numel()==(qn+2*kn)*dim*hidden
                assert torch.equal(ours(inp)[0],oracle(inp)[0])
                cases.append({'kv':kv,'packed':packed,'bias':bias,'q_heads':qn,'kv_heads':kn,
                              'replicas':rep,'kv_source_head':kstart//dim})
    return cases


def merged_tiny(rank,world,device,ref):
    from minivllm.layers.linear import MergedColumnParallelLinear
    sizes=[8,12,4];hidden=6
    weights=[torch.full((size,hidden),float(10*(idx+1)))+torch.arange(size).view(-1,1) for idx,size in enumerate(sizes)]
    biases=[w[:,0].clone() for w in weights]
    cases=[]
    for packed in (False,True):
        for bias in (False,True):
            with torch.device(device):
                ours=MergedColumnParallelLinear(hidden,sizes,bias=bias)
                oracle=ref.MergedColumnParallelLinear(hidden,sizes,bias=bias)
            with torch.no_grad():
                for par,refpar,values in [(ours.weight,oracle.weight,weights)]+([(ours.bias,oracle.bias,biases)] if bias else []):
                    if packed:
                        data=torch.cat(values)
                        ours.weight_loader(par,data);oracle.weight_loader(refpar,data)
                    else:
                        for i in (2,0,1):
                            ours.weight_loader(par,values[i],i);oracle.weight_loader(refpar,values[i],i)
            ours.validate_loaded_shards({0,1,2})
            assert torch.equal(ours.weight,oracle.weight)
            if bias:assert torch.equal(ours.bias,oracle.bias)
            expected=torch.cat([w[rank*(n//world):(rank+1)*(n//world)] for w,n in zip(weights,sizes)])
            assert torch.equal(ours.weight.cpu(),expected)
            x=torch.arange(hidden,dtype=torch.float32,device=device).view(1,-1)
            assert torch.equal(ours(x)[0],oracle(x)[0])
            cases.append({'packed':packed,'bias':bias,'local_rows':sum(n//world for n in sizes)})
    return cases


def qwen_mlp(rank,world,device):
    from minivllm.models.qwen3 import Qwen2MLP
    from torch.nn import functional as F
    cfg={'hidden_size':4,'intermediate_size':8,'hidden_act':'silu'}
    gate=torch.arange(32,dtype=torch.float32).reshape(8,4)*0.01
    up=(torch.arange(32,dtype=torch.float32).reshape(8,4)+1)*0.02
    down=(torch.arange(32,dtype=torch.float32).reshape(4,8)+1)*0.005
    with torch.device(device): layer=Qwen2MLP(cfg)
    layer.gate_up_proj.weight_loader(layer.gate_up_proj.weight,gate,0)
    layer.gate_up_proj.weight_loader(layer.gate_up_proj.weight,up,1)
    layer.down_proj.weight_loader(layer.down_proj.weight,down)
    x=torch.tensor([[0.1,0.2,0.3,0.4]],device=device)
    with torch.no_grad():
        got=layer(x)
        expected=F.linear(F.silu(F.linear(x,gate.to(device)))*F.linear(x,up.to(device)),down.to(device))
    torch.testing.assert_close(got,expected,atol=1e-5,rtol=1e-5)
    return {'max_abs':float((got-expected).abs().max()),'local_intermediate':8//world}


def qwen_shapes(rank,world,device):
    from minivllm.models.qwen3 import Qwen3Attention
    config={'hidden_size':32,'num_attention_heads':8,'num_key_value_heads':2,'head_dim':8,
            'max_position_embeddings':64,'rms_norm_eps':1e-6}
    with torch.device(device): layer=Qwen3Attention(config)
    layer = layer.to(device)  # 对齐 BaseModelLoader._initialize_model 的最终设备迁移
    assert layer.total_num_heads==8 and layer.total_num_kv_heads==2
    assert layer.num_heads==8//world and layer.num_kv_heads==max(1,2//world)
    assert layer.q_size==(8//world)*8 and layer.q_norm.weight.numel()==8 and layer.k_norm.weight.numel()==8
    assert layer.o_proj.weight.shape==(32,64//world)
    assert layer.attn.num_heads==layer.num_heads and layer.attn.num_kv_heads==layer.num_kv_heads
    with torch.no_grad():
        layer.qkv_proj.weight.fill_(0.01)
        hidden = torch.ones(2,32,device=device)
        positions = torch.tensor([0,1],device=device)
        q,k,v = layer.attention_pre(positions,hidden)
    assert q.shape==(2,layer.q_size)
    assert k.shape==v.shape==(2,layer.qkv_proj.kv_size)
    assert torch.isfinite(q).all() and torch.isfinite(k).all() and torch.isfinite(v).all()
    return {'explicit_head_dim':8,'hidden_size':32,'heads':layer.num_heads,'kv_heads':layer.num_kv_heads,'attention_pre_shapes':True}


def real_layer(rank,world,device,which):
    from minivllm.layers.linear import QKVParallelLinear,MergedColumnParallelLinear
    from minivllm.layers.layernorm import RMSNorm
    from minivllm.model_loader.weight_utils import default_weight_loader
    model,revision=MODELS[which]
    root=Path('/data/syh/.cache/huggingface/hub')/f'models--Qwen--{model}'/'snapshots'/revision
    cfg=json.loads((root/'config.json').read_text());index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    keys={name:'model.layers.0.'+name+'.weight' for name in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','mlp.gate_proj','mlp.up_proj','self_attn.q_norm','self_attn.k_norm')}
    files={index[k] for k in keys.values()}; assert len(files)==1
    d=cfg['head_dim'];h=cfg['hidden_size'];q=cfg['num_attention_heads'];kv=cfg['num_key_value_heads'];inter=cfg['intermediate_size']
    old=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            qkv=QKVParallelLinear(h,d,q,kv)
            merged=MergedColumnParallelLinear(h,[inter,inter])
            q_norm, k_norm = RMSNorm(d), RMSNorm(d)
    finally:torch.set_default_dtype(old)
    assert q%world==0 and kv%world==0 and inter%world==0
    q_width=(q//world)*d;kv_width=(kv//world)*d;merged_width=inter//world
    assert qkv.num_heads==q//world and qkv.num_kv_heads==kv//world
    assert tuple(qkv.weight.shape)==(q_width+2*kv_width,h)
    assert tuple(merged.weight.shape)==(2*merged_width,h)
    destination={'q':(0,q_width),'k':(q_width,kv_width),'v':(q_width+kv_width,kv_width)}
    with safe_open(root/next(iter(files)),framework='pt',device='cpu') as f:
        for name,shard in [('self_attn.q_proj','q'),('self_attn.k_proj','k'),('self_attn.v_proj','v')]:
            source=f.get_tensor(keys[name]); assert source.device.type=='cpu'
            qkv.weight_loader(qkv.weight,source,shard)
            off,n=destination[shard]
            assert torch.equal(qkv.weight[off:off+n].cpu(),source[rank*n:(rank+1)*n])
        qkv.validate_loaded_shards({'q','k','v'})
        for idx,name in enumerate(('mlp.gate_proj','mlp.up_proj')):
            source=f.get_tensor(keys[name]);merged.weight_loader(merged.weight,source,idx)
            off,n=idx*merged_width,merged_width
            assert torch.equal(merged.weight[off:off+n].cpu(),source[rank*n:(rank+1)*n])
        merged.validate_loaded_shards({0,1})
        for name, norm in [('self_attn.q_norm',q_norm),('self_attn.k_norm',k_norm)]:
            source=f.get_tensor(keys[name])
            assert source.numel()==d and norm.weight.numel()==d
            default_weight_loader(norm.weight,source)
            assert torch.equal(norm.weight.cpu(),source)
    return {'model':model,'revision':revision,'source_file':str(root/next(iter(files))),
            'qkv_shape':list(qkv.weight.shape),'merged_shape':list(merged.weight.shape),
            'head_dim':d,'total_q_heads':q,'total_kv_heads':kv,'local_q_heads':q//world,
            'local_kv_heads':kv//world,'norm_elements':d,'real_checkpoint_slices':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['tiny','8b','14b'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'));torch.cuda.set_device(local);device=f'cuda:{local}'
    from minivllm.distributed import init_distributed_environment,initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment
    init_distributed_environment(world,rank,'env://',local,timeout=timedelta(seconds=90));record={'rank':rank,'world':world,'complete':False}
    try:
        initialize_model_parallel(world,1,timeout=timedelta(seconds=90))
        from vllm.model_executor.layers import linear as ref
        from vllm.model_executor import parameter as refparam
        frozen=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0'
        for module,rel in ((ref,'model_executor/layers/linear.py'),(refparam,'model_executor/parameter.py')):
            assert hashlib.sha256(Path(module.__file__).read_bytes()).digest()==hashlib.sha256((frozen/rel).read_bytes()).digest()
        with ExitStack() as stack:
            for module in (ref,refparam):
                stack.enter_context(patch.object(module,'get_tensor_model_parallel_rank',return_value=rank))
                stack.enter_context(patch.object(module,'get_tensor_model_parallel_world_size',return_value=world))
            if a.case=='tiny':
                record.update(qkv=qkv_tiny(rank,world,device,ref),merged=merged_tiny(rank,world,device,ref),qwen=qwen_shapes(rank,world,device),mlp=qwen_mlp(rank,world,device))
            else:record.update(real_layer(rank,world,device,a.case))
        record.update(complete=True,device_uuid=str(torch.cuda.get_device_properties(local).uuid))
    except BaseException:
        record['error']=traceback.format_exc();raise
    finally:
        a.out.mkdir(parents=True,exist_ok=True);(a.out/f'rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        destroy_model_parallel();destroy_distributed_environment()
    return 0
if __name__=='__main__':sys.exit(main())
