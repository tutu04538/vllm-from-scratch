"""Real AWQ checkpoint rank-local slicing and frozen Marlin numerical check.

Runs logical ranks serially on one GPU; this is local-shard evidence, not a
multi-process collective or full-model TP test.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from minivllm.layers import linear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.model_loader.post_load import process_weights_after_loading
from benchmarks.check_step090_reference import verify_oracle
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tp', type=int, choices=(2, 4), required=True)
a = p.parse_args()
verify_oracle()
torch.set_default_dtype(torch.bfloat16)
hf = json.loads((a.model / 'config.json').read_text())
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
qcfg = resolve_quantization_config(None, hf['quantization_config'])
hidden, inter = hf['hidden_size'], hf['intermediate_size']
heads, kv_heads, dim = hf['num_attention_heads'], hf['num_key_value_heads'], hf['head_dim']
rows = sorted({0, hf['num_hidden_layers']//2, hf['num_hidden_layers']-1})
cases = {'qkv': [('q','self_attn.q_proj'),('k','self_attn.k_proj'),('v','self_attn.v_proj')],
         'o_proj': [(None,'self_attn.o_proj')],
         'gate_up': [(0,'mlp.gate_proj'),(1,'mlp.up_proj')],
         'down_proj': [(None,'mlp.down_proj')]}
results = []
for layer_id in rows:
    for case, projections in cases.items():
        source = {}
        for shard, proj in projections:
            source[shard] = {}
            for name in ('qweight','qzeros','scales'):
                key=f'model.layers.{layer_id}.{proj}.{name}'
                with safe_open(a.model/index[key],framework='pt',device='cpu') as f:
                    source[shard][name]=f.get_tensor(key)
        for rank in range(a.tp):
            linear._linear_tp_group = lambda disable_tp: (SimpleNamespace(), rank, a.tp)
            with torch.device('cuda:0'):
                if case == 'qkv':
                    layer=linear.QKVParallelLinear(hidden,dim,heads,kv_heads,bias=False,
                                                   prefix='qkv',quant_config=qcfg)
                elif case == 'gate_up':
                    layer=linear.MergedColumnParallelLinear(hidden,[inter,inter],bias=False,
                                                            prefix='gate_up',quant_config=qcfg)
                else:
                    k,n=(hidden,hidden) if case=='o_proj' else (inter,hidden)
                    layer=linear.RowParallelLinear(k,n,bias=False,reduce_results=False,
                                                   prefix=case,quant_config=qcfg)
            for name in ('qweight','qzeros','scales'):
                param=getattr(layer,name)
                for shard,_ in projections:
                    value=source[shard][name]
                    param.weight_loader(param,value,shard)
                param.validate_loaded()
                if case in ('o_proj','down_proj'):
                    value=source[None][name]
                    width=value.shape[0]//a.tp
                    expected=value.narrow(0,rank*width,width)
                else:
                    chunks=[]
                    for shard,_ in projections:
                        value=source[shard][name]
                        copies=max(1,a.tp//kv_heads) if shard in ('k','v') else 1
                        num_shards=a.tp//copies
                        width=value.shape[1]//num_shards
                        chunks.append(value.narrow(1,(rank//copies)*width,width))
                    expected=torch.cat(chunks,dim=1)
                assert torch.equal(param.cpu(),expected), (layer_id,case,rank,name)
            with torch.inference_mode():
                process_weights_after_loading(nn.Sequential(layer),None,'cuda:0')
                n,k=layer.marlin_logical_nk
                pn,pk=layer.marlin_padded_nk
                assert (n,k)==(pn,pk), 'this real-model test expects no tile padding'
                torch.manual_seed(124)
                x=torch.randn(2,k,device='cuda',dtype=torch.bfloat16).contiguous()
                actual=layer.quant_method.apply(layer,x)
                reference=ops.marlin_gemm(x,None,layer.qweight,None,layer.scales,None,None,
                    layer.qzeros,layer.g_idx,layer.g_idx_sort_indices,layer.quant_method.workspace,
                    scalar_types.uint4,2,pn,pk,True,False,True,False)[...,:n]
                difference=(actual.float()-reference.float()).abs().max().item()
                assert difference == 0.0, (layer_id,case,rank,difference)
            results.append({'layer':layer_id,'case':case,'rank':rank,'max_abs':difference})
            del layer,actual,reference,x
            torch.cuda.empty_cache()
        del source
        print(f'{a.model.name} TP{a.tp} layer={layer_id} {case}: passed',flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'model':str(a.model),'tp':a.tp,'cases':results},indent=2)+'\n')
