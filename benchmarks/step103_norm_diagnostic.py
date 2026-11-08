"""Compare local and frozen CUDA RMSNorm on the exact real embedding row."""
import json,sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.check_step090_reference import verify_oracle
from minivllm.layers.ops.layernorm import rms_norm,rms_norm_native


def main():
 verify_oracle()
 import vllm._custom_ops as ops
 path,_=resolve_snapshot('8b');index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
 name='model.embed_tokens.weight'
 with safe_open(path/index[name],framework='pt',device='cpu') as handle:
  x=handle.get_slice(name)[198:199].to('cuda')
  residual=handle.get_slice(name)[14880:14881].to('cuda')
 name='model.layers.0.input_layernorm.weight'
 with safe_open(path/index[name],framework='pt',device='cpu') as handle: weight=handle.get_tensor(name).to('cuda')
 eps=json.loads((path/'config.json').read_text())['rms_norm_eps']
 with torch.inference_mode():
  mini=rms_norm(x,weight,eps);native=rms_norm_native(x,weight,eps);up=torch.empty_like(x);ops.rms_norm(up,x,weight,eps)
  fp32=x.float();one_cast=(fp32*torch.rsqrt(fp32.square().mean(dim=-1,keepdim=True)+eps)*weight.float()).to(x.dtype)
  true=(x.double()*torch.rsqrt(x.double().square().mean(dim=-1,keepdim=True)+eps)*weight.double())
  report={}
  for label,out in [('mini',mini),('upstream',up),('native',native)]:
   diff=(out.double()-true).abs()
   report[label]={'max_abs_fp64':float(diff.max()),'mean_abs_fp64':float(diff.mean())}
  diff=(mini.float()-up.float()).abs();idx=int(diff.argmax())
  report['mini_vs_upstream']={'max_abs':float(diff.flatten()[idx]),'index':idx,'mini':float(mini.flatten()[idx]),'upstream':float(up.flatten()[idx]),'unequal_values':int((mini!=up).sum())}
  upx=x.clone();upr=residual.clone();ops.fused_add_rms_norm(upx,upr,weight,eps)
  rounded=(x.float()+residual.float()).to(x.dtype).float()
  fused_one=(rounded*torch.rsqrt(rounded.square().mean(dim=-1,keepdim=True)+eps)*weight.float()).to(x.dtype)
  full=x.float()+residual.float()
  fused_full=(full*torch.rsqrt(full.square().mean(dim=-1,keepdim=True)+eps)*weight.float()).to(x.dtype)
  fused_mini,mini_resid=rms_norm(x,weight,eps,residual)
  report['fused']={'one_round_unequal':int((fused_one!=upx).sum()),'full_sum_unequal':int((fused_full!=upx).sum()),'mini_unequal':int((fused_mini!=upx).sum()),'residual_equal':bool(torch.equal(mini_resid,upr))}
  report['single_round_vs_upstream']={'unequal_values':int((one_cast!=up).sum()),'max_abs':float((one_cast.float()-up.float()).abs().max())}
  report['native_vs_mini_equal']=bool(torch.equal(native,mini));report['native_vs_upstream_equal']=bool(torch.equal(native,up))
 out=ROOT/'benchmarks/results/step103/norm_diagnostic.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
