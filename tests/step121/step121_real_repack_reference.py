"""在隔离的冻结 vLLM 进程对标准 packed 输入运行 CUDA repack。"""
import argparse
import hashlib
from pathlib import Path
import torch
from vllm import _custom_ops

p=argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
frozen=Path(__file__).resolve().parents[2]/'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/_custom_ops.py'
if hashlib.sha256(Path(_custom_ops.__file__).read_bytes()).digest()!=hashlib.sha256(frozen.read_bytes()).digest():
    raise RuntimeError('上游 _custom_ops 与冻结源码不一致')
d=torch.load(a.input,map_location='cpu',weights_only=True)
q=d['standard'].cuda()
perm=torch.empty(0,device='cuda',dtype=torch.int32)
result=_custom_ops.gptq_marlin_repack(q,perm,d['k'],d['n'],4,False)
torch.save(result.cpu(),a.output)
