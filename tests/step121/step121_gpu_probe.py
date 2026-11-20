"""独立进程生成相同输入并运行本地或冻结 vLLM 的真实 CUDA repack。"""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=('mini', 'vllm'), required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
if not torch.cuda.is_available():
    raise RuntimeError('真实 CUDA GPU 不可用')
if args.backend == 'mini':
    from minivllm.quant_ops.ops import gptq_marlin_repack
    from minivllm.layers.quantization.marlin import marlin_make_workspace_new
    first = marlin_make_workspace_new(torch.device('cuda:0'))
    second = marlin_make_workspace_new(torch.device('cuda:0'))
    assert first.data_ptr() != second.data_ptr()
    first.fill_(7)
    second.fill_(9)
    address = first.data_ptr()
    reused = marlin_make_workspace_new(torch.device('cuda:0'), existing=first)
    assert reused.data_ptr() == address and torch.count_nonzero(reused).item() == 0
    assert torch.all(second == 9).item()
    def repack(q, perm, k, n):
        return gptq_marlin_repack(q, perm, k, n)
else:
    from vllm import _custom_ops
    def repack(q, perm, k, n):
        return _custom_ops.gptq_marlin_repack(q, perm, k, n, 4, False)

results = {}
for k, n in ((128, 64), (256, 128), (384, 192)):
    generator = torch.Generator(device='cpu').manual_seed(k * 1000 + n)
    for pattern in ('random', 'regular'):
        if pattern == 'random':
            q = torch.randint(-(2**31), 2**31, (k // 8, n), dtype=torch.int32, generator=generator)
        else:
            q = torch.arange(k // 8 * n, dtype=torch.int32).reshape(k // 8, n) * 7654321
        q = q.cuda()
        perm = torch.empty(0, dtype=torch.int32, device=q.device)
        results[f'{k}x{n}_{pattern}'] = repack(q, perm, k, n).cpu()
torch.cuda.synchronize()
torch.save(results, args.output)
