"""公开 8B AWQ 首层完整投影形状的局部 CUDA GEMM 与上游同 packed 差分。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from minivllm.layers.linear import ColumnParallelLinear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.model_loader.post_load import process_weights_after_loading
from minivllm.layers.quantization import marlin as marlin_module
from torch import nn

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--case', choices=('qkv', 'o_proj', 'gate_up', 'down_proj'), required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--trace', action='store_true')
p.add_argument('--full-oracle', action='store_true', help='在 CUDA 上用 FP32 检查全部输出列')
p.add_argument('--repack-compare', action='store_true', help='隔离对照真实权重的整个 repack 输出')
a = p.parse_args()
config = json.loads((a.model / 'config.json').read_text())
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
prefix = 'model.layers.0.'
projections = {'qkv': ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
               'o_proj': ['self_attn.o_proj'], 'gate_up': ['mlp.gate_proj','mlp.up_proj'],
               'down_proj': ['mlp.down_proj']}[a.case]
items = {}
for suffix in ('qweight', 'qzeros', 'scales'):
    parts = []
    for proj in projections:
        key = f'{prefix}{proj}.{suffix}'
        with safe_open(a.model / index[key], framework='pt', device='cpu') as f:
            parts.append(f.get_tensor(key))
    items[suffix] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
k = items['qweight'].shape[0]
n = items['qweight'].shape[1] * 8
assert n % 64 == 0 and k % 128 == 0
cfg = resolve_quantization_config(None, config['quantization_config'])
with torch.device('cuda:0'):
    layer = ColumnParallelLinear(k, n, bias=False, prefix='real_projection', quant_config=cfg,
                                 params_dtype=torch.bfloat16)
for name in ('qweight','qzeros','scales'):
    param = getattr(layer, name)
    param.weight_loader(param, items[name].cuda())
repack_snapshot = {}
if a.repack_compare:
    original_repack = marlin_module.gptq_marlin_repack
    def capture_repack(q, perm, size_k, size_n):
        repack_snapshot['standard'] = q.cpu()
        repack_snapshot['k'], repack_snapshot['n'] = size_k, size_n
        return original_repack(q, perm, size_k, size_n)
    marlin_module.gptq_marlin_repack = capture_repack
with torch.inference_mode():
    process_weights_after_loading(nn.Sequential(layer), None, 'cuda:0')
    torch.manual_seed(122)
    x = (torch.randn((128, k), device='cuda', dtype=torch.bfloat16) * .1).contiguous()
    outputs = {}
    for m in (1, 2, 7, 16, 32, 128):
        outputs[str(m)] = layer(x[:m])[0].cpu()
    trace_summary = None
    if a.trace:
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            layer(x[:7])
            torch.cuda.synchronize()
        trace_path = a.output.with_suffix('.trace.json')
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        names = [event.get('name', '') for event in json.loads(trace_path.read_text())['traceEvents']]
        assert any('minivllm_quant::marlin_gemm' in name for name in names)
        assert any('Marlin<' in name for name in names)
        assert not any('gptq_marlin_repack' in name or name == 'aten::linear' for name in names)
        trace_summary = {'path':str(trace_path), 'local_gemm':True, 'marlin_kernel':True,
                         'repack_in_forward':False, 'aten_linear':False}
    payload = {'k': k, 'n': n, 'weight': layer.qweight.detach().cpu(),
               'zeros': layer.qzeros.detach().cpu(), 'scales': layer.scales.detach().cpu(),
               'bias': torch.zeros(n, dtype=torch.bfloat16), 'x': x.cpu(),
               'empty': torch.empty(0, dtype=torch.int32)}
# 以原 checkpoint 独立构建 FP32 oracle；full-oracle 覆盖全部输出列。
columns = n if a.full_oracle else 16
order = [0,4,1,5,2,6,3,7]
shifts = torch.arange(0,32,4,dtype=torch.int32)
def unpack(t):
    return ((t[:, :columns // 8].unsqueeze(-1) >> shifts) & 15)[:, :, order].reshape(t.shape[0],columns).float()
q = unpack(items['qweight'])
z = unpack(items['qzeros']).repeat_interleave(128, dim=0)
s = items['scales'][:, :columns].float().repeat_interleave(128,dim=0)
w = (q-z)*s
with tempfile.TemporaryDirectory() as d:
    inp, out = Path(d)/'input.pt', Path(d)/'reference.pt'
    if a.repack_compare:
        repack_in, repack_out = Path(d)/'repack_in.pt', Path(d)/'repack_out.pt'
        torch.save(repack_snapshot, repack_in)
        subprocess.run([sys.executable, str(ROOT/'tests/step121/step121_real_repack_reference.py'),
                        '--input',str(repack_in),'--output',str(repack_out)],
                       env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), check=True, timeout=600)
        ref_repack = torch.load(repack_out,map_location='cpu',weights_only=True)
        if not torch.equal(payload['weight'], ref_repack):
            raise AssertionError('真实层完整 repack 与冻结 vLLM 不一致')
        del ref_repack, repack_snapshot['standard']
    torch.save(payload, inp)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(ROOT/'tests/step122/step122_gpu_probe.py'), '--backend','vllm',
                    '--input',str(inp),'--output',str(out)], env=env, check=True, timeout=600)
    reference = torch.load(out, map_location='cpu', weights_only=True)
report = {}
if a.full_oracle:
    torch.backends.cuda.matmul.allow_tf32 = False
    w_gpu = w.cuda()
for key, actual in outputs.items():
    m = int(key)
    oracle = (payload['x'][:m].cuda().float() @ w_gpu).cpu() if a.full_oracle else payload['x'][:m].float() @ w
    torch.testing.assert_close(actual.float(), reference[key].float(), rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(actual[:, :columns].float(), oracle, rtol=2e-2, atol=5e-2)
    report[key] = {'upstream_max_abs': (actual.float()-reference[key].float()).abs().max().item(),
                   'oracle_max_abs': (actual[:,:columns].float()-oracle).abs().max().item()}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({'status':'passed', 'model':str(a.model.resolve()), 'case':a.case,
                                'k':k,'n':n,'m':report,'oracle_columns':columns,'trace':trace_summary, 'repack_exact':a.repack_compare}, indent=2)+'\n')
print(json.dumps({'status':'passed','case':a.case,'k':k,'n':n,'m':report}))
