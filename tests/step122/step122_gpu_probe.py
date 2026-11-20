"""独立子进程消费同一 packed 输入，运行本地或冻结 vLLM GEMM。"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
p = argparse.ArgumentParser()
p.add_argument('--backend', choices=('mini', 'vllm'), required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
d = torch.load(args.input, map_location='cpu', weights_only=True)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA 不可用')
items = {key: value.cuda() for key, value in d.items() if isinstance(value, torch.Tensor)}
k, n = d['k'], d['n']
workspace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count, device='cuda', dtype=torch.int32)
if args.backend == 'mini':
    from minivllm.quant_ops.ops import marlin_gemm_w4a16, prepare_full_extension
    prepare_full_extension()
    def run(x):
        return marlin_gemm_w4a16(x, items['weight'], items['scales'], items['zeros'],
                                items['empty'], items['empty'], workspace, size_n=n, size_k=k,
                                bias=items['bias'])
else:
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types
    import hashlib
    import vllm.scalar_type as vllm_scalar
    frozen = Path(__file__).resolve().parents[2] / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm'
    for installed, source in ((ops.__file__, frozen / '_custom_ops.py'),
                              (vllm_scalar.__file__, frozen / 'scalar_type.py')):
        if hashlib.sha256(Path(installed).read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise RuntimeError(f'安装的 vLLM 与冻结源码不一致: {source.name}')
    def run(x):
        return ops.marlin_gemm(x, None, items['weight'], items['bias'], items['scales'],
                               None, None, items['zeros'], items['empty'], items['empty'],
                               workspace, scalar_types.uint4, x.shape[0], n, k, True, False, True, False)
outputs = {}
for m in (1, 2, 7, 16, 32, 128):
    outputs[str(m)] = run(items['x'][:m]).cpu()
if args.backend == 'mini':
    # 不同 M 连续交错；同步拿到输出用于检查历史状态污染。
    for iteration in range(100):
        m = (1, 2, 7, 16, 32, 128)[iteration % 6]
        torch.testing.assert_close(run(items['x'][:m]).cpu(), outputs[str(m)], rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stream_out = run(items['x'][:7])
        finished = stream.record_event()
    torch.cuda.current_stream().wait_event(finished)
    torch.testing.assert_close(stream_out.cpu(), outputs['7'], rtol=0, atol=0)
    try:
        marlin_gemm_w4a16(items['x'][:1], items['weight'], items['scales'], items['zeros'],
                          items['empty'], items['empty'], workspace[:1], size_n=n, size_k=k)
    except ValueError as exc:
        assert '容量' in str(exc)
    else:
        raise AssertionError('不足容量 workspace 没有被拒绝')
torch.cuda.synchronize()
torch.save(outputs, args.output)
