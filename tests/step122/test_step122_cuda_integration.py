"""真实 CUDA 的 AWQ 加载、Marlin 重排、局部 GEMM 和 Graph；不开放模型推理。"""
import pytest
import torch
from torch import nn

from minivllm.layers.linear import ColumnParallelLinear
from minivllm.layers.quantization import resolve_quantization_config
from minivllm.model_loader.post_load import process_weights_after_loading

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='需真实 CUDA；不算 GPU 通过')


def make_layer(n=72, skip_bias_add=False, dtype=torch.bfloat16):
    k, group = 256, 128
    generator = torch.Generator().manual_seed(1228 + n)
    q = torch.randint(0, 16, (k, n), dtype=torch.int32, generator=generator)
    z = torch.randint(0, 16, (k // group, n), dtype=torch.int32, generator=generator)
    scales = (torch.rand((k // group, n), generator=generator) * .02 + .01).to(dtype)
    bias = (torch.randn(n, generator=generator) * .1).to(dtype)
    # 独立构造 AWQ checkpoint：每个 32-bit word 按 AWQ 物理位序存列。
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    def pack(values):
        return (values.reshape(values.shape[0], n // 8, 8)[:, :, order] << shifts).sum(-1, dtype=torch.int32)
    config = resolve_quantization_config(None, {'quant_method': 'awq', 'bits': 4, 'group_size': 128,
                                                'zero_point': True, 'version': 'gemm'})
    with torch.device('cuda:0'):
        layer = ColumnParallelLinear(k, n, bias=True, prefix='test', quant_config=config,
                                     params_dtype=dtype, skip_bias_add=skip_bias_add)
    for name, value in [('qweight', pack(q)), ('qzeros', pack(z)), ('scales', scales)]:
        param = getattr(layer, name)
        param.weight_loader(param, value.to('cuda'))
    with torch.no_grad():
        layer.bias.copy_(bias.to('cuda'))
    process_weights_after_loading(nn.Sequential(layer), None, 'cuda:0')
    assert layer.quant_method.post_load_layout == 'kernel'
    assert all(getattr(layer, p).layout_state == 'kernel' for p in ('qweight','qzeros','scales'))
    assert layer.g_idx.numel() == layer.g_idx_sort_indices.numel() == 0
    assert layer.g_idx.is_cuda and layer.g_idx_sort_indices.is_cuda
    weight_fp32 = (q.float() - z.repeat_interleave(group, dim=0).float()) * scales.repeat_interleave(group, dim=0).float()
    return layer, weight_fp32, bias


@pytest.mark.parametrize('n', [64, 72, 128])
@pytest.mark.parametrize('skip_bias_add', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_real_awq_to_local_forward_and_bias(n, skip_bias_add, dtype):
    layer, oracle_weight, bias = make_layer(n, skip_bias_add, dtype)
    generator = torch.Generator().manual_seed(456)
    for m in (1, 2, 7, 16, 32, 128):
        x = (torch.randn((m, 256), generator=generator) * .3).to(dtype).cuda()
        with torch.inference_mode():
            output, returned_bias = layer(x)
        oracle = x.cpu().float() @ oracle_weight
        if skip_bias_add:
            assert returned_bias.shape == (n,)
            torch.testing.assert_close(returned_bias.cpu(), bias, rtol=0, atol=0)
        else:
            assert returned_bias is None
            oracle += bias.float()
        assert output.shape == (m, n)
        torch.testing.assert_close(output.cpu().float(), oracle, rtol=2e-2, atol=5e-2)


def test_two_layers_workspace_and_graph_replay():
    first, oracle_weight, bias = make_layer()
    second, _, _ = make_layer()
    assert first.quant_method.workspace.data_ptr() != second.quant_method.workspace.data_ptr()
    w_address = first.qweight.data_ptr()
    ws_address = first.quant_method.workspace.data_ptr()
    torch.manual_seed(122)
    x = torch.randn((7, 256), device='cuda', dtype=torch.bfloat16) * .3
    with torch.inference_mode():
        first(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, _ = first(x)
        graph.replay()
        for i in range(3):
            x.copy_(torch.randn_like(x) * .3)
            second(x)
            graph.replay()
            torch.cuda.synchronize()
            oracle = x.cpu().float() @ oracle_weight + bias.float()
            torch.testing.assert_close(output.cpu().float(), oracle, rtol=2e-2, atol=5e-2)
            assert first.qweight.data_ptr() == w_address
            assert first.quant_method.workspace.data_ptr() == ws_address


def test_profile_uses_local_gemm_without_repack(tmp_path):
    import json
    layer, _, _ = make_layer()
    x = torch.randn((7, 256), device='cuda', dtype=torch.bfloat16)
    with torch.inference_mode():
        layer(x)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            layer(x)
            torch.cuda.synchronize()
    path = tmp_path / 'step122_trace.json'
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    names = [event.get('name', '') for event in trace['traceEvents']]
    assert any('minivllm_quant::marlin_gemm' in name for name in names), names
    assert any('Marlin<' in name for name in names), names
    assert not any('gptq_marlin_repack' in name for name in names), names
