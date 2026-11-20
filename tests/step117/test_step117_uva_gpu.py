"""量化后处理替换 UVA 参数时的 CUDA 后端选择与 host 内存所有权。"""
import gc
import weakref
import pytest
import torch
from torch import nn

from minivllm.model_loader.post_load import device_loading_context
from minivllm import uva_ext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='需要真实 CUDA UVA')


def test_replaced_uva_parameter_keeps_pinned_owner_and_default_cpp(monkeypatch):
    monkeypatch.delenv('MINIVLLM_UVA_BACKEND', raising=False)
    assert uva_ext.resolve_backend() == 'cpp'
    original = torch.arange(8, dtype=torch.float32).pin_memory()
    view = uva_ext.get_cuda_view_from_cpu_tensor(original)
    layer = nn.Module()
    layer.weight = nn.Parameter(view, requires_grad=False)
    layer.weight._vllm_is_uva_offloaded = True
    # 与常规 UVA buffer 一样，初始 owner 显式保持存活。
    layer.weight._vllm_uva_cpu_data = original
    calls=[]
    real_get = uva_ext.get_cuda_view_from_cpu_tensor
    def monitored(cpu):
        calls.append(cpu)
        return real_get(cpu)
    monkeypatch.setattr(uva_ext, 'get_cuda_view_from_cpu_tensor', monitored)
    expected=torch.arange(8,device='cuda',dtype=torch.float32)+10
    with device_loading_context(layer, torch.device('cuda:0')):
        layer.weight=nn.Parameter(expected,requires_grad=False)
    assert len(calls) == 1
    assert layer.weight._vllm_is_uva_offloaded
    owner=layer.weight._vllm_uva_cpu_data
    assert owner.is_pinned() and owner.device.type == 'cpu'
    del expected
    gc.collect()
    torch.testing.assert_close(layer.weight.cpu(),torch.arange(8,dtype=torch.float32)+10)
    owner[0] = 99
    torch.cuda.synchronize()
    assert layer.weight[0].item() == 99
    owner_ref = weakref.ref(owner)
    calls.clear()
    del owner, layer
    gc.collect()
    assert owner_ref() is None  # 不把 pinned 内存永久放进全局 keepalive。


def test_cpu_parameter_restored_after_gpu_processing_failure():
    layer=nn.Linear(4,4,bias=False)
    with pytest.raises(RuntimeError,match='injected'):
        with device_loading_context(layer,torch.device('cuda:0')):
            assert layer.weight.device.type == 'cuda'
            raise RuntimeError('injected')
    assert layer.weight.device.type == 'cpu'


def test_reoffload_never_returns_wrong_device(monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip('该用例需要两张 CUDA GPU')
    monkeypatch.delenv('MINIVLLM_UVA_BACKEND', raising=False)
    torch.cuda.set_device(0)
    owner=torch.arange(8,dtype=torch.float32).pin_memory()
    layer=nn.Module()
    layer.weight=nn.Parameter(uva_ext.get_cuda_view_from_cpu_tensor(owner),requires_grad=False)
    layer.weight._vllm_is_uva_offloaded=True
    layer.weight._vllm_uva_cpu_data=owner
    try:
        with device_loading_context(layer,torch.device('cuda:1')):
            layer.weight=nn.Parameter(torch.arange(8,device='cuda:1',dtype=torch.float32)+20,requires_grad=False)
    except RuntimeError as exc:
        assert 'UVA 视图设备' in str(exc) and '不一致' in str(exc)
        assert torch.cuda.current_device() == 0
        return  # 该环境明确拒绝，不能记为跨设备 UVA 已支持。
    assert layer.weight.device == torch.device('cuda:1')
    assert torch.cuda.current_device() == 0
    assert layer.weight._vllm_uva_cpu_data.is_pinned()
    torch.testing.assert_close(layer.weight.cpu(),torch.arange(8,dtype=torch.float32)+20)
