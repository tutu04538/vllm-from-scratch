"""加载后按冻结上游顺序处理线性方法、attention 和模型级 hook。"""
from contextlib import contextmanager

import torch

from ..layers.parameter import BasevLLMParameter
from ..layers.quantization.methods import QuantizeMethodBase


@contextmanager
def device_loading_context(module, target_device):
    target_device = torch.device(target_device)
    if target_device.type == 'cpu':
        yield
        return
    original = {}
    uva_offloaded = set()
    try:
        for name, param in module.named_parameters():
            if getattr(param, '_vllm_is_uva_offloaded', False):
                uva_offloaded.add(name)
            if param.device.type == 'cpu':
                original[name] = param.device
                param.data = param.data.to(target_device)
        yield
    finally:
        for name, param in module.named_parameters():
            if name in original:
                param.data = param.data.to(original[name])
            if name in uva_offloaded and not getattr(param, '_vllm_is_uva_offloaded', False):
                from ..utils.platform_utils import is_pin_memory_available
                from ..utils.torch_utils import get_accelerator_view_from_cpu_tensor
                cpu_data = param.data.to(device='cpu')
                if is_pin_memory_available():
                    cpu_data = cpu_data.pin_memory()
                param.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                param._vllm_is_uva_offloaded = True


def process_weights_after_loading(model, model_config, target_device):
    from ..attention.layer import Attention
    state = getattr(model, '_quant_postload_state', None)
    if state not in (None, 'loading'):
        raise RuntimeError(f'模型后处理只能执行一次，当前状态: {state}')
    model._quant_postload_state = 'processing'
    layers = [module for module in model.modules() if isinstance(
        getattr(module, 'quant_method', None), QuantizeMethodBase)]
    for module in layers:
        if hasattr(module, '_method_weights_ready'):
            module._method_weights_ready = False
    try:
        for module in layers:
            for param in module.parameters(recurse=False):
                if isinstance(param, BasevLLMParameter):
                    param.validate_loaded()
            original_params = dict(module.named_parameters(recurse=False))
            with device_loading_context(module, target_device):
                module.quant_method.process_weights_after_loading(module)
                for name, new_param in module.named_parameters(recurse=False):
                    old_param = original_params.get(name)
                    if old_param is not None and new_param is not old_param:
                        # 重排可能替换 Parameter；仍需重载时沿用其原加载器。
                        if not hasattr(new_param, 'weight_loader') and hasattr(old_param, 'weight_loader'):
                            new_param.weight_loader = old_param.weight_loader
                        if isinstance(old_param, BasevLLMParameter) and isinstance(new_param, BasevLLMParameter):
                            if not new_param._loaded_shards:
                                new_param._loaded_shards = old_param._loaded_shards.copy()
                            if new_param.required_shards is None:
                                new_param.required_shards = old_param.required_shards
            if hasattr(module, 'update_param_tp_status'):
                module.update_param_tp_status()
            else:
                for param in module.parameters(recurse=False):
                    if isinstance(param, BasevLLMParameter):
                        param.update_tp(getattr(module, 'tp_rank', 0), getattr(module, 'tp_size', 1))
        for module in model.modules():
            if isinstance(module, Attention) and callable(getattr(module, 'process_weights_after_loading', None)):
                with device_loading_context(module, target_device):
                    module.process_weights_after_loading(model_config.dtype)
        hook = getattr(model, 'process_weights_after_loading', None)
        if callable(hook):
            hook()
        for module in layers:
            for param in module.parameters(recurse=False):
                if isinstance(param, BasevLLMParameter):
                    expected = getattr(module.quant_method, 'post_load_layout', 'kernel')
                    if param.layout_state != expected:
                        raise RuntimeError(f'{param.logical_name}: 量化方法未完成 {expected} 布局转换')
        for module in layers:
            if hasattr(module, '_method_weights_ready'):
                module._method_weights_ready = True
        model._quant_postload_state = ('representation_ready' if any(
            not getattr(m.quant_method, 'supports_execution', True) for m in layers) else 'ready')
    except BaseException:
        model._quant_postload_state = 'failed'
        raise
