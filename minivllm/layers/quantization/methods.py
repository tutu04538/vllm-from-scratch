"""量化方法生命周期与尚未开放的执行边界。"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.nn import Parameter


class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer, x, bias=None):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        return None


class LinearMethodBase(QuantizeMethodBase):
    """线性层局部参数和矩阵计算接口；通信仍由线性层完成。"""


class UnquantizedLinearMethod(LinearMethodBase):
    """为当前 rank 创建普通权重，并使用 F.linear 计算。"""
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes), input_size_per_partition,
                                       dtype=params_dtype))
        for name, value in weight_attrs.items():
            setattr(weight, name, value)
        layer.register_parameter('weight', weight)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class PendingQuantizedLinearMethod(LinearMethodBase):
    """仅描述已选方法；后续关卡实现参数和 GPU 计算。"""
    def __init__(self, config, prefix):
        self.config = config
        self.prefix = prefix

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError(
            f"量化执行尚未开放: {self.config.get_name()}，层 {self.prefix}（{self.config.source}）")

    def apply(self, layer, x, bias=None):
        raise NotImplementedError(f"量化执行尚未开放: 层 {self.prefix}")
