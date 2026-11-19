"""量化方法生命周期与尚未开放的执行边界。"""
from abc import ABC, abstractmethod

import torch.nn.functional as F


class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer, x, bias=None):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        return None


class UnquantizedLinearMethod(QuantizeMethodBase):
    """被量化配置明确排除的线性层，沿用现有普通权重。"""
    def create_weights(self, layer, *args, **kwargs):
        return None

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class PendingQuantizedLinearMethod(QuantizeMethodBase):
    """仅描述已选方法；后续关卡实现参数和 GPU 计算。"""
    def __init__(self, config, prefix):
        self.config = config
        self.prefix = prefix

    def create_weights(self, layer, *args, **kwargs):
        raise NotImplementedError(
            f"量化执行尚未开放: {self.config.get_name()}，层 {self.prefix}（{self.config.source}）")

    def apply(self, layer, x, bias=None):
        raise NotImplementedError(f"量化执行尚未开放: 层 {self.prefix}")
