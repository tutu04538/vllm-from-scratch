"""Linear method 的基本构造与加载生命周期；GPU 差分另行执行。"""
import pytest
import torch
from torch import nn

from minivllm.layers.linear import (ReplicatedLinear, ColumnParallelLinear, RowParallelLinear,
                                    QKVParallelLinear, MergedColumnParallelLinear)
from minivllm.layers.quantization.methods import UnquantizedLinearMethod
from minivllm.model_loader.base_loader import BaseModelLoader


@pytest.mark.parametrize('layer', [
    lambda: ReplicatedLinear(8, 12),
    lambda: ColumnParallelLinear(8, 12),
    lambda: RowParallelLinear(8, 12),
    lambda: QKVParallelLinear(8, 2, 4, 2),
    lambda: MergedColumnParallelLinear(8, [4, 8]),
])
def test_method_creates_rank_local_parameter(layer):
    linear = layer()
    assert isinstance(linear.quant_method, UnquantizedLinearMethod)
    assert linear.weight.weight_loader.__self__ is linear
    assert linear.weight.shape[1] == 8


def test_loader_postprocess_failure_does_not_open_forward():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ReplicatedLinear(2, 2)

        def load_weights(self, weights):
            with pytest.raises(RuntimeError, match='尚未完成加载'):
                self.proj(torch.ones(1, 2))
            return {'proj.weight'}

    model = Model()
    def fail(layer):
        raise ValueError('postprocess failure')
    model.proj.quant_method.process_weights_after_loading = fail

    class Loader(BaseModelLoader):
        def _initialize_model(self, model_config, device):
            return model
        def get_all_weights(self, model):
            return iter(())

    with pytest.raises(ValueError, match='postprocess failure'):
        Loader().load_model(None, 'cpu')
    with pytest.raises(RuntimeError, match='尚未完成加载'):
        model.proj(torch.ones(1, 2))
