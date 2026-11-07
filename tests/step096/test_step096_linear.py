"""096 semantic boundaries and independently launched GPU ranks."""
import importlib.util
from pathlib import Path
import pytest
import torch
from minivllm.layers.linear import ColumnParallelLinear, RowParallelLinear

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('step096_acceptance',ROOT/'benchmarks/check_step096_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_single_gpu_free_compatibility_and_bias():
    w=torch.tensor([[1.,1.,1.,1.],[1.,2.,3.,4.]])
    x=torch.tensor([[1.,2.,3.,4.]])
    for cls in (ColumnParallelLinear,RowParallelLinear):
        layer=cls(4,2,bias=True,skip_bias_add=True)
        layer.weight_loader(layer.weight,w);layer.weight_loader(layer.bias,torch.tensor([1.,2.]))
        y,b=layer(x)
        assert torch.equal(y,torch.tensor([[10.,30.]]))
        assert torch.equal(b,torch.tensor([1.,2.]))
    with pytest.raises(ValueError):RowParallelLinear(4,2,bias=True,reduce_results=False)
    with pytest.raises(NotImplementedError):ColumnParallelLinear(4,2,quant_config=object())
    with pytest.raises(ValueError):ColumnParallelLinear(0,2)
    with pytest.raises(ValueError):RowParallelLinear(4,0)
    layer=ColumnParallelLinear(4,2)
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,torch.ones(3,4))
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,torch.ones(2,4),'q')
    class SpecialParameter(torch.nn.Parameter):
        pass
    with pytest.raises(NotImplementedError):layer.weight_loader(SpecialParameter(torch.empty(2,4)),torch.ones(2,4))
    assert ColumnParallelLinear(4,2,params_dtype=torch.float64).weight.dtype == torch.float64
    assert RowParallelLinear(4,2,params_dtype=torch.float64).weight.dtype == torch.float64


@pytest.mark.parametrize('world',[1,2,4])
def test_real_cuda_frozen_vllm(world,tmp_path):
    value=runner.execute(world,tmp_path,180)
    assert value['passed'],Path(value['log']).read_text()[-20000:]
    assert len(value['ranks'])==world
