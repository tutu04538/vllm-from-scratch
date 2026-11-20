"""真实 CUDA 的 scale/zero/padding 与冻结源码 CPU 参考逐值比较。"""
import ast
from pathlib import Path
import numpy
import pytest
import torch
from minivllm.layers.quantization.marlin import (marlin_pad_qweight, marlin_pad_scales,
    marlin_padded_nk, marlin_permute_scales, marlin_zero_points)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='需真实 CUDA')
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/model_executor/layers/quantization/utils/marlin_utils.py'


def frozen():
    tree = ast.parse(SOURCE.read_text())
    wanted = {'get_scale_perms', 'marlin_permute_scales', 'marlin_zero_points'}
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in wanted], type_ignores=[])
    def pack_cols(q,bits,groups,n):
        a=q.numpy().astype(numpy.uint32)
        result=numpy.zeros((groups,n//8),dtype=numpy.uint32)
        for i in range(8):
            result |= a[:,i::8] << (4*i)
        return torch.from_numpy(result.view(numpy.int32).copy())
    env={'torch':torch,'numpy':numpy,'pack_cols':pack_cols}
    exec(compile(module,str(SOURCE),'exec'),env)
    return env


@pytest.mark.parametrize('k,n', [(128,64),(256,72),(256,128),(384,192)])
def test_gpu_each_intermediate_exact(k,n):
    pn,pk = marlin_padded_nk(n,k,128)
    groups=k//128
    q=torch.arange(k//8*n,dtype=torch.int32).reshape(k//8,n)
    z=(torch.arange(groups*n,dtype=torch.int32).reshape(groups,n)*7+5)%16
    s=(torch.arange(groups*n,dtype=torch.float32).reshape(groups,n)/19).to(torch.bfloat16)
    pq=marlin_pad_qweight(q.cuda(),n,k,pn,pk)
    pz=marlin_pad_scales(z.cuda(),n,k,pn,pk,128)
    ps=marlin_pad_scales(s.cuda(),n,k,pn,pk,128)
    torch.testing.assert_close(pq[:,:n].cpu(),q,rtol=0,atol=0)
    torch.testing.assert_close(pz[:,:n].cpu(),z,rtol=0,atol=0)
    torch.testing.assert_close(ps[:,:n].cpu(),s,rtol=0,atol=0)
    assert torch.count_nonzero(pq[:,n:]) == 0
    assert torch.count_nonzero(pz[:,n:]) == 0
    assert torch.count_nonzero(ps[:,n:]) == 0
    upstream=frozen()
    expected_z=upstream['marlin_zero_points'](pz.cpu(),pk//128,pn,4)
    expected_s=upstream['marlin_permute_scales'](ps.cpu(),pk,pn,128)
    torch.testing.assert_close(marlin_zero_points(pz,pn).cpu(),expected_z,rtol=0,atol=0)
    torch.testing.assert_close(marlin_permute_scales(ps,pk,pn,128).cpu(),expected_s,rtol=0,atol=0)
