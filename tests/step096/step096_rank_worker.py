"""096 real GPU rank, including hash-matched frozen vLLM differential."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    from minivllm.distributed import (init_distributed_environment, initialize_model_parallel,
        get_tp_group, destroy_model_parallel, destroy_distributed_environment)
    from minivllm.layers.linear import ColumnParallelLinear, RowParallelLinear
    from minivllm.model_loader.auto_weights_loader import AutoWeightsLoader
    init_distributed_environment(world, rank, 'env://', local, timeout=timedelta(seconds=90))
    record = {'rank':rank, 'world':world, 'complete':False}
    try:
        if world > 1:
            for constructor in (lambda: ColumnParallelLinear(4,4), lambda: RowParallelLinear(4,4)):
                try:
                    constructor()
                except RuntimeError as exc:
                    assert 'TP 组尚未初始化' in str(exc)
                else:
                    raise AssertionError('缺失 TP 组不能静默降级')
        initialize_model_parallel(world, 1, timeout=timedelta(seconds=90))
        group = get_tp_group()
        assert group.rank_in_group == rank
        from vllm.model_executor.layers import linear as ref
        from vllm.model_executor import parameter as refparam
        frozen = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0'
        for module, relative in ((ref, 'model_executor/layers/linear.py'),
                                 (refparam, 'model_executor/parameter.py')):
            assert hashlib.sha256(Path(module.__file__).read_bytes()).digest() == hashlib.sha256((frozen / relative).read_bytes()).digest()
        weight = torch.tensor([[1.,1.,1.,1.],[1.,2.,3.,4.],[2.,0.,0.,0.],[0.,0.,0.,1.]])
        bias = torch.tensor([1.,2.,3.,4.])
        x = torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.]], device=f'cuda:{local}')
        # 实际 safetensors 从 CPU 流式读取；每层只加载本 rank 的权重切片。
        from safetensors.torch import save_file
        from minivllm.model_loader.weight_utils import iter_weights
        checkpoint = a.out / f'checkpoint-rank{rank}'
        checkpoint.mkdir(parents=True, exist_ok=True)
        save_file({'weight':weight, 'bias':bias}, str(checkpoint/'model.safetensors'))
        assert all(t.device.type == 'cpu' for _,t in iter_weights(str(checkpoint)))
        for cls in (ColumnParallelLinear, RowParallelLinear):
            with torch.device(f'cuda:{local}'):
                layer = cls(4,4,bias=True)
            assert AutoWeightsLoader(layer).load_weights(iter_weights(str(checkpoint))) == {'weight','bias'}
            n=4//world
            expected = weight[rank*n:(rank+1)*n] if cls is ColumnParallelLinear else weight[:,rank*n:(rank+1)*n]
            assert torch.equal(layer.weight, expected.to(layer.weight.device))
        assert torch.equal(torch.nn.functional.linear(x,weight.to(x.device))[0], torch.tensor([10.,30.,2.,4.],device=x.device))
        if world > 1:
            for constructor in (lambda: ColumnParallelLinear(4,5), lambda: RowParallelLinear(5,4)):
                try:
                    constructor()
                except ValueError:
                    pass
                else:
                    raise AssertionError('不可整除维度必须初始化失败')
        cases = 0
        with ExitStack() as stack:
            for module in (ref, refparam):
                stack.enter_context(patch.object(module, 'get_tensor_model_parallel_rank', return_value=rank))
                stack.enter_context(patch.object(module, 'get_tensor_model_parallel_world_size', return_value=world))
            stack.enter_context(patch.object(ref, 'tensor_model_parallel_all_reduce', side_effect=group.all_reduce))
            stack.enter_context(patch.object(ref, 'tensor_model_parallel_all_gather', side_effect=group.all_gather))
            for cls, reference in ((ColumnParallelLinear, ref.ColumnParallelLinear),
                                   (RowParallelLinear, ref.RowParallelLinear)):
                with torch.device(f'cuda:{local}'):
                    unsharded = cls(4,4,bias=True,disable_tp=True)
                    ref_unsharded = reference(4,4,bias=True,disable_tp=True)
                values = [('weight',weight),('bias',bias)]
                AutoWeightsLoader(unsharded).load_weights(iter(values))
                with torch.no_grad():
                    ref_unsharded.weight.weight_loader(ref_unsharded.weight,weight)
                    ref_unsharded.bias.weight_loader(ref_unsharded.bias,bias)
                assert unsharded.tp_rank == 0 and unsharded.tp_size == 1
                assert tuple(unsharded.weight.shape) == (4,4)
                assert torch.equal(unsharded(x)[0],ref_unsharded(x)[0])
            for gather in (False, True):
                for use_bias in (False, True):
                    for skip in (False, True):
                        for ret in (False, True):
                            with torch.device(f'cuda:{local}'):
                                ours = ColumnParallelLinear(4,4,bias=use_bias,gather_output=gather,
                                      skip_bias_add=skip,return_bias=ret)
                                oracle = ref.ColumnParallelLinear(4,4,bias=use_bias,gather_output=gather,
                                      skip_bias_add=skip,return_bias=ret)
                            values = [('weight', weight)] + ([('bias',bias)] if use_bias else [])
                            assert AutoWeightsLoader(ours).load_weights(iter(values)) == {n for n,_ in values}
                            with torch.no_grad():
                                oracle.weight.weight_loader(oracle.weight,weight)
                                if use_bias: oracle.bias.weight_loader(oracle.bias,bias)
                            n=4//world; assert tuple(ours.weight.shape)==(n,4)
                            assert torch.equal(ours.weight,weight[rank*n:(rank+1)*n].to(x.device))
                            assert torch.equal(ours.weight,oracle.weight)
                            if use_bias: assert torch.equal(ours.bias,oracle.bias)
                            actual, expected = ours(x), oracle(x)
                            if ret:
                                actual, actual_bias = actual; expected, expected_bias = expected
                                assert (actual_bias is None)==(expected_bias is None)
                                if actual_bias is not None: assert torch.equal(actual_bias,expected_bias)
                            assert torch.equal(actual,expected), ('column',rank,world,gather,use_bias,skip,ret,actual,expected)
                            whole = torch.nn.functional.linear(x,weight.to(x.device),bias.to(x.device) if use_bias and not skip else None)
                            target = whole if gather else whole[:,rank*n:(rank+1)*n]
                            assert torch.equal(actual,target)
                            assert ours.weight.untyped_storage().nbytes() == 16 * 4 // world
                            cases += 1
            for parallel in (False, True):
                for reduce in (False, True):
                    for use_bias in (False, True):
                        for skip in (False, True):
                            if not reduce and use_bias and not skip: continue
                            for ret in (False, True):
                                with torch.device(f'cuda:{local}'):
                                    ours = RowParallelLinear(4,4,bias=use_bias,input_is_parallel=parallel,
                                          reduce_results=reduce,skip_bias_add=skip,return_bias=ret)
                                    oracle = ref.RowParallelLinear(4,4,bias=use_bias,input_is_parallel=parallel,
                                          reduce_results=reduce,skip_bias_add=skip,return_bias=ret)
                                values = [('weight',weight)] + ([('bias',bias)] if use_bias else [])
                                assert AutoWeightsLoader(ours).load_weights(iter(values)) == {n for n,_ in values}
                                with torch.no_grad():
                                    oracle.weight.weight_loader(oracle.weight,weight)
                                    if use_bias: oracle.bias.weight_loader(oracle.bias,bias)
                                n=4//world; assert tuple(ours.weight.shape)==(4,n)
                                assert torch.equal(ours.weight,weight[:,rank*n:(rank+1)*n].to(x.device))
                                assert torch.equal(ours.weight,oracle.weight)
                                if use_bias: assert torch.equal(ours.bias,oracle.bias)
                                inp=x[:,rank*n:(rank+1)*n].contiguous() if parallel else x
                                actual, expected=ours(inp),oracle(inp)
                                if ret:
                                    actual,actual_bias=actual;expected,expected_bias=expected
                                    assert (actual_bias is None)==(expected_bias is None)
                                    if actual_bias is not None:assert torch.equal(actual_bias,expected_bias)
                                assert torch.equal(actual,expected),('row',rank,world,parallel,reduce,use_bias,skip,ret,actual,expected)
                                if reduce:
                                    whole=torch.nn.functional.linear(x,weight.to(x.device),bias.to(x.device) if use_bias and not skip else None)
                                    assert torch.equal(actual,whole)
                                else:
                                    local_bias = bias.to(x.device) if use_bias and not skip and rank == 0 else None
                                    part = torch.nn.functional.linear(x[:,rank*n:(rank+1)*n], weight[:,rank*n:(rank+1)*n].to(x.device), local_bias)
                                    assert torch.equal(actual,part)
                                assert ours.weight.untyped_storage().nbytes() == 16 * 4 // world
                                cases += 1
            # BF16 still uses the same local shard and exact integral fixture.
            xbf = x.to(torch.bfloat16)
            for cls, reference, options in (
                (ColumnParallelLinear, ref.ColumnParallelLinear, {'gather_output':True}),
                (RowParallelLinear, ref.RowParallelLinear, {'input_is_parallel':False})):
                with torch.device(f'cuda:{local}'):
                    ours = cls(4,4,bias=True,params_dtype=torch.bfloat16,**options)
                    oracle = reference(4,4,bias=True,params_dtype=torch.bfloat16,**options)
                AutoWeightsLoader(ours).load_weights(iter([('weight',weight),('bias',bias)]))
                with torch.no_grad():
                    oracle.weight.weight_loader(oracle.weight,weight)
                    oracle.bias.weight_loader(oracle.bias,bias)
                assert torch.equal(ours(xbf)[0],oracle(xbf)[0])
                assert torch.equal(ours(xbf)[0],torch.nn.functional.linear(xbf,weight.to(xbf),bias.to(xbf)))
                cases += 1
        if world == 2:
            # 需求中的原样 2×4 手算例：Column 各一行，Row 各两列。
            small_weight = weight[:2]
            single = x[:1]
            with torch.device(f'cuda:{local}'):
                small_column = ColumnParallelLinear(4,2,bias=False)
                small_row = RowParallelLinear(4,2,bias=False,input_is_parallel=False,reduce_results=False)
            small_column.weight_loader(small_column.weight,small_weight)
            small_row.weight_loader(small_row.weight,small_weight)
            column_part = small_column(single)[0]
            row_part = small_row(single)[0]
            assert torch.equal(column_part,torch.tensor([[10.] if rank==0 else [30.]],device=single.device))
            assert torch.equal(row_part,torch.tensor([[3.,5.] if rank==0 else [7.,25.]],device=single.device))
            assert torch.equal(group.all_gather(column_part,-1),torch.tensor([[10.,30.]],device=single.device))
            assert torch.equal(group.all_reduce(row_part),torch.tensor([[10.,30.]],device=single.device))
        record.update(complete=True,cases=cases,tp_rank=group.rank_in_group,
                      device_uuid=str(torch.cuda.get_device_properties(local).uuid),
                      column_local_elements=16//world,row_local_elements=16//world,
                      real_safetensors_cpu_stream=True, missing_tp_group_rejected=world>1,
                      nondivisible_rejected=world>1, bf16_oracle=True, exact_step096_fixture=world==2)
    except BaseException:
        record['error']=traceback.format_exc()
        raise
    finally:
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/f'rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        destroy_model_parallel();destroy_distributed_environment()
    return 0

if __name__ == '__main__':
    sys.exit(main())
