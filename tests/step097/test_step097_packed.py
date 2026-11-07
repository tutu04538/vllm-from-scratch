"""097 slicing, completeness, unsupported Gemma, and true GPU tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from minivllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('step097_acceptance',ROOT/'benchmarks/check_step097_acceptance.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('rank,kv_head',[(0,0),(1,0),(2,1),(3,1)])
def test_replication_rank_mapping(rank,kv_head,monkeypatch):
    monkeypatch.setattr('minivllm.distributed.parallel_state.get_tp_group_if_initialized',
                        lambda:SimpleNamespace(rank_in_group=rank,world_size=4))
    layer=QKVParallelLinear(3,2,8,2,bias=False)
    q=torch.arange(16*3,dtype=torch.float32).view(16,3)
    k=100+torch.arange(4*3,dtype=torch.float32).view(4,3)
    v=200+torch.arange(4*3,dtype=torch.float32).view(4,3)
    layer.weight_loader(layer.weight,torch.cat([q,k,v]))
    expected=torch.cat([q[rank*4:(rank+1)*4],k[kv_head*2:(kv_head+1)*2],v[kv_head*2:(kv_head+1)*2]])
    assert torch.equal(layer.weight,expected)
    assert (layer.num_heads,layer.num_kv_heads,layer.num_kv_head_replicas)==(2,1,2)
    layer.validate_loaded_shards({'q','k','v'})


def test_packed_merged_shards_and_failures(monkeypatch):
    monkeypatch.setattr('minivllm.distributed.parallel_state.get_tp_group_if_initialized',
                        lambda:SimpleNamespace(rank_in_group=1,world_size=2))
    layer=MergedColumnParallelLinear(2,[4,4],bias=True)
    gate=torch.tensor([[10.,10.],[11.,11.],[12.,12.],[13.,13.]])
    up=torch.tensor([[20.,20.],[21.,21.],[22.,22.],[23.,23.]])
    layer.weight_loader(layer.weight,torch.cat([gate,up]))
    assert torch.equal(layer.weight,torch.cat([gate[2:],up[2:]]))
    with pytest.raises(ValueError,match='bias.*缺少'):layer.validate_loaded_shards({0,1})
    layer.weight_loader(layer.bias,torch.arange(8.))
    assert torch.equal(layer.bias,torch.tensor([2.,3.,6.,7.]))
    layer.validate_loaded_shards({0,1})
    layer.weight_loader(layer.bias,torch.arange(4.),0)
    with pytest.raises(ValueError,match='bias.*重复'):layer.validate_loaded_shards({0,1})
    layer.weight_loader(layer.weight,gate,0)
    with pytest.raises(ValueError,match='重复'):layer.validate_loaded_shards({0,1})
    layer.reset_load_state()
    layer.weight_loader(layer.weight,gate,0)
    with pytest.raises(ValueError,match='缺少'):layer.validate_loaded_shards({0,1})
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,up,2)
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,torch.zeros(7,2))
    with pytest.raises(ValueError):MergedColumnParallelLinear(2,[4,3])
    with pytest.raises(ValueError):QKVParallelLinear(2,2,7,2)
    with pytest.raises(ValueError):QKVParallelLinear(2,2,8,3)


def test_qkv_missing_duplicate_bias_and_invalid():
    layer=QKVParallelLinear(3,2,4,2,bias=True)
    q=torch.ones(8,3);k=torch.ones(4,3);v=torch.ones(4,3)
    layer.weight_loader(layer.weight,q,'q')
    layer.weight_loader(layer.bias,torch.ones(8),'q')
    with pytest.raises(ValueError,match='缺少'):layer.validate_loaded_shards({'q','k','v'})
    layer.weight_loader(layer.weight,k,'k');layer.weight_loader(layer.weight,v,'v')
    with pytest.raises(ValueError,match='bias.*缺少'):layer.validate_loaded_shards({'q','k','v'})
    layer.weight_loader(layer.bias,torch.ones(4),'k');layer.weight_loader(layer.bias,torch.ones(4),'v')
    layer.validate_loaded_shards({'q','k','v'})
    layer.weight_loader(layer.bias,torch.ones(4),'k')
    with pytest.raises(ValueError,match='bias.*重复'):layer.validate_loaded_shards({'q','k','v'})
    layer.weight_loader(layer.weight,q,'q')
    with pytest.raises(ValueError,match='重复'):layer.validate_loaded_shards({'q','k','v'})
    layer.reset_load_state()
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,torch.ones(15,3))
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,k,'z')
    with pytest.raises(ValueError):layer.weight_loader(layer.weight,torch.ones(3,3),'k')


def test_gemma_tp_rejected_at_configuration(monkeypatch):
    from minivllm.config import ModelConfig, DeviceConfig, ParallelConfig, VllmConfig
    monkeypatch.setenv('VLLM_USE_V2_MODEL_RUNNER','1')
    hf={'model_type':'gemma4_text','num_attention_heads':8,'num_key_value_heads':2,
        'num_hidden_layers':4}
    model=ModelConfig('unused',hf_config=hf,enforce_eager=True)
    with pytest.raises(NotImplementedError,match='Gemma'):
        VllmConfig(model,device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2))


@pytest.mark.parametrize('case',list(runner.CASES))
def test_true_gpu_and_frozen_or_real(case,tmp_path):
    result=runner.execute(case,tmp_path,180)
    assert result['passed'],Path(result['log']).read_text()[-15000:]
    assert len(result['ranks'])==result['world']


def test_illegal_replication_factor_rejected(monkeypatch):
    monkeypatch.setattr('minivllm.distributed.parallel_state.get_tp_group_if_initialized',
                        lambda:SimpleNamespace(rank_in_group=0,world_size=4))
    # KV=3<TP=4，无法让每个 KV 头拥有同样的整数复制因子。
    with pytest.raises(ValueError):
        QKVParallelLinear(4,2,8,3)
    with pytest.raises(ValueError):
        QKVParallelLinear(4,2,6,2)
