import types
import ast
from pathlib import Path

import pytest
import torch

from minivllm.attention.metadata import AttentionMetadata
from minivllm.forward_context import set_forward_context
from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.models.utils import PPMissingLayer, get_pp_indices
from minivllm.testing.tiny_models import _weights, tiny_qwen3_config
import minivllm.distributed.parallel_state as parallel


def model(monkeypatch, config, rank, size, weights):
    monkeypatch.setattr(parallel, "_PP", types.SimpleNamespace(rank_in_group=rank, world_size=size))
    result = Qwen3ForCausalLM(config, prefix="")
    result.load_weights(iter(sorted(weights.items())))
    return result


def run(model, ids, positions, intermediate=None):
    metadata = AttentionMetadata(torch.tensor([0, len(ids)], dtype=torch.int32),
        torch.tensor([len(ids)], dtype=torch.int32), torch.tensor([[0]], dtype=torch.int32),
        torch.arange(len(ids)), 16, 1)
    attn = {}
    for layer in model.model.layers[model.model.start_layer:model.model.end_layer]:
        a = layer.self_attn.attn
        a.kv_cache = torch.zeros(2, 1, 16, a.num_kv_heads, a.head_size)
        attn[a.layer_name] = metadata
    with torch.inference_mode(), set_forward_context(attn, num_tokens=len(ids)):
        return model(ids, positions, intermediate)


@pytest.mark.parametrize("layers,size,counts", [(4,2,[2,2]), (5,2,[3,2]),
                                                (4,4,[1,1,1,1]), (5,4,[1,1,2,1])])
def test_partitions(layers, size, counts, monkeypatch):
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    pairs = [get_pp_indices(layers, rank, size) for rank in range(size)]
    assert [end-start for start,end in pairs] == counts
    assert pairs[0][0] == 0 and pairs[-1][1] == layers


def test_explicit_partitions(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,3,1")
    assert [get_pp_indices(5, i, 3) for i in range(3)] == [(0,1),(1,4),(4,5)]
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,3")
    with pytest.raises(ValueError): get_pp_indices(5, 0, 3)


def test_frozen_partition_helper(monkeypatch):
    path = (Path(__file__).resolve().parents[2] / "learning_notes/14_vllm_from_scratch/"
            "分布式并行完整需求/参考资料/vllm_0_28_0/distributed/utils.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_pp_indices")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    env = types.SimpleNamespace(VLLM_PP_LAYER_PARTITION=None)
    namespace = {"envs": env, "logger": types.SimpleNamespace(info=lambda *args: None)}
    exec(compile(module, str(path), "exec"), namespace)
    frozen = namespace["get_pp_indices"]
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    for layers in (0, 2, 4, 5):
        for size in (1, 2, 4):
            for rank in range(size):
                assert get_pp_indices(layers, rank, size) == frozen(layers, rank, size)
    env.VLLM_PP_LAYER_PARTITION = "1,3,1"
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", env.VLLM_PP_LAYER_PARTITION)
    assert [get_pp_indices(5, i, 3) for i in range(3)] == [frozen(5, i, 3) for i in range(3)]
    with pytest.raises(ValueError): get_pp_indices(5, 3, 3)


@pytest.mark.parametrize("layers,size", [(4,2), (5,2), (4,4), (5,4)])
@pytest.mark.parametrize("tied", [False, True])
def test_split_matches_full(monkeypatch, layers, size, tied):
    config = tiny_qwen3_config("tiny_gqa")
    config.update(num_hidden_layers=layers, tie_word_embeddings=tied)
    weights = _weights(config, 7)
    full = model(monkeypatch, config, 0, 1, weights)
    stages = [model(monkeypatch, config, rank, size, weights) for rank in range(size)]
    ids = torch.tensor([1,2,3]); positions = torch.arange(3)
    boundaries = {}
    hooks = []
    for stage in stages[:-1]:
        index = stage.model.end_layer - 1
        def save_boundary(module, args, output, index=index):
            boundaries[index] = tuple(value.clone() for value in output)
        hooks.append(full.model.layers[index].register_forward_hook(save_boundary))
    expected = run(full, ids, positions)
    for hook in hooks: hook.remove()
    intermediate = None
    for rank, stage in enumerate(stages):
        intermediate = run(stage, ids, positions, intermediate)
        if rank < size - 1:
            assert set(intermediate.tensors) == {"hidden_states", "residual"}
            assert intermediate["hidden_states"].shape == intermediate["residual"].shape == (3, config["hidden_size"])
            assert intermediate["residual"].dtype == torch.float32
            reference_hidden, reference_residual = boundaries[stage.model.end_layer - 1]
            torch.testing.assert_close(intermediate["hidden_states"], reference_hidden, rtol=0, atol=0)
            torch.testing.assert_close(intermediate["residual"], reference_residual, rtol=0, atol=0)
    torch.testing.assert_close(intermediate, expected, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(stages[-1].compute_logits(intermediate), full.compute_logits(expected), rtol=0, atol=0)
    for rank, stage in enumerate(stages):
        start,end = stage.model.start_layer,stage.model.end_layer
        assert all(isinstance(layer, PPMissingLayer) == (not start <= i < end)
                   for i,layer in enumerate(stage.model.layers))
        assert isinstance(stage.lm_head, PPMissingLayer) == (rank != size - 1)
    if tied:
        assert stages[-1].lm_head.weight is stages[-1].model.embed_tokens.weight
        assert stages[0].model.embed_tokens.weight is not stages[-1].model.embed_tokens.weight
        torch.testing.assert_close(stages[0].model.embed_tokens.weight,
                                   stages[-1].model.embed_tokens.weight, rtol=0, atol=0)


def test_residual_and_local_missing(monkeypatch):
    config = tiny_qwen3_config("tiny_gqa"); config["num_hidden_layers"] = 4
    weights = _weights(config, 9)
    first = model(monkeypatch, config, 0, 2, weights)
    last = model(monkeypatch, config, 1, 2, weights)
    ids = torch.tensor([1,2,3]); positions = torch.arange(3)
    boundary = run(first, ids, positions)
    expected = run(last, ids, positions, boundary)
    boundary["residual"] = torch.zeros_like(boundary["residual"])
    wrong = run(last, ids, positions, boundary)
    assert not torch.allclose(wrong, expected, rtol=1e-4, atol=1e-4)
    missing = {name: value for name,value in weights.items() if name != "model.layers.2.self_attn.q_norm.weight"}
    with pytest.raises(ValueError, match="没有被加载"):
        model(monkeypatch, config, 1, 2, missing)
    model(monkeypatch, config, 0, 2, missing)
    empty = last.make_empty_intermediate_tensors(7, torch.float32, "cpu")
    assert empty["hidden_states"].shape == empty["residual"].shape == (7, config["hidden_size"])
