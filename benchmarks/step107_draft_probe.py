"""TP draft FC and mapped logits on an already recorded 106 feature fixture."""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    from minivllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                      destroy_model_parallel, destroy_distributed_environment)
    from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
    init_distributed_environment(world, rank, 'env://', local, timeout=timedelta(seconds=180))
    try:
        initialize_model_parallel(world, 1, timeout=timedelta(seconds=180))
        config = json.loads((args.draft / 'config.json').read_text())
        weights = load_file(str(args.draft / 'model.safetensors'))
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('cuda'):
            model = Eagle3ForCausalLM(config)
        loaded = model.load_weights(iter(sorted(weights.items())))
        model.to('cuda').eval()
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
        with torch.inference_mode():
            combined = model.combine_hidden_states(fixture['aux'].to('cuda'))
            logits_input = fixture.get('logits_input')
            mapped = model.compute_logits(combined if logits_input is None else logits_input.to('cuda'))
            qkv_input = fixture.get('qkv_input')
            qkv = None if qkv_input is None else model.model.layers[0].self_attn.qkv_proj(qkv_input.to('cuda'))[0].cpu()
            oracle_qkv = None
            if qkv_input is not None:
                from vllm.model_executor.layers import linear as ref
                from vllm.model_executor import parameter as refparam
                frozen = ROOT / 'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/vllm_0_28_0'
                for module, relative in ((ref, 'model_executor/layers/linear.py'),
                                         (refparam, 'model_executor/parameter.py')):
                    assert hashlib.sha256(Path(module.__file__).read_bytes()).digest() == hashlib.sha256((frozen / relative).read_bytes()).digest()
                with ExitStack() as stack:
                    for module in (ref, refparam):
                        stack.enter_context(patch.object(module, 'get_tensor_model_parallel_rank', return_value=rank))
                        stack.enter_context(patch.object(module, 'get_tensor_model_parallel_world_size', return_value=world))
                    with torch.device('cuda'):
                        layer = ref.QKVParallelLinear(config['hidden_size'] * 2, config['head_dim'],
                                                     config['num_attention_heads'], config['num_key_value_heads'],
                                                     bias=False)
                    for shard in ('q', 'k', 'v'):
                        layer.weight_loader(layer.weight, weights[f'midlayer.self_attn.{shard}_proj.weight'], shard)
                    assert torch.equal(layer.weight, model.model.layers[0].self_attn.qkv_proj.weight)
                    oracle_qkv = layer(qkv_input.to('cuda'))[0].cpu()
            attention_output = attention_kv = layer_output = layer_residual = None
            layer_logits = None
            attention_input = fixture.get('attention_input')
            if attention_input is not None:
                from minivllm.attention.backends.triton_attn import TritonAttentionImpl, TritonAttentionMetadataBuilder
                from minivllm.forward_context import set_forward_context
                attn = model.model.layers[0].self_attn.attn
                attn.impl = TritonAttentionImpl(attn.num_heads, attn.head_size, attn.scale, attn.num_kv_heads)
                length = attention_input[0].shape[0]
                block_size = 16
                assert length % block_size == 0
                nblocks = length // block_size
                attn.kv_cache = torch.zeros((2, nblocks + 1, block_size, attn.num_kv_heads, attn.head_size),
                                            device='cuda', dtype=torch.bfloat16)
                builder = TritonAttentionMetadataBuilder(block_size)
                meta = builder.build(query_start_loc=torch.tensor([0, length], device='cuda', dtype=torch.int32),
                                     seq_lens=torch.tensor([length], device='cuda', dtype=torch.int32),
                                     block_table=torch.arange(1, nblocks + 1, device='cuda', dtype=torch.int32).view(1, -1),
                                     slot_mapping=torch.arange(block_size, block_size + length, device='cuda', dtype=torch.int64),
                                     num_reqs=1, max_query_len=length)
                q, k, v = attention_input
                def shard(x, total_heads, local_heads):
                    replica = max(1, world // total_heads)
                    start = (rank // replica) * local_heads * attn.head_size
                    return x[:, start:start + local_heads * attn.head_size].to('cuda')
                with set_forward_context({attn.layer_name: meta}, num_tokens=length):
                    attention_output = attn(shard(q, config['num_attention_heads'], attn.num_heads),
                                            shard(k, config['num_key_value_heads'], attn.num_kv_heads),
                                            shard(v, config['num_key_value_heads'], attn.num_kv_heads)).cpu()
                attention_kv = attn.kv_cache[:, 1:].permute(1, 2, 0, 3, 4).reshape(
                    length, 2, attn.num_kv_heads, attn.head_size).cpu()
                layer_inputs = fixture.get('layer_inputs')
                if layer_inputs is not None:
                    attn.kv_cache.zero_()
                    with set_forward_context({attn.layer_name: meta}, num_tokens=length):
                        local_output, local_residual = model.model.layers[0](
                            layer_inputs['positions'].to('cuda'), layer_inputs['embeds'].to('cuda'),
                            layer_inputs['hidden_states'].to('cuda'), None)
                    layer_output, layer_residual = local_output.cpu(), local_residual.cpu()
                    normalized, _ = model.model.norm(local_output, local_residual)
                    layer_logits = model.compute_logits(normalized[-1:]).cpu()
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({'combined': combined.cpu(), 'mapped_logits': mapped.cpu(), 'qkv': qkv, 'oracle_qkv': oracle_qkv,
                    'attention_output': attention_output, 'attention_kv': attention_kv,
                    'layer_output': layer_output, 'layer_residual': layer_residual,
                    'layer_logits': layer_logits},
                   args.output / f'rank{rank}.pt')
        draft_attention = model.model.layers[0].self_attn
        report = {'rank': rank, 'world': world,
                  'uuid': str(torch.cuda.get_device_properties(local).uuid),
                  'loaded_names': sorted(loaded),
                  'parameter_shapes': {name: list(param.shape) for name, param in model.named_parameters()},
                  'parameter_bytes': sum(param.numel() * param.element_size() for param in model.parameters()),
                  'q_heads': draft_attention.num_heads, 'kv_heads': draft_attention.num_kv_heads,
                  'kv_bytes': 0 if draft_attention.attn.kv_cache is None else
                              draft_attention.attn.kv_cache.numel() * draft_attention.attn.kv_cache.element_size(),
                  'embedding_shared': 'not linked: recorded target embeddings supplied to this module probe',
                  'lm_head_has_own_weight': model.has_own_lm_head,
                  'mapping_dtypes': {'d2t': str(model.d2t.dtype), 't2d': str(model.t2d.dtype)}}
        (args.output / f'rank{rank}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'rank': rank, 'world': world, 'status': 'complete',
                          'uuid': str(torch.cuda.get_device_properties(local).uuid),
                          'scope': 'draft FC, mapped logits, optional local QKV and prefill attention'}), flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == '__main__':
    main()
