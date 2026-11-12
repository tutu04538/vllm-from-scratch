"""Exercise EAGLE3 rank-local heads and global vocabulary mapping over NCCL."""
import json
import os
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from safetensors.torch import load_file
from minivllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  destroy_model_parallel, destroy_distributed_environment)
from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
from minivllm.testing.tiny_models import tiny_eagle3_dir


def main():
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    init_distributed_environment(world, rank, 'env://', rank, timeout=timedelta(seconds=90))
    try:
        initialize_model_parallel(world, 1, timeout=timedelta(seconds=90))
        path = Path(tiny_eagle3_dir())
        config = json.loads((path / 'config.json').read_text())
        config.update(vocab_size=129, draft_vocab_size=67)
        weights = load_file(str(path / 'model.safetensors'))
        generator = torch.Generator().manual_seed(107)
        weights['lm_head.weight'] = torch.randn(67, config['hidden_size'], generator=generator)
        ids = torch.arange(67) + torch.arange(67) * 62 // 66
        weights['d2t'] = ids - torch.arange(67)
        weights['t2d'] = torch.zeros(config['vocab_size'], dtype=torch.bool)
        weights['t2d'][ids] = True
        with torch.device('cuda'):
            model = Eagle3ForCausalLM(config)
        model.to('cuda')
        model.load_weights(iter(sorted(weights.items())))
        assert torch.equal(model.d2t.cpu(), weights['d2t'])
        assert torch.equal(model.t2d.cpu(), weights['t2d'])
        # Sharing uses this worker's own target shard and keeps the reduced head separate.
        from minivllm.layers.linear import VocabParallelEmbedding, ParallelLMHead
        from minivllm.worker.gpu.spec_decode.eagle import utils
        with torch.device('cuda'):
            target_embed = VocabParallelEmbedding(config['vocab_size'], config['hidden_size'])
            target_head = ParallelLMHead(config['vocab_size'], config['hidden_size'])
        target = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embed), lm_head=target_head)
        draft_head = model.lm_head
        target_kv = torch.zeros((2, 2, 16, max(1, 2 // world), 16), device='cuda')
        draft_kv = torch.zeros_like(target_kv)
        from minivllm.attention import Attention
        target_attention = Attention(4 // world, 16, num_kv_heads=max(1, 2 // world),
                                     layer_name='target.attn')
        target_attention.kv_cache = target_kv
        model.model.layers[0].self_attn.attn.kv_cache = draft_kv
        assert draft_kv.data_ptr() != target_attention.kv_cache.data_ptr()
        cfg = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=object()),
                              device_config=SimpleNamespace(device='cuda'))
        with patch.object(utils, 'get_model', return_value=model):
            assert utils.load_eagle_model(target, cfg) is model
        assert model.model.embed_tokens is target_embed
        assert model.lm_head is draft_head and model.lm_head is not target_head
        assert model.model.layers[0].self_attn.attn.kv_cache is draft_kv
        attn = model.model.layers[0].self_attn
        assert attn.num_heads == config['num_attention_heads'] // world
        assert attn.num_kv_heads == max(1, config['num_key_value_heads'] // world)
        assert attn.attn.num_heads == attn.num_heads
        assert attn.attn.num_kv_heads == attn.num_kv_heads
        assert attn.q_size == attn.num_heads * config['head_dim']
        class CaptureAttention(torch.nn.Module):
            def forward(self, q, k, v):
                assert q.shape == (3, attn.q_size)
                assert k.shape == v.shape == (3, attn.num_kv_heads * attn.head_dim)
                return q
        # Exercise projection, local split, q/k norm, RoPE and output collective;
        # the stand-in intentionally does not validate actual attention or KV.
        attn.attn = CaptureAttention()
        inputs = torch.randn(3, config['hidden_size'] * 2, generator=generator).cuda()
        projected = attn(torch.arange(3, device='cuda'), inputs)
        assert projected.shape == (3, config['hidden_size'])
        assert torch.isfinite(projected).all()
        assert tuple(model.model.fc.weight.shape) == tuple(weights['fc.weight'].shape)
        assert torch.equal(model.model.fc.weight.cpu(), weights['fc.weight'])
        # Poison padded local rows: they must be discarded before d2t scatter.
        with torch.no_grad():
            model.lm_head.weight[model.lm_head.shard_indices.num_org_elements:].fill_(1e5)
        hidden = torch.randn(3, config['hidden_size'], generator=generator).cuda()
        logits = model.compute_logits(hidden)
        expected = torch.nn.functional.linear(hidden, weights['lm_head.weight'].cuda())
        torch.testing.assert_close(logits[:, ids], expected, rtol=1e-5, atol=1e-5)
        invalid = torch.ones(config['vocab_size'], dtype=torch.bool, device='cuda')
        invalid[ids] = False
        assert torch.isneginf(logits[:, invalid]).all()
        probabilities = logits.softmax(-1)
        assert torch.allclose(probabilities.sum(-1), torch.ones(3, device='cuda'))
        torch.testing.assert_close(probabilities[:, ids], expected.softmax(-1), rtol=1e-5, atol=1e-6)
        assert probabilities[:, invalid].count_nonzero() == 0
        gathered = [torch.empty_like(logits) for _ in range(world)]
        torch.distributed.all_gather(gathered, logits)
        assert all(torch.equal(logits, other) for other in gathered)
        print(f'step107 rank={rank} world={world} local_heads={attn.num_heads}/{attn.num_kv_heads} passed', flush=True)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == '__main__':
    main()
