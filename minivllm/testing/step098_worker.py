"""098 test-only Worker: exercise the V2 local KV path with tiny sharded weights."""
import os
from minivllm.worker.worker import Worker


class Step098Worker(Worker):
    def load_model(self):
        from minivllm.worker.gpu.model_runner import GPUModelRunner
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)
        self.model_runner.load_model()
        self.attention_inputs = []
        original = self.model_runner.prepare_attn

        def capture(batch):
            tables, slots = original(batch)
            self.attention_inputs.append({
                'positions': batch.positions[:batch.num_tokens].cpu().tolist(),
                'input_ids': batch.input_ids[:batch.num_tokens].cpu().tolist(),
                'slot_mapping': slots[:, :batch.num_tokens].cpu().tolist(),
                'block_tables': [t[:batch.num_reqs].cpu().tolist() for t in tables],
            })
            return tables, slots

        self.model_runner.prepare_attn = capture

    def determine_available_memory(self):
        available = super().determine_available_memory()
        caps = os.environ.get('STEP098_CAPS')
        if caps:
            group = self.get_cache_config().kv_cache_groups[0]
            per_block = group.kv_cache_spec.page_size_bytes * len(group.layer_names)
            available = min(available, int(caps.split(',')[self.rank]) * per_block)
        return available

    def inspect_kv(self):
        return {name: tensor.detach().cpu() for name, tensor in self.model_runner.kv_caches.items()}

    def test_padding(self):
        import torch
        from minivllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
        for cache in self.model_runner.kv_caches.values():
            before = cache.clone()
            key = torch.full((1, cache.shape[-2], cache.shape[-1]), 123., device=cache.device, dtype=cache.dtype)
            slots = torch.tensor([-1], device=cache.device, dtype=torch.int64)
            triton_reshape_and_cache_flash(key, key, cache[0], cache[1], slots)
            if not torch.equal(cache, before):
                raise AssertionError('padding 写入了真实 KV')
        return True

    def inspect_attention_inputs(self):
        return self.attention_inputs
