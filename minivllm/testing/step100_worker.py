"""100 专用探针：仅在测试 Worker 中导出 V2 采样与状态快照。"""
import numpy as np
import torch
from minivllm.testing.step098_worker import Step098Worker


def _cpu(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    return value


class Step100Worker(Step098Worker):
    def load_model(self):
        super().load_model()
        runner = self.model_runner
        # 模拟不同 rank 的初始化随机消耗；生产 warmup 后必须重置模型 seed。
        for _ in range(self.rank + 1):
            np.random.random()
            torch.rand(1, device=self.device)
        self.sample_trace = []
        original_sample = runner.sample
        original_logits = runner.model.compute_logits

        def capture_logits(hidden):
            logits = original_logits(hidden)
            self._raw_logits = logits.detach().cpu().tolist()
            return logits

        def capture_sample(hidden, batch, grammar):
            result = original_sample(hidden, batch, grammar)
            sampled, num_sampled, num_rejected = result
            states = runner.sampler.sampling_states
            indices = batch.idx_mapping_np.tolist()
            self.sample_trace.append({
                'req_ids': list(batch.req_ids),
                'slots': indices,
                'positions': _cpu(batch.positions[batch.logits_indices]),
                'expanded_slots': _cpu(batch.expanded_idx_mapping),
                'seeds': [int(states.seeds.np[i]) for i in indices],
                'logits': self._raw_logits,
                'sampled': _cpu(sampled.sampled_token_ids),
                'num_sampled': _cpu(num_sampled),
                'num_rejected': _cpu(num_rejected),
                'logprobs': _cpu(sampled.logprobs_tensors),
            })
            return result

        runner.model.compute_logits = capture_logits
        runner.sample = capture_sample

    def inspect_sampling(self):
        states = self.model_runner.req_states
        snapshots = {}
        for req, idx in states.req_id_to_index.items():
            length = int(states.total_len.gpu[idx].item())
            snapshots[req] = {
                'slot': idx,
                'total_len': length,
                'computed': int(states.num_computed_tokens.gpu[idx].item()),
                'tokens': _cpu(states.all_token_ids.gpu[idx, :length]),
                'last': _cpu(states.last_sampled_tokens[idx]),
            }
        return {'trace': self.sample_trace, 'states': snapshots}
