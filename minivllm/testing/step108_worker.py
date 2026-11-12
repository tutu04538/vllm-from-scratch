"""Test-only worker: expose per-rank TP speculative execution snapshots."""
import torch

from ..worker.worker import Worker


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    return value


def assert_rank_consistent(snapshots):
    reference = snapshots[0]
    for other in snapshots[1:]:
        if other['state'] != reference['state']:
            raise AssertionError(f"rank {other['rank']} request state diverged")
        if len(other['trace']) != len(reference['trace']):
            raise AssertionError(f"rank {other['rank']} trace length diverged")
        for index, (expected, actual) in enumerate(zip(reference['trace'], other['trace'])):
            for field in expected:
                if expected[field] != actual.get(field):
                    raise AssertionError(f"rank {other['rank']} event {index} field {field} diverged")


class Step108Worker(Worker):
    def load_model(self):
        super().load_model()
        self.step108_trace = []
        self.step108_forcing = None
        self.step108_proposals = 0
        self.step108_fail_proposal = False
        runner = self.model_runner
        speculator = runner.speculator
        if speculator is None:
            return
        sample = runner.sample
        propose = speculator.propose
        combine = speculator.model.combine_hidden_states
        prefill = speculator._prefill
        from ..worker.gpu.spec_decode.autoregressive import speculator as speculator_module
        prepare = speculator_module.prepare_prefill_inputs

        def prepare_trace(last_indices, current_step, buffers, batch, num_sampled,
                          num_rejected, last_sampled, next_prefill, max_requests):
            result = prepare(last_indices, current_step, buffers, batch, num_sampled,
                             num_rejected, last_sampled, next_prefill, max_requests)
            starts = _cpu(batch.query_start_loc[:batch.num_reqs + 1])
            rejected = _cpu(num_rejected)
            active_slots = _cpu(batch.idx_mapping[:batch.num_reqs])
            draft_ids = _cpu(buffers.input_ids[:batch.num_tokens])
            draft_positions = _cpu(buffers.positions[:batch.num_tokens])
            # Rejected trailing rows are physically present but do not have a
            # draft value; never compare their uninitialized contents.
            for req in range(batch.num_reqs):
                for row in range(starts[req + 1] - rejected[req], starts[req + 1]):
                    draft_ids[row] = draft_positions[row] = None
            self.step108_trace.append({'kind': 'prefill', 'req_ids': list(batch.req_ids),
                'target_input_ids': _cpu(batch.input_ids[:batch.num_tokens]),
                'target_positions': _cpu(batch.positions[:batch.num_tokens]),
                'draft_input_ids': draft_ids, 'draft_positions': draft_positions,
                'query_start_loc': starts,
                'last_indices': _cpu(last_indices[:batch.num_reqs]),
                'num_sampled': _cpu(num_sampled), 'num_rejected': rejected,
                'last_sampled': _cpu(last_sampled.reshape(-1)[batch.idx_mapping[:batch.num_reqs]]),
                'next_prefill': _cpu(next_prefill.reshape(-1)[batch.idx_mapping[:batch.num_reqs]]),
                'slots': active_slots})
            return result

        speculator_module.prepare_prefill_inputs = prepare_trace

        def combine_trace(aux):
            result = combine(aux)
            self.step108_trace.append({'kind': 'combine', 'target_aux': _cpu(aux),
                                       'combined_hidden': _cpu(result)})
            return result

        def prefill_trace(num_reqs, num_tokens, *args, **kwargs):
            self.step108_trace.append({'kind': 'prefill_hidden',
                                       'draft_hidden': _cpu(speculator.hidden_states[:num_tokens])})
            return prefill(num_reqs, num_tokens, *args, **kwargs)

        speculator.model.combine_hidden_states = combine_trace
        speculator._prefill = prefill_trace

        def sample_trace(hidden, batch, grammar):
            result = sample(hidden, batch, grammar)
            output, num_sampled, num_rejected = result
            sampled = _cpu(output.sampled_token_ids)
            counts = _cpu(num_sampled)
            self.step108_trace.append({'kind': 'sample', 'req_ids': list(batch.req_ids),
                'input_ids': _cpu(batch.input_ids[:batch.num_tokens]),
                'positions': _cpu(batch.positions[:batch.num_tokens]),
                'query_start_loc': _cpu(batch.query_start_loc[:batch.num_reqs + 1]),
                'cu_num_logits': _cpu(batch.cu_num_logits[:batch.num_reqs + 1]),
                'slots': _cpu(batch.idx_mapping[:batch.num_reqs]),
                'seeds': _cpu(runner.sampler.sampling_states.seeds.gpu[batch.idx_mapping[:batch.num_reqs]]),
                'sampled': [row[:count] for row, count in zip(sampled, counts)],
                'num_sampled': counts,
                'num_rejected': _cpu(num_rejected)})
            return result

        def propose_trace(batch, metadata, slots, *args, **kwargs):
            if self.step108_fail_proposal and self.rank == 1:
                raise RuntimeError('step108 injected proposal failure on rank 1')
            original_inputs = _cpu(batch.input_ids[:batch.num_tokens])
            original_positions = _cpu(batch.positions[:batch.num_tokens])
            tokens = propose(batch, metadata, slots, *args, **kwargs)
            if self.step108_forcing is not None and self.step108_proposals == 0:
                for row, req_id in enumerate(batch.req_ids):
                    replacement = self.step108_forcing[req_id]
                    tokens[row, :len(replacement)] = torch.tensor(replacement, device=tokens.device,
                                                                  dtype=tokens.dtype)
            self.step108_proposals += 1
            self.step108_trace.append({'kind': 'propose', 'req_ids': list(batch.req_ids),
                'target_input_ids': original_inputs, 'target_positions': original_positions,
                'slot_mappings': _cpu(slots), 'tokens': _cpu(tokens),
                'draft_logits': (_cpu(speculator.draft_logits[batch.idx_mapping[:batch.num_reqs]])
                                 if speculator.draft_logits is not None else None)})
            return tokens

        runner.sample = sample_trace
        speculator.propose = propose_trace

    def step108_set_forcing(self, replacements):
        self.step108_forcing = replacements
        self.step108_proposals = 0
        self.step108_trace.clear()

    def step108_fail_next_proposal(self):
        self.step108_fail_proposal = True

    def step108_skew_rank1(self, field, delta):
        if self.rank != 1:
            return
        states = self.model_runner.req_states
        index = next(iter(states.req_id_to_index.values()))
        if field == 'seed':
            seeds = self.model_runner.sampler.sampling_states.seeds
            seeds.np[index] += delta
            seeds.copy_to_uva()
        elif field == 'length':
            states.total_len.gpu[index] += delta
        else:
            raise ValueError(field)

    def step108_snapshot(self):
        states = self.model_runner.req_states
        runner = self.model_runner
        state, kv = {}, {}
        target_cache = next(iter(runner.kv_caches.values()))
        draft_cache = None
        if runner.speculator is not None:
            draft_attention = runner.speculator.model.model.layers[0].self_attn.attn
            draft_cache = draft_attention.kv_cache
            assert draft_cache is not target_cache and draft_cache.data_ptr() != target_cache.data_ptr()
        for req_id, index in states.req_id_to_index.items():
            length = int(states.total_len.gpu[index].item())
            computed = int(states.num_computed_tokens.gpu[index].item())
            table = runner.block_tables.block_tables[0].gpu[index].detach().cpu().tolist()
            block_size = runner.block_size
            positions = [(int(table[pos // block_size]), pos % block_size) for pos in range(computed)]
            target_values = [target_cache[:, block, offset] for block, offset in positions]
            draft_values = [] if draft_cache is None else [draft_cache[:, block, offset] for block, offset in positions]
            state[req_id] = {'slot': index, 'length': length,
                'computed': computed, 'blocks': table[:(computed + block_size - 1)//block_size],
                'seed': int(runner.sampler.sampling_states.seeds.np[index]),
                'tokens': _cpu(states.all_token_ids.gpu[index, :length])}
            kv[req_id] = {'target': _cpu(torch.stack(target_values)) if target_values else [],
                          'draft': _cpu(torch.stack(draft_values)) if draft_values else []}
        return {'rank': self.rank, 'trace': self.step108_trace, 'state': state, 'local_kv': kv}
