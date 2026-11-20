"""Test-only RPC snapshots from the unmodified production Worker and Runner."""

def install(worker):
    runner = worker.model_runner
    runner._step102_trace = []
    original = runner.prepare_attn
    def traced(batch):
        tables, slots = original(batch)
        n = batch.num_tokens
        runner._step102_trace.append({
            'requests': list(batch.req_ids),
            'slots': batch.idx_mapping_np.tolist(),
            'counts': batch.num_scheduled_tokens.tolist(),
            'query_start': batch.query_start_loc_np.tolist(),
            'computed': batch.num_computed_tokens_np.tolist(),
            'positions': batch.positions[:n].cpu().tolist(),
            'input_ids': batch.input_ids[:n].cpu().tolist(),
            'slot_mapping': slots[:, :n].cpu().tolist(),
            'block_tables': [table[:batch.num_reqs].cpu().tolist() for table in tables],
        })
        return tables, slots
    runner.prepare_attn = traced
    return worker.rank


def snapshot(worker):
    import torch
    runner = worker.model_runner
    state = runner.req_states
    tables = runner.block_tables
    reqs = {}
    for req_id, slot in state.req_id_to_index.items():
        count = int(tables.num_blocks.np[0, slot])
        length = int(state.total_len.gpu[slot].item())
        reqs[req_id] = {
            'slot': slot,
            'computed': int(state.num_computed_tokens.gpu[slot].item()),
            'total_len': length,
            'tokens': state.all_token_ids.gpu[slot, :length].cpu().tolist(),
            'blocks': tables.block_tables[0].gpu[slot, :count].cpu().tolist(),
            'seed': int(runner.sampler.sampling_states.seeds.np[slot]),
        }
    return {'rank': worker.rank or 0, 'requests': reqs,
            'trace': runner._step102_trace[-1] if runner._step102_trace else None,
            'trace_count': len(runner._step102_trace),
            'kv': {name: tensor.detach().cpu().clone() for name,tensor in runner.kv_caches.items()}}


def fail_rank_one(worker):
    if worker.rank == 1:
        def fail(*args, **kwargs):
            raise RuntimeError('step102 injected rank 1 sample failure')
        worker.model_runner.sample = fail
    return worker.rank
