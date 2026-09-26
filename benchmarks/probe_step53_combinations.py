"""诊断旧第五十四关：仅在当前进程绕过 priority/prefix 配置限制，不改实现文件。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step53.engine as engine_module
from step53 import Engine

original_check = engine_module._check_speculative

def relaxed_check(mode, k, n, policy, prefix, backend, graph):
    return original_check(mode, k, n, 'fcfs', False, backend, graph)

engine_module._check_speculative = relaxed_check
DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2, num_kv_heads=1,
            num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def check_pool(e, immutable):
    p = e.kv_cache_pool
    refs = Counter()
    for q in e.scheduler.running + e.scheduler.waiting:
        if q.cache is None or q.cache.block_table is None:
            continue
        refs.update(q.cache.block_table)
        assert len(q.cache.block_table) == math.ceil(q.cache.length / p.block_size)
        assert 0 <= q.cache.length < len(q.all_token_ids)
        prev = b''
        for i, actual in enumerate(q.block_hashes):
            tokens = tuple(q.all_token_ids[i*p.block_size:(i+1)*p.block_size])
            expected = hashlib.sha256(json.dumps((prev.hex(), tokens)).encode()).digest()
            assert actual == expected
            prev = expected
    assert p.block_usage == [refs[b] for b in range(p.num_kv_blocks)]
    chain, prev = [], p._SENTINEL_HEAD
    cur = p.block_next[prev]
    while cur != p._SENTINEL_TAIL:
        assert cur not in chain and p.block_prev[cur] == prev
        chain.append(cur)
        prev, cur = cur, p.block_next[cur]
    assert p.block_prev[cur] == prev and len(chain) == p.num_allocatable
    assert set(chain) == {b for b, count in enumerate(p.block_usage) if count == 0}
    assert len(p.block_to_hash) == len(p.hash_to_block)
    now = {}
    for h, b in p.hash_to_block.items():
        assert p.block_to_hash[b] == h
        key = (h.hex(), b)
        if key in immutable:
            old_k, old_v = immutable[key]
            assert torch.equal(old_k, p.k_cache[:, b]) and torch.equal(old_v, p.v_cache[:, b])
            now[key] = immutable[key]
        else:
            now[key] = (p.k_cache[:, b].clone(), p.v_cache[:, b].clone())
    immutable.clear()
    immutable.update(now)
    return sum(count > 1 for count in p.block_usage)


def run(seed, policy, prefix, spec, device='cpu'):
    torch.manual_seed(seed)
    block = (2, 4)[seed % 2]
    slots = (1, 2, 4)[seed % 3]
    budget = (2, 5, 12)[seed % 3]
    num_blocks = math.ceil((9 + 10 - 1) / block) + (seed % 3) * 2
    e = Engine(device=device, max_num_seqs=slots, max_num_batched_tokens=budget,
               block_size=block, num_kv_blocks=num_blocks, enable_prefix_caching=prefix,
               scheduling_policy=policy, speculative_mode=spec, num_speculative_tokens=3,
               prompt_lookup_n=1 + seed % 3, **DIMS)
    events, records, saved = [], [], {}
    e.on_token = lambda ev: events.append(dict(ev))
    e.scheduler.on_finished = lambda rec: records.append(dict(rec))
    reqs = [dict(request_id=f'R{i}', prompt_ids=[1, 2, 3]*3, max_new_tokens=10,
                 priority=0 if i < 2 else -1, temperature=0) for i in range(4)]
    arrivals = {0: reqs[:2], 2: reqs[2:]}
    counts = dict(draft_rounds=0, shared_observations=0, reused_tokens=0)
    all_seqs = {}
    for step in range(500):
        for r in arrivals.get(step, []):
            e.add_request(r)
        if not e.has_unfinished_requests():
            if step < max(arrivals):
                continue
            break
        for q in e.scheduler.waiting + e.scheduler.running:
            all_seqs[q.request_id] = q
        e.step()
        counts['shared_observations'] += check_pool(e, saved)
        counts['draft_rounds'] += sum(bool(it['draft_ids']) for it in e.scheduler.scheduled_items)
        assert sum(it['num_scheduled_tokens'] for it in e.scheduler.scheduled_items) <= budget
    else:
        raise AssertionError('超过 500 步')
    assert len(records) == 4 and all(not r.get('error') for r in records)
    assert all(u == 0 for u in e.kv_cache_pool.block_usage)
    outputs = {r['request_id']: r['output_ids'] for r in records}
    for rid, tokens in outputs.items():
        evs = [v for v in events if v['request_id'] == rid]
        assert [v['token_id'] for v in evs] == tokens
        assert [v['output_index'] for v in evs] == list(range(len(tokens)))
    counts.update(preemptions=e.scheduler.num_preemptions,
                  priority_preemptions=e.scheduler.num_priority_preemptions,
                  reused_tokens=sum(q.reused_tokens for q in all_seqs.values()))
    return outputs, counts


if __name__ == '__main__':
    torch.set_num_threads(1)
    cases = []
    for device, seeds in [('cpu', range(24))] + ([('cuda', (7, 11))] if torch.cuda.is_available() else []):
        for seed in seeds:
            for policy in ('fcfs', 'priority'):
                for prefix in (False, True):
                    plain, _ = run(seed, policy, prefix, None, device)
                    speculative, stats = run(seed, policy, prefix, 'ngram', device)
                    assert plain == speculative, (seed, policy, prefix, device)
                    cases.append(dict(seed=seed, policy=policy, prefix=prefix, device=device, **stats))
    path = Path('/tmp/step53_combinations/probe.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(change='process-local config check bypass only', cases=cases), indent=2)+'\n')
    print(f'PASS {len(cases)} 普通/投机对照；只绕过配置校验，无实现改动')
    for name in ('draft_rounds', 'shared_observations', 'reused_tokens', 'preemptions', 'priority_preemptions'):
        print(name, sum(c[name] for c in cases))
    print('结果', path)
