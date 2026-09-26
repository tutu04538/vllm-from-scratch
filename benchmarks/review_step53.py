"""第五十三关独立功能验收；不测吞吐。结果默认写 /tmp/step53_acceptance/review.json。"""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step53 import Engine
from step52 import Engine as Engine52

DIMS = dict(vocab_size=64, d_model=16, max_seq_len=64, num_q_heads=2,
            num_kv_heads=1, num_layers=2, intermediate_size=32, head_dim=8, eos_token_ids=[63])


def pool_check(e):
    p = e.kv_cache_pool
    refs = Counter()
    for q in e.scheduler.running + e.scheduler.waiting:
        if q.cache is None or q.cache.block_table is None:
            continue
        refs.update(q.cache.block_table)
        assert 0 <= q.cache.length < len(q.all_token_ids)
        assert len(q.cache.block_table) == math.ceil(q.cache.length / p.block_size)
        assert list(q.all_token_ids) == q.prompt_ids + list(q.output_ids)
    assert p.block_usage == [refs[i] for i in range(p.num_kv_blocks)]
    chain, prev = [], p._SENTINEL_HEAD
    cur = p.block_next[prev]
    while cur != p._SENTINEL_TAIL:
        assert cur not in chain and p.block_prev[cur] == prev
        chain.append(cur)
        prev, cur = cur, p.block_next[cur]
    assert p.block_prev[cur] == prev
    assert set(chain) == {i for i, r in enumerate(p.block_usage) if r == 0}
    assert len(chain) == p.num_allocatable
    assert not p.hash_to_block and not p.block_to_hash


def run(e, arrivals, limit=500):
    events, finished, trace = [], [], []
    e.on_token = lambda ev: events.append(dict(ev))
    e.scheduler.on_finished = lambda ev: finished.append(dict(ev))
    shrink = [0]
    original = e.scheduler._shrink_draft
    def shrink_draft(item):
        shrink[0] += 1
        return original(item)
    e.scheduler._shrink_draft = shrink_draft
    calls = [0]
    forward = e.model._forward_append
    def counted(*args, **kwargs):
        calls[0] += 1
        return forward(*args, **kwargs)
    e.model._forward_append = counted
    for step in range(limit):
        for req in arrivals.get(step, []):
            e.add_request(dict(req))
        if not e.has_unfinished_requests():
            if step < max(arrivals):
                continue
            break
        before, old_calls = len(events), calls[0]
        e.step()
        items = e.scheduler.scheduled_items
        assert calls[0] - old_calls == int(bool(items))
        assert sum(it['num_scheduled_tokens'] for it in items) <= e.scheduler.max_num_batched_tokens
        assert all(it['num_scheduled_tokens'] > 0 for it in items)
        picked = [it for it in items if it['can_sample']]
        order = {it['request'].request_id: i for i, it in enumerate(picked)}
        emitted = events[before:]
        assert [order[ev['request_id']] for ev in emitted] == sorted(order[ev['request_id']] for ev in emitted)
        pool_check(e)  # 立即检查状态；不能到请求结束后再访问同一对象来代替逐步断言
        trace.append(dict(drafts=[len(it['draft_ids']) for it in picked],
                          events=emitted, preemptions=e.scheduler.num_preemptions))
    else:
        raise AssertionError('超过 step 上限')
    assert not e.has_unfinished_requests() and all(u == 0 for u in e.kv_cache_pool.block_usage)
    outs = {}
    for rec in finished:
        assert not rec.get('error') and rec['request_id'] not in outs
        rid = rec['request_id']
        actual = [v for v in events if v['request_id'] == rid]
        assert [v['token_id'] for v in actual] == rec['output_ids']
        assert [v['output_index'] for v in actual] == list(range(len(actual)))
        outs[rid] = rec['output_ids']
    assert len(outs) == sum(map(len, arrivals.values()))
    return dict(outputs=outs, trace=trace, shrink=shrink[0], preemptions=e.scheduler.num_preemptions)


def real_case(seed, device='cpu', dtype=torch.float32):
    block = (1, 2, 4, 8)[seed % 4]
    budget = (1, 2, 5, 12)[(seed // 4) % 4]
    reqs = [dict(request_id=f'R{i}', prompt_ids=([1, 2, 3] if i % 2 == 0 else [4, 4, 5]) * 2,
                 max_new_tokens=8 + i, temperature=0) for i in range(4)]
    feasible = math.ceil((6 + 11 - 1) / block)
    blocks = feasible + (seed % 3) * math.ceil(6 / block)
    arrivals = {0: reqs[:3], 2: reqs[3:]}
    results = []
    for spec in (None, 'ngram'):
        torch.manual_seed(seed)
        e = Engine(device=device, dtype=dtype, max_num_seqs=4,
                   max_num_batched_tokens=budget, block_size=block, num_kv_blocks=blocks,
                   enable_prefix_caching=False, speculative_mode=spec,
                   prompt_lookup_n=1 + seed % 3, num_speculative_tokens=1 + seed % 4, **DIMS)
        results.append(run(e, arrivals))
    assert results[0]['outputs'] == results[1]['outputs'], (seed, device, dtype, results)
    return dict(seed=seed, device=device, dtype=str(dtype), block_size=block, budget=budget,
                pool_blocks=blocks, preemptions=results[1]['preemptions'], shrink=results[1]['shrink'],
                draft_rounds=sum(bool(k) for t in results[1]['trace'] for k in t['drafts']))


class OracleModel:
    """真实模型负责 KV 写入，目标 logits 按逻辑位置的脚本给出，控制各请求接受分支。"""
    def __init__(self, inner, outputs):
        self.inner, self.outputs, self.scheduler = inner, outputs, None
    def __getattr__(self, key):
        return getattr(self.inner, key)
    def _forward_append(self, ids, sizes, caches, pool, sample_rows=None):
        items = self.scheduler.scheduled_items
        predictions = []
        for it in items:
            q = it['request']
            for pos in range(it['start_cache_length'], it['start_cache_length'] + it['num_scheduled_tokens']):
                idx = pos - len(q.prompt_ids) + 1
                predictions.append(self.outputs[q.request_id][idx] if idx >= 0 else 0)
        self.inner._forward_append(ids, sizes, caches, pool, sample_rows=sample_rows)
        chosen = predictions if sample_rows is None else [predictions[i] for i in sample_rows]
        logits = torch.zeros(len(chosen), self.vocab_size, device=self.device)
        for row, tok in enumerate(chosen):
            logits[row, tok] = 10
        return logits


def mixed_case():
    outputs = dict(A=[1, 2, 3, 7, 8, 9], C=[1, 9, 8, 7, 6, 5],
                   D=[1, 2, 9, 7, 8, 6], E=[1, 2, 63, 4, 5, 6])
    torch.manual_seed(11)
    e = Engine(device='cpu', max_num_seqs=4, max_num_batched_tokens=32,
               block_size=4, num_kv_blocks=24, enable_prefix_caching=False,
               speculative_mode='ngram', num_speculative_tokens=2, prompt_lookup_n=2, **DIMS)
    wrapped = OracleModel(e.model, outputs)
    wrapped.scheduler = e.scheduler
    e.model = wrapped
    reqs = [dict(request_id=rid, prompt_ids=[1, 2, 63 if rid == 'E' else 3, 4] * 2,
                 max_new_tokens=6, temperature=0) for rid in outputs]
    r = run(e, {0: reqs})
    second = r['trace'][1]
    assert second['drafts'] == [2, 2, 2, 2]
    assert [(ev['request_id'], ev['token_id']) for ev in second['events']] == [
        ('A', 2), ('A', 3), ('A', 7), ('C', 9), ('D', 2), ('D', 9), ('E', 2), ('E', 63)]
    assert r['outputs'] == {rid: vals[:vals.index(63)+1] if 63 in vals else vals
                            for rid, vals in outputs.items()}
    return r


def iterable_case():
    outs = []
    for factory in (list, tuple, iter, lambda x: (t for t in x)):
        torch.manual_seed(3)
        e = Engine(device='cpu', enable_prefix_caching=False, **DIMS)
        req = dict(request_id='G', prompt_ids=factory([1, 2]), max_new_tokens=2, temperature=0)
        original = req['prompt_ids']
        e.add_request(req)
        assert req['prompt_ids'] is original and e.scheduler.waiting[0].prompt_ids == [1, 2]
        finished = []
        while e.has_unfinished_requests():
            finished += e.step()
        outs.append(finished)
    assert all(v == outs[0] for v in outs)
    return 'PASS: list / tuple / iterator / generator; request dictionary not replaced'



def sampler_regression():
    results = []
    configs = [('cpu', torch.float32, 'torch', False)]
    if torch.cuda.is_available():
        configs += [('cuda', torch.bfloat16, 'torch', False),
                    ('cuda', torch.float32, 'triton', True)]
    penalties = [{}, {'repetition_penalty': 1.3},
                 {'presence_penalty': 0.3, 'frequency_penalty': 0.2},
                 {'repetition_penalty': 1.2, 'presence_penalty': 0.3, 'frequency_penalty': 0.2}]
    for device, dtype, backend, graph in configs:
        for penalty in penalties:
            outputs = []
            for cls in (Engine52, Engine):
                torch.manual_seed(19)
                events, finished = [], []
                e = cls(device=device, dtype=dtype, attention_backend=backend, use_cuda_graph=graph,
                        max_num_seqs=2, max_num_batched_tokens=8, block_size=4, num_kv_blocks=16,
                        enable_prefix_caching=True, speculative_mode=None,
                        on_token=lambda ev: events.append(dict(ev)),
                        on_finished=lambda ev: finished.append(dict(ev)), **DIMS)
                for rid, seed in [('A', 3), ('B', 7)]:
                    e.add_request(dict(request_id=rid, prompt_ids=[1, 2, 3, 1, 2, 3],
                                       max_new_tokens=7, temperature=0.8, top_k=12, top_p=0.9,
                                       seed=seed, **penalty))
                for _ in range(100):
                    if not e.has_unfinished_requests():
                        break
                    e.step()
                else:
                    raise AssertionError('sampler regression step limit')
                assert len(finished) == 2 and all(u == 0 for u in e.kv_cache_pool.block_usage)
                outputs.append((events, finished))
            assert outputs[0] == outputs[1], (device, dtype, backend, graph, penalty, outputs)
            results.append(dict(device=device, dtype=str(dtype), backend=backend, graph=graph,
                                penalties=penalty, result='PASS'))
    return results


def source_digest(pkg):
    h = hashlib.sha256()
    for p in sorted((ROOT / pkg).rglob('*.py')):
        h.update(str(p.relative_to(ROOT / pkg)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


if __name__ == '__main__':
    torch.set_num_threads(1)
    cases = [real_case(seed) for seed in range(64)]
    if torch.cuda.is_available():
        cases += [real_case(seed, 'cuda', dt) for dt in (torch.float32, torch.bfloat16)
                  for seed in (11, 29)]
        torch.cuda.synchronize()
    report = dict(source_digest={p: source_digest(p) for p in ('step52', 'step53')},
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  cases=cases, simultaneous_accept_reject_eos=mixed_case(), iterable=iterable_case(),
                  sampler_regression=sampler_regression())
    path = Path('/tmp/step53_acceptance/review.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS {len(cases)} 组普通/投机逐请求对照；'
          f'草稿轮数 {sum(c["draft_rounds"] for c in cases)}；'
          f'抢占 {sum(c["preemptions"] for c in cases)}；缩草稿 {sum(c["shrink"] for c in cases)}')
    print(f'PASS {len(report["sampler_regression"])} 组采样/惩罚/Graph 旧路径差分')
    print('PASS 同批全接受/首拒绝/部分接受/EOS；PASS 可迭代输入；结果', path)
