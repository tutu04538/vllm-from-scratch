"""验收方独立检查：真实模型逐 token 对照 + 每一步 KV 状态，非吞吐基准。"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from step51 import Engine as Engine51
from step52 import Engine as Engine52


def digest(package):
    h = hashlib.sha256()
    for p in sorted((ROOT / package).rglob('*.py')):
        h.update(str(p.relative_to(ROOT / package)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def assert_pool(engine):
    pool = engine.kv_cache_pool
    refs = Counter(b for seq in engine.scheduler.running
                   for b in seq.cache.block_table)
    assert pool.block_usage == [refs[b] for b in range(pool.num_kv_blocks)]
    chain = []
    prev = pool._SENTINEL_HEAD
    cur = pool.block_next[prev]
    while cur != pool._SENTINEL_TAIL:
        assert cur not in chain and pool.block_prev[cur] == prev
        chain.append(cur)
        prev, cur = cur, pool.block_next[cur]
    assert pool.block_prev[cur] == prev
    assert set(chain) == {b for b, ref in enumerate(pool.block_usage) if ref == 0}
    assert len(chain) == pool.num_allocatable
    assert not pool.hash_to_block and not pool.block_to_hash
    for seq in engine.scheduler.running:
        assert 0 <= seq.cache.length < len(seq.all_token_ids)
        if seq.output_ids:
            assert seq.cache.length == len(seq.all_token_ids) - 1
        assert len(seq.cache.block_table) == math.ceil(seq.cache.length / pool.block_size)
        assert list(seq.all_token_ids) == seq.prompt_ids + list(seq.output_ids)
        assert seq.recomputed_tokens == seq.num_preemptions == 0


def run_pair(seed, device, dtype, block_size, budget, n, k):
    prompt = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    reqs = [dict(request_id='A', prompt_ids=prompt, max_new_tokens=15, temperature=0),
            dict(request_id='B', prompt_ids=[2, 3, 1] * 3, max_new_tokens=7, temperature=0)]
    outputs, calls, draft_rounds, multi_rounds = [], [], 0, 0
    for cls in (Engine51, Engine52):
        events, finished = [], []
        torch.manual_seed(seed)
        cfg = dict(device=device, dtype=dtype, vocab_size=32, d_model=16,
                   num_q_heads=2, num_kv_heads=1, head_dim=8, num_layers=2,
                   intermediate_size=32, eos_token_ids=[31], max_seq_len=32,
                   max_num_seqs=1, max_num_batched_tokens=budget, block_size=block_size,
                   num_kv_blocks=math.ceil((len(prompt) + 15 - 1) / block_size),
                   enable_prefix_caching=False, attention_backend='torch',
                   on_token=lambda ev: events.append(dict(ev)),
                   on_finished=lambda ev: finished.append(dict(ev)))
        if cls is Engine52:
            cfg.update(speculative_mode='ngram', num_speculative_tokens=k, prompt_lookup_n=n)
        e = cls(**cfg)
        for r in reqs:
            e.add_request(r)
        forward_calls = 0
        original_forward = e.model._forward_append
        def forward(*args, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            return original_forward(*args, **kwargs)
        e.model._forward_append = forward
        for step in range(200):
            if not e.has_unfinished_requests():
                break
            before = len(events)
            e.step()
            assert sum(it['num_scheduled_tokens'] for it in e.scheduler.scheduled_items) <= budget
            if cls is Engine52:
                draft_rounds += sum(bool(it['draft_ids']) for it in e.scheduler.scheduled_items)
                multi_rounds += int(len(events) - before > 1)
                assert_pool(e)  # 即刻检查，不能保存请求引用到结束后才断言
            assert all(not r.get('error') for r in e.scheduler.step_done)
        else:
            raise AssertionError('超过 200 步，疑似无法完成')
        assert len(finished) == 2 and e.scheduler.num_preemptions == 0
        for r in finished:
            actual = [ev for ev in events if ev['request_id'] == r['request_id']]
            assert [ev['output_index'] for ev in actual] == list(range(len(r['output_ids'])))
            assert [ev['token_id'] for ev in actual] == r['output_ids']
        assert all(ref == 0 for ref in e.kv_cache_pool.block_usage)
        outputs.append((finished, events))
        calls.append(forward_calls)
    assert outputs[0] == outputs[1], (seed, device, dtype, block_size, budget, n, k, outputs)
    return dict(seed=seed, device=device, dtype=str(dtype), block_size=block_size,
                budget=budget, n=n, k=k, forward_calls=calls,
                draft_rounds=draft_rounds, multi_token_rounds=multi_rounds)


def iterable_probe():
    e = Engine52(device='cpu', enable_prefix_caching=False, speculative_mode='ngram')
    e.add_request(dict(request_id='G', prompt_ids=iter([1, 2]), max_new_tokens=2, temperature=0))
    prompt = e.scheduler.waiting[0].prompt_ids
    try:
        e.step()
    except Exception as exc:
        return dict(queued_prompt=prompt, error=f'{type(exc).__name__}: {exc}')
    return dict(queued_prompt=prompt, error=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/step52_acceptance/review.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for seed in range(8):
        for index, (block, budget) in enumerate(((1, 1), (2, 3), (4, 8), (8, 16))):
            results.append(run_pair(seed, 'cpu', torch.float32, block, budget,
                                    n=1 + seed % 3, k=1 + index))
    if torch.cuda.is_available():
        for dtype in (torch.float32, torch.bfloat16):
            for seed in (7, 29):
                results.append(run_pair(seed, 'cuda', dtype, 4, 8, n=2, k=3))
        torch.cuda.synchronize()
    report = dict(source_digest={p: digest(p) for p in ('step51', 'step52')},
                  torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  cases=results, iterable_probe=iterable_probe())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS: {len(results)} 个真实模型对照；'
          f'草稿轮数={sum(r["draft_rounds"] for r in results)}；'
          f'多 token 提交轮数={sum(r["multi_token_rounds"] for r in results)}')
    print('迭代器接口诊断（与 list 输入的主验收分开）：', report['iterable_probe'])
    print('结果：', args.output)
