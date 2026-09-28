"""验收补测：真实优先级抢占后的采样随机流；最后合法事件接受 EOS。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step56 import Engine, TinyCausalLM
from step56.rejection import BatchedRejectionSampler, RejectionItem
from step56.rejection_rng import EVENT_INDEX_LIMIT, event_uniform
from step56.sampling import SamplingParams, TorchSampler
from step56.speculative import verify_drafts_random


def run_recovery(interrupt):
    dims = dict(vocab_size=64, d_model=16, num_q_heads=2, num_kv_heads=1,
                head_dim=8, num_layers=1, intermediate_size=32, max_seq_len=64,
                eos_token_ids=[63], device='cuda', max_num_query_tokens=16)
    torch.manual_seed(1)
    target, draft = TinyCausalLM(**dims), TinyCausalLM(**dims)
    with torch.no_grad():
        target.lm_head.weight.zero_()
        draft.lm_head.weight.zero_()
    # draft 池容不下 A 的历史，使两组的 A 都一直 K=0，排除 K 改变随机事件序列的影响。
    engine = Engine(model=target, draft_model=draft, max_num_seqs=1,
                    max_num_batched_tokens=4, block_size=4, num_kv_blocks=16,
                    draft_num_kv_blocks=1, draft_max_num_batched_tokens=4,
                    enable_prefix_caching=False, speculative_mode='draft_model',
                    rejection_backend='triton', num_speculative_tokens=2,
                    scheduling_policy='priority')
    engine.add_request(dict(request_id='A', prompt_ids=[1,2,3,4,5,6,7],
                            max_new_tokens=6, temperature=0.8, seed=7, priority=10))
    finished, trace, outputs = [], [], []
    engine.on_token = lambda ev: outputs.append(ev['token_id']) if ev['request_id'] == 'A' else None
    runtime = engine.sample_runtime
    original = runtime.run

    def observed(logits, picked, on_token=None):
        before = []
        for item in picked:
            seq = item['request']
            if seq.request_id == 'A':
                before.append((item, len(seq.output_ids), seq.rejection_rng_counter,
                               seq.sampling_state.generator.get_state().clone()))
        result = original(logits, picked, on_token)
        for item, count, counter, rng in before:
            seq = item['request']
            trace.append(dict(outputs_before=count, preemptions=seq.num_preemptions,
                              start=item['start_cache_length'], inputs=item['num_scheduled_tokens'],
                              actual_k=len(item['draft_ids']), speculative=item['speculative'],
                              counter_before=counter, counter_after=seq.rejection_rng_counter,
                              old_generator_changed=not torch.equal(rng, seq.sampling_state.generator.get_state()),
                              cache_length=seq.cache.length))
        return result
    runtime.run = observed
    inserted = False
    for _ in range(60):
        if not engine.has_unfinished_requests():
            break
        if interrupt and not inserted and len(outputs) == 3:
            engine.add_request(dict(request_id='B', prompt_ids=[1], max_new_tokens=1,
                                    temperature=0.0, priority=0))
            inserted = True
        finished.extend(engine.step())
    assert not engine.has_unfinished_requests()
    assert all(x == 0 for x in engine.kv_cache_pool.block_usage + engine.draft_kv_pool.block_usage)
    assert all(r['actual_k'] == 0 for r in trace)
    return dict(outputs=outputs, trace=trace, preemptions=engine.scheduler.num_priority_preemptions)


def terminal_event():
    # 最后合法事件用于接受 EOS，之后没有 bonus/correction，所以本轮没有用越界事件。
    seed, counter = 11, EVENT_INDEX_LIMIT - 1
    u = event_uniform(seed, counter, 0)
    p = torch.tensor([0.01, 0.01, 0.98], device='cuda')
    q = torch.tensor([0.0, 0.0, 1.0], device='cuda')
    assert u < 0.98
    seq = SimpleNamespace(rejection_seed=seed, rejection_rng_counter=counter,
                          sampling_params=SamplingParams(temperature=0.8, seed=7), request_id='EOS')
    entry = RejectionItem(plan={'request': seq}, mode='distribution', draft_ids=[2],
                          remaining_outputs=2, row_probs=[p, p], draft_probs=[q])
    backend = BatchedRejectionSampler('triton', {2}, TorchSampler(), device='cuda')
    actual = backend.materialize_results(backend.verify_batch([entry]))[0]
    events = []
    def draw_uniform():
        events.append(counter)
        return event_uniform(seed, counter, 0)
    def draw_token(_):
        raise AssertionError('接受 EOS 后不应抽 bonus')
    expected = verify_drafts_random([2], [p.cpu(), p.cpu()], {2}, 2,
                                    draw_uniform, draw_token, draft_probs=[q.cpu()])
    return dict(counter=counter, uniform=u, cpu_tokens=expected.committed_ids,
                cpu_events=events, gpu_tokens=actual.committed_ids,
                gpu_consumed=actual.rng_consumed, gpu_error=actual.error)


def main():
    torch.set_num_threads(1)
    control, interrupted = run_recovery(False), run_recovery(True)
    assert interrupted['preemptions'] > 0
    restored = [r for r in interrupted['trace'] if r['preemptions'] > 0 and r['outputs_before'] > 0]
    eos = terminal_event()
    checks = [
        ('抢占恢复后继续采样不切回旧随机流', bool(restored) and all(
            not r['old_generator_changed'] and r['counter_after'] == r['counter_before'] + 1
            for r in restored)),
        ('固定 K=0/相同 logits 时，插入抢占不改变 A 的输出', control['outputs'] == interrupted['outputs']),
        ('最后合法事件接受 EOS 不应误报溢出', eos['gpu_error'] is None and
         eos['gpu_tokens'] == eos['cpu_tokens'] and eos['gpu_consumed'] == len(eos['cpu_events'])),
    ]
    output = dict(control=control, interrupted=interrupted, terminal_event=eos,
                  checks=[dict(name=n, passed=ok) for n,ok in checks])
    path = Path('/tmp/step56_recheck/recovery_rng.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    for name, ok in checks:
        print('PASS' if ok else 'FAIL', name)
    print('control:', control['outputs'])
    print('preempt:', interrupted['outputs'])
    print('EOS:', eos)
    return 0 if all(ok for _,ok in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
