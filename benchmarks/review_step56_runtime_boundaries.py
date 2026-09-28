"""独立验收：真实运行时的 K=0 路由、结果回传次数和 RNG 边界。

仅包装观测，不修改 step56 实现。失败项对应第五十六关的现有要求。
"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step56 import Engine, TinyCausalLM
from step56.rejection import BatchedRejectionSampler, RejectionItem
from step56.rejection_rng import event_word
from step56.sampling import SamplingParams, TorchSampler


def runtime_trace(mode, budget, greedy=False):
    dims = dict(vocab_size=64, d_model=16, num_q_heads=2, num_kv_heads=1,
                head_dim=8, num_layers=1, intermediate_size=32, max_seq_len=64,
                eos_token_ids=[63], device='cuda', max_num_query_tokens=max(16, budget))
    torch.manual_seed(1)
    target = TinyCausalLM(**dims)
    draft = TinyCausalLM(**dims) if mode == 'draft_model' else None
    with torch.no_grad():
        target.lm_head.weight.zero_()
        if draft is not None:
            draft.lm_head.weight.zero_()
    engine = Engine(model=target, draft_model=draft, max_num_seqs=1,
                    max_num_batched_tokens=budget, block_size=4, num_kv_blocks=16,
                    draft_num_kv_blocks=16 if draft else None,
                    draft_max_num_batched_tokens=16 if draft else None,
                    enable_prefix_caching=False, speculative_mode=mode,
                    rejection_backend='triton', num_speculative_tokens=2)
    runtime = engine.sample_runtime
    original = runtime.run
    trace = []

    def observed(logits, picked, on_token=None):
        snapshots = []
        for item in picked:
            seq = item['request']
            generator = seq.sampling_state.generator
            snapshots.append((item, len(seq.output_ids), seq.rejection_rng_counter,
                              generator.get_state().clone() if generator is not None else None))
        reads = []
        def wrap(name, method):
            def wrapped(tensor, *args, **kwargs):
                if tensor.is_cuda:
                    caller = traceback.extract_stack(limit=3)[-2]
                    reads.append(dict(method=name, shape=list(tensor.shape),
                                      caller=f'{Path(caller.filename).name}:{caller.lineno}'))
                return method(tensor, *args, **kwargs)
            return wrapped
        with ExitStack() as stack:
            for name in ('cpu', 'tolist', 'item'):
                stack.enter_context(patch.object(torch.Tensor, name, wrap(name, getattr(torch.Tensor, name))))
            result = original(logits, picked, on_token)
        for item, previous_outputs, counter, gen_state in snapshots:
            seq = item['request']
            trace.append(dict(previous_outputs=previous_outputs,
                              reserved=item['num_reserved_drafts'], actual_k=len(item['draft_ids']),
                              uses_verification=runtime._needs_verification(item),
                              counter_before=counter, counter_after=seq.rejection_rng_counter,
                              old_generator_changed=gen_state is not None and not torch.equal(
                                  gen_state, seq.sampling_state.generator.get_state()),
                              device_reads=reads))
        return result

    runtime.run = observed
    engine.add_request(dict(request_id='A', prompt_ids=[1, 2, 3, 4, 5, 6, 7],
                            max_new_tokens=6, temperature=0.0 if greedy else 0.8, seed=7))
    for _ in range(40):
        if not engine.has_unfinished_requests():
            break
        engine.step()
    assert not engine.has_unfinished_requests()
    return trace


def counter_boundaries():
    backend = BatchedRejectionSampler('triton', set(), TorchSampler(), device='cuda')
    weights = torch.full((64,), 1 / 64, device='cuda')
    entries = []
    for counter in (0, 1 << 32, -1):
        seq = SimpleNamespace(rejection_seed=11, rejection_rng_counter=counter,
                              sampling_params=SamplingParams(temperature=0.8, seed=7))
        entries.append(RejectionItem(plan={'request': seq}, mode='distribution',
                                     draft_ids=[], remaining_outputs=1, row_probs=[weights]))
    rows = []
    for entry in entries:
        try:
            result = backend.materialize_results(backend.verify_batch([entry]))[0]
            row = dict(tokens=result.committed_ids, error=result.error,
                       consumed=result.rng_consumed)
        except ValueError as error:
            row = dict(tokens=[], error=str(error), consumed=0)
        row['counter'] = entry.plan['request'].rejection_rng_counter
        rows.append(row)
    try:
        event_word(11, 1 << 32, 0)
        cpu_rejects = False
    except ValueError:
        cpu_rejects = True
    return dict(cpu_rejects=cpu_rejects, rows=rows)


def main():
    torch.set_num_threads(1)
    results = {}
    checks = []
    for mode in ('draft_model', 'ngram'):
        trace = runtime_trace(mode, budget=1 if mode == 'draft_model' else 16)
        results[mode] = trace
        fallback = [r for r in trace if r['previous_outputs'] > 0 and r['actual_k'] == 0]
        checks.append((f'{mode}: decode K=0 使用 counter RNG', bool(fallback) and all(
            r['uses_verification'] and r['counter_after'] == r['counter_before'] + 1
            and not r['old_generator_changed'] for r in fallback)))
    greedy = runtime_trace('draft_model', budget=16, greedy=True)
    results['greedy'] = greedy
    speculative = [r for r in greedy if r['actual_k'] > 0]
    checks.append(('greedy GPU 验证整轮仅一次结果回传', bool(speculative) and all(
        len(r['device_reads']) == 1 and r['device_reads'][0]['method'] == 'cpu'
        for r in speculative)))
    boundaries = counter_boundaries()
    results['counter_boundaries'] = boundaries
    checks.append(('GPU 拒绝越界 counter，和 CPU 契约一致', boundaries['cpu_rejects'] and all(
        r['error'] is not None for r in boundaries['rows'][1:])))
    results['checks'] = [{'name': name, 'passed': ok} for name, ok in checks]
    dest = Path('/tmp/step56_review/runtime_boundaries.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    for name, ok in checks:
        print('PASS' if ok else 'FAIL', name)
    print('trace:', dest)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
