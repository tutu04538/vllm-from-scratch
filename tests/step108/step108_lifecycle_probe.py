"""Exercise EAGLE3 TP preemption, cancellation, slot reuse and failure."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.step108_worker import Step108Worker, assert_rank_consistent
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir, tiny_eagle3_dir


def check_padding_write(worker):
    import torch
    from minivllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

    device = worker.device
    slots = torch.tensor([-1, 3, 4, -1], dtype=torch.int64, device=device)
    caches = []
    for poison in (123., -999.):
        key = torch.full((4, 1, 32), poison, device=device)
        value = torch.full_like(key, poison * 2)
        key[1:3] = 7
        value[1:3] = 11
        cache = torch.full((2, 2, 4, 1, 32), -17., device=device)
        triton_reshape_and_cache_flash(key, value, cache[0], cache[1], slots)
        caches.append(cache)
    assert torch.equal(caches[0], caches[1])
    assert torch.all(caches[0][:, 0, :3] == -17)
    assert torch.all(caches[0][:, 1, 1:] == -17)
    assert torch.all(caches[0][0, 0, 3] == 7) and torch.all(caches[0][0, 1, 0] == 7)
    assert torch.all(caches[0][1, 0, 3] == 11) and torch.all(caches[0][1, 1, 0] == 11)
    return {'rank': worker.rank, 'padding_skipped': True, 'cross_block': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp', type=int, choices=[2, 4], required=True)
    p.add_argument('--mode', choices=['preempt', 'cancel', 'failure', 'padding'], required=True)
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    draft = Path(tiny_eagle3_dir())
    spec = SpeculativeConfig(method='eagle3', num_speculative_tokens=2,
        draft_model_config=ModelConfig(str(draft), dtype='float32', max_model_len=20,
                                       hf_config=json.loads((draft/'config.json').read_text())))
    cfg = VllmConfig(model_config=ModelConfig(tiny_qwen3_dir('tiny_gqa'), dtype='float32',
                       max_model_len=20, hf_config=tiny_qwen3_config('tiny_gqa'), seed=123),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=5 if args.mode == 'preempt' else 12,
                                 enable_prefix_caching=args.prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=24),
        device_config=DeviceConfig('cuda'), parallel_config=ParallelConfig(tensor_parallel_size=args.tp),
        compilation_config=CompilationConfig(cudagraph_mode='none'), attention_backend='triton',
        speculative_config=spec)
    ex = MultiprocExecutor(cfg, _worker_cls=Step108Worker)
    engine = LLMEngine(cfg, ex, tokenizer=None)
    try:
        core = engine.engine_core.engine_core
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, seed=17)
        engine.add_request('a', [1, 2, 3, 5, 6, 7], params)
        if args.mode == 'padding':
            ranks = ex.collective_rpc(check_padding_write)
            assert len(ranks) == args.tp and all(r['padding_skipped'] and r['cross_block'] for r in ranks)
            result = {'mode': args.mode, 'tp': args.tp, 'ranks': ranks}
        elif args.mode == 'failure':
            ex.collective_rpc('step108_fail_next_proposal')
            before = list(core.scheduler.requests['a'].output_token_ids)
            try:
                engine.step()
            except RuntimeError as exc:
                assert 'step108 injected proposal failure' in str(exc)
            else:
                raise AssertionError('rank failure not propagated')
            assert list(core.scheduler.requests['a'].output_token_ids) == before
            steps = core.scheduler.num_steps
            try:
                core.step()
            except RuntimeError:
                pass
            else:
                raise AssertionError('failed engine accepted another step')
            assert core.scheduler.num_steps == steps
            result = {'mode': args.mode, 'tp': args.tp, 'failure': core.failure}
        else:
            engine.step()
            first = ex.collective_rpc('step108_snapshot')
            assert_rank_consistent(first)
            initial_slot = first[0]['state']['a']['slot']
            if args.mode == 'cancel':
                engine.abort_request(['a'])
                name, prompt = 'c', [10, 9, 8, 7, 6]
            else:
                name, prompt = 'b', [1, 2, 3, 5, 6, 7]
            engine.add_request(name, prompt, params)
            rounds = []
            for _ in range(50):
                if not engine.has_unfinished_requests():
                    break
                engine.step()
                ranks = ex.collective_rpc('step108_snapshot')
                assert_rank_consistent(ranks)
                rounds.append({'ranks': ranks, 'preemptions': core.scheduler.num_preemptions})
            assert not engine.has_unfinished_requests()
            if args.mode == 'preempt':
                assert core.scheduler.num_preemptions > 0
                if not args.prefix:
                    # A is decoding while B's longer prompt is prefilling.
                    mixed = [event for snapshot in rounds
                             for event in snapshot['ranks'][0]['trace']
                             if event['kind'] == 'sample'
                             and event['req_ids'] == ['a', 'b']
                             and event['query_start_loc'][1] <= 3
                             and event['query_start_loc'][2] - event['query_start_loc'][1] > 3]
                    assert mixed, 'expected an actual mixed prefill/decode batch'
            else:
                seen = [snapshot['ranks'][0]['state'][name]['slot'] for snapshot in rounds
                        if name in snapshot['ranks'][0]['state']]
                assert seen and initial_slot in seen
            result = {'mode': args.mode, 'tp': args.tp, 'rounds': rounds,
                      'preemptions': core.scheduler.num_preemptions}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result)+'\n')
        print(json.dumps({'status': 'passed', 'mode': args.mode, 'tp': args.tp}))
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
