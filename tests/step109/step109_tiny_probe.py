"""Launch a tiny EAGLE3 TP engine; parent creates no CUDA objects."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.step108_worker import Step108Worker, assert_rank_consistent
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir, tiny_eagle3_dir


def step109_graph_stats(worker):
    runner = worker.model_runner
    spec = runner.speculator
    def stat(manager):
        return None if manager is None else {'captured': len(manager.graphs),
                                             'replays': manager.num_replays}
    return {'rank': worker.rank, 'target': stat(runner.target_cudagraph_manager),
            'prefill': stat(spec.prefill_cudagraph_manager) if spec else None,
            'decode': stat(spec.decode_cudagraph_manager) if spec else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tp', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--k', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--graph', action='store_true')
    p.add_argument('--batch', type=int, choices=[1, 2, 4], default=2)
    p.add_argument('--accepted', type=int, default=None)
    p.add_argument('--mixed', action='store_true', help='alternate accepted prefix length by request')
    p.add_argument('--probabilistic', action='store_true')
    p.add_argument('--detect-divergence', action='store_true')
    p.add_argument('--eos', action='store_true', help='exercise EOS truncation')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    target = tiny_qwen3_dir('tiny_gqa')
    draft = Path(tiny_eagle3_dir('tiny_gqa'))
    if args.probabilistic:
        from safetensors.torch import load_file, save_file
        import torch
        reduced = Path(tempfile.mkdtemp(prefix='step108_reduced_'))
        draft_config = json.loads((draft/'config.json').read_text())
        draft_config['draft_vocab_size'] = 7
        (reduced/'config.json').write_text(json.dumps(draft_config))
        weights = load_file(str(draft/'model.safetensors'))
        ids = torch.tensor([0, 1, 3, 5, 6, 8, 10])
        weights['lm_head.weight'] = weights['lm_head.weight'][ids].contiguous()
        weights['d2t'] = ids - torch.arange(len(ids))
        weights['t2d'] = torch.zeros(11, dtype=torch.bool); weights['t2d'][ids] = True
        save_file(weights, str(reduced/'model.safetensors'))
        draft = reduced
    hf = tiny_qwen3_config('tiny_gqa')
    if args.mixed and (args.batch < 2 or args.accepted is not None):
        p.error('mixed requires batch >= 2 and excludes --accepted')
    if args.probabilistic and (args.mixed or args.accepted is not None):
        p.error('probabilistic mode does not force candidate IDs')
    if args.eos and (args.probabilistic or args.mixed or args.accepted is not None):
        p.error('EOS mode does not force candidates')
    if args.accepted is not None and not 0 <= args.accepted <= args.k:
        p.error('accepted must be in 0..K')

    def run(speculative, baseline=None):
        spec = None if not speculative else SpeculativeConfig(method='eagle3', num_speculative_tokens=args.k,
            draft_sample_method='probabilistic' if args.probabilistic else 'greedy',
            draft_model_config=ModelConfig(str(draft), dtype='float32', max_model_len=64,
                                           hf_config=json.loads((draft/'config.json').read_text())))
        cfg = VllmConfig(model_config=ModelConfig(target, dtype='float32', max_model_len=64, hf_config=hf),
            cache_config=CacheConfig(block_size=16, num_gpu_blocks=16),
            scheduler_config=SchedulerConfig(max_num_seqs=args.batch, max_num_batched_tokens=64),
            device_config=DeviceConfig(device='cuda'), compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if args.graph and speculative else 'none', cudagraph_capture_sizes=sorted({1, 2, 4, *((args.k + 1) * n for n in (1, 2, 4))}) if args.graph and speculative else None),
            speculative_config=spec, parallel_config=ParallelConfig(tensor_parallel_size=args.tp),
            attention_backend='triton')
        executor = MultiprocExecutor(cfg, _worker_cls=Step108Worker)
        engine = LLMEngine(cfg, executor, tokenizer=None)
        outputs, rounds, metrics, per_step_graph = {}, [], [], []
        try:
            if speculative:
                forcing = None
                if args.accepted is not None or args.mixed:
                    forcing = {}
                    for index, (name, history) in enumerate(sorted(baseline.items())):
                        accepted = (0 if index % 2 == 0 else args.k) if args.mixed else args.accepted
                        candidates = history[1:1+args.k]
                        if accepted < args.k:
                            candidates[accepted] = (candidates[accepted] + 1) % 11
                        forcing[name] = candidates
                executor.collective_rpc('step108_set_forcing', args=(forcing,))
            for index in range(args.batch):
                name = chr(ord('a') + index)
                prompt = [1 + (index % 4), 2, 3, 4, 5, 6]
                engine.add_request(name, prompt, SamplingParams(max_tokens=12,
                    temperature=.8 if args.probabilistic else 0, seed=123+index,
                    eos_token_id=1 if args.eos else None, ignore_eos=not args.eos))
            for _ in range(100):
                if not engine.has_unfinished_requests():
                    break
                for output in engine.step():
                    outputs[output.request_id] = list(output.token_ids)
                if speculative:
                    stats = engine.last_scheduler_stats.spec_decoding_stats
                    metrics.append(None if stats is None else {
                        'num_drafts': stats.num_drafts, 'num_draft_tokens': stats.num_draft_tokens,
                        'num_accepted_tokens': stats.num_accepted_tokens,
                        'num_accepted_tokens_per_pos': list(stats.num_accepted_tokens_per_pos),
                        'num_draft_tokens_per_pos': list(stats.num_draft_tokens_per_pos)})
                    rounds.append(executor.collective_rpc('step108_snapshot'))
                    if args.graph:
                        per_step_graph.append(executor.collective_rpc(step109_graph_stats))
                    if args.detect_divergence and len(rounds) == 1:
                        for field in ('seed', 'length'):
                            executor.collective_rpc('step108_skew_rank1', args=(field, 1))
                            altered = executor.collective_rpc('step108_snapshot')
                            try:
                                assert_rank_consistent(altered)
                            except AssertionError:
                                pass
                            else:
                                raise AssertionError(f'failed to diagnose rank {field} divergence')
                            executor.collective_rpc('step108_skew_rank1', args=(field, -1))
                        assert_rank_consistent(executor.collective_rpc('step108_snapshot'))
                else:
                    rounds.append(executor.collective_rpc('step108_snapshot'))
            assert not engine.has_unfinished_requests()
            assert len(outputs) == args.batch
            assert all(1 <= len(value) <= 12 for value in outputs.values())
            if not args.eos:
                assert all(len(value) == 12 for value in outputs.values())
            graph_stats = executor.collective_rpc(step109_graph_stats) if speculative else []
            if args.graph and speculative:
                for first in per_step_graph[0]:
                    assert first['target']['replays'] == 0 and first['prefill']['replays'] == 0, first
                for snapshot in per_step_graph:
                    for before, after in zip(snapshot, graph_stats):
                        for name in ('target', 'prefill', 'decode'):
                            assert before[name]['captured'] == after[name]['captured']
            return outputs, rounds, metrics, graph_stats
        finally:
            engine.shutdown()

    baseline, baseline_rounds, _, _ = run(False)
    outputs, rounds, metrics, graph_stats = run(True, baseline)
    if args.graph:
        for stats in graph_stats:
            assert stats['target']['captured'] and stats['target']['replays'], stats
            assert stats['prefill']['captured'] and stats['prefill']['replays'], stats
            if args.k > 1:
                assert stats['decode']['captured'] and stats['decode']['replays'], stats
    kv_comparisons = 0
    if not args.probabilistic:
        assert outputs == baseline, (baseline, outputs)
        import torch
        for snapshots in rounds:
            for rank, snapshot in enumerate(snapshots):
                for req_id, state in snapshot['state'].items():
                    matching = [old[rank] for old in baseline_rounds
                                if req_id in old[rank]['state']
                                and old[rank]['state'][req_id]['computed'] == state['computed']]
                    if matching and state['computed']:
                        reference = torch.tensor(matching[-1]['local_kv'][req_id]['target'])
                        actual = torch.tensor(snapshot['local_kv'][req_id]['target'])
                        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
                        kv_comparisons += 1
        assert kv_comparisons >= args.tp * args.batch
    if args.eos:
        assert outputs['a'][-1] == 1 and len(outputs['a']) < 12
    for snapshots in rounds:
        assert_rank_consistent(snapshots)
    trace = rounds[-1][0]['trace']
    combined = [event for event in trace if event['kind'] == 'combine']
    prefilled = [event for event in trace if event['kind'] == 'prefill_hidden']
    assert combined
    if not args.graph:
        assert len(combined) == len(prefilled)
        for c, f in zip(combined, prefilled):
            assert c['combined_hidden'] == f['draft_hidden'][:len(c['combined_hidden'])]
    for event in (event for event in trace if event['kind'] == 'prefill'):
        for row in range(len(event['req_ids'])):
            start, end = event['query_start_loc'][row:row+2]
            valid_end = end - event['num_rejected'][row]
            assert event['last_indices'][row] == valid_end - 1
            assert event['draft_positions'][start:valid_end] == event['target_positions'][start:valid_end]
            assert event['draft_input_ids'][start:valid_end-1] == event['target_input_ids'][start+1:valid_end]
            expected_tail = event['last_sampled'][row] if event['num_sampled'][row] else event['next_prefill'][row]
            assert event['draft_input_ids'][valid_end-1] == expected_tail, (event, row, expected_tail)
            assert event['draft_input_ids'][valid_end:end] == [None] * (end - valid_end)
    sample = [event for event in trace if event['kind'] == 'sample']
    if args.probabilistic:
        import torch
        for event in (event for event in trace if event['kind'] == 'propose'):
            logits = torch.tensor(event['draft_logits'])
            assert logits.shape == (len(event['req_ids']), args.k, 11)
            probability = (logits / .8).softmax(-1)
            torch.testing.assert_close(probability.sum(-1), torch.ones_like(probability[..., 0]))
            assert probability[..., [2, 4, 7, 9]].count_nonzero() == 0
    if args.accepted is not None or args.mixed:
        assert len(sample) >= 3
        second = sample[1]
        assert metrics[1] is not None
        accepted_values = []
        for index, req_id in enumerate(second['req_ids']):
            request_index = ord(req_id) - ord('a')
            accepted = (0 if request_index % 2 == 0 else args.k) if args.mixed else args.accepted
            accepted_values.append(accepted)
            assert second['num_sampled'][index] == accepted + 1
            assert second['num_rejected'][index] == args.k - accepted
            third = sample[2]
            next_index = third['req_ids'].index(req_id)
            assert third['positions'][third['query_start_loc'][next_index]] == (
                second['positions'][second['query_start_loc'][index]] + accepted + 1)
        assert metrics[1]['num_drafts'] == len(accepted_values)
        assert metrics[1]['num_draft_tokens'] == len(accepted_values) * args.k
        assert metrics[1]['num_accepted_tokens'] == sum(accepted_values)
        assert metrics[1]['num_accepted_tokens_per_pos'] == [sum(r > pos for r in accepted_values)
                                                              for pos in range(args.k)]
        assert metrics[1]['num_draft_tokens_per_pos'] == [len(accepted_values)] * args.k
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'tp': args.tp, 'k': args.k, 'batch': args.batch,
        'accepted': args.accepted, 'probabilistic': args.probabilistic,
        'outputs': outputs, 'ranks': rounds, 'metrics': metrics, 'graph_stats': graph_stats,
        'kv_comparisons': kv_comparisons})+'\n')
    print(json.dumps({'status': 'passed', 'tp': args.tp, 'k': args.k, 'outputs': outputs}))


if __name__ == '__main__':
    main()
