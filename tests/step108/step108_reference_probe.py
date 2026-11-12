"""Record tiny EAGLE3 per-rank frozen or project execution on the same fixture."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
from benchmarks.step106_reference_probe import install, flush
from benchmarks.check_step090_reference import verify_oracle
from step108_fixture import create_pair


def flush_rank(worker, directory):
    import torch.distributed as dist
    return flush(worker, str(Path(directory)/f'rank{dist.get_rank()}.pt'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=['frozen', 'mini'], required=True)
    p.add_argument('--tp', type=int, choices=[2, 4], required=True)
    p.add_argument('--k', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alias-probe', action='store_true')
    p.add_argument('--force-first', help='JSON list of B=1 candidate IDs')
    args = p.parse_args()
    forced = None if args.force_first is None else [json.loads(args.force_first)]
    if forced is not None and len(forced[0]) != args.k:
        p.error('force-first must contain exactly K IDs')
    target, draft = args.fixture/'target', args.fixture/'draft'
    if args.implementation == 'frozen':
        if not (target/'config.json').exists():
            create_pair(args.fixture)
        verify_oracle()
    elif not (target/'config.json').exists() or not (draft/'config.json').exists():
        p.error('the frozen run must create the shared fixture first')
    args.output.mkdir(parents=True, exist_ok=True)
    prompt = [1, 2, 3, 4, 5, 6]
    if args.implementation == 'frozen':
        from vllm import LLM, SamplingParams
        engine = LLM(model=str(target.resolve()), skip_tokenizer_init=True, dtype='float32',
            tensor_parallel_size=args.tp, enforce_eager=True, max_model_len=64, max_num_seqs=1,
            max_num_batched_tokens=64, gpu_memory_utilization=.25, kv_cache_memory_bytes=33554432,
            attention_config={'backend': 'TRITON_ATTN'}, enable_prefix_caching=False,
            disable_log_stats=True, seed=0,
            speculative_config={'method': 'eagle3', 'model': str(draft.resolve()),
                                'num_speculative_tokens': args.k})
        identities = engine.collective_rpc(install, args=(True, None, None, 32, forced))
        if args.alias_probe:
            from step108_alias_hooks import install_alias
            engine.collective_rpc(install_alias)
        output = engine.generate([{'prompt_token_ids': prompt}],
            SamplingParams(temperature=0, max_tokens=12, ignore_eos=True, seed=123), use_tqdm=False)
        tokens = list(output[0].outputs[0].token_ids)
        counts = engine.collective_rpc(flush_rank, args=(str(args.output.resolve()),))
        if args.alias_probe:
            from step108_alias_hooks import flush_alias
            engine.collective_rpc(flush_alias, args=(str(args.output.resolve()),))
    else:
        from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                              SchedulerConfig, SpeculativeConfig, VllmConfig)
        from minivllm.config import CompilationConfig, ParallelConfig
        from minivllm.executor.multiproc_executor import MultiprocExecutor
        cfg = VllmConfig(model_config=ModelConfig(str(target), dtype='float32', max_model_len=64,
                           hf_config=json.loads((target/'config.json').read_text()), seed=0),
            cache_config=CacheConfig(block_size=16, num_gpu_blocks=8),
            scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=64),
            device_config=DeviceConfig('cuda'), parallel_config=ParallelConfig(tensor_parallel_size=args.tp),
            compilation_config=CompilationConfig(cudagraph_mode='none'), attention_backend='triton',
            speculative_config=SpeculativeConfig(method='eagle3', num_speculative_tokens=args.k,
                draft_model_config=ModelConfig(str(draft), dtype='float32', max_model_len=64,
                    hf_config=json.loads((draft/'config.json').read_text()))))
        executor = MultiprocExecutor(cfg); engine = LLMEngine(cfg, executor, tokenizer=None)
        try:
            identities = executor.collective_rpc(install, args=(False, None, None, 32, forced))
            engine.add_request('a', prompt, SamplingParams(temperature=0, max_tokens=12,
                                                            ignore_eos=True, seed=123))
            tokens = []
            while engine.has_unfinished_requests():
                for output in engine.step():
                    tokens = list(output.token_ids)
            counts = executor.collective_rpc(flush_rank, args=(str(args.output.resolve()),))
        finally:
            engine.shutdown()
    (args.output/'summary.json').write_text(json.dumps({'implementation': args.implementation, 'tp': args.tp,
        'k': args.k, 'forced_first': forced, 'tokens': tokens, 'counts': counts, 'identities': identities}, indent=2)+'\n')
    print(json.dumps({'status': 'complete', 'implementation': args.implementation, 'tokens': tokens}))


if __name__ == '__main__':
    main()
