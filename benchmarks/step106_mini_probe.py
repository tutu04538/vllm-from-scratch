"""106：真实 Qwen3-8B + EAGLE3 单卡 eager 诊断运行。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig,
                      SamplingParams, SchedulerConfig, SpeculativeConfig, VllmConfig)
from minivllm.config import CompilationConfig
from minivllm.executor.uniproc_executor import UniProcExecutor
from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.step103_workload import prompts
from benchmarks.step106_reference_probe import install, flush


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forced-from', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target, _ = resolve_snapshot('8b')
    draft = Path('/data/syh/models/converted/AngelSlim/Qwen3-8B_eagle3/9629dfce7a4a10564dd48d3e5485c3976095653c')
    target_cfg = json.loads((target / 'config.json').read_text())
    draft_cfg = json.loads((draft / 'config.json').read_text())
    forced_history = forced_drafts = None
    if args.forced_from:
        import torch
        prior = json.loads((args.forced_from / 'summary.json').read_text())
        if prior['k'] != args.k:
            raise ValueError('固定历史的 K 不匹配')
        forced_history = prior['runs'][0]['tokens']
        events = torch.load(prior['runs'][0]['trace'], map_location='cpu', weights_only=True)
        forced_drafts = [event['output'].tolist() for event in events if event['event'] == 'propose']
    spec = SpeculativeConfig(method='eagle3', num_speculative_tokens=args.k,
                             draft_model_config=ModelConfig(str(draft), dtype='bfloat16',
                                                            max_model_len=2048, hf_config=draft_cfg))
    config = VllmConfig(model_config=ModelConfig(str(target), dtype='bfloat16', max_model_len=2048,
                                                hf_config=target_cfg, seed=0),
                        cache_config=CacheConfig(block_size=16, num_gpu_blocks=192,
                                                 enable_prefix_caching=False),
                        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
                        device_config=DeviceConfig(device='cuda'),
                        compilation_config=CompilationConfig(cudagraph_mode='none'),
                        speculative_config=spec, attention_backend='triton')
    start = time.monotonic()
    executor = UniProcExecutor(config)
    engine = None
    try:
        engine = LLMEngine(config, executor, tokenizer=None)
        identity = install(executor.driver_worker, upstream=False,
                           forced_history=forced_history, forced_drafts=forced_drafts)
        prompt = prompts('8b', 1)[0]
        engine.add_request('request0', prompt, SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
        tokens = None
        for _ in range(256):
            if not engine.has_unfinished_requests():
                break
            for item in engine.step():
                tokens = list(item.token_ids)
        else:
            raise RuntimeError('request did not finish')
        path = (args.output / 'run0.pt').resolve()
        count = flush(executor.driver_worker, str(path))
        payload = {'status': 'mini_run_complete', 'k': args.k, 'identity': identity,
                   'prompt_token_ids': prompt, 'tokens': tokens, 'trace': str(path),
                   'events': count, 'elapsed_seconds': time.monotonic() - start}
        (args.output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'status': payload['status'], 'k': args.k, 'events': count}))
    finally:
        if engine is not None:
            engine.shutdown()
        else:
            executor.shutdown()


if __name__ == '__main__':
    main()
