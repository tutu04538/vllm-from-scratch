"""Record real EAGLE3 layer inputs/outputs from a frozen vLLM TP engine."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
from benchmarks.step106_reference_probe import install, flush
from benchmarks.check_step090_reference import verify_oracle
from benchmarks.step103_frozen_probe import resolve_snapshot, prompts


def flush_rank(worker, path):
    import torch.distributed as dist
    return flush(worker, str(Path(path) / f'rank{dist.get_rank()}.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['8b', '14b'], required=True)
    parser.add_argument('--tp', type=int, choices=[2, 4], required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.75)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    oracle = verify_oracle()
    target, _ = resolve_snapshot(args.model)
    draft = (Path('/data/syh/models/converted/AngelSlim') / {
        '8b': 'Qwen3-8B_eagle3/9629dfce7a4a10564dd48d3e5485c3976095653c',
        '14b': 'Qwen3-14B_eagle3/3d13517724e81cb409ddf1d4650772ec52f1e18e'}[args.model])
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(target), skip_tokenizer_init=True, dtype='bfloat16',
              tensor_parallel_size=args.tp, enforce_eager=True, max_model_len=2048,
              max_num_seqs=1, max_num_batched_tokens=128,
              gpu_memory_utilization=args.gpu_memory_utilization, attention_config={'backend': 'TRITON_ATTN'},
              enable_prefix_caching=False, disable_log_stats=True, seed=0,
              speculative_config={'method': 'eagle3', 'model': str(draft), 'num_speculative_tokens': 2})
    identities = llm.collective_rpc(install, args=(True, None, None))
    prompt = prompts(args.model, 1)[0]
    runs = []
    for index in range(2):
        path = args.output / f'run{index}'
        path.mkdir(exist_ok=True)
        outputs = llm.generate([{'prompt_token_ids': prompt}],
                               SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), use_tqdm=False)
        counts = llm.collective_rpc(flush_rank, args=(str(path.resolve()),))
        runs.append({'tokens': list(outputs[0].outputs[0].token_ids), 'events_per_rank': counts})
    (args.output / 'summary.json').write_text(json.dumps({'oracle': oracle, 'model': args.model,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'tp': args.tp, 'identities': identities, 'runs': runs, 'prompt': prompt}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': 'reference_runs_complete', 'model': args.model, 'tp': args.tp}))


if __name__ == '__main__':
    main()
