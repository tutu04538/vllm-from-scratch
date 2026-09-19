"""CPU attention-only comparison, including gather/block views and position metadata."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import importlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.benchmark import Timer
PROJECT = Path(__file__).resolve().parents[1]
RECORDS = Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')
sys.path[:0]=[str(p) for p in PROJECT.glob("step[0-9][0-9]") if p.is_dir()]+[str(PROJECT)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['step20', 'step21'], required=True)
    parser.add_argument('--workload', choices=['decode32', 'chunk13', 'prefill32'], required=True)
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(0)
    m = importlib.import_module(args.mode); source = Path(m.__file__); digest = sha(source)
    fixtures = {}
    for kind in ['model', 'prefix', 'disabled_engine'] + (['attention'] if args.mode == 'step21' else []):
        path = sorted(RECORDS.glob(args.mode+'_'+kind+'_contract_*.json'))[-1]
        d = json.loads(path.read_text())
        assert d['status'] == 'passed' and d['source_sha256'] == digest
        fixtures[kind] = str(path)
    length, count = {'decode32': (32, 1), 'chunk13': (13, 5), 'prefill32': (32, 32)}[args.workload]
    dim = 8; size = 4
    model = m.TinyCausalLM(d_model=dim).eval()
    pool = m.KVCachePool(size, 12, dim, torch.device('cpu'))
    table = list(range(9, 1, -1))
    cache = m.CacheConfig(block_table=table, length=length)
    q = torch.randn(count, dim); k = torch.randn(length, dim); v = torch.randn(length, dim)
    input_hash = hashlib.sha256(b''.join(t.numpy().tobytes() for t in (q,k,v))).hexdigest()
    pool.k_cache.fill_(10000); pool.v_cache.fill_(-10000)
    for pos in range(length):
        pool.k_cache[table[pos//size], pos%size] = k[pos]
        pool.v_cache[table[pos//size], pos%size] = v[pos]
    saved = (pool.k_cache.clone(), pool.v_cache.clone())
    pointers = (pool.k_cache.data_ptr(), pool.v_cache.data_ptr())
    scores = q.double() @ k.double().T / math.sqrt(dim)
    positions = torch.arange(length-count, length)
    scores.masked_fill_(torch.arange(length)[None,:] > positions[:,None], float('-inf'))
    want = torch.softmax(scores, dim=-1) @ v.double()

    def run():
        if args.mode == 'step21':
            query_positions = torch.arange(length-count, length, device=q.device)
            return model.block_attention(q, cache, pool, query_positions)
        # Same attention fragment as step20._forward_append; projection/write/lm_head excluded in both modes.
        past_k, past_v = pool.gather(cache)
        score = torch.matmul(q, past_k.transpose(-1,-2)) / (dim**0.5)
        query_pos = torch.arange(length-count, length, device=q.device).unsqueeze(-1)
        key_pos = torch.arange(length, device=q.device).unsqueeze(0)
        score = score.masked_fill(key_pos>query_pos, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        return torch.matmul(weights, past_v)

    with torch.inference_mode():
        before = run()
        torch.testing.assert_close(before.double(), want, atol=1e-5, rtol=1e-5)
        t = Timer(stmt='fn()', globals={'fn':run}, num_threads=1).blocked_autorange(min_run_time=0.5)
        after = run()
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        torch.testing.assert_close(after.double(), want, atol=1e-5, rtol=1e-5)
    assert all(torch.equal(a,b) for a,b in zip(saved,(pool.k_cache,pool.v_cache)))
    assert pointers == (pool.k_cache.data_ptr(),pool.v_cache.data_ptr())
    assert sha(source) == digest
    report = dict(mode=args.mode, workload=args.workload, source_sha256=digest, script_sha256=sha(Path(__file__)),
                  fixtures=fixtures, device='cpu', dtype='float32', threads=1, torch=torch.__version__,
                  seed=0, input_hash=input_hash, length=length, query_count=count, d_model=dim, block_size=size,
                  timing=dict(attention=dict(median_us=t.median*1e6, iqr_us=t.iqr*1e6, iqr_over_median=t.iqr/t.median,
                                             high_variance=t.iqr/t.median>0.1, number_per_run=t.number_per_run, raw_times=t.raw_times)),
                  validation=dict(max_abs_error=(after.double()-want).abs().max().item(), pool_unchanged=True),
                  scope='Attention only: query/key position construction, gather or direct block access, score/mask/normalization/value accumulation. Excludes model/pool construction, QKV projection, KV append, lm_head. CPU FP32 inference_mode for both. step20 mathematical fragment copied from _forward_append; step21 uses its block_attention.')
    path = PROJECT/'benchmarks/results'/('step21_attention_'+args.workload+'_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(args.workload,args.mode,round(t.median*1e6,3),path)


if __name__ == '__main__':
    main()
