"""CPU fixed KV workload; include address construction, no observer in timing."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.benchmark import Timer
PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(p) for p in PROJECT.glob("step[0-9][0-9]") if p.is_dir()]+[str(PROJECT)]
RECORDS = Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['step19', 'step20'], required=True)
    parser.add_argument('--workload', choices=['single_short', 'mixed_chunks', 'three_long_decode'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    m = importlib.import_module(args.mode)
    source = Path(m.__file__); digest = sha(source)
    kinds = ['model', 'prefix', 'disabled_engine'] + (['slots'] if args.mode == 'step20' else [])
    fixtures = {}
    for kind in kinds:
        path = sorted(RECORDS.glob(args.mode+'_'+kind+'_contract_*.json'))[-1]
        record = json.loads(path.read_text())
        assert record['status'] == 'passed' and record['source_sha256'] == digest
        fixtures[kind] = str(path)
    lengths, counts = {'single_short': ([3], [1]), 'mixed_chunks': ([17,3,0], [1,5,7]),
                       'three_long_decode': ([24,16,8], [1,1,1])}[args.workload]
    pool = m.KVCachePool(4, 24, 8, torch.device('cpu'))
    tables = [list(range(i*8, i*8+8))[::-1] for i in range(len(counts))]
    caches = [m.CacheConfig(block_table=t, length=n) for t, n in zip(tables, lengths)]
    for table in tables:
        for b in table:
            pool.block_usage[b] = 1
    values = torch.arange(pool.k_cache.numel(), dtype=torch.float32).reshape_as(pool.k_cache)
    pool.k_cache.copy_(values); pool.v_cache.copy_(-values)
    initial = (pool.k_cache.clone(), pool.v_cache.clone())
    pointers = (pool.k_cache.data_ptr(), pool.v_cache.data_ptr())
    k = torch.arange(sum(counts)*8, dtype=torch.float32).reshape(-1, 8)+10000
    v = -k
    offsets = [0]
    for n in counts:
        offsets.append(offsets[-1]+n)

    def append():
        # Reset lengths is included equally; history/new values are identical each iteration.
        for cache, length in zip(caches, lengths):
            cache.length = length
        if args.mode == 'step20':
            pool.append_batch(caches, counts, k, v)
        else:
            for cache, start, end in zip(caches, offsets[:-1], offsets[1:]):
                pool.append(cache, k[start:end], v[start:end])

    def gather():
        return [pool.gather(cache) for cache in caches]

    def combined():
        append()
        return gather()

    slots = [table[p//4]*4+p%4 for table, old, n in zip(tables, lengths, counts) for p in range(old, old+n)]
    expected = []
    for a, new in zip(initial, (k, v)):
        gold = a.clone()
        for slot, row in zip(slots, new):
            gold[slot//4, slot%4] = row
        expected.append(gold)

    def check():
        result = combined()
        assert [c.length for c in caches] == [n+c for n, c in zip(lengths, counts)]
        assert pointers == (pool.k_cache.data_ptr(), pool.v_cache.data_ptr())
        for actual, want in zip((pool.k_cache, pool.v_cache), expected):
            assert torch.equal(actual, want)
        for cache, pair in zip(caches, result):
            for actual, storage in zip(pair, expected):
                want = torch.stack([storage[cache.block_table[p//4], p%4] for p in range(cache.length)])
                assert torch.equal(actual, want)

    timing = {}
    with torch.inference_mode():
        check()
        for name, fn in [('append_with_length_reset', append), ('gather', gather), ('append_and_gather_with_length_reset', combined)]:
            t = Timer(stmt='fn()', globals={'fn': fn}, num_threads=1).blocked_autorange(min_run_time=0.5)
            timing[name] = dict(median_us=t.median*1e6, iqr_us=t.iqr*1e6, iqr_over_median=t.iqr/t.median,
                                high_variance=t.iqr/t.median>0.1, number_per_run=t.number_per_run, raw_times=t.raw_times)
        check()
    assert sha(source) == digest
    report = dict(mode=args.mode, workload=args.workload, source_sha256=digest, script_sha256=sha(Path(__file__)),
                  fixtures=fixtures, device='cpu', dtype='float32', threads=1, torch=torch.__version__,
                  block_size=4, d_model=8, tables=tables, lengths_before=lengths, counts=counts, slots=slots,
                  validation='Exact whole-pool and gathered K/V before and after timing; stable pointers', timing=timing,
                  scope='Pool initialization and tensor data construction excluded. Append includes length reset, address generation, writes and length increments; gather includes address generation and copies. Same input values overwrite the same new slots each iteration; history remains fixed. Both use inference_mode.')
    path = PROJECT/'benchmarks/results'/('step20_kv_'+args.workload+'_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())+'.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(args.workload, args.mode, {k:round(v['median_us'],3) for k,v in timing.items()}, path)


if __name__ == '__main__':
    main()
