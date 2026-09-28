"""独立复验：GPU 抢占恢复后，两套有效 KV 与已提交前缀单独重算一致。"""
from collections import Counter
from pathlib import Path
import json
import math
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_step56_recovery_rng as scenarios
from step56 import Engine
from step56.cache import CacheConfig, KVCachePool


def compare(model, pool, cache, tokens):
    length = cache.length
    if length == 0:
        return 0.0
    def gather(p, c):
        probe = CacheConfig(block_table=list(c.block_table), length=0)
        slots = p.build_slot_mapping([probe], [length])
        return p.k_flat[:, slots].clone(), p.v_flat[:, slots].clone()
    actual = gather(pool, cache)
    fresh = KVCachePool(pool.block_size, math.ceil(length/pool.block_size)+1,
                        model.num_kv_heads, model.head_dim, model.device, False,
                        num_layers=model.num_layers, dtype=model.dtype)
    ref = CacheConfig()
    fresh.ensure_blocks_for(ref, length)
    with torch.inference_mode():
        model._forward_append(torch.tensor(list(tokens[:length]), dtype=torch.long),
                              [length], [ref], fresh, sample_rows=[])
    return max(float((a-b).abs().max()) for a,b in zip(actual,gather(fresh,ref)))


def main():
    torch.set_num_threads(1)
    reports=[]
    for budget,prefix in ((4,False),(9,False),(3,True)):
        stats=dict(budget=budget,prefix=prefix,checked=0,target_error=0.,draft_error=0.,
                   max_reused_tokens=0,nonempty_draft_checks=0)
        def factory(**kwargs):
            kwargs.update(max_num_batched_tokens=budget,enable_prefix_caching=prefix)
            engine=Engine(**kwargs)
            original=engine.step
            def observed_step():
                result=original()
                if engine.scheduler.num_preemptions:
                    for attr,model_attr,pool_attr in (('cache','model','kv_cache_pool'),
                                                      ('draft_cache','draft_model','draft_kv_pool')):
                        pool=getattr(engine,pool_attr)
                        refs=Counter(b for seq in engine.scheduler.running
                                     for b in (getattr(seq,attr).block_table or []))
                        assert pool.block_usage == [refs[b] for b in range(pool.num_kv_blocks)]
                        for seq in engine.scheduler.running:
                            cache=getattr(seq,attr)
                            stats['max_reused_tokens']=max(stats['max_reused_tokens'],seq.reused_tokens)
                            if attr=='draft_cache' and cache.length:
                                stats['nonempty_draft_checks']+=1
                            assert len(cache.block_table or []) == math.ceil(cache.length/pool.block_size)
                            error=compare(getattr(engine,model_attr),pool,cache,seq.all_token_ids)
                            key='target_error' if attr=='cache' else 'draft_error'
                            stats[key]=max(stats[key],error)
                            stats['checked']+=1
                return result
            engine.step=observed_step
            return engine
        scenarios.Engine=factory
        result=scenarios.run_recovery(True)
        ok=result['preemptions']>0 and stats['checked']>0 and max(stats['target_error'],stats['draft_error'])<1e-4
        stats.update(passed=ok,preemptions=result['preemptions'])
        reports.append(stats)
        print('PASS' if ok else 'FAIL','恢复后 GPU 有效 KV/持块/引用核对',stats)
    path=Path('/tmp/step56_final_review/recovery_kv.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(reports,indent=2)+'\n')
    return 0 if all(x['passed'] for x in reports) else 1


if __name__=='__main__':
    sys.exit(main())
