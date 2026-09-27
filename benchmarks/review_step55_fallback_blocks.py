"""验收补测：预留 K>0、实际 K=0 的 fallback 必须归还 target 多预留整块。"""
from collections import Counter
import json
import math
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step55 import Engine, TinyCausalLM


def run(draft_budget):
    dims = dict(vocab_size=32, d_model=16, num_q_heads=2, num_kv_heads=1,
                head_dim=8, num_layers=1, intermediate_size=32, max_seq_len=64,
                eos_token_ids=[31], device='cpu', max_num_query_tokens=16)
    torch.manual_seed(1)
    target, draft = TinyCausalLM(**dims), TinyCausalLM(**dims)
    # 固定输出 0，排除随机初始化模型提前 EOS 的干扰。实际前向与 KV 写入不替换。
    with torch.no_grad():
        target.lm_head.weight.zero_()
        draft.lm_head.weight.zero_()
    e = Engine(model=target, draft_model=draft, max_num_seqs=1,
               max_num_batched_tokens=16, block_size=4, num_kv_blocks=8,
               draft_num_kv_blocks=8, draft_max_num_batched_tokens=draft_budget,
               enable_prefix_caching=False, speculative_mode='draft_model',
               num_speculative_tokens=2)
    e.add_request(dict(request_id='A', prompt_ids=[1,2,3,4,5,6,7], max_new_tokens=8))
    trace, finished, events = [], [], []
    e.on_token = lambda ev: events.append(dict(ev))
    for step in range(1, 40):
        if not e.has_unfinished_requests():
            break
        finished += e.step()
        items = e.scheduler.scheduled_items
        for q in e.scheduler.running:
            it = next(it for it in items if it['request'] is q)
            trace.append(dict(step=step, target_length=q.cache.length,
                              target_blocks=list(q.cache.block_table),
                              expected_blocks=math.ceil(q.cache.length / 4),
                              reserved_drafts=it['num_reserved_drafts'],
                              actual_drafts=list(it['draft_ids']),
                              draft_length=q.draft_cache.length))
        for pool, field in ((e.kv_cache_pool,'cache'),(e.draft_kv_pool,'draft_cache')):
            refs = Counter(b for q in e.scheduler.running for b in (getattr(q,field).block_table or []))
            assert pool.block_usage == [refs[i] for i in range(pool.num_kv_blocks)]
    else:
        raise AssertionError('超过 step 上限')
    assert len(finished)==1 and finished[0]['output_ids']==[0]*8
    assert [ev['output_index'] for ev in events]==list(range(8))
    assert all(u==0 for u in e.kv_cache_pool.block_usage+e.draft_kv_pool.block_usage)
    bad=[t for t in trace if len(t['target_blocks']) != t['expected_blocks']]
    return dict(draft_budget=draft_budget, result='FAIL' if bad else 'PASS',
                mismatches=bad, trace=trace,
                proposed_tokens=e.draft_proposer.num_proposed_tokens,
                output=finished[0]['output_ids'])


if __name__=='__main__':
    torch.set_num_threads(1)
    results=[run(1),run(16)]
    for r in results:
        print(r['result'], 'draft_budget=',r['draft_budget'],
              'proposed_tokens=',r['proposed_tokens'], 'mismatches=',r['mismatches'])
    path=Path('/tmp/step55_acceptance/fallback_blocks.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    sys.exit(1 if any(r['mismatches'] for r in results) else 0)
