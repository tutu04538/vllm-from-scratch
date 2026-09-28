"""复用原真实模型测试的构造/请求，只驻留一个双模型引擎检验 GPU 随机回放。"""
from pathlib import Path

source = Path(__file__).with_name('check_step56_real_qwen3.py')
# 原脚本没有 main 保护：只装载 helper 定义，不执行会同时持有多套模型的顶层测试。
helpers, _ = source.read_text().split('tokenizer, prompts = encode(PROMPTS)', 1)
exec(compile(helpers, str(source), 'exec'), globals())

tokenizer, prompts = encode(PROMPTS)
engine, counters = build('draft_model', num_speculative_tokens=3, rejection_backend='triton')
sampling = dict(temperature=0.8, top_k=20, top_p=0.9, seed=7)
a, events_a, steps_a = run(engine, prompts, max_new_tokens=16, sampling=sampling)
accepted_a = engine.sample_runtime.num_accepted_drafts
b, events_b, steps_b = run(engine, prompts, max_new_tokens=16, sampling=sampling)
accepted_b = engine.sample_runtime.num_accepted_drafts - accepted_a
check('GPU 随机：相同 seed 的新请求复用引擎后输出可复现', a == b, str(a))
check('GPU 随机：两轮都实际接受草稿', accepted_a > 0 and accepted_b > 0,
      f'accepted={accepted_a},{accepted_b}')
check('GPU 随机：两轮回调序号连续', all(
    [idx for rid, _, idx in events if rid == key] == list(range(len(outputs[key])))
    for outputs, events in ((a, events_a), (b, events_b)) for key in outputs))
check('GPU 随机：两套池活动引用归零', all(x == 0 for x in engine.kv_cache_pool.block_usage)
      and all(x == 0 for x in engine.draft_kv_pool.block_usage))
sys.exit(1 if FAIL else 0)
