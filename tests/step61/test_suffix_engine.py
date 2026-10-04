"""step61：真引擎端到端（方法='suffix'）——候选链路、统计、生命周期。

数值正确性口径与 57/58/59/60 一致：**greedy 下投机与不投机的输出必须逐 token 相同**
（草稿只是猜测，target 验证决定最终答案）。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suffix_helpers import make_suffix_engine, step59_helpers  # noqa: E402

from minivllm import SamplingParams  # noqa: E402

# 有重复模式（"1 2 3 4" 出现两次）→ suffix 树里有可匹配的后缀，能真的产出候选
REPEAT_PROMPT = [1, 2, 3, 4, 1, 2, 3, 4]


def _run(engine, core, requests, *, max_tokens=6, collect_stats=True):
    """加请求 → 跑到没有未完成请求；返回 `(每条输出, 逐步 stats)`。"""
    for req_id, prompt in requests:
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                          eos_token_id=999))
    outputs, stats = {}, []
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        if collect_stats:
            stats.append(core.scheduler.spec_decoding_stats)
    return outputs, stats


def _totals(stats):
    """把逐步 stats 累加成 `(drafts, draft_tokens, accepted)`。"""
    drafts = sum(s.num_drafts for s in stats if s is not None)
    draft_tokens = sum(s.num_draft_tokens for s in stats if s is not None)
    accepted = sum(s.num_accepted_tokens for s in stats if s is not None)
    return drafts, draft_tokens, accepted


@pytest.fixture
def suffix_engine(tiny_dir, hf_config, device):
    engines = []

    def build(**kwargs):
        engine, core, runner = make_suffix_engine(tiny_dir=tiny_dir, hf_config=hf_config,
                                                  **kwargs)
        engines.append(engine)
        return engine, core, runner

    yield build, device
    for engine in engines:
        engine.shutdown()


def test_greedy_output_matches_non_speculative(tiny_dir, hf_config, device):
    """greedy 下 method='suffix' 与不开投机的输出逐 token 相同。"""
    prompts = [("a", REPEAT_PROMPT), ("b", [5, 6, 5, 6, 5, 6])]
    spec = step59_helpers.run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts,
                                      max_tokens=8, temperature=0.0, method="suffix", spec_k=8,
                                      device=device)
    baseline = step59_helpers.run_prompts(tiny_dir=tiny_dir, hf_config=hf_config, prompts=prompts,
                                          max_tokens=8, temperature=0.0, spec_k=None,
                                          device=device)
    assert spec[0] == baseline[0], (spec[0], baseline[0])


def test_draft_path_is_engaged(suffix_engine):
    """真跑一轮：提议确实产出候选（num_drafts / num_draft_tokens > 0），不是空转。"""
    build, device = suffix_engine
    engine, core, runner = build(spec_k=8, budget=64, max_num_seqs=2)
    assert type(runner.proposer).__name__ == "SuffixDecodingProposer"
    assert runner.proposer.num_speculative_tokens == 8
    outputs, stats = _run(engine, core, [("a", REPEAT_PROMPT)])
    drafts, draft_tokens, accepted = _totals(stats)
    assert outputs["a"], "请求必须正常产出"
    assert drafts > 0 and draft_tokens > 0, _totals(stats)
    # 全局缓存按请求记录（跨请求复用的来源）
    assert "a" in runner.proposer.suffix_cache.cached_requests
    print(f"[step61] drafts={drafts} draft_tokens={draft_tokens} accepted={accepted} "
          f"output={outputs['a']}")


def test_request_ids_can_be_reused_after_finish(suffix_engine):
    """同 ID 两轮请求：第一轮结束后提议者状态被清掉，第二轮不报"already active"。"""
    build, device = suffix_engine
    engine, core, runner = build(spec_k=8, budget=64, max_num_seqs=2)
    first, _ = _run(engine, core, [("reuse", REPEAT_PROMPT)], max_tokens=4)
    # 触发一次"没有请求"的清理轮（finish 通知在这一轮被 Runner 处理）
    for out in engine.step():
        pass
    assert "reuse" not in runner.proposer.suffix_cache.active_requests
    second, stats = _run(engine, core, [("reuse", REPEAT_PROMPT)], max_tokens=4)
    assert second["reuse"], "第二轮必须正常产出（ID 重用不能继承旧树）"
    assert _totals(stats)[1] > 0, "第二轮照旧要能提候选"


def test_two_requests_share_the_global_tree(suffix_engine, tiny_dir, hf_config, device):
    """两条请求同时在跑：都产出、都进全局缓存，greedy 输出与不投机逐 token 相同。"""
    build, _device = suffix_engine
    engine, core, runner = build(spec_k=8, budget=64, max_num_seqs=2)
    requests = [("a", REPEAT_PROMPT), ("b", [1, 2, 3, 4, 1, 2, 3, 4, 9])]
    outputs, stats = _run(engine, core, requests)
    drafts, draft_tokens, accepted = _totals(stats)
    assert set(outputs) == {"a", "b"} and all(outputs.values())
    assert drafts > 0 and draft_tokens > 0, _totals(stats)
    cached = set(runner.proposer.suffix_cache.cached_requests)
    assert {"a", "b"} <= cached, cached
    baseline, _ = step59_helpers.run_prompts(tiny_dir=tiny_dir, hf_config=hf_config,
                                             prompts=requests, max_tokens=6, temperature=0.0,
                                             spec_k=None, device=device)
    assert outputs == baseline, (outputs, baseline)
    print(f"[step61] 双请求 drafts={drafts} draft_tokens={draft_tokens} accepted={accepted}")
