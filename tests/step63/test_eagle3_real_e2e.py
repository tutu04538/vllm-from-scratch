"""step63 收尾：**真实 EAGLE3 checkpoint 的端到端**（需求 063 §3.3，也是 §5 里那条待办）。

为什么这条一直没跑：真实 draft 的 `lm_head` 是 **32000** 宽（`draft_vocab_size`），而 target 词表是
**151936**，checkpoint 里带 `d2t (32000,)` / `t2d (151936,) bool` 的偏移映射。提议者直接对
`compute_logits()` 的结果 `argmax`，所以映射必须发生在**那个方法里面**（上游
`llama_eagle3.py:339-356` 就是这么写的）；否则 32000 个 draft id 会被当成 target id 用——
它们都 < 151936，**不报错**，只是每一枚草稿都指到了别的字（`d2t` 里只有 0.4% 的偏移是 0）。

这条在 66 关收尾时补上（`compute_logits()` 现在 scatter 回 target 宽度），本文件在**真实权重**上钉住：

  1. 开了投机（K=2）与非投机的 greedy 输出**逐 token 相同**；
  2. 草稿 id 全部落在 `t2d` 标记的可用集合里（id 空间正确的直接证据）；
  3. 反证：同一份 hidden 上取 **draft 空间的 argmax**（= 不做映射时会交出的 id），逐枚与
     `argmax + d2t` 对照 —— 不映射时 id 会变（`d2t` 里 99.6% 的偏移非 0），也就是"不报错、
     只是每枚草稿都指到别的字"。⚠️ 判据**不是**"这些 id 在不在 `t2d` 里"：`t2d` 的低位
     target id 大部分是恒等映射，用那个当判据会时灵时不灵（2026-10-08 修自回归位置时踩过）。

真实权重在 `models/`（不入库）；缺权重或没有 CUDA 时**跳过并写明原因**（不是"通过"）。

**实测的接受情况（本机，本 prompt，K=2）**：14 个请求·轮里 drafted=28、accepted=1（接受长度 ≈1.07）。
官方模型卡（`AngelSlim/Qwen3-1.7B_eagle3`）在 Qwen3-1.7B 上报的**接受长度是 2.13~2.2**（K=2/4），
所以**草稿质量这条还有差距**，而且差距不在 id 空间（id 空间已由本文件钉住），更像 draft 解码层
与上游的逐值差异——那正是 63 关 §5 里"要等 69/70 的 forward 上下文基建"的那条待办。
本文件**不对接受率作断言**（它是候选质量，不是正确性）。
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,  # noqa: E402
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

TARGET_DIR = ROOT / "models" / "Qwen3-1.7B"
DRAFT_DIR = ROOT / "models" / "Qwen3-1.7B-eagle3"
PROMPT = "用一句话解释什么是 KV cache。"
K = 2
MAX_TOKENS = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (TARGET_DIR / "model.safetensors.index.json").is_file()
    or not (DRAFT_DIR / "model.safetensors").is_file(),
    reason="真实 EAGLE3 端到端需要 CUDA + models/Qwen3-1.7B[-eagle3]（本机权重不入库）：待验，不是通过")


def make_engine(*, with_spec: bool, num_gpu_blocks: int = 64, max_model_len: int = 512):
    """真实 1.7B target（+ eagle3 draft）的引擎。"""
    target_hf = json.loads((TARGET_DIR / "config.json").read_text())
    draft_hf = json.loads((DRAFT_DIR / "config.json").read_text())
    spec = None
    if with_spec:
        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=K,
            draft_model_config=ModelConfig(model=str(DRAFT_DIR), dtype="bfloat16",
                                           max_model_len=max_model_len, hf_config=draft_hf))
    config = VllmConfig(
        model_config=ModelConfig(model=str(TARGET_DIR), dtype="bfloat16",
                                 max_model_len=max_model_len, hf_config=target_hf),
        cache_config=CacheConfig(block_size=16, num_gpu_blocks=num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
        device_config=DeviceConfig(device="cuda"), speculative_config=spec)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def run(engine, core, prompt_ids):
    """跑完这条请求。顺手把每步的 `spec_decoding_stats` 累加起来（Scheduler 只留最近一步的）。"""
    engine.add_request("a", list(prompt_ids),
                       SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0,
                                      eos_token_id=999))
    outputs, totals = {}, {"drafted": 0, "accepted": 0, "rounds": 0}
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        stats = core.scheduler.spec_decoding_stats
        if stats is not None:
            totals["drafted"] += stats.num_draft_tokens
            totals["accepted"] += stats.num_accepted_tokens
            totals["rounds"] += stats.num_drafts
    return outputs["a"], totals


def install_draft_spy(runner):
    """记录每枚草稿 id，**并**记录"不做 d2t 映射时会交出去的那个 id"。

    后者是反证的关键：提议者拿到 `compute_logits()` 的结果就 `argmax`。把映射去掉时它 argmax 的
    是 **draft 空间**的 logits（`draft_vocab_logits()`），得到的 id 会被当成 target id 使用。
    这个量**只取决于模型在这批 hidden 上的输出**，与"哪几枚草稿被接受/被拒"无关——所以反证
    不会因为草稿质量变化而时灵时不灵（63 关修自回归位置时就踩过：草稿一变，原来的"逆表反推"
    反证偶然全落在合法集合里，断言失效）。
    """
    drafts, unmapped = [], []
    original = runner.take_draft_token_ids

    def spy():
        result = original()
        if result is not None:
            drafts.extend(int(token) for row in result.draft_token_ids for token in row)
        return result

    runner.take_draft_token_ids = spy

    model = runner.proposer.model
    original_logits = model.compute_logits

    def logits_spy(hidden_states, *args, **kwargs):
        out = original_logits(hidden_states, *args, **kwargs)
        if getattr(model, "d2t", None) is not None:
            raw = model.draft_vocab_logits(hidden_states).argmax(dim=-1)
            unmapped.extend(int(token) for token in raw.tolist())
        return out

    model.compute_logits = logits_spy
    return drafts, unmapped


@pytest.fixture(scope="module")
def real_run():
    """跑一遍非投机与 K=2 投机；返回输出、草稿 id 与 d2t/t2d 表。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TARGET_DIR), local_files_only=True)
    prompt_ids = tokenizer(PROMPT)["input_ids"]

    plain_engine, plain_core, _ = make_engine(with_spec=False)
    try:
        plain, _ = run(plain_engine, plain_core, prompt_ids)
    finally:
        plain_engine.shutdown()

    spec_engine, core, runner = make_engine(with_spec=True)
    drafts, unmapped = install_draft_spy(runner)
    try:
        spec, totals = run(spec_engine, core, prompt_ids)
        model = runner.proposer.model
        d2t, t2d = model.d2t.clone(), model.t2d.clone()
        draft_vocab_size, target_vocab_size = model.draft_vocab_size, model.target_vocab_size
    finally:
        spec_engine.shutdown()
    print(f"[step63-real] 非投机 {len(plain)} token == 投机 {len(spec)} token: {plain == spec}；"
          f"草稿 {len(drafts)} 枚；整段统计 drafted={totals['drafted']} "
          f"accepted={totals['accepted']}（{totals['rounds']} 个请求·轮）")
    return {"plain": plain, "spec": spec, "drafts": drafts, "unmapped": unmapped,
            "totals": totals, "d2t": d2t, "t2d": t2d,
            "draft_vocab_size": draft_vocab_size,
            "target_vocab_size": target_vocab_size}


def test_real_checkpoint_greedy_matches_non_speculative(real_run):
    """真实 draft 的 32000 词表 + d2t 映射下，开了投机的 greedy 输出必须与非投机逐 token 相同。"""
    assert len(real_run["spec"]) == MAX_TOKENS and len(real_run["plain"]) == MAX_TOKENS
    assert real_run["spec"] == real_run["plain"]
    assert real_run["drafts"], "一枚草稿都没提出来？这条端到端就没验到 d2t 那条路"


def test_real_draft_probs_are_in_target_space(real_run):
    """草稿概率 `q` 也必须是 **target 空间**的（宽度 = target 词表）——不需要再做事后换算。

    原因：`d2t` 的映射发生在 `compute_logits()` 里、也就是 **softmax 之前**，所以
    `softmax(32000 个值)` 与"scatter 到 151936 宽再 softmax"逐位相同（非 d2t 位置是 -inf → 概率恰好 0）。
    拒绝采样器按 **target 词表**的步长索引 `draft_probs`，宽度对不上就是越界读（内核里是
    `draft_probs_ptr + token_idx * vocab_size + vocab_offset`，只断言了 ndim），所以这条必须有断言盯着。
    """
    runner_engine, _, runner = make_engine(with_spec=True)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(TARGET_DIR), local_files_only=True)
        prompt_ids = tokenizer(PROMPT)["input_ids"]
        _, totals = run(runner_engine, runner_engine.engine_core.engine_core, prompt_ids)
        pending = runner.pending_draft_probs
        assert pending is not None and pending.draft_probs is not None
        assert tuple(pending.draft_probs.shape)[1] == runner.input_batch.vocab_size == 151936
        # 贪心草稿的 q 是**点质量**（一行只有一个非零），这正是 `draft_probs=None` 那条分支的等价物
        assert int((pending.draft_probs[0] > 0).sum()) == 1
        assert totals["drafted"] > 0
    finally:
        runner_engine.shutdown()


def test_real_draft_ids_live_in_target_space(real_run):
    """草稿 id 必须是 **target 空间**、且落在 `t2d` 标记的集合里（映射接对了的直接证据）。

    反证（同一条用例里做）：把这些 id 用 `d2t` 的**逆表**退回 draft 空间，就得到"不映射时提议者
    会交出去的那些 id"——它们只有约 21%（32000/151936）能落在 `t2d` 里，整批几乎必然越界。
    """
    ids = torch.tensor(real_run["drafts"], dtype=torch.long)
    t2d, d2t = real_run["t2d"], real_run["d2t"]
    draft_vocab_size, target_vocab_size = real_run["draft_vocab_size"], real_run["target_vocab_size"]
    assert draft_vocab_size == 32000 and target_vocab_size == 151936
    assert bool(t2d[ids].all()), "有草稿 id 不在 t2d 标记的可用集合里"
    assert int(ids.max()) < target_vocab_size

    # 逐枚对照：`compute_logits()` = 把 draft 空间的 logits scatter 到 target 宽度，所以
    # **贪心交出的 id 必须恰好等于 `draft 空间 argmax + d2t[argmax]`**（scatter 是一一映射，
    # 非映射位置是 -inf）。这一条把"映射发生在 softmax/argmax 之前"钉死，比反推逆表直接得多。
    unmapped = real_run["unmapped"]
    assert unmapped, "没记到 draft 空间的 argmax，反证没有证据"
    assert len(unmapped) == len(ids), (len(unmapped), len(ids))
    raw = torch.tensor(unmapped, dtype=torch.long)
    assert int(raw.max()) < draft_vocab_size
    assert torch.equal(raw + d2t[raw], ids), (
        "交出去的草稿 id 不等于『draft 空间 argmax + d2t』：映射没接在 compute_logits 里，"
        "或者 argmax 与映射的先后顺序反了")

    # 反证：**不做映射时交出去的就是 `raw`**——`d2t` 里 99.6% 的偏移非 0，也就是那些 id 会指到
    # 别的字（都 < 151936，不报错，只是每枚草稿都错）。
    # 注意**不能**用"这些 id 在不在 `t2d` 里"当判据：`t2d` 的前 20813 个 target id 恰好是
    # 恒等映射，而模型偏好的 draft id 多半落在这一段，所以"落在 t2d 里"并不代表"映射没起作用"。
    offsets = d2t[raw]
    changed = int((offsets != 0).sum())
    # 运行级：**至少有一枚**草稿在不映射时会指到别的字（本机实测 32 枚里 10 枚；
    # 这个比例取决于模型 argmax 落在哪一段，所以只断言 > 0，比例写进断言消息备查）。
    assert changed > 0, (
        f"{len(unmapped)} 枚草稿在不做映射时 id 一个都不会变（{offsets.tolist()}）："
        f"这条反证在本轮没有区分度")
    # 表级（与运行无关的确定性事实）：draft 词表里 99.6% 的 id 都需要偏移，
    # 也就是说"不映射"不是少几个 token，而是绝大多数的 id 都指到别的字。
    assert float((d2t != 0).float().mean()) > 0.99, "d2t 表本身几乎全是恒等映射？那映射就不是关键"
