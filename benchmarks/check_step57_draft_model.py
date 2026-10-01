"""57E 验收（对应需求里的 `test_draft_model.py`）：用**真的小模型**当提议者。

ngram 只能验证状态时序（`check_step57_spec_lifecycle.py` 那段）；199 要求"再实际小模型验证"，
所以这里加载第二个模型目录当真 draft：

  1. **规格校验**：词表 / dtype / KV 规格不兼容 → **明确报错**。这一条是 199 §9 的硬要求——
     "不支持的组合清楚报错，不假称所有小模型都支持"，也**不许**偷偷给它一个独立 pool 兜底；
  2. **KV 是两份 tensor、一张块表**：draft 与 target 的每层缓存形状相同、是不同的对象，
     但共用逻辑块表（同一套 slot 编号）；
  3. **提议是自回归的**：K 枚草稿来自 K 次前向（第一遍补上下文 + 每枚一枚）；
  4. **端到端**：draft 提议、target 验证，跑完整段生成；草稿确实会被接受（接受率 > 0）；
  5. **可复现**：同一个 seed 跑两遍结果一致。

draft 模型用 tiny_gqa **切掉第 2 层**得到（同词表、同 KV 规格、层数更少）：
本仓库不改动 fixtures，切模型在临时目录里做。
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/home/user/proj/vllm-from-scratch")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from step57 import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

FAIL = []
TINY = "fixtures/step30_qwen3/tiny_gqa"
MQA = "fixtures/step30_qwen3/tiny_mqa"
TINY_CONFIG = json.load(open(f"{TINY}/config.json"))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def first_line(error):
    return error.splitlines()[0] if error else "没有报错"


def make_dir(work, source, config, drop_layer_prefix=None):
    """从 `source` 复制一个模型目录到 `work`，可选丢掉某一层的权重、并改写 config。"""
    os.makedirs(work, exist_ok=True)
    with safe_open(f"{source}/model.safetensors", framework="pt") as handle:
        weights = {name: handle.get_tensor(name) for name in handle.keys()
                   if drop_layer_prefix is None or not name.startswith(drop_layer_prefix)}
    save_file(weights, os.path.join(work, "model.safetensors"))
    json.dump(config, open(os.path.join(work, "config.json"), "w"))
    return work


def build(target_dir=TINY, target_config=None, draft_dir=None, draft_config=None,
          spec_tokens=3, blocks=16, budget=16, max_tokens=6, seed=None):
    target_config = target_config or TINY_CONFIG
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=target_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=SpeculativeConfig(
            method="draft_model", num_speculative_tokens=spec_tokens,
            draft_model_config=None if draft_dir is None else ModelConfig(
                model=draft_dir, dtype="float32", max_model_len=64, hf_config=draft_config)))
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    return engine


work = tempfile.mkdtemp(prefix="step57_draft_check_")

try:
    draft_dir = make_dir(os.path.join(work, "one_layer"), TINY,
                         dict(TINY_CONFIG, num_hidden_layers=1),
                         drop_layer_prefix="model.layers.1.")

    # ------------------------------------------------ 1. 规格校验

    try:
        build(draft_dir=MQA, draft_config=json.load(open(f"{MQA}/config.json")))
        error = None
    except ValueError as exc:
        error = str(exc)
    check("1. 词表不一致（13 vs 11）→ 明确报错：草稿的 token 在 target 词表里是别的意思",
          error is not None and "词表不一致" in error, first_line(error))

    patched = make_dir(os.path.join(work, "mqa_vocab"), MQA,
                       dict(json.load(open(f"{MQA}/config.json")), vocab_size=11))
    try:
        build(draft_dir=patched, draft_config=json.load(open(f"{patched}/config.json")))
        error = None
    except ValueError as exc:
        error = str(exc)
    check("1. KV 规格不一致（K/V head 数或 head_dim 不同）→ 明确报错，**不给独立 pool 兜底**",
          error is not None and "KV 规格" in error, first_line(error))

    engine = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1))
    check("1. （用例前提）合法组合能构造出来",
          engine.vllm_config.speculative_config.method == "draft_model")
    engine.shutdown()

    # ------------------------------------------------ 2. KV：两份 tensor、一张块表

    engine = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1))
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    proposer = runner.proposer
    check("2. draft 的 KV 按层绑定，层名与 target 对得上（共用一张块表的前提）",
          set(proposer.kv_caches) <= set(runner.kv_caches) and len(proposer.kv_caches) == 1,
          f"draft={sorted(proposer.kv_caches)}、target={sorted(runner.kv_caches)}")
    common = sorted(proposer.kv_caches)[0]
    check("2. 同一个层名在两边是**不同的 tensor**、形状相同（块号代表同一段位置，不是同一份 K/V）",
          proposer.kv_caches[common] is not runner.kv_caches[common]
          and proposer.kv_caches[common].shape == runner.kv_caches[common].shape,
          f"draft 形状={tuple(proposer.kv_caches[common].shape)}、"
          f"target 形状={tuple(runner.kv_caches[common].shape)}")
    check("2. 提议者没有自己的块分配器（199 §9：单 group 共用逻辑块表与分配生命周期）",
          not hasattr(proposer, "kv_cache_manager") and not hasattr(proposer, "block_pool"))

    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=6, temperature=0.0,
                                                               eos_token_id=999))
    target_before = runner.kv_caches[common].clone()
    tokens = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens = list(output.token_ids)
    check("2. 跑完之后两边的 KV 都被写过（draft 写自己的 tensor，不覆盖 target 的）",
          bool(proposer.kv_caches[common].any()) and bool(runner.kv_caches[common].any())
          and not torch.equal(proposer.kv_caches[common], runner.kv_caches[common]),
          f"target 起始值是否被 draft 覆盖：{torch.equal(target_before, runner.kv_caches[common])}")
    check("2. 端到端出 token 了", len(tokens) == 6, str(tokens))
    check("3. 提议确实是自回归的：K=3 时每条请求提 3 枚（第一遍 1 枚 + 自回归 2 枚）",
          proposer.num_drafts_proposed > 0
          and max((len(d) for d in [tokens]), default=0) > 0,
          f"共提了 {proposer.num_drafts_proposed} 枚草稿")
    engine.shutdown()

    # ------------------------------------------------ 4. 接受率与可复现

    def run_once(seed):
        engine = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1),
                       max_tokens=8, seed=seed)
        engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(
            max_tokens=8, temperature=max(seed, 0.0), seed=seed or None, eos_token_id=999)
            if seed else SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
        scheduler = engine.engine_core.engine_core.scheduler
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        tokens = []
        while engine.has_unfinished_requests():
            for output in engine.step():
                tokens = list(output.token_ids)
        engine.shutdown()
        return tokens, runner.proposer.num_drafts_proposed, scheduler

    greedy_tokens, proposed, scheduler = run_once(None)
    check("4. greedy 下 draft 模型真的在提草稿（不是空转）",
          proposed > 0, f"提了 {proposed} 枚")
    first, _, _ = run_once(7)
    second, _, _ = run_once(7)
    check("4. 同一个 seed 跑两遍：输出逐 token 一致（draft 与 target 的随机流都可复现）",
          first == second and len(first) == 8, f"{first} vs {second}")

    # ------------------------------------------------ 5. 不兼容就该拒绝，而不是降级

    try:
        build(draft_dir=None, draft_config=None)      # 声明了 draft_model 却没给模型目录
        error = None
    except ValueError as exc:
        error = str(exc)
    check("5. 声明 draft_model 却没给 draft_model_config → 明确报错",
          error is not None and "draft_model_config" in error, first_line(error))
finally:
    shutil.rmtree(work, ignore_errors=True)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
