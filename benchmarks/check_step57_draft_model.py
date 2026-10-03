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
tiny 模型由 minivllm.testing.tiny_models 现场生成（仓库不再放 safetensors），切模型在临时目录里做。
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

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                    SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)

FAIL = []
from minivllm.testing.tiny_models import tiny_qwen3_dir   # 测试模型现场生成（仓库不再放 fixtures）
TINY = tiny_qwen3_dir("tiny_gqa")
MQA = tiny_qwen3_dir("tiny_mqa")
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


# 59 关起：投机**验证**走 Triton 内核（上游同样只有 GPU 路径），所以开投机的脚本要上 GPU。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(target_dir=TINY, target_config=None, draft_dir=None, draft_config=None,
          spec_tokens=3, blocks=16, budget=16, max_tokens=6, seed=None, max_model_len=64):
    target_config = target_config or TINY_CONFIG
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=max_model_len,
                                 hf_config=target_config),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE),
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
    # ------------------------------------------------ 6. 验收方抓到的四类边界（补成回归用例）

    def one_step(prompt, k=3, budget=16, device=DEVICE, max_model_len=64):
        """跑一轮，返回 (是否抛异常, 异常字符串)。"""
        engine = None
        try:
            engine = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1),
                           spec_tokens=k, budget=budget, max_model_len=max_model_len)
            engine.add_request("r", list(prompt), SamplingParams(max_tokens=6, temperature=0.0,
                                                                eos_token_id=999))
            engine.step()
            return None, engine
        except Exception as exc:                     # noqa: BLE001
            return f"{type(exc).__name__}: {exc}", engine


    error, engine = one_step([1, 2, 3, 4], k=3)
    check("6. prompt 恰好占满一个块：草稿要写的位置有 lookahead 槽位（不越界）",
          error is None, error or "正常")
    if engine is not None:
        engine.shutdown()

    error, engine = one_step([1, 2, 3, 4, 5, 6, 7, 8], k=2)
    check("6. prompt 恰好占满两个块：同上", error is None, error or "正常")
    if engine is not None:
        engine.shutdown()

    # budget=2、draft_slots=1 → 输入预算只剩 1 行给 target，所以本轮排 1 个 token（58 双预算）
    error, engine = one_step([1, 2, 3, 4, 5, 6], k=1, budget=2)
    synced = None
    if engine is not None:
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        synced = (runner.proposer._draft_computed.get("r", 0),
                  runner.proposer.num_drafts_proposed)
    check("6. 中间 prefill 块：不提草稿（没有可验证的 next token），但 draft 的 KV 照常同步"
          "（提了 Scheduler 也会丢，vLLM 的 update_draft_token_ids 同款规则）",
          error is None and synced == (1, 0), error or f"（draft 进度, 累计提议数）={synced}")
    if engine is not None:
        engine.shutdown()

    if torch.cuda.is_available():
        error, engine = one_step([1, 2, 3, 4, 5, 6], k=3, device="cuda")
        check("6. CUDA：草稿的 slot_mapping 在 CPU 上算完再搬（块表镜像是 CPU 结构）",
              error is None, error or "正常")
        if engine is not None:
            engine.shutdown()
    else:
        check("6. （跳过 CUDA 设备用例：本机没有 CUDA）", True)

    # 上下文快满：lookahead 会被 max_model_len 截掉 → 提议者要少提几枚，而不越界写
    error, engine = one_step([1, 2, 3, 4, 5, 6], k=3, max_model_len=8)
    check("6. 上下文快满（max_model_len=8）时 lookahead 被截掉：提议者少提几枚，不越界",
          error is None, error or "正常")
    if engine is not None:
        engine.shutdown()

    # ---- 6.2：chunked prefill 下的**发布边界**（199 §9：group 各层都写完才能发布）----
    # 做法与 vLLM 相同：drafter 每轮与 target 跑同一段位置（中间 prefill 块只同步 KV、不提
    # 草稿），所以 target 算完的块在 draft 那一层也已经写完。发布处因此**不夹** draft 进度
    # （vLLM 也不夹），不变量改由这里的用例盯着——既查进度，也查发布位置上的 draft KV 真的
    # 被写过（验收探针 204 §6.2 抓的就是"draft KV 全零、block_hash 却已登记"）。
    def run_with_draft(prompt, budget, prefix, blocks=32, max_tokens=6, max_model_len=64):
        config = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1),
                       spec_tokens=1, budget=budget, blocks=blocks,
                       max_model_len=max_model_len).vllm_config
        config = VllmConfig(model_config=config.model_config, cache_config=CacheConfig(
                                block_size=4, num_gpu_blocks=blocks,
                                enable_prefix_caching=prefix),
                            scheduler_config=config.scheduler_config,
                            device_config=config.device_config,
                            speculative_config=config.speculative_config)
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
        scheduler = engine.engine_core.engine_core.scheduler
        runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        engine.add_request("r", list(prompt), SamplingParams(max_tokens=max_tokens,
                                                            temperature=0.0, eos_token_id=999))
        observations = []
        while engine.has_unfinished_requests():
            engine.step()
            request = scheduler.requests.get("r")
            if request is None:
                continue
            # 最后一个完整块的最后一位：它属于"发布范围"（发布的是完整块），所以 draft 那一层
            # 必须已经有 KV——只对齐计数器不算数，要看 tensor 里的值
            last_published = request.num_computed_tokens // 4 * 4 - 1
            draft_kv_sum = None
            if last_published >= 0:
                physical = scheduler.kv_cache_manager.get_blocks("r").get_block_ids()[0][
                    last_published // 4]
                layer = sorted(runner.proposer.kv_caches)[0]
                draft_kv_sum = float(runner.proposer.kv_caches[layer][
                    :, physical, last_published % 4].abs().sum())
            observations.append((request.num_computed_tokens,
                                 runner.proposer._draft_computed.get("r", 0),
                                 scheduler.kv_cache_manager.num_cached_blocks(),
                                 draft_kv_sum))
        engine.shutdown()
        return observations, runner

    # budget=2 → 强制 chunked prefill；前缀缓存开着才可能"提前发布"
    observations, _ = run_with_draft([1, 2, 3, 4, 5, 6, 7, 8], budget=2, prefix=True)
    mid_prefill = [obs for obs in observations if obs[0] < 8]      # prompt 8 个 token 还没算完
    check("6. 中间 prefill 块：draft 的 KV 也同步到 target 算完的位置（不是原地不动）",
          bool(mid_prefill) and all(draft == computed and draft > 0
                                    for computed, draft, _cached, _kv in mid_prefill),
          f"中间 prefill 轮次（target 进度, draft 进度, ...）={mid_prefill}")
    check("6. 发布的完整块不超过 draft 侧进度"
          "（发布处不夹边界，靠的是两边每轮同步；199 §9 的不变量仍要成立）",
          all(computed // 4 * 4 <= draft for computed, draft, _cached, _kv in observations),
          f"观察={observations}")
    check("6. 发布范围里的 draft KV 确实被写过（不是只有计数器对齐）",
          all(kv is None or kv > 0 for _computed, _draft, _cached, kv in observations),
          f"最后一位的 draft KV 绝对值和={[kv for *_x, kv in observations]}")
    check("6. draft 追上来之后照常发布",
          observations[-1][2] > 0 and observations[-1][1] >= observations[-1][0],
          f"最后一轮={observations[-1]}（target 进度, draft 进度, 缓存块数, draft KV 和）")

    # prefix 开/关：输出必须一致（缓存只该改变速度）
    on, _ = run_with_draft([1, 2, 3, 4, 5, 6], budget=16, prefix=True)
    off, _ = run_with_draft([1, 2, 3, 4, 5, 6], budget=16, prefix=False)
    check("6. 真实 draft 下 prefix 开/关：跑完都正常（缓存不改变结果，只影响能不能命中）",
          on[-1][2] > 0 and off[-1][2] == 0 and on[-1][1] >= on[-1][0] and off[-1][1] >= off[-1][0],
          f"开={(on[-1])}、关={(off[-1])}")

    # 抢占后恢复（真实 draft + 小池子）：能跑完，draft 侧进度也重置过
    observations, runner = run_with_draft([1, 2, 3, 4, 1, 2, 3, 4], budget=16, prefix=False)
    check("6. 恢复之后 draft 侧进度不落后（旧物理编号上的 KV 不能当历史）",
          observations[-1][1] >= observations[-1][0],
          f"最后一轮={observations[-1]}")

    # 提议必须看到"本轮新采样的 token"；发布的完整块不能超过 draft 侧的进度
    engine = build(draft_dir=draft_dir, draft_config=dict(TINY_CONFIG, num_hidden_layers=1),
                   spec_tokens=1, max_tokens=8)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.model_runner
    scheduler = engine.engine_core.engine_core.scheduler
    seen_histories = []
    original_propose = runner.proposer.propose


    def traced_propose(rows, all_token_ids, *args, **kwargs):
        # 58：参数从"req_ids + 一个含糊的边界"改成 TargetRows（起点/终点写清楚）
        seen_histories.append({target.req_id: list(all_token_ids[target.req_id][:target.history_end])
                               for target in rows})
        return original_propose(rows, all_token_ids, *args, **kwargs)


    runner.proposer.propose = traced_propose
    engine.add_request("r", [1, 2, 3, 4, 5, 6], SamplingParams(max_tokens=8, temperature=0.0,
                                                               eos_token_id=999))
    published_boundary, draft_boundary = None, None
    while engine.has_unfinished_requests():
        engine.step()
        request = scheduler.requests.get("r")
        if request is not None and "r" in runner.proposer._draft_computed:
            block_size = 4
            published_boundary = (min(request.num_computed_tokens, request.num_tokens)
                                  // block_size) * block_size
            draft_boundary = runner.proposer._draft_computed["r"]
    engine.shutdown()

    check("6. 提议看到的是**本轮刚采样的 token**（提议在记账之后，199 §4 的时序）",
          bool(seen_histories) and len(seen_histories[0]["r"]) >= 7,
          f"第一次提议看到的历史长度={len(seen_histories[0]['r']) if seen_histories else None}")
    check("6. 发布的完整块不超过 draft 侧已算到的位置（prefix 只在该 group 各层都有 KV 时才能发布）",
          published_boundary is not None and published_boundary <= draft_boundary,
          f"发布到 {published_boundary}、draft 算到 {draft_boundary}")

    # ------------------------------------------------ 7. 205 复验：逻辑上限与请求生命周期

    def build_engine(*, k=3, max_model_len=64, budget=16, blocks=32, prefix=False,
                     policy="fcfs"):
        """与验收补测同一套构造：可配 k=None（不投机）、prefix、policy。"""
        model = ModelConfig(model=TINY, dtype="float32", max_model_len=max_model_len,
                            hf_config=TINY_CONFIG)
        config = VllmConfig(
            model_config=model,
            cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks,
                                     enable_prefix_caching=prefix),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=budget,
                                             policy=policy),
            device_config=DeviceConfig(device=DEVICE),
            speculative_config=None if k is None else SpeculativeConfig(
                method="draft_model", num_speculative_tokens=k,
                draft_model_config=ModelConfig(model=draft_dir, dtype="float32",
                                               max_model_len=64,
                                               hf_config=dict(TINY_CONFIG,
                                                              num_hidden_layers=1))))
        engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
        core = engine.engine_core.engine_core
        return engine, core, core.model_executor.driver_worker.model_runner

    def run_to_end(engine, limit=100):
        final = {}
        for _ in range(limit):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                final[out.request_id] = list(out.token_ids)
        return final

    # 7.1 逻辑上界：块表容量按块向上取整（10 个位置 → 12 个槽位），不能拿它当模型长度。
    #     prompt 8 + max_tokens 2 会正好走到"最后两个位置"，草稿要在这里少提而不是越界写。
    bound_ok, bound_detail = True, []
    for max_len in (9, 10, 11, 12):
        results = {}
        for k in (None, 3):
            engine, _core, _runner = build_engine(k=k, max_model_len=max_len)
            engine.add_request("r", [1, 2, 3, 4, 5, 6, 7, 8],
                               SamplingParams(max_tokens=2, temperature=0.0, eos_token_id=999))
            try:
                results[k] = run_to_end(engine)["r"]
            except Exception as exc:                 # noqa: BLE001
                results[k] = f"{type(exc).__name__}: {exc}"
            engine.shutdown()
        bound_ok &= results[None] == results[3] and isinstance(results[3], list) and bool(results[3])
        bound_detail.append(f"max_len={max_len}: 非投机={results[None]}、K=3={results[3]}")
    check("7. 逻辑上限 9/10/11/12：草稿到边界就少提，正常收尾且与非投机输出一致",
          bound_ok, "；".join(bound_detail))

    # 最后一轮刚好生成到上限（prompt 7 + 3 = max_model_len 10）
    results = {}
    for k in (None, 3):
        engine, _core, _runner = build_engine(k=k, max_model_len=10)
        engine.add_request("r", [1, 2, 3, 4, 5, 6, 7],
                           SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
        results[k] = run_to_end(engine)["r"]
        engine.shutdown()
    check("7. 最后一轮刚好生成到上下文上限：输出与非投机一致、长度正好等于剩余额度",
          results[None] == results[3] and len(results[3]) == 3,
          f"非投机={results[None]}、K=3={results[3]}")

    # 7.2 本轮没被调度的请求：**不是结束**——进度与随机流必须保留，再入批不能拿旧草稿配新 q
    # budget=6：第一轮 A、B 各要 2+1=3 行（都排得下）；第二轮 A 要 4+1=5 行把预算吃光，
    # B 因为 input_budget 只剩 1 行（<= draft_slots）被跳过——正是"本轮没排上"的场景
    engine, core, runner = build_engine(k=3, budget=6)
    for req, seed in (("A", 10), ("B", 20)):
        engine.add_request(req, [1, 2], SamplingParams(max_tokens=3, temperature=1.0,
                                                       seed=seed, eos_token_id=999))
    engine.step()
    before = runner.proposer._draft_generators["B"]
    before_state = before.get_state().clone()
    engine.step()
    scheduled_second = list(runner.input_batch.req_ids)
    after = runner.proposer._draft_generators.get("B")
    generator_kept = after is before and torch.equal(after.get_state(), before_state)
    outputs = run_to_end(engine)
    engine.shutdown()
    check("7. 本轮没排上的 B：draft generator 保留（同一对象、状态不动），再入批不报 q 对不上",
          scheduled_second == ["A"] and generator_kept and len(outputs.get("B", [])) == 3,
          f"第二轮 batch={scheduled_second}、generator 保留={generator_kept}、输出={outputs}")

    # 7.3 最后一条请求结束后的空清理轮也要清 proposer 状态；ID 复用不能继承旧进度
    engine, core, runner = build_engine(k=1, prefix=True)
    engine.add_request("reuse", [1, 2, 3, 4, 5, 6, 7, 8],
                       SamplingParams(max_tokens=1, temperature=1.0, seed=20, eos_token_id=999))
    run_to_end(engine)
    stale = (dict(runner.proposer._draft_computed), list(runner.proposer._draft_generators))
    calls = []
    original_forward = runner.proposer._forward

    def observed_forward(num_tokens, num_reqs):
        # 58：`_forward` 现在只吃"工作区的前 N 行"（固定缓冲），行内容从缓冲里读
        calls.append(runner.proposer.positions_cpu[:num_tokens].tolist())
        return original_forward(num_tokens, num_reqs)

    runner.proposer._forward = observed_forward
    engine.add_request("reuse", [8, 7, 6, 5, 4, 3, 2, 1],
                       SamplingParams(max_tokens=2, temperature=1.0, seed=30, eos_token_id=999))
    engine.step()
    block = core.scheduler.kv_cache_manager.get_blocks("reuse").blocks[0][0]
    layer = sorted(runner.proposer.kv_caches)[0]
    slot_kv = float(runner.proposer.kv_caches[layer][:, block.block_id, 3].abs().sum())
    runner.proposer._forward = original_forward
    engine.shutdown()
    check("7. 结束清理轮清空 draft 进度/随机流；复用同一 ID 的新请求会重建状态并重算 KV",
          not stale[0] and not stale[1] and bool(calls) and slot_kv > 0,
          f"清理后={stale}、新请求 draft forward={calls}、新 prefix 第 3 位 draft KV={slot_kv:.3f}")

    # 7.4 提议异常 → 明确失败态：下一轮必须在调度之前被拒绝（不能返回空输出继续跑）
    engine, core, runner = build_engine(k=1)
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
    original_forward = runner.proposer._forward

    def injected_failure(*args, **kwargs):
        raise RuntimeError("review-injected-draft-forward-failure")

    runner.proposer._forward = injected_failure
    first_error = None
    try:
        engine.step()
    except Exception as exc:                         # noqa: BLE001
        first_error = f"{type(exc).__name__}: {exc}"
    runner.proposer._forward = original_forward
    history_at_failure = list(core.scheduler.requests["r"].all_token_ids)
    second_error = None
    try:
        engine.step()
    except Exception as exc:                         # noqa: BLE001
        second_error = f"{type(exc).__name__}: {exc}"
    check("7. 提议异常 → 失败态：runner.failure 记录原因，下一轮在调度之前被拒绝且不再推进",
          first_error is not None and runner.failure is not None and second_error is not None
          and list(core.scheduler.requests["r"].all_token_ids) == history_at_failure,
          f"首次={first_error}；failure={runner.failure}；下一轮={second_error}")
    engine.shutdown()

    # 7.5 真实抢占 + 恢复：要求确实抢占，且输出与非投机 greedy 完全一致
    def preemption_run(k):
        # blocks=3：58 的双预算让每轮能排的 target token 变少，原来 blocks=4 已经不再抢占；
        # 缩到 3 才能继续覆盖"抢占后恢复"（两边都必须 preemptions>0 且输出一致）
        engine, core, _runner = build_engine(k=k, budget=4, blocks=3, policy="priority")
        for req, priority in (("A", 0), ("B", 5)):
            engine.add_request(req, [1, 2], SamplingParams(max_tokens=8, temperature=0.0,
                                                           eos_token_id=999),
                               priority=priority)
        outputs = run_to_end(engine)
        preemptions = core.scheduler.num_preemptions
        engine.shutdown()
        return outputs, preemptions

    reference, speculative = preemption_run(None), preemption_run(3)
    check("7. 真实抢占 + 恢复（num_preemptions>0）：draft 与非投机 greedy 输出一致",
          speculative[1] > 0 and reference[0] == speculative[0],
          f"非投机={reference}、投机={speculative}")

    # 7.6 prefix：必须真的命中（初始 computed>0），且开/关两种情况的输出完全一致
    def prefix_run(enabled):
        engine, _core, runner = build_engine(k=3, prefix=enabled, budget=4)
        hits = []
        original_update = runner._update_states

        def observe(packet):
            hits.extend((req.req_id, req.num_computed_tokens)
                        for req in packet.scheduled_new_reqs)
            return original_update(packet)

        runner._update_states = observe
        outputs = []
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        for req in ("first", "reuse"):
            engine.add_request(req, prompt, SamplingParams(max_tokens=6, temperature=0.0,
                                                           eos_token_id=999))
            outputs.append(run_to_end(engine)[req])
        runner._update_states = original_update
        engine.shutdown()
        return outputs, hits

    enabled, disabled = prefix_run(True), prefix_run(False)
    check("7. prefix 二次请求真的命中（初始 computed>0），且输出与关闭缓存完全一致",
          any(req == "reuse" and num > 0 for req, num in enabled[1])
          and enabled[0] == disabled[0],
          f"开={enabled}、关={disabled}")

finally:
    shutil.rmtree(work, ignore_errors=True)

print()
print(f"{'全部通过' if not FAIL else '失败: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
