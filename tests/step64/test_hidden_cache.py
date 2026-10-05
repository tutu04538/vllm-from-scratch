"""step64：`extract_hidden_states` 的 cache-only 执行路径（需求 064 §4）。

验收口径分三层：

1. **单元层**：缓存形状/层身份/哨兵槽位/错误路径（不建引擎，快）；
2. **物理 slot 层**：跑 tiny 引擎，把"提议者写进缓存的 `[T, L, H]`"与**独立重算的特征参考**
   逐槽位比——验证层、请求、位置都没有互换（需求 §4 第一条）；
3. **生命周期层**：chunked prefill、prefix 命中、拒绝尾部、请求释放/复用，以及端到端
   greedy 与非投机逐 token 相同、返回宽度只取第 0 列。

参考特征怎么来：同一份权重、同一份本轮元数据**重跑一次 target 前向**（`reference_aux()`）。
它独立于"提议者拿到的那份特征"，所以能同时盯住两件事——提议者拿到的是**本轮真实特征**，
以及它们被写到了**正确的槽位**。
"""

import pytest
import torch

from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                      SchedulerConfig, SpeculativeConfig, UniProcExecutor, VllmConfig, Worker)
from minivllm.attention.forward_context import set_forward_context
from minivllm.attention.metadata import AttentionMetadata
from minivllm.models.extract_hidden_states import (CacheOnlyAttentionBackend,
                                                  CacheOnlyAttentionLayer,
                                                  CacheOnlyAttentionMetadata,
                                                  ExtractHiddenStatesModel, basic_cache)
from minivllm.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_LAYERS = [0, 1]
PROMPTS = [("a", [1, 2, 3, 4, 1, 2, 3, 4]), ("b", [5, 6, 5, 6])]


# ---------------------------------------------------------------- 夹具


def target_config(*, block_size=4, num_gpu_blocks=32, max_num_seqs=2, budget=64,
                  max_model_len=64, prefix_caching=False, with_spec=True):
    target_dir = tiny_qwen3_dir("tiny_gqa")
    target_hf = tiny_qwen3_config("tiny_gqa")
    spec = None
    if with_spec:
        spec = SpeculativeConfig(
            method="extract_hidden_states", num_speculative_tokens=1,
            draft_model_config=ModelConfig(
                model=target_dir, dtype="float32", max_model_len=max_model_len,
                hf_config={"eagle_aux_hidden_state_layer_ids": list(AUX_LAYERS)}))
    return VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32",
                                 max_model_len=max_model_len, hf_config=target_hf),
        cache_config=CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks,
                                 enable_prefix_caching=prefix_caching),
        scheduler_config=SchedulerConfig(max_num_seqs=max_num_seqs,
                                         max_num_batched_tokens=budget),
        device_config=DeviceConfig(device=DEVICE), speculative_config=spec)


def make_engine(config=None, *, with_spec=True, **kwargs):
    """起一个 tiny 引擎（默认带 extract 投机）。返回 (engine, core, runner)。"""
    if config is None:
        config = target_config(with_spec=with_spec, **kwargs)
    engine = LLMEngine(config, UniProcExecutor(config, Worker(config)))
    core = engine.engine_core.engine_core
    return engine, core, core.model_executor.driver_worker.model_runner


def add_request(engine, req_id, prompt, max_tokens=6):
    engine.add_request(req_id, list(prompt),
                       SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                      eos_token_id=999))


def run(engine, core, prompts, *, max_tokens=6):
    """跑完这些请求。返回 `(每条请求的输出, 每轮的投机统计)`。"""
    for req_id, prompt in prompts:
        add_request(engine, req_id, prompt, max_tokens=max_tokens)
    outputs, stats = {}, []
    for _ in range(400):
        if not engine.has_unfinished_requests():
            break
        for out in engine.step():
            outputs[out.request_id] = list(out.token_ids)
        if core.scheduler.spec_decoding_stats is not None:
            stats.append(core.scheduler.spec_decoding_stats)
    return outputs, stats


def install_spy(runner):
    """记录每次 `propose()` 的入参/返回值，以及"哪条请求的哪个位置落在哪个槽位"。

    `positions` 是从**本轮元数据**独立推出来的：每个批行的起点 = `seq_lens - 该行 query 长度`
    （所以 prefix 命中的请求起点就是命中末尾），槽位 = `block_table[row, pos//bs]*bs + pos%bs`。
    它和 `slot_mapping` 是同一条公式的两个来源，对不上就说明位置/槽位错位。
    """
    proposer = runner.proposer
    calls = []
    original = proposer.propose

    def spy(num_speculative_tokens, sampled_token_ids, target_hidden_states,
            common_attn_metadata):
        # 在提议者把它拷进常驻缓冲**之前**抓一份（拷进去之后缓冲会被下一轮覆盖）
        stacked = proposer._stack_hidden_states(target_hidden_states).clone()
        result = original(num_speculative_tokens=num_speculative_tokens,
                          sampled_token_ids=sampled_token_ids,
                          target_hidden_states=target_hidden_states,
                          common_attn_metadata=common_attn_metadata)
        block_size = runner.block_size
        query_start_loc = common_attn_metadata.query_start_loc.tolist()
        seq_lens = common_attn_metadata.seq_lens.tolist()
        block_table = common_attn_metadata.block_table.cpu()
        positions = []
        for row, req_id in enumerate(runner.input_batch.req_ids):
            query_len = query_start_loc[row + 1] - query_start_loc[row]
            start = seq_lens[row] - query_len
            for offset in range(query_len):
                position = start + offset
                slot = int(block_table[row, position // block_size]) * block_size \
                    + position % block_size
                positions.append((req_id, position, slot))
        calls.append({
            "stacked": stacked,
            "slot_mapping": common_attn_metadata.slot_mapping.clone(),
            "sampled": sampled_token_ids.clone(),
            "returned": result.clone(),
            "metadata": common_attn_metadata,
            "positions": positions,
        })
        return result

    proposer.propose = spy
    return calls


def install_round_spy(runner):
    """记录每轮调度快照（每条请求本轮排了几个 token、从哪个位置开始、带了哪些草稿）。"""
    rounds = []
    original = runner.execute_model

    def spy(scheduler_output):
        if scheduler_output.total_num_scheduled_tokens > 0:
            # 新请求的起点就是 prefix 命中长度（`NewRequestData.num_computed_tokens` 是本轮
            # 计算**之前**的快照），续跑请求的起点由协议校正——两边都记下来
            starts = {data.req_id: data.num_computed_tokens
                      for data in scheduler_output.scheduled_new_reqs}
            for index, req_id in enumerate(scheduler_output.scheduled_cached_reqs.req_ids):
                starts[req_id] = scheduler_output.scheduled_cached_reqs.num_computed_tokens[index]
            rounds.append({
                "num_scheduled": dict(scheduler_output.num_scheduled_tokens),
                "starts": starts,
                "spec": {req_id: list(tokens) for req_id, tokens
                         in scheduler_output.scheduled_spec_decode_tokens.items()},
            })
        return original(scheduler_output)

    runner.execute_model = spy
    return rounds


def cache_of(runner):
    proposer = runner.proposer
    assert len(proposer.attn_layer_names) == 1
    return proposer.kv_caches[proposer.attn_layer_names[0]]


def slot_of(runner, req_id, position, layer_index=0):
    """请求的绝对位置 → 物理槽位（用执行端镜像里的块表）。请求必须还在批里（未结束）。"""
    block_ids = runner.requests[req_id].block_ids[layer_index]
    block_size = runner.block_size
    return block_ids[position // block_size] * block_size + position % block_size


def read_slot(runner, slot):
    cache = cache_of(runner)
    block_size = cache.shape[1]
    return cache[slot // block_size, slot % block_size]


def reference_aux(runner, metadata, input_ids, positions):
    """独立重算：同一份权重 + 同一轮元数据再跑一次 target 前向，返回 `[T, L, H]`。

    重跑会按同一份 `slot_mapping` 再写一遍 target 的 KV——**写的是同样的值**（同 token、同位置、
    同槽位），所以对引擎状态没有影响；换来的是一个不依赖提议者的特征参考。
    """
    model = runner.model
    layer_names = list(runner._attention_layers())
    inputs = torch.tensor(input_ids, dtype=torch.int64, device=runner.device)
    positions_t = torch.tensor(positions, dtype=torch.int64, device=runner.device)
    with torch.inference_mode(), set_forward_context(
            {name: metadata for name in layer_names}, num_tokens=len(input_ids)):
        _, aux = model(inputs, positions_t)
    flat = torch.cat(list(aux), dim=-1)
    return flat.view(flat.shape[0], len(AUX_LAYERS), model.config["hidden_size"])


# ---------------------------------------------------------------- 1. 单元层


def test_config_derivation_and_constraints():
    """配置期：方法识别、K=1、辅助层编号必需、派生配置的四个字段。"""
    config = target_config()
    spec = config.speculative_config
    assert spec.uses_extract_hidden_states() and not spec.use_eagle()
    assert spec.eagle3_use_aux_hidden_state() is False
    assert spec.model == "extract_hidden_states", "上游把 model 换成字面量标记"
    # 不跑 draft 模型、也不额外留 KV 位置（与上游 max_num_new_slots_for_drafting 的表一致）
    assert spec.max_num_new_slots_for_drafting == 0

    derived = spec.derive_extract_hidden_states_config(config.model_config, config.cache_config)
    assert derived.model == config.model_config.model, "draft 目录 = target 目录（cache-only 无权重）"
    assert derived.dtype == config.model_config.dtype
    hf = derived.hf_config
    assert hf["architectures"] == ["ExtractHiddenStatesModel"]
    assert hf["eagle_aux_hidden_state_layer_ids"] == AUX_LAYERS
    assert hf["hidden_size"] == config.model_config.hf_config["hidden_size"]
    assert hf["cache_block_size"] == config.cache_config.block_size
    assert hf["torch_dtype"] == str(config.model_config.dtype)

    def spec_with(**kwargs):
        args = dict(method="extract_hidden_states", num_speculative_tokens=1,
                    draft_model_config=ModelConfig(
                        model="/tmp", hf_config={"eagle_aux_hidden_state_layer_ids": AUX_LAYERS}))
        args.update(kwargs)
        return SpeculativeConfig(**args)

    with pytest.raises(ValueError, match="num_speculative_tokens=1"):
        spec_with(num_speculative_tokens=2)
    with pytest.raises(ValueError, match="eagle_aux_hidden_state_layer_ids"):
        SpeculativeConfig(method="extract_hidden_states", num_speculative_tokens=1)
    assert spec_with().uses_extract_hidden_states()


def test_cache_shape_and_layer_identity():
    """缓存形状 = [blocks, block_size, L, H]；层名与两个维度与源码语义一致。"""
    engine, core, runner = make_engine()
    try:
        proposer = runner.proposer
        assert isinstance(proposer, ExtractHiddenStatesProposer)
        assert runner.capture_aux_hidden_states is True
        assert runner.eagle_aux_hidden_state_layers == tuple(AUX_LAYERS)
        assert runner.aux_hidden_states is None, "还没跑过一轮：特征缓冲是空的"
        assert proposer.num_hidden_states == len(AUX_LAYERS)
        assert proposer.hidden_size == 32
        # 上游同款：特征缓冲按 "max_num_batched_tokens + max_num_seqs" 开
        assert proposer.hidden_states.shape == (64 + 2, len(AUX_LAYERS), 32)

        assert proposer.attn_layer_names == ["cache_only_layers.2"], "层名带 target 的层数"
        layer = proposer.model.cache_only_layers["2"]
        assert isinstance(layer, CacheOnlyAttentionLayer)
        assert (layer.num_heads, layer.head_size) == (len(AUX_LAYERS), 32), \
            "num_heads <- 辅助层数 L，head_size <- hidden_size H"
        cache = cache_of(runner)
        assert tuple(cache.shape) == (32, 4, len(AUX_LAYERS), 32)
        assert tuple(cache.shape) == tuple(CacheOnlyAttentionBackend.get_kv_cache_shape(
            32, 4, len(AUX_LAYERS), 32))
        assert cache.dtype == torch.float32
        # 与 target 的 KV 不是同一份张量（同一张逻辑块表、各自的物理张量）
        assert all(cache is not kv for kv in runner.kv_caches.values())
    finally:
        engine.shutdown()


def test_basic_cache_padding_slot_goes_to_null_block():
    """哨兵槽位 -1 落到 0 号块（垃圾桶），有效槽位照常散射。"""
    block_size, num_heads, head_size = 4, 2, 3
    kv_cache = torch.zeros(3, block_size, num_heads, head_size)
    # +1：让每个元素都非零，才能用 count_nonzero 数"写了几个位置"
    to_cache = torch.arange(1, 2 * num_heads * head_size + 1,
                            dtype=torch.float32).view(2, num_heads, head_size)
    basic_cache(to_cache, kv_cache, torch.tensor([-1, 5], dtype=torch.int64))
    assert torch.equal(kv_cache[0, 0], to_cache[0]), "-1 重定向到 0 号块的第 0 个槽位"
    assert torch.equal(kv_cache[1, 1], to_cache[1]), "槽位 5 = 块 1 的第 1 个位置"
    assert int(torch.count_nonzero(kv_cache)) == 2 * num_heads * head_size, "只写了这两个槽位"


def test_layer_and_impl_error_paths():
    """错误路径：没有 forward 上下文 / 没绑定缓存 / 不许调 impl.forward / dtype 不一致。"""
    layer = CacheOnlyAttentionLayer(num_heads=2, head_size=4, block_size=4,
                                    kv_cache_torch_dtype=torch.float32,
                                    layer_name="cache_only_layers.2")
    features = torch.zeros(3, 2, 4)
    with pytest.raises(RuntimeError, match="forward 上下文"):
        layer(features)                                  # 不在 forward 上下文里
    with set_forward_context({}, num_tokens=3):
        with pytest.raises(KeyError, match="没有层"):
            layer(features)                              # 上下文里没有这一层的 metadata
    with pytest.raises(RuntimeError, match="kv_cache 还没绑定"):
        with set_forward_context({layer.layer_name: CacheOnlyAttentionMetadata(
                torch.zeros(3, dtype=torch.int64))}, num_tokens=3):
            layer(features)
    with pytest.raises(RuntimeError, match="不算 attention"):
        layer.impl.forward()
    # dtype 不一致：静默转换会变成另一种数值，必须报错
    kv_cache = torch.zeros(2, 4, 2, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        layer.impl.do_kv_cache_update(layer, features, kv_cache, torch.zeros(3, dtype=torch.int64))


def test_model_does_not_iterate_weights():
    """`load_weights()` 不许去读 target 的权重（迭代器是惰性的，读了就是白读几 GB）。"""
    config = target_config()
    derived = config.speculative_config.derive_extract_hidden_states_config(
        config.model_config, config.cache_config)
    model = ExtractHiddenStatesModel(derived.hf_config)

    class Exploding:
        def __iter__(self):
            raise AssertionError("cache-only 模型不该迭代权重")

    assert model.load_weights(Exploding()) == set()
    with pytest.raises(ValueError, match="eagle_aux_hidden_state_layer_ids"):
        ExtractHiddenStatesModel({**derived.hf_config, "eagle_aux_hidden_state_layer_ids": []})


def test_stacking_matches_upstream_stack():
    """`[T, L*H]`（本仓库 target 的拼接布局）与上游的 `stack(list, dim=1)` 逐值相同。"""
    config = target_config()
    proposer = ExtractHiddenStatesProposer(config, DEVICE)
    proposer.load_model()
    torch.manual_seed(0)
    layers = [torch.randn(5, proposer.hidden_size) for _ in range(len(AUX_LAYERS))]
    flat = torch.cat(layers, dim=-1)
    from_list = proposer._stack_hidden_states(layers)
    from_flat = proposer._stack_hidden_states(flat)
    assert from_list.shape == (5, len(AUX_LAYERS), proposer.hidden_size)
    assert torch.equal(from_list, from_flat)
    with pytest.raises(ValueError, match=r"与 L\*H"):
        proposer._stack_hidden_states(torch.zeros(5, proposer.hidden_size))


def test_propose_protocol_returns_first_column_only():
    """特殊协议：宽度 >1 时**只返回第 0 列**，并且真的把特征写进了缓存。"""
    config = target_config()
    proposer = ExtractHiddenStatesProposer(config, DEVICE)
    proposer.load_model()
    features = torch.arange(1, 2 * len(AUX_LAYERS) * proposer.hidden_size + 1,
                            dtype=torch.float32).view(2, len(AUX_LAYERS), proposer.hidden_size)
    slots = torch.tensor([3, 9], dtype=torch.int64)
    metadata = AttentionMetadata(query_start_loc=torch.tensor([0, 2]),
                                 seq_lens=torch.tensor([2]),
                                 block_table=torch.zeros(1, 4, dtype=torch.int64),
                                 slot_mapping=slots, block_size=4, num_reqs=1)
    sampled = torch.tensor([[7, 9], [11, 13]], dtype=torch.int32)      # K=1：两列
    # 上游形态：每层一个 [T, H] 的**列表**（本仓库 target 传的是拼接后的 [T, L*H]）
    returned = proposer.propose(num_speculative_tokens=1, sampled_token_ids=sampled,
                                target_hidden_states=[features[:, 0], features[:, 1]],
                                common_attn_metadata=metadata)
    assert returned.shape == (2, 1), "宽度必须退化到 1 列"
    assert returned[:, 0].tolist() == [7, 11], "返回的是第 0 列（target 自己采出的那个）"
    cache = proposer.kv_caches[proposer.attn_layer_names[0]]
    for index, slot in enumerate(slots.tolist()):
        assert torch.equal(cache[slot // 4, slot % 4], features[index].to(cache.device))


# ---------------------------------------------------------------- 2. 物理 slot 层


def test_physical_slots_layer_request_position():
    """两请求、两辅助层：读物理 slot，验证"哪一层的哪一行写到了哪一个槽位"。"""
    engine, core, runner = make_engine()
    calls = install_spy(runner)
    try:
        outputs, _ = run(engine, core, PROMPTS, max_tokens=4)
        assert outputs, "两个请求都该有输出"
        call = calls[0]                       # 第一轮：两条请求的 prefill
        stacked, slots = call["stacked"], call["slot_mapping"]
        assert stacked.shape[0] == len(PROMPTS[0][1]) + len(PROMPTS[1][1])
        # (a) 每一行都写到了它自己的槽位（直接读物理槽位）
        for index in range(stacked.shape[0]):
            slot = int(slots[index])
            assert torch.equal(read_slot(runner, slot), stacked[index]), \
                f"第 {index} 行没有落在槽位 {slot}"
        # (b) 有区分力：两层特征不同、两条请求的行不同——否则"互换"也看不出来
        assert not torch.equal(stacked[:, 0], stacked[:, 1]), "两层特征必须不同"
        assert not torch.equal(stacked[0], stacked[8]), "两条请求的特征必须不同"
        # (c) 独立参考：同一轮元数据 + 同一份权重重跑 → 与提议者写进缓存的那份逐值相同
        input_ids, positions = [], []
        for _, prompt in PROMPTS:
            input_ids.extend(prompt)
            positions.extend(range(len(prompt)))
        reference = reference_aux(runner, call["metadata"], input_ids, positions)
        assert torch.equal(reference, stacked), "缓存里的特征必须就是本轮 target 算出的特征"
        # (d) 层/请求/位置 → 槽位：从元数据独立推出来的三元组与写入顺序逐行对得上
        assert [entry[2] for entry in call["positions"]] == slots.tolist()
        assert [entry[:2] for entry in call["positions"]] == [
            (req_id, position) for req_id, prompt in PROMPTS
            for position in range(len(prompt))]
    finally:
        engine.shutdown()


# ---------------------------------------------------------------- 3. 生命周期层


def test_chunked_prefill_covers_every_position():
    """chunked prefill：分几轮算的 prompt，每个位置的特征最后都要在缓存里、而且是对的。

    参考来自**另一个引擎**（同样的权重、预算够大 → 一次装下整个 prompt）：换一套 batch 形状
    重算同一段 prompt，是真正独立的对照。
    """
    prompt = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 5, 6]        # 12 token，预算 8 → 两轮
    engine, core, runner = make_engine(budget=8, max_num_seqs=1)
    rounds = install_round_spy(runner)
    try:
        add_request(engine, "a", prompt, max_tokens=8)
        for _ in range(4):
            engine.step()
            if len(rounds) >= 2:
                break
        assert len(rounds) >= 2, f"预算 8 装不下 12 token 的 prompt，应该分块：{rounds}"
        assert rounds[0]["num_scheduled"]["a"] == 8, "第一块 8 个 token"
        cached = {position: read_slot(runner, slot_of(runner, "a", position)).clone()
                  for position in range(len(prompt))}
    finally:
        engine.shutdown()

    ref_engine, ref_core, ref_runner = make_engine(budget=64, max_num_seqs=1)
    ref_calls = install_spy(ref_runner)
    try:
        add_request(ref_engine, "a", prompt, max_tokens=8)
        ref_engine.step()
        reference = ref_calls[0]["stacked"]                # 一次算完的整段特征
        assert reference.shape[0] == len(prompt)
    finally:
        ref_engine.shutdown()

    for position in range(len(prompt)):
        # 两次前向的 query 行数不同（8 行 vs 12 行）→ 允许浮点累加顺序的微小差异；
        # 层/行/位置写错会差到 1e-1 以上，这个容差拦得住
        assert torch.allclose(cached[position], reference[position], atol=1e-6, rtol=1e-5), \
            f"位置 {position} 的特征与独立参考不符"


def test_rejected_tail_overwritten_and_release_reuse():
    """拒绝尾部被下一轮重算覆盖；请求释放、块被复用后读到的是新请求的特征。"""
    engine, core, runner = make_engine(budget=64, max_num_seqs=1)
    calls = install_spy(runner)
    rounds = install_round_spy(runner)
    try:
        _, stats = run(engine, core, [PROMPTS[0]], max_tokens=6)
        assert any(r["spec"] for r in rounds), "必须真的调度过草稿（K=1 → 每轮 2 行）"
        assert any(s.num_draft_tokens > 0 for s in stats)
        # 同一槽位被写过多次（草稿行那一格，下一轮被重算的正确 token 覆盖）
        write_counts = {}
        for call in calls:
            for slot in call["slot_mapping"].tolist():
                write_counts[slot] = write_counts.get(slot, 0) + 1
        assert max(write_counts.values()) >= 2, \
            f"应该有槽位被重写（拒绝尾部被覆盖），实际写入次数 {sorted(write_counts.values())}"
        # 最终内容 = 最后一次写入的那份（没有被更早的残留盖回去）
        last_write = {}
        for call in calls:
            for index, slot in enumerate(call["slot_mapping"].tolist()):
                last_write[slot] = call["stacked"][index]
        for slot, expected in last_write.items():
            assert torch.equal(read_slot(runner, slot), expected), f"槽位 {slot} 的最终内容不对"

        # 释放 + 复用：新请求写在复用的块上 → 它的槽位装的是它自己的特征
        calls.clear()
        outputs, _ = run(engine, core, [("c", [3, 1, 4, 1, 5])], max_tokens=2)
        assert outputs.get("c"), "新请求要有输出"
        assert calls, "新请求也要经过提议者"
        for call in calls:
            for index, slot in enumerate(call["slot_mapping"].tolist()):
                assert torch.equal(read_slot(runner, slot), call["stacked"][index]), \
                    f"复用块上的槽位 {slot} 没有写成当轮的特征"
    finally:
        engine.shutdown()


def test_prefix_hit_keeps_cached_features():
    """prefix 命中：命中段不重算，但命中段的特征必须已经在缓存里；新算的那段要写对。"""
    engine, core, runner = make_engine(prefix_caching=True, max_num_seqs=1)
    try:
        prompt_a = PROMPTS[0][1]                                  # 8 token = 两个完整块
        run(engine, core, [("a", prompt_a)], max_tokens=1)         # 跑完并发布块
        calls = install_spy(runner)
        rounds = install_round_spy(runner)
        prompt_b = prompt_a + [7, 8, 9, 10]                        # 前 8 个 token 与 a 相同
        add_request(engine, "b", prompt_b, max_tokens=8)
        engine.step()                                             # b 的 prefill 轮（b 还在批里）
        first = rounds[0]
        assert first["starts"].get("b") == len(prompt_a), \
            f"第二条请求应该命中 8 个 token 的 prefix，实际起点 {first['starts']}"
        assert first["num_scheduled"]["b"] == len(prompt_b) - len(prompt_a), \
            "命中段不重算：本轮只排后面的 token"
        cache = cache_of(runner)
        for position in range(len(prompt_a)):
            got = read_slot(runner, slot_of(runner, "b", position))
            assert torch.count_nonzero(got) > 0, f"命中段位置 {position} 的特征是空的"
        # 本轮新算的那一段：与独立参考逐值相同
        reference = reference_aux(runner, calls[0]["metadata"],
                                  list(prompt_b[len(prompt_a):]),
                                  list(range(len(prompt_a), len(prompt_b))))
        for index, position in enumerate(range(len(prompt_a), len(prompt_b))):
            assert torch.equal(read_slot(runner, slot_of(runner, "b", position)),
                               reference[index]), f"本轮新算的位置 {position} 的特征不对"
    finally:
        engine.shutdown()


def test_end_to_end_greedy_and_draft_protocol():
    """端到端：greedy 与非投机逐 token 相同；返回的就是采样张量的第 0 列。"""
    engine, core, runner = make_engine()
    calls = install_spy(runner)
    try:
        spec_outputs, stats = run(engine, core, PROMPTS)
    finally:
        engine.shutdown()
    base_engine, base_core, _ = make_engine(with_spec=False)
    try:
        base_outputs, _ = run(base_engine, base_core, PROMPTS)
    finally:
        base_engine.shutdown()
    assert spec_outputs == base_outputs, f"{spec_outputs} != {base_outputs}"
    assert calls, "提议者必须被调用过"
    assert stats and sum(s.num_draft_tokens for s in stats) > 0, "草稿必须真的进过调度器"
    for call in calls:
        assert call["returned"].shape[1] == 1
        assert torch.equal(call["returned"], call["sampled"][:, :1]), \
            "返回的就是采样张量第 0 列（宽度 >1 也只取这一列）"


def test_drafts_are_target_sampled_tokens():
    """每一枚草稿都是 target 自己采出的 token（不是模型猜的），且没有概率 q。"""
    engine, core, runner = make_engine(max_num_seqs=1)
    drafts_seen = []
    original = runner._propose_extract_hidden_states

    def spy(state, sampled_by_row):
        drafts = original(state, sampled_by_row)
        drafts_seen.append(drafts)
        return drafts

    runner._propose_extract_hidden_states = spy
    calls = install_spy(runner)
    try:
        run(engine, core, [PROMPTS[0]], max_tokens=4)
    finally:
        engine.shutdown()
    assert drafts_seen and calls and len(drafts_seen) == len(calls)
    for drafts in drafts_seen:
        assert drafts.draft_probs is None, "点质量 q：extract 方法没有概率"
        assert all(len(tokens) <= 1 for tokens in drafts.draft_token_ids)
    # 同一轮里：草稿 = 那一行采样张量的第 0 列（就是 target 本轮采出的 token）
    paired = 0
    for call, drafts in zip(calls, drafts_seen):
        for row, req_id in enumerate(drafts.req_ids):
            tokens = drafts.draft_token_ids[row]
            first = int(call["sampled"][row, 0])
            if tokens:
                assert tokens == [first], f"{req_id} 的草稿 {tokens} 不是采样第 0 列 {first}"
                paired += 1
    assert paired > 0, "至少要有一条请求真的拿到了草稿"
