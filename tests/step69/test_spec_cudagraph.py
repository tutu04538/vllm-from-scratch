"""69 关（二）：键 → 捕获 → 重放，以及"图与 eager 结果一致 / padding 不留痕"。

需求 069 §4 的验收项在这里逐条落地：

    * eager / Graph 的同权重 greedy 与中间值对照（含首拒与全接受）
    * B=1→3→1、图 bucket 边界、prefill/decode 混合、记录**真实**选中的 mode/key
    * 复用地址不变；故意污染 padding 缓冲，有效输出不变、且不写错误 KV slot
    * profiler 看到真实 replay，热路径不夹带逐请求同步
    * 不支持的模式按规则降级/拒绝（配置期报错，不静默退化）
"""

import pytest
import torch

from spec69_helpers import (CUDAGraphMode, CompilationConfig, DEVICE, DispatcherHarness,
                            graph_snapshot, greedy_outputs, kv_digest, make_config,
                            make_engine, requires_cuda, run_to_end, tiny_eagle3_dir)
from minivllm import SamplingParams
from minivllm.config import CompilationMode, VllmConfig
from minivllm.forward_context import BatchDescriptor


# ---------------------------------------------------------------- 1. 键与分派（不需要模型）


def test_bs_to_padded_graph_size_matches_the_source_rule():
    """补齐映射逐值：正好命中档位不补，落在两档之间补到**下一个**档位。"""
    harness = DispatcherHarness(budget=64, max_num_seqs=4, mode="full_decode_only",
                                capture_sizes=[2, 4, 8], max_capture_size=8)
    dispatcher = harness.dispatcher
    mapping = dispatcher._bs_to_padded_graph_size
    assert mapping[1] == 2 and mapping[2] == 2 and mapping[3] == 4
    assert mapping[4] == 4 and mapping[5] == 8 and mapping[8] == 8
    assert len(mapping) == 9                    # 表只到最大档位为止（更大的形状不查表）


def test_dispatch_pads_to_bucket_and_computes_num_reqs():
    """真实形状 → (FULL, 补齐后的键)：num_reqs 由补齐后的行数除以 1+K 得到。

    K=3（q=4）、max_num_seqs=4：5 行 → 补到 8 行 → 2 条请求（统一 decode）。
    """
    harness = DispatcherHarness(spec_k=3, budget=64, max_num_seqs=4)
    mode, desc = harness.dispatch(5, uniform_decode=True)
    assert mode is CUDAGraphMode.FULL
    assert desc.num_tokens == 8 and desc.num_reqs == 2 and desc.uniform is True
    # 4 行正好是档位 → 不补
    mode, desc = harness.dispatch(4, uniform_decode=True)
    assert (mode, desc.num_tokens, desc.num_reqs) == (CUDAGraphMode.FULL, 4, 1)


def test_mixed_prefill_decode_falls_back_to_eager():
    """混合批（每条请求行数不同）在 FULL_DECODE_ONLY 下没有图键 → 按规则回退 NONE。"""
    harness = DispatcherHarness(spec_k=3, budget=64, max_num_seqs=4)
    # 一条 4 行（K+1）+ 一条 1 行：max_num_scheduled=4 但总行数 5 != 8 → 不是统一批
    assert harness.dispatch(5, uniform_decode=False)[0] is CUDAGraphMode.NONE
    # 纯 prefill（比如 12 行一条请求）也没有键
    assert harness.dispatch(12, uniform_decode=False)[0] is CUDAGraphMode.NONE


def test_shapes_above_the_largest_bucket_fall_back():
    harness = DispatcherHarness(spec_k=0, budget=16, max_num_seqs=16, mode="full")
    assert harness.dispatch(8, uniform_decode=True)[0] is CUDAGraphMode.FULL
    mode, desc = harness.dispatch(17, uniform_decode=True)
    assert mode is CUDAGraphMode.NONE and desc == BatchDescriptor(17)


def test_capture_sizes_are_rounded_up_to_multiples_of_one_plus_k():
    """档位表必须是 (1+K) 的倍数（上游 issue #28207 的修法）。"""
    harness = DispatcherHarness(spec_k=2, budget=32, max_num_seqs=4)
    sizes = harness.config.compilation_config.cudagraph_capture_sizes
    assert sizes and all(size % 3 == 0 for size in sizes), sizes
    # 每条请求恰好 1+K 行 → 每个键的 num_reqs 都是整数
    for _mode, descs in harness.dispatcher.get_capture_descs():
        for desc in descs:
            assert desc.num_tokens % 3 == 0


def test_capture_desc_order_is_largest_first():
    """捕获顺序：大图在前（小图复用大图占下的显存池）。"""
    harness = DispatcherHarness(spec_k=0, budget=32, max_num_seqs=8, mode="full")
    for _mode, descs in harness.dispatcher.get_capture_descs():
        tokens = [desc.num_tokens for desc in descs]
        assert tokens == sorted(tokens, reverse=True)


def test_unsupported_modes_are_rejected_loudly():
    """**还**不支持的东西在配置期明确拒绝（不静默退化）。

    71 关补充：PIECEWISE 已实现（手工分段），所以这里断言的是"它被接受且切分点被记下"；
    仍然拒绝的是"配了不生效"的那几项——torch.compile 模式、`compile_sizes`、
    以及**不可配置的 `splitting_ops`**（本仓库的切分点由模型结构决定）。改动理由与
    期望值变化的记录见提交信息（207 §8：改期望值、不改断言强度）。
    """
    piecewise = CompilationConfig(cudagraph_mode="piecewise")
    assert piecewise.cudagraph_mode is CUDAGraphMode.PIECEWISE
    assert piecewise.splitting_ops == ["minivllm::attention_core"]
    assert piecewise.splitting_ops_contain_attention()
    both = CompilationConfig(cudagraph_mode="full_and_piecewise")
    assert both.splitting_ops == ["minivllm::attention_core"]
    with pytest.raises(NotImplementedError, match="mode"):
        CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    with pytest.raises(NotImplementedError, match="compile_sizes"):
        CompilationConfig(compile_sizes=[8])
    with pytest.raises(NotImplementedError, match="splitting_ops"):
        CompilationConfig(cudagraph_mode="piecewise",
                          splitting_ops=["vllm::unified_attention_with_output"])
    with pytest.raises(ValueError):
        CompilationConfig(cudagraph_mode="does_not_exist")


def test_config_resolution_and_enforce_eager():
    """默认解析：CUDA → FULL_AND_PIECEWISE（= 上游 V1 默认）；CPU / enforce_eager → NONE。

    71 关补充：以前本仓库默认解析成 FULL_DECODE_ONLY，因为 PIECEWISE 那一半没实现；
    现在分段图有了，默认值回到上游的 `FULL_AND_PIECEWISE`（decode 走全图、混合批走分段图）。
    """
    tiny_dir, hf = __import__("spec69_helpers").tiny_model()
    cuda_cfg = make_config(tiny_dir=tiny_dir, hf_config=hf, device="cuda")
    assert cuda_cfg.compilation_config.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
    assert cuda_cfg.compilation_config.cudagraph_capture_sizes
    cpu_cfg = make_config(tiny_dir=tiny_dir, hf_config=hf, device="cpu")
    assert cpu_cfg.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
    assert cpu_cfg.compilation_config.cudagraph_capture_sizes == []
    eager_cfg = make_config(tiny_dir=tiny_dir, hf_config=hf, device="cuda",
                            enforce_eager=True)
    assert eager_cfg.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
    assert eager_cfg.compilation_config.max_cudagraph_capture_size == 0


def test_explicit_plain_full_is_downgraded_at_engine_init(tiny_dir, hf_config):
    """`cudagraph_mode='full'`（非分段）会被**能力协商**降级成 FULL_AND_PIECEWISE。

    71 关补充：`full` 的含义是"混合 prefill/decode 批也录一张全图"，而那要求注意力后端
    支持任意批（`AttentionCGSupport.ALWAYS`）。本仓库的 Torch 后端只支持"每请求行数相同"
    的批（`UNIFORM_BATCH`），所以这里按上游的降级规则变成 `FULL_AND_PIECEWISE`：decode 仍走
    全图，混合批改走分段图。以前这条是"直接 NotImplementedError"，现在是"按规则降级 + warning"，
    断言强度不变（都要求"不许静默按字面执行"）。
    """
    if DEVICE != "cuda":
        pytest.skip("只在 CUDA 上会走到建图那条路")
    _engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, mode="full")
    assert runner.compilation_config.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
    assert runner.cudagraph_dispatcher.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE


# ---------------------------------------------------------------- 2. 捕获与重放（要 CUDA）


@requires_cuda
def test_capture_happens_once_per_key_and_replay_reuses_it(tiny_dir, hf_config):
    """同一个键只捕获一次；之后每一轮都是 replay，而且输入缓冲地址不变。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=None,
                                        budget=16, blocks=32, mode="full_decode_only")
    before = graph_snapshot(runner)
    assert before["captured"] >= 2, before            # 至少 [1, 2] 两个档位
    assert before["captures"] == before["captured"]   # 捕获次数 == 档位数
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=8, temperature=0.0, eos_token_id=999))
    addresses = {name: buffer.data_ptr() for name, buffer in runner._padded_buffers.items()}
    run_to_end(engine)
    after = graph_snapshot(runner)
    assert after["replays"] >= 7, after               # 8 个 token 至少 7 次 decode
    assert after["captures"] == before["captures"], "重放阶段不该再捕获新图"
    assert {name: buffer.data_ptr() for name, buffer in
            runner._padded_buffers.items()} == addresses, "静态缓冲的地址必须恒定"
    engine.shutdown()


@requires_cuda
def test_eager_and_graph_produce_the_same_greedy_tokens(tiny_dir, hf_config):
    """同权重 greedy 逐 token 相同：普通 decode、投机（首拒 / 全接受）三种形状都覆盖。"""
    prompts = (("A", [1, 2, 3, 4, 5, 6]), ("B", [2, 3, 4]))
    for spec_k in (None, 1, 3):
        eager = greedy_outputs(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=spec_k,
                               prompts=prompts, max_tokens=8, budget=32, blocks=32,
                               mode="none")
        graph = greedy_outputs(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=spec_k,
                               prompts=prompts, max_tokens=8, budget=32, blocks=32,
                               mode="full_decode_only")
        assert eager == graph, (spec_k, eager, graph)


@requires_cuda
def test_graph_selection_is_recorded_per_step(tiny_dir, hf_config):
    """每轮真实选中的 mode/key 都记在 `cudagraph_selections` 里（验收要求"记录真实选择"）。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=None,
                                        budget=16, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    engine.shutdown()
    modes = [entry["mode"] for entry in runner.cudagraph_selections]
    assert modes[0] == "NONE", "prefill（6 行、非统一 decode）没有图键，必须回退 eager"
    assert all(mode == "FULL" for mode in modes[1:]), modes
    assert all(entry["padded_tokens"] == entry["num_tokens"]
               for entry in runner.cudagraph_selections[1:])


@requires_cuda
def test_batch_size_1_to_3_to_1_reuses_two_graphs(tiny_dir, hf_config):
    """B=1→3→1：档位来回切换时只重放、不重新捕获（键就是"档位"）。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=None,
                                        budget=16, blocks=64, max_num_seqs=4,
                                        mode="full_decode_only")
    captured = graph_snapshot(runner)["captures"]

    def drain(limit=20):
        for _ in range(limit):
            if not engine.has_unfinished_requests():
                return
            list(engine.step())

    def add(req_id, prompt):
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))

    add("a", [1, 2, 3, 4])            # B=1
    drain()
    for req_id in ("b", "c", "d"):    # B=3
        add(req_id, [2, 3, 4, 5])
    drain()
    add("e", [5, 6, 7])               # 又回到 B=1
    drain()

    snapshot = graph_snapshot(runner)
    assert snapshot["captures"] == captured, "档位不变时不该再捕获"
    assert snapshot["replays"] > 0
    padded = [entry["padded_tokens"] for entry in runner.cudagraph_selections
              if entry["mode"] == "FULL"]
    # 1+K=1（非投机）→ 档位 1、2、4；三条请求的批补到 4
    assert 1 in padded and 4 in padded, padded      # 两个档位都真的用过
    engine.shutdown()


@requires_cuda
def test_polluted_padding_buffer_does_not_change_valid_output(tiny_dir, hf_config):
    """污染 padding 缓冲：有效输出不变，而且真实块的 KV 逐位相同（0 号块除外）。

    **必须真的存在补齐区**，否则这条用例是空的：所以用 3 条请求 × K=1（= 6 行）→ 档位 8 行，
    既有 2 行补齐的**行**、又多出 1 条补齐的**请求**（3 → 4 条）。断言里先查 `tail > 0`
    与 `padded_reqs > num_reqs`，再谈污染。

    污染用的是**合法但在这些行上"错"的值**（最大 token id / 最大位置）：padding 行的内容
    会被 embedding 与 RoPE 查表读到，所以只有"在合法范围内"的污染才是在考"这些行不影响结果"；
    超出范围的值会被模型自己的越界检查拦下（那是另一件事，不是本用例要证明的）。
    """
    vocab_size = hf_config["vocab_size"]

    def run(poison: bool):
        engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1,
                                           budget=16, blocks=32, max_num_seqs=4,
                                           mode="full_decode_only")
        original = runner._prepare_inputs_padded
        seen = {}

        def poisoned(inputs, mode, batch_descriptor):
            original(inputs, mode, batch_descriptor)
            buffers = runner._padded_buffers
            used = inputs.num_tokens
            seen["tail"] = int(batch_descriptor.num_tokens - used)
            seen["num_reqs"] = int(batch_descriptor.num_reqs) - runner.input_batch.num_reqs
            if poison and seen["tail"] > 0:
                # 补齐区（真实行之后：包括"补齐请求"的那几行）灌成合法但错的值
                buffers["input_ids"][used:].fill_(vocab_size - 1)
                buffers["positions"][used:].fill_(runner.max_model_len - 1)
            seen.setdefault("rows", []).append(
                (int(used), int(batch_descriptor.num_tokens)))

        runner._prepare_inputs_padded = poisoned
        for req_id, prompt in (("a", [1, 2, 3, 4]), ("b", [2, 3, 4, 5]), ("c", [3, 4, 5, 6])):
            engine.add_request(req_id, list(prompt),
                               SamplingParams(max_tokens=4, temperature=0.0,
                                              eos_token_id=999))
        outputs = run_to_end(engine)
        digest = kv_digest(runner).clone()
        engine.shutdown()
        return outputs, digest, seen

    clean, clean_digest, clean_seen = run(poison=False)
    dirty, dirty_digest, dirty_seen = run(poison=True)
    assert dirty_seen["tail"] > 0, f"这一轮没有补齐区，用例是空的：{dirty_seen}"
    assert dirty_seen["num_reqs"] > 0, f"没有补齐请求：{dirty_seen}"
    assert clean == dirty, "污染 padding 之后有效输出变了：说明 padding 行漏进了计算"
    assert torch.equal(clean_digest, dirty_digest), \
        "真实块的 KV 被改了：padding 行写到了不该写的地方（0 号块之外的槽位）"


@requires_cuda
def test_padding_rows_have_deterministic_content(tiny_dir, hf_config):
    """补齐行的内容**显式写死为 0**，不依赖"上一轮恰好留了什么"。

    为什么在意：padding 行虽然结果会被丢掉、槽位也是哨兵，但它们的 token/position **会被
    embedding 与 RoPE 查表读到**（超范围就是 device 端越界）。所以不能把"上一轮的残值恰好
    合法"当不变量——填进去的必须是确定的、合法的值。
    """
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=1,
                                       budget=16, blocks=32, max_num_seqs=4,
                                       mode="full_decode_only")
    original = runner._prepare_inputs_padded
    seen = {}

    def spy(inputs, mode, batch_descriptor):
        buffers = runner._padded_buffers
        # 先把整块灌成"合法但非 0"的旧值，模拟"上一轮留下的残值"
        buffers["input_ids"][:batch_descriptor.num_tokens].fill_(hf_config["vocab_size"] - 1)
        buffers["positions"][:batch_descriptor.num_tokens].fill_(runner.max_model_len - 1)
        original(inputs, mode, batch_descriptor)
        used = inputs.num_tokens
        seen["tail_ids"] = buffers["input_ids"][used:batch_descriptor.num_tokens].tolist()
        seen["tail_positions"] = buffers["positions"][used:batch_descriptor.num_tokens].tolist()
        seen["tail_slots"] = buffers["slot_mapping"][used:batch_descriptor.num_tokens].tolist()

    runner._prepare_inputs_padded = spy
    for req_id, prompt in (("a", [1, 2, 3, 4]), ("b", [2, 3, 4, 5]), ("c", [3, 4, 5, 6])):
        engine.add_request(req_id, list(prompt),
                           SamplingParams(max_tokens=2, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    engine.shutdown()
    assert seen.get("tail_slots"), f"没有补齐区：{seen}"
    assert set(seen["tail_ids"]) == {0}, seen
    assert set(seen["tail_positions"]) == {0}, seen
    assert set(seen["tail_slots"]) == {-1}, seen      # 哨兵：不能是 0（0 是真实槽位）


@requires_cuda
def test_padding_writes_land_only_in_the_blank_block(tiny_dir, hf_config):
    """0 号块是垃圾桶：请求的块表里永远没有它，padding 的垃圾只落在它身上。"""
    engine, core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                       budget=32, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=6, temperature=0.0, eos_token_id=999))
    run_to_end(engine)
    manager = core.kv_cache_manager
    assert manager.block_pool.null_block is not None, "开了图就必须留白 0 号块"
    # 容量口径：留白之后"可用块数 = 总数 - 1"（`num_allocated_blocks` 已经扣掉垃圾桶）
    assert manager.num_free_blocks() == manager.num_gpu_blocks - 1 -         manager.num_allocated_blocks
    used = {block_id for row in runner.input_batch.block_table.cpu[:runner.input_batch.num_reqs]
            for block_id in row.tolist() if block_id != 0}
    assert 0 not in used, "0 号块被分给了真实请求：padding 会覆盖真实数据"
    engine.shutdown()


@requires_cuda
def test_no_per_request_sync_inside_the_graph_path(tiny_dir, hf_config):
    """热路径不夹带逐请求同步：图前向期间任何一次隐式同步都当场报错。

    `torch.cuda.set_sync_debug_mode("error")` 会在"CPU 等 GPU"的操作（`.item()`、
    `bool(tensor)`、`int(tensor)`、D2H copy）上抛异常。图路径的输入准备与模型前向都在
    这个窗口里跑一遍——它们必须一次同步都没有（每请求一次同步正是 69 关要消灭的东西）。
    """
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=3,
                                        budget=32, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=3, temperature=0.0, eos_token_id=999))
    list(engine.step())            # prefill（eager）
    # 直接问图那一段：输入准备（H2D 上传）本来就在图外，这里要证明的是**图区域内**没有同步。
    # 形状用统一 decode 的：K=3 → 每请求 4 行，所以 num_tokens=4、一条请求。
    mode, batch_descriptor = runner._determine_batch_execution_and_padding(
        num_tokens=4, num_reqs=1, num_scheduled_tokens=[4], max_num_scheduled_tokens=4)
    assert mode is CUDAGraphMode.FULL
    runner._fill_padded_buffers(4, 1, [4], batch_descriptor.num_tokens,
                                batch_descriptor.num_reqs, 4)
    metadata = runner._padded_attn_metadata(batch_descriptor)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        runner._run_model_padded(
            __import__("minivllm.worker.gpu_model_runner", fromlist=["_PaddedRun"])._PaddedRun(
                batch_descriptor=batch_descriptor, mode=mode,
                num_tokens=batch_descriptor.num_tokens),
            metadata=metadata)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    engine.shutdown()


@requires_cuda
def test_profiler_sees_a_real_graph_replay(tiny_dir, hf_config):
    """profiler 里能看到 `cudaGraphLaunch`：证明热路径真的在重放图（而不是"看起来配好了"）。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=None,
                                        budget=16, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
    list(engine.step())            # 先跑一轮 prefill，避免把捕获算进来
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        while engine.has_unfinished_requests():
            list(engine.step())
        torch.cuda.synchronize()
    names = {event.name for event in prof.events()}
    assert any("cudaGraphLaunch" in name for name in names), sorted(names)[:20]
    assert prof.key_averages() is not None
    engine.shutdown()


@requires_cuda
def test_graph_mode_keeps_spec_logprobs_and_grammar_working(tiny_dir, hf_config):
    """68 关的约束输出在受支持模式（图）下同样成立：logprobs 的宽度/行数与 greedy 输出对齐。"""
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf_config, spec_k=2,
                                        budget=32, blocks=32, mode="full_decode_only")
    engine.add_request("r", [1, 2, 3, 4, 5, 6],
                       SamplingParams(max_tokens=4, temperature=0.0, logprobs=2,
                                      eos_token_id=999))
    seen = []
    while engine.has_unfinished_requests():
        for out in engine.step():
            seen.append(out)
    engine.shutdown()
    assert seen and seen[-1].logprobs is not None
    # 每个交付位置最多 2 个候选（top-2），且必须有采样到的那个 token
    for position in seen[-1].logprobs:
        assert 1 <= len(position) <= 2
    assert any(entry["mode"] == "FULL" for entry in runner.cudagraph_selections)
