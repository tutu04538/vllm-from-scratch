"""69 关补充（71 关前置）：PIECEWISE 分段图 —— 键、分段点、填充口径、与 eager 逐值一致。

需求 71 §3.5 要求"动态投机长度与 full graph 不能共存 → 降级到 PIECEWISE"，而 69 关当时
只做了 FULL 那一半。这一份把另一半补上，验收点与 69 关同源：

    * 配置：PIECEWISE / FULL_AND_PIECEWISE 被接受；`splitting_ops` 不可配置（配了不生效＝报错）
    * 能力协商：后端能力档位（`AttentionCGSupport`）决定最终模式，`full` 被降级且不静默
    * 键：PIECEWISE 的键**不带请求数**（`num_reqs=None`）——注意力在图外，图只认 token 数
    * 填充口径：PIECEWISE 只补 token 行，请求级缓冲（query_start_loc/seq_lens/块表）保持真实数
    * 数值：eager / PIECEWISE / FULL_AND_PIECEWISE / FULL_DECODE_ONLY 四种模式 greedy 逐 token 相同
    * 工作区：段间静态缓冲按输入工作区容量开一次，地址稳定、不随档位重建
    * 统计：每一轮（含"没走图"）都进 `CUDAGraphStat` 表
"""

import pytest
import torch

from spec69_helpers import (CUDAGraphMode, DEVICE, make_engine, requires_cuda,
                            run_to_end, tiny_model)
from minivllm import SamplingParams
from minivllm.attention.backend import AttentionCGSupport, min_cudagraph_support
from minivllm.compilation import CUDAGraphLogging, CUDAGraphStat
from minivllm.config import CompilationConfig, CompilationMode


# ---------------------------------------------------------------- 1. 配置与能力协商（不需 GPU）


def test_piecewise_is_accepted_and_split_ops_are_recorded():
    """PIECEWISE 现在被接受，切分点被记下；显式配 `splitting_ops` 仍报错（配了不生效）。"""
    cfg = CompilationConfig(cudagraph_mode="full_and_piecewise")
    assert cfg.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
    assert cfg.splitting_ops_contain_attention()
    assert cfg.splitting_ops == ["minivllm::attention_core"]
    pure = CompilationConfig(cudagraph_mode="piecewise")
    assert pure.cudagraph_mode is CUDAGraphMode.PIECEWISE
    with pytest.raises(NotImplementedError, match="splitting_ops"):
        CompilationConfig(cudagraph_mode="piecewise", splitting_ops=[])
    # 编译这一轴仍然不做（PIECEWISE 在本仓库靠手工分段，不靠 torch.compile）
    with pytest.raises(NotImplementedError, match="mode"):
        CompilationConfig(mode=CompilationMode.VLLM_COMPILE)


def test_attention_capability_min_is_the_most_conservative():
    """多个 KV cache group 取**最保守**的一档（一个不能进图，整批就不能进图）。"""
    assert min_cudagraph_support([AttentionCGSupport.ALWAYS,
                                  AttentionCGSupport.UNIFORM_BATCH]) \
        is AttentionCGSupport.UNIFORM_BATCH
    assert min_cudagraph_support([AttentionCGSupport.UNIFORM_BATCH,
                                  AttentionCGSupport.NEVER]) is AttentionCGSupport.NEVER
    with pytest.raises(ValueError):
        min_cudagraph_support([])


def test_torch_backend_declares_uniform_batch_support():
    """本仓库的 Torch 后端声明的档位是 UNIFORM_BATCH（图内注意力只认"每请求行数相同"）。"""
    from minivllm.attention.backends.torch_sdpa import TorchAttentionBackend

    builder_cls = TorchAttentionBackend.get_builder_cls()
    assert builder_cls.get_cudagraph_support(None) is AttentionCGSupport.UNIFORM_BATCH
    # 基类默认 NEVER（新的后端不声明能力就不能进图，是保守的一侧）
    from minivllm.attention.metadata import AttentionMetadataBuilder

    assert AttentionMetadataBuilder.get_cudagraph_support(None) \
        is AttentionCGSupport.NEVER


def test_graph_stat_table_counts_every_mode_including_none():
    """统计表按 (真实行数, 补齐行数, 模式) 聚合，**含"没走图"那一类**。"""
    logging_ = CUDAGraphLogging(CUDAGraphMode.FULL_AND_PIECEWISE, [1, 2, 4])
    logging_.observe(CUDAGraphStat(1, 1, 0, "FULL"))
    logging_.observe(CUDAGraphStat(1, 1, 0, "FULL"))
    logging_.observe(CUDAGraphStat(6, 8, 2, "PIECEWISE"))
    logging_.observe(CUDAGraphStat(9, 9, 0, "NONE"))
    table = logging_.generate_metric_table()
    assert "FULL_AND_PIECEWISE" in table and "[1, 2, 4]" in table
    lines = [line for line in table.splitlines() if line.startswith("| ")]
    body = "\n".join(lines)
    assert "| 1               | 1             | 0            | FULL         | 2" in body
    assert "| 6               | 8             | 2            | PIECEWISE    | 1" in body
    assert "NONE" in body                     # 没走图的那一轮也在账上
    logging_.log(lambda _msg: None)           # log() 打完清空
    assert logging_.stats == []


def _inner_qwen3(runner):
    """runner.model（可能有 FULL 包装器）→ `Qwen3Model`（分段缓冲与分段包装器都在它上面）。"""
    model = runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    return getattr(model, "model", model)          # Qwen3ForCausalLM.model 或 Qwen3Model 本身


# ---------------------------------------------------------------- 2. 分派与填充口径（要 CUDA）


@requires_cuda
def test_plain_full_is_downgraded_by_capability_negotiation():
    """`cudagraph_mode='full'` + 本后端（UNIFORM_BATCH）→ 降级成 FULL_AND_PIECEWISE。"""
    tiny_dir, hf = tiny_model()
    _engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf, mode="full",
                                         budget=32, max_num_seqs=4)
    try:
        assert runner.compilation_config.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
        assert runner.cudagraph_dispatcher.cudagraph_mode is CUDAGraphMode.FULL_AND_PIECEWISE
        # 降级不是"字面执行"：混合批走分段图、decode 走全图，两张表都有键
        descs = runner.cudagraph_dispatcher.get_capture_descs()
        modes = {mode for mode, _ in descs}
        assert modes == {CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL}
    finally:
        _engine.shutdown()


@requires_cuda
def test_piecewise_keys_drop_num_reqs_and_only_pad_tokens():
    """PIECEWISE 的键没有请求数；填充只补 token 行、请求级缓冲保持真实数。"""
    tiny_dir, hf = tiny_model()
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf, mode="piecewise",
                                        budget=32, max_num_seqs=4, blocks=64)
    try:
        keys = [desc for _mode, descs in runner.cudagraph_dispatcher.get_capture_descs()
                for desc in descs]
        assert keys and all(desc.num_reqs is None for desc in keys)
        # 一条 6 行的 prefill → 补到档位 8；请求数保持真实（1），不是 8
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=2, temperature=0.0, eos_token_id=999))
        run_to_end(engine, limit=4)
        selection = runner.cudagraph_selections[0]
        assert selection["mode"] == "PIECEWISE"
        assert (selection["num_tokens"], selection["padded_tokens"]) == (6, 8)
        buffers = runner._padded_buffers
        assert buffers["slot_mapping"][:8].tolist()[:6] != [] and \
            buffers["slot_mapping"][6:8].tolist() == [-1, -1]      # padding 行是哨兵
        assert buffers["input_ids"][6:8].tolist() == [0, 0]        # padding 行 token 归零
        assert buffers["positions"][6:8].tolist() == [0, 0]        # padding 行位置归零
        # 请求级缓冲保持真实数：PIECEWISE 交给注意力的元数据按真实请求数**切片**，
        # 补齐行之外的残值读不到（"有效长度用切片表达"是 AGENTS §8 的那条约定）。
        meta = runner._piecewise_attn_metadata(1, 8)
        assert meta.num_reqs == 1 and meta.uniform_query_len is None
        assert meta.slot_mapping.shape[0] == 8          # token 级：按档位补齐
        assert meta.seq_lens.shape[0] == 1              # 请求级：不补
        assert meta.query_start_loc.shape[0] == 2
        assert meta.block_table.shape[0] == 1
    finally:
        engine.shutdown()


# ---------------------------------------------------------------- 3. 分段图真的在跑（要 CUDA）


@requires_cuda
def test_pieces_capture_once_per_key_and_replay_afterwards():
    """每层两段各按 PIECEWISE 的键捕获一次，之后同一键只重放；段间缓冲地址稳定。"""
    tiny_dir, hf = tiny_model()
    engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf,
                                        mode="full_and_piecewise", budget=32,
                                        max_num_seqs=4, blocks=64)
    try:
        model = _inner_qwen3(runner)
        layer = model.layers[0]
        assert layer._piece_pre.runtime_mode is CUDAGraphMode.PIECEWISE
        assert layer._piece_post.runtime_mode is CUDAGraphMode.PIECEWISE
        num_keys = len(runner.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE])
        assert layer._piece_pre.num_captures == num_keys          # 每个键一张图
        stage_ptr = layer._attn_out_stage.data_ptr()
        hidden_ptr = model._hidden_in_stage.data_ptr()
        assert layer._attn_out_stage.shape[0] == \
            runner.vllm_config.scheduler_config.max_num_batched_tokens
        engine.add_request("A", [1, 2, 3, 4, 5, 6],
                           SamplingParams(max_tokens=4, temperature=0.0, eos_token_id=999))
        run_to_end(engine)
        assert layer._piece_pre.num_replays > 0                   # 真的重放过（不是每轮重捕获）
        assert layer._piece_pre.num_captures == num_keys          # 运行期没有偷偷再捕获
        assert layer._attn_out_stage.data_ptr() == stage_ptr      # 缓冲不重建
        assert model._hidden_in_stage.data_ptr() == hidden_ptr
        assert "PIECEWISE" in {rec["mode"] for rec in runner.cudagraph_selections}
    finally:
        engine.shutdown()


@requires_cuda
def test_all_modes_agree_with_eager_token_by_token():
    """eager / PIECEWISE / FULL_AND_PIECEWISE / FULL_DECODE_ONLY：先 prefill 再加第二条请求
    （制造**混合 prefill+decode 批**），四种模式的 greedy 输出必须逐 token 相同。

    K=0 与 K=2 都测：投机批的每请求行数是 1+K，分段图要能吃下"行数不同"的混合批。
    """
    tiny_dir, hf = tiny_model()
    results = {}
    selections = {}
    for mode in ("none", "piecewise", "full_and_piecewise", "full_decode_only"):
        for spec_k in (None, 2):
            engine, _core, runner = make_engine(
                tiny_dir=tiny_dir, hf_config=hf, mode=mode, spec_k=spec_k,
                budget=32, max_num_seqs=4, blocks=64)
            try:
                sampling = lambda: SamplingParams(          # noqa: E731
                    max_tokens=6, temperature=0.0, eos_token_id=999)
                engine.add_request("A", [1, 2, 3, 4, 5, 6], sampling())
                final = run_to_end(engine, limit=3)
                engine.add_request("B", [2, 3, 4], sampling())
                final.update(run_to_end(engine))
            finally:
                engine.shutdown()
            results[(mode, spec_k)] = {k: list(v) for k, v in sorted(final.items())}
            selections[(mode, spec_k)] = [rec["mode"]
                                          for rec in runner.cudagraph_selections]
    base = results[("none", None)]
    assert base["A"] and base["B"], base
    for spec_k in (None, 2):
        for mode in ("piecewise", "full_and_piecewise", "full_decode_only"):
            assert results[(mode, spec_k)] == results[("none", spec_k)], (mode, spec_k)
    # 混合批确实走了分段图（不是"碰巧都回退 eager 所以一致"）
    assert "PIECEWISE" in selections[("full_and_piecewise", None)]
    assert "PIECEWISE" in selections[("piecewise", None)]


@requires_cuda
def test_piecewise_disabled_leaves_the_model_path_untouched():
    """没开分段图时**一个包装器都不装**（`_piece_pre is None`）：FULL 走全图、NONE 走 eager。"""
    tiny_dir, hf = tiny_model()
    for mode in ("full_decode_only", "none"):
        engine, _core, runner = make_engine(tiny_dir=tiny_dir, hf_config=hf, mode=mode,
                                            budget=16, max_num_seqs=2)
        try:
            inner = _inner_qwen3(runner)
            assert inner._hidden_in_stage is None
            assert all(layer._piece_pre is None for layer in inner.layers)
            assert all(layer._attn_out_stage is None for layer in inner.layers)
        finally:
            engine.shutdown()
