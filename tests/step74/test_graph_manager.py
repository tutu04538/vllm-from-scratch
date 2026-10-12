"""74 关：图管理器与捕获期输入的**单元**判据（不跑引擎，CPU 也能跑）。

覆盖需求 074 §3 里"不跑端到端也能钉住"的那几条：

    U1  候选表：档位 → 图键的取整与上限（prefill 图 = 1+K 行/请求、decode 图 = 1 行/请求）
    U2  兼容性：`allow_shape_padding` 的两档（草稿 prefill 图必须精确命中；decode 图可补齐）
    U3  分派：没捕获完 / token 数为 0 / 候选表没有这个 token 数 → 一律回退 NONE（不猜）
    U4  `make_dummy`：写的是**常驻缓冲**，补齐行是"写了也没人读"的安全值
    U5  假块表/假槽位：块表补零行、槽位全哨兵（0 号垃圾桶）
    U6  capability 转发：`AttentionGroup` 把"能不能原地更新"问到**构造器**上
    U7  `set_attn()` 定下的"融合 / 逐步"（`_configure_fused_multi_step_decode`）与构造器的
        声明一致；不伪造 capability（声明 False 时基类的更新方法必须报错）
"""

import pytest
import torch

from spec74_helpers import DEVICE, make_config  # noqa: E402

from minivllm.attention.backends.torch_sdpa import (  # noqa: E402
    TorchAttentionBackend,
    TorchAttentionMetadataBuilder,
)
from minivllm.attention.metadata import AttentionMetadata  # noqa: E402
from minivllm.config import CUDAGraphMode  # noqa: E402
from minivllm.testing.fake_attn_backend import (  # noqa: E402
    NoDraftMetadataUpdateBuilder,
    draft_attn_groups,
)
from minivllm.worker.gpu.block_table import PAD_SLOT_ID, BlockTables  # noqa: E402
from minivllm.worker.gpu.cudagraph_utils import (  # noqa: E402
    BatchExecutionDescriptor,
    CudaGraphManager,
    _is_compatible,
)
from minivllm.worker.gpu.input_batch import InputBatch, InputBuffers  # noqa: E402
from minivllm.worker.utils import (  # noqa: E402
    AttentionGroup,
    get_uniform_decode_token_count,
    is_uniform_query_len,
)


def _make_buffers(max_num_reqs=4, max_num_tokens=32, device=DEVICE):
    return InputBuffers(max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens,
                        device=device)


def _make_block_tables(max_num_reqs=4, max_num_tokens=32, max_num_blocks=4,
                       device=DEVICE):
    return BlockTables(block_sizes=[4], max_num_reqs=max_num_reqs,
                       max_num_batched_tokens=max_num_tokens,
                       max_num_blocks_per_group=[max_num_blocks], device=device,
                       kernel_block_sizes=[4])


# -------- U1/U2/U3：候选表、兼容性、分派 --------

def test_candidates_round_to_uniform_decode_query_len():
    """U1：prefill 图的档位按 1+K 向上取整，请求数 = 行数 /(1+K)，超上限的丢掉。"""
    config = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 3, 4, 8])
    manager = CudaGraphManager(config, torch.device("cpu"),
                               CUDAGraphMode.FULL_DECODE_ONLY,
                               decode_query_len=4, allow_shape_padding=False)
    descs = manager._capture_descs[CUDAGraphMode.FULL]
    shapes = sorted((d.num_tokens, d.num_reqs) for d in descs)
    # 1/2/3/4 → 4（1 条请求）；8 → 8（2 条请求）
    assert shapes == [(4, 1), (8, 2)]
    assert all(d.uniform_token_count == 4 for d in descs)


def test_decode_candidates_use_one_token_per_request():
    """U1：decode 图的档位就是"请求数"，每请求 1 行。"""
    config = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 3, 4, 8])
    manager = CudaGraphManager(config, torch.device("cpu"),
                               CUDAGraphMode.FULL_DECODE_ONLY,
                               decode_query_len=1, allow_shape_padding=True)
    shapes = sorted((d.num_tokens, d.num_reqs) for d in manager._capture_descs[CUDAGraphMode.FULL])
    # max_num_seqs=3 → 4/8 两个档位的请求数超上限，必须丢掉（那种形状不可能出现）
    assert shapes == [(1, 1), (2, 2), (3, 3)]
    assert all(d.uniform_token_count == 1 for d in manager._capture_descs[CUDAGraphMode.FULL])


def test_shape_padding_flag_controls_compatibility():
    """U2：`allow_shape_padding=False` 时只认精确形状（草稿 prefill 图复用 target 元数据）。"""
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8,
                                    num_reqs=2, uniform_token_count=4)
    # 精确命中：两档都兼容
    assert _is_compatible(desc, 2, 8, 4, 0, None, allow_shape_padding=True)
    assert _is_compatible(desc, 2, 8, 4, 0, None, allow_shape_padding=False)
    # 更小的批：允许补齐的一档兼容；要求精确形状的那一档**必须拒绝**（否则图会去读
    # target 缓冲里没被写过的行）
    assert _is_compatible(desc, 1, 4, 4, 0, None, allow_shape_padding=True)
    assert not _is_compatible(desc, 1, 4, 4, 0, None, allow_shape_padding=False)
    # uniform_token_count 必须精确相等（统一 decode 图的形状就是 1+K 行）
    assert not _is_compatible(desc, 2, 8, 1, 0, None, allow_shape_padding=True)


def test_dispatch_falls_back_to_eager_when_shape_not_captured():
    """U3：没捕获完 / 没有这个档位 → NONE，且回退描述符带**未补齐**的数字。"""
    config = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 3, 4, 8])
    manager = CudaGraphManager(config, torch.device("cpu"),
                               CUDAGraphMode.FULL_DECODE_ONLY,
                               decode_query_len=4, allow_shape_padding=False)
    # 还没捕获 → NONE（`_graphs_captured` 之前一律 eager）
    desc = manager.dispatch(1, 4, 4)
    assert desc.cg_mode == CUDAGraphMode.NONE and desc.num_tokens == 4

    manager._graphs_captured = True
    # 候选表里没有 12 这个 token 数（只有 4/8 有图）→ NONE
    assert manager.dispatch(3, 12, 4).cg_mode == CUDAGraphMode.NONE
    # 有图但不兼容（非统一形状）→ NONE
    assert manager.dispatch(1, 8, None).cg_mode == CUDAGraphMode.NONE
    # 兼容 → 返回那张图的键
    assert manager.dispatch(1, 4, 4) == manager._capture_descs[CUDAGraphMode.FULL][-1]


def test_record_selection_is_the_only_writer():
    """验收要"断言实际选中"：账本由分派处唯一写入，且 `replayed` 说清是不是真的重放。"""
    config = make_config(spec_k=3, mode="full_decode_only", capture_sizes=[1, 2, 3, 4, 8])
    manager = CudaGraphManager(config, torch.device("cpu"),
                               CUDAGraphMode.FULL_DECODE_ONLY,
                               decode_query_len=1, allow_shape_padding=True)
    manager._graphs_captured = True
    desc = manager.dispatch(2, 2, 1)
    manager.record_selection(desc, 2, 2)
    assert manager.selections == [{
        "num_reqs": 2, "num_tokens": 2, "desc_num_reqs": 2, "desc_num_tokens": 2,
        "mode": "FULL", "replayed": False,   # 没真捕获过图 → 不算重放
    }]


# -------- U4/U5：捕获期输入 --------

def test_make_dummy_writes_persistent_buffers():
    """U4：`make_dummy` 只写常驻缓冲，并把补齐行写成"写了也没人读"的安全值。"""
    buffers = _make_buffers(max_num_reqs=4, max_num_tokens=32)
    buffers.input_ids[:8] = 7          # 先弄脏，确认会被覆盖
    buffers.positions[:8] = 5
    batch = InputBatch.make_dummy(2, 8, buffers)
    assert batch.num_reqs == 2 and batch.num_tokens == 8
    # 每请求 4 行（8 / 2）；seq_lens / query_start_loc 与运行期同一条口径
    assert buffers.seq_lens[:2].tolist() == [4, 4]
    assert buffers.seq_lens[2:].tolist() == [0, 0]        # 补齐请求 seq_len = 0
    assert buffers.query_start_loc[:3].tolist() == [0, 4, 8]
    assert buffers.query_start_loc[3:].tolist() == [8, 8]  # 非递减（后端要求）
    # token / position 清零（它们会被 embedding / RoPE 查表读到）
    assert buffers.input_ids[:8].tolist() == [0] * 8
    assert buffers.positions[:8].tolist() == [0] * 8
    assert batch.uniform_query_len == 4                    # 图里的形状
    assert batch.logits_indices.tolist() == [3, 7]
    # 缓冲是**同一批张量**（图里记的是指针）
    assert batch.input_ids.data_ptr() == buffers.input_ids.data_ptr()


def test_make_dummy_reports_non_uniform_batches():
    """U4 的另一半：摊不匀的假批（行数不是请求数的整数倍）不能声明统一形状。"""
    buffers = _make_buffers()
    batch = InputBatch.make_dummy(3, 8, buffers)   # 8 / 3 → 3,3,2
    assert batch.uniform_query_len is None
    assert batch.max_query_len == 3


def test_dummy_block_tables_and_slot_mappings_are_persistent_and_safe():
    """U5：假块表补零行（0 号块 = 垃圾桶）、假槽位全哨兵，且都是常驻缓冲的视图。"""
    tables = _make_block_tables()
    tables.input_block_tables[0][:2] = 9
    views = tables.get_dummy_block_tables(2)
    assert views[0].data_ptr() == tables.input_block_tables[0].data_ptr()
    assert tables.input_block_tables[0][:2].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]

    tables.slot_mappings.fill_(3)
    slot_view = tables.get_dummy_slot_mappings(5)
    assert slot_view.data_ptr() == tables.slot_mappings.data_ptr()
    # 填的是**整个**缓冲（内核可能访问到请求范围之外），不只是前 5 行
    assert (tables.slot_mappings == PAD_SLOT_ID).all()
    assert slot_view.shape == (1, 5)


# -------- U6/U7：capability 与融合/回退的门 --------

def test_attention_group_forwards_draft_metadata_capability():
    """U6：capability 挂在**构造器**上，由 `AttentionGroup` 转发（上游同名属性）。"""
    group = AttentionGroup(backend=TorchAttentionBackend, layer_names=["l0"],
                           metadata_builders=[TorchAttentionMetadataBuilder(4)])
    assert group.supports_draft_decode_metadata_update is True
    fake = AttentionGroup(backend=TorchAttentionBackend, layer_names=["l0"],
                          metadata_builders=[NoDraftMetadataUpdateBuilder(4)])
    assert fake.supports_draft_decode_metadata_update is False


def test_base_builder_refuses_to_update_draft_metadata():
    """U7：声明不支持的后端，`update_draft_decode_metadata()` 必须**报错**（不伪造）。"""
    builder = NoDraftMetadataUpdateBuilder(4)
    metadata = AttentionMetadata(query_start_loc=torch.zeros(2, dtype=torch.int32),
                                seq_lens=torch.zeros(1, dtype=torch.int32),
                                block_table=torch.zeros(1, 1, dtype=torch.int32),
                                slot_mapping=torch.zeros(1, dtype=torch.int64),
                                block_size=4, num_reqs=1, uniform_query_len=1)
    with pytest.raises(NotImplementedError, match="不支持草稿 decode 元数据的原地更新"):
        builder.update_draft_decode_metadata(metadata)
    # 生产后端是**空操作**：步相关字段都是常驻缓冲的切片（与上游 Triton 后端同款理由）
    TorchAttentionMetadataBuilder(4).update_draft_decode_metadata(metadata)


def test_uniform_decode_helpers():
    """U6 的前置判据：`is_uniform_query_len` / `get_uniform_decode_token_count`。"""
    assert is_uniform_query_len(2, 8, 4)
    assert not is_uniform_query_len(2, 9, 4)
    assert not is_uniform_query_len(0, 0, 4)
    assert get_uniform_decode_token_count(2, 8, 4, has_prefill=False) == 4
    # 有 prefill 的批即使形状像 decode 也不算（提示词分块会长成 decode 的样子）
    assert get_uniform_decode_token_count(2, 8, 4, has_prefill=True) is None


def test_gate_matches_capability_declaration():
    """U7：`set_attn()` 定下的路 = 构造器的 capability 声明（两个替身后端各一遍）。"""
    from minivllm.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )

    class Stub(AutoRegressiveSpeculator):
        def load_draft_model(self, target_model):
            raise NotImplementedError

    for supports_update, expected in ((True, True), (False, False)):
        config = make_config(spec_k=3)
        speculator = Stub(config, torch.device("cpu"))
        speculator.draft_attn_layer_names = ["l0"]
        speculator.attn_groups = draft_attn_groups(["l0"], 4,
                                                   supports_update=supports_update)
        speculator._configure_fused_multi_step_decode()
        assert speculator.use_fused_multi_step_decode is expected
