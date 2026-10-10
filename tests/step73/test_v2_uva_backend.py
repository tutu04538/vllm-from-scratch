"""73 关：UVA 设备视图的**两条后端**（对应上游 `_C.get_cuda_view_from_cpu_tensor`）。

判据只有一条、但必须真做出来：**设备张量与宿主张量别名同一块内存**——
GPU 内核写进去、CPU 立刻读得到；CPU 改一行、内核也立刻看得到。
"拷了一份看起来一样的数据"不算通过（那正是 UVA 要避免的东西）。

两条后端（`MINIVLLM_UVA_BACKEND` 显式选择，默认 cpp）：
    cpp     `minivllm/uva_ext.py` 用 `load_inline` 现编的上游同构算子
    python  ctypes + DLPack 兜底（没有 C++ 工具链的机器）
"""

import os

import pytest
import torch

from spec73_helpers import requires_cuda  # noqa: E402

from minivllm import uva_ext  # noqa: E402


def _uva_buffer(size=(8,), dtype=torch.int32):
    from minivllm.worker.gpu.buffer_utils import UvaBuffer

    return UvaBuffer(size, dtype)


@requires_cuda
def test_default_backend_is_cpp_and_aliases_memory(monkeypatch):
    """默认后端 = cpp（与上游同级），并且**双向可见**（真别名，不是拷贝）。"""
    monkeypatch.delenv("MINIVLLM_UVA_BACKEND", raising=False)
    assert uva_ext.resolve_backend() == "cpp"
    buf = _uva_buffer()
    # 上游同款形态：宿主是 torch 的 pinned 张量，设备视图与它同一个虚拟地址
    assert buf.cpu.is_pinned()
    assert buf.uva.is_cuda
    assert buf.uva.data_ptr() == buf.cpu.data_ptr()

    # GPU 内核写 → CPU 读
    buf.uva.fill_(7)
    torch.cuda.synchronize()
    assert buf.np.tolist() == [7] * 8
    # CPU 写 → 内核读（下一次内核运行会读到新值）
    buf.np[3] = 21
    torch.cuda.synchronize()
    assert buf.uva[3].item() == 21


@requires_cuda
def test_python_backend_also_aliases_memory(monkeypatch):
    """python 兜底后端给出同样的别名语义（换实现不能换语义）。"""
    monkeypatch.setenv("MINIVLLM_UVA_BACKEND", "python")
    assert uva_ext.resolve_backend() == "python"
    buf = _uva_buffer()
    assert buf.uva.data_ptr() == buf.cpu.data_ptr()
    buf.uva.fill_(5)
    torch.cuda.synchronize()
    assert buf.np.tolist() == [5] * 8
    buf.np[1] = 9
    torch.cuda.synchronize()
    assert buf.uva[1].item() == 9


def test_unknown_backend_fails_loudly(monkeypatch):
    """写错后端名 → 当场报错（不静默挑一个）。"""
    monkeypatch.setenv("MINIVLLM_UVA_BACKEND", "nope")
    with pytest.raises(ValueError, match="MINIVLLM_UVA_BACKEND 只能是"):
        uva_ext.resolve_backend()


@requires_cuda
def test_staged_write_lands_in_host_memory(monkeypatch):
    """`StagedWriteTensor(uva_instead_of_gpu=True)` 的落盘真的写进宿主内存（UVA 的用处）。

    这条是 73 关把 `all_token_ids` 放 UVA 的**根本理由**：采样内核按 slot 追加 token 之后，
    CPU 侧（下一轮的 prefill 取历史 / 惩罚计数）要能立刻读到，而这份表可能几 GB、不能进显存。
    """
    monkeypatch.delenv("MINIVLLM_UVA_BACKEND", raising=False)
    from minivllm.worker.gpu.buffer_utils import StagedWriteTensor

    tensor = StagedWriteTensor((2, 8), dtype=torch.int32, device=torch.device("cuda"),
                               uva_instead_of_gpu=True)
    tensor.stage_write(1, 0, [11, 12, 13])
    tensor.apply_write()
    torch.cuda.synchronize()
    assert tensor._uva_buf.np[1, :3].tolist() == [11, 12, 13]
