"""块表镜像（对应 vLLM `v1/worker/block_table.py` 的单 group 子集）。

**它是一份镜像，不是真相**：逻辑块号 → 物理块号的权威映射在 Scheduler 那边
（`KVCacheManager`），执行侧只是把协议送来的块号记在定长缓冲里，再整块拷到 GPU 给 attention
kernel 用。三种角色不能混（198 §2）：

    控制端权威状态（Scheduler/KVCacheManager）  →  执行端镜像（本类）  →  GPU 输入副本

为什么要一份定长 CPU 缓冲而不是每轮现搭张量：**块表是每轮都要上传的东西**，定长 + 原位写入
+ 一次 H2D 拷贝，比"每轮创建新张量再上传"少一次分配、也少一次同步。缓冲按行复用：

    [max_num_reqs, max_num_blocks_per_req]   行 = batch 行，列 = 该请求的第几个逻辑块

**有效列数单独存**（`num_blocks_per_row`），不能靠"块号 0 表示空"来判断：**物理块 0 是合法
块**，用它当哨兵会让"第一个块恰好是 0 号"的请求少数一块（vLLM 同样单独维护这张表）。
"""

import torch


class BlockTable:
    """一个 KV group 的块表镜像。行内容由协议决定，这里只负责存、拷、算。"""

    def __init__(self, block_size: int, max_num_reqs: int, max_num_blocks_per_req: int,
                 device: str = "cpu") -> None:
        if max_num_blocks_per_req <= 0:
            raise ValueError("max_num_blocks_per_req 必须为正")
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        # CPU 侧是权威副本（协议写这里），GPU 侧只在 commit 之后有效
        self.cpu = torch.zeros((max_num_reqs, max_num_blocks_per_req), dtype=torch.int64)
        self.num_blocks_per_row = torch.zeros(max_num_reqs, dtype=torch.int64)
        self.gpu = torch.zeros((max_num_reqs, max_num_blocks_per_req), dtype=torch.int64,
                               device=device)

    def num_blocks(self, row_index: int) -> int:
        return int(self.num_blocks_per_row[row_index])

    # -------- 写入（都在 CPU 副本上做）--------

    def add_row(self, row_index: int, block_ids) -> None:
        """给一行写入**初始**块表（新请求 / 镜像重建）。其余列补 0。

        补 0 而不是留着上一条请求的块号：attention 只读有效列，但"读到残留"一旦发生就是静默的
        错（读到别人的 KV），所以缓冲复用时要显式清零。
        """
        if len(block_ids) > self.max_num_blocks_per_req:
            raise ValueError(
                f"块表放不下：第 {row_index} 行有 {len(block_ids)} 个块，"
                f"上限 {self.max_num_blocks_per_req}（= ceil(max_model_len / block_size)）")
        self.cpu[row_index].zero_()
        self.cpu[row_index, :len(block_ids)] = torch.tensor(list(block_ids), dtype=torch.int64)
        self.num_blocks_per_row[row_index] = len(block_ids)

    def set_row(self, row_index: int, block_ids) -> None:
        """**整表替换**（resumed 请求：旧物理编号必须一个都不留）。"""
        self.add_row(row_index, block_ids)

    def append_to_row(self, row_index: int, block_ids) -> None:
        """续跑：把**新增**块接到已有块后面（普通路径，不替换）。"""
        existing = self.cpu[row_index, :self.num_blocks(row_index)].tolist()
        self.add_row(row_index, existing + list(block_ids))

    def clear_row(self, row_index: int) -> None:
        self.cpu[row_index].zero_()
        self.num_blocks_per_row[row_index] = 0

    def move_row(self, src: int, dst: int) -> None:
        """batch 行重排（`condense`）时把一行搬到另一行。"""
        self.cpu[dst] = self.cpu[src]
        self.num_blocks_per_row[dst] = self.num_blocks_per_row[src]
        self.clear_row(src)

    # -------- 上传 --------

    def commit_block_table(self, num_reqs: int) -> None:
        """把前 `num_reqs` 行拷到 GPU。**整个批一次拷**，不是每请求一次。"""
        self.gpu[:num_reqs].copy_(self.cpu[:num_reqs])

    # -------- 派生：slot mapping --------

    def compute_slot_mapping(self, num_reqs: int, positions: torch.Tensor,
                             req_indices: torch.Tensor) -> torch.Tensor:
        """`slot = block_table[req, pos // block_size] * block_size + pos % block_size`。

        这是"分页 KV 写到哪"的**唯一**答案（198 §4 的数字就是按它算的）。用 Torch 算术而不是
        kernel：这一层要能逐值对照，先不引入 Triton。

        越界检查不是可选的：`pos // block_size` 超出这一行已有的块数，说明 Scheduler 少分配了块
        （控制面 bug）或者 positions 算错了。此时按 0 号块去写会把别人的 KV 覆盖掉，必须报错。
        """
        block_index = (positions // self.block_size).to(torch.int64)
        block_offset = (positions % self.block_size).to(torch.int64)
        table = self.cpu[:num_reqs]
        used_blocks = self.num_blocks_per_row[:num_reqs]
        overrun = block_index >= used_blocks[req_indices]
        if bool(overrun.any()):
            bad = int(overrun.nonzero()[0])
            row = int(req_indices[bad])
            raise IndexError(
                f"第 {row} 行的第 {int(block_index[bad])} 个块不在块表里"
                f"（该行只有 {int(used_blocks[row])} 个块）：位置 {int(positions[bad])} "
                f"没有对应的物理槽位。块分配属于控制面，这里出现说明协议给少了块或 positions 算错了")
        physical = table[req_indices, block_index]
        return physical * self.block_size + block_offset
