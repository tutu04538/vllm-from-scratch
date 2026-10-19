"""DFlash 的 V1 提议者（对应 vLLM `v1/spec_decode/dflash.py::DFlashProposer`）。

它继承 `SpecDecodeBaseProposer`（**不**继承 `DraftModelProposer`，与上游一致）——因为 DFlash
不吃 token 序列、也不跑自回归：第一遍是"上下文 KV 预计算 + 一次 query forward"，采样只有
一轮（`parallel_drafting=True` 时 `propose()` 走早退分支，本仓库由配置强制打开）。

与 EAGLE/PARD 的三处结构差别（这就是本文件存在的理由）：

    set_inputs_first_pass   只把 **1 + K 行 query** 写进输入工作区；上下文行（target 本轮的
                            行）单独记下它们的**位置与槽位**，不进模型
    _forward                先 `precompute_and_store_context_kv()`（用 target hidden 翻译成
                            K/V，直接散射进草稿缓存），再把 query 行交给模型
    采样                    只采 mask 行（anchor 行只贡献 K/V）；`sample_positions` 是它们
                            自己的 query 位置（mask 预测"自己这个位置"，不是下一个位置）

**注意力因果性**：每层的 `causal` 来自 `dflash_config.causal` / `is_causal`，默认非因果
（"标准" DFlash checkpoint 就是这样）。本仓库只支持全因果或全非因果两种（混合滑窗/全注意力
在配置期就拒绝），所以这里只需要一个布尔值。

**边界**（与 `docs/step76_alignment.md` §5 一致）：`sample_from_anchor=True`、混合滑窗、
独立 `mask_embedding.pt` 都不支持——前两者在配置期拒绝，后者在模型加载时拒绝。
"""

import torch

from .draft_model import SpecDecodeBaseProposer
from .utils import (FirstPassPlan, TargetRows, DFlashFirstPass, expand_dflash_inputs,
                    compute_slot_mapping_for_rows)


class DFlashProposer(SpecDecodeBaseProposer):
    # DFlash 的模型 `forward(input_ids, positions)` **不吃** hidden states（上下文 K/V 由
    # 预计算写入缓存），所以这里保持 False——但特征并没有浪费：`set_inputs_first_pass()`
    # 会把它投成上下文 KV（上游 `pass_hidden_states_to_model=True` 只影响基类的行数预算，
    # 而 `method == "dflash"` 在预算公式里本来就被减 0）。
    pass_hidden_states_to_model = False

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        super().__init__(spec_config, vllm_config, device)
        if spec_config.draft_model_config is None:
            raise ValueError("DFlash 投机必须给 draft_model_config（草稿是独立权重）")
        self.draft_model_config = spec_config.draft_model_config
        # 上游 `DFlashProposer.__init__`：query = 每请求 `1 + K` 行（bonus + K 个 mask）。
        # K 可以逐轮变小（71 关），所以真正用的是 `set_inputs_first_pass()` 里按本轮 K 算的值；
        # 这里只记容量上界。
        self.max_query_tokens = self.max_num_reqs * (1 + self.num_speculative_tokens)
        # 上下文行：位置/槽位两份独立缓冲（上游 `_context_positions_buffer` /
        # `_context_slot_mapping_buffer`）。它们**不进输入工作区**：`_upload()` 只推 query 行，
        # 上下文那份由 `_precompute_context_kv()` 单独推（图外的一次拷贝）。
        self.context_positions_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64)
        self.context_slot_mapping_cpu = torch.zeros(self.max_num_tokens, dtype=torch.int64)
        if device == "cpu":
            self.context_positions = self.context_positions_cpu
            self.context_slot_mapping = self.context_slot_mapping_cpu
        else:
            self.context_positions = torch.zeros(self.max_num_tokens, dtype=torch.int64,
                                                 device=device)
            self.context_slot_mapping = torch.zeros(self.max_num_tokens, dtype=torch.int64,
                                                    device=device)
        # 本轮 target 的 hidden（已过 `combine_hidden_states()`）与上下文行数
        self._dflash_context_states: torch.Tensor | None = None
        self._dflash_num_context = 0
        # 每层因果性在 `load_model()` 里读模型（配置期已经保证全层一致）
        self.attention_causal = True
        self.dflash_causal = True

    # -------- 加载 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        causal_flags = self.model.get_draft_attn_causal()
        if len(set(causal_flags)) != 1:
            raise NotImplementedError(
                f"DFlash 的逐层因果性不一致（{causal_flags}）：本仓库只支持全部因果或全部"
                f"非因果（混合滑窗/全注意力需要多 KV group，在配置期就拒绝）")
        self.dflash_causal = bool(causal_flags[0])
        self.attention_causal = self.dflash_causal
        return self.model

    def model_returns_tuple(self) -> bool:
        # 上游 `model_returns_tuple()` 的规则：`method not in ("mtp", "draft_model", "dflash")`
        # → DFlash 只返回一个 hidden。
        return False

    # -------- 第一遍输入 --------

    def set_inputs_first_pass(self, rows: list[TargetRows], all_token_ids,
                              target_hidden_states=None,
                              target_token_ids: torch.Tensor | None = None,
                              target_positions: torch.Tensor | None = None) -> FirstPassPlan:
        """对应上游 `set_inputs_first_pass()` 的 DFlash 分支（L139-226）。

        上游在这里启动 `copy_and_expand_dflash_inputs_kernel` 并顺手构造新的
        `CommonAttentionMetadata`；本仓库的 V1 第一遍仍走 CPU staging（58 关的布局），所以
        对应物是 `expand_dflash_inputs()`（内核的逐行等价）+ `compute_slot_mapping_for_rows()`
        （内核里那段槽位查表）。规则、顺序、输出含义逐条相同。
        """
        if target_token_ids is None or target_positions is None:
            raise ValueError(
                "DFlash 的第一遍需要本轮 target 的原始输入（target_token_ids / target_positions）："
                "上游内核就是在这两份缓冲上展开的，本仓库不自己重建行")
        if target_hidden_states is None:
            raise ValueError(
                "DFlash 需要 target 的辅助层 hidden states（上下文 KV 由它们翻译出来）；"
                "runner 没给特征说明 `capture_aux_hidden_states` 没打开")
        num_query = 1 + self.num_speculative_tokens
        num_tokens_in = sum(int(target.target_rows) for target in rows)
        tokens = [int(token) for token in target_token_ids[:num_tokens_in]]
        positions_in = [int(position) for position in target_positions[:num_tokens_in]]
        if len(tokens) != num_tokens_in or len(positions_in) != num_tokens_in:
            raise RuntimeError(
                f"本轮 target 的输入行数 {len(tokens)} 与调度快照的行数之和 {num_tokens_in} "
                f"不一致：控制面/执行面口径不一致（不是模型问题）")

        expanded: DFlashFirstPass = expand_dflash_inputs(
            positions_in, rows,
            num_query_per_req=num_query,
            mask_token_id=self.parallel_drafting_token_id)
        num_tokens = len(expanded.input_ids)
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"DFlash 第一遍要 {num_tokens} 行 query，超过输入工作区 "
                f"{self.max_num_tokens} 行：Scheduler 的 input_budget 没按 net=K 兜住")

        # ---- query 行：token / 位置 / 槽位（上游内核的 out_query_* 三个输出）----
        num_reqs = len(rows)
        block_table = self.block_table_cpu[:num_reqs]
        query_positions = torch.tensor(expanded.positions, dtype=torch.int64)
        query_req_index = torch.arange(num_reqs).repeat_interleave(num_query)
        query_slots = compute_slot_mapping_for_rows(
            query_positions, query_req_index, block_table, self.block_size,
            self.max_model_len)
        self.input_ids_cpu[:num_tokens] = torch.tensor(expanded.input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = query_positions
        self.slot_mapping_cpu[:num_tokens] = query_slots
        # DFlash 没有"被拒尾部"这个概念：被拒的行留在 context 里（不写 query 行）；
        # mask 行是 query 行，但**不是** padding，所以 is_rejected 全 0。
        self.is_rejected_token_mask_cpu[:num_tokens] = False
        is_masked = [False] * num_tokens
        for request_index in range(num_reqs):
            base = request_index * num_query
            for off in range(base + 1, base + num_query):
                is_masked[off] = True
        self.is_masked_token_mask_cpu[:num_tokens] = torch.tensor(is_masked, dtype=torch.bool)
        self.query_start_loc_cpu[:num_reqs + 1] = torch.tensor(
            expanded.query_start_loc, dtype=torch.int64)
        self.seq_lens_cpu[:num_reqs] = torch.tensor(expanded.seq_lens, dtype=torch.int64)

        # ---- context 行：位置 / 槽位（上游 out_context_* 两个输出）----
        self._dflash_num_context = expanded.num_context
        if expanded.num_context > self.max_num_tokens:
            raise RuntimeError(
                f"本轮 context 行数 {expanded.num_context} 超过工作区 {self.max_num_tokens}")
        context_positions = torch.tensor(expanded.context_positions, dtype=torch.int64)
        context_req_index = torch.arange(num_reqs).repeat_interleave(
            torch.tensor([int(target.target_rows) for target in rows], dtype=torch.int64))
        context_slots = compute_slot_mapping_for_rows(
            context_positions, context_req_index, block_table, self.block_size,
            self.max_model_len)
        self.context_positions_cpu[:expanded.num_context] = context_positions
        self.context_slot_mapping_cpu[:expanded.num_context] = context_slots

        # ---- 上下文特征：target 的多辅助层拼接 → fc 投到草稿宽度（上游在 propose() 里做）----
        stacked = torch.cat([target_hidden_states[target.req_id] for target in rows], dim=0)
        self._dflash_context_states = self.model.combine_hidden_states(
            stacked.to(self.device))

        self._check_valid_positions(rows)
        ready = [index for index, target in enumerate(rows) if target.ready]
        sample_rows: list[int] = []
        sample_req_ids: list[str] = []
        sample_positions: list[int] = []
        for request_index in ready:
            target = rows[request_index]
            base = request_index * num_query
            for off in range(1, num_query):
                sample_rows.append(base + off)
                sample_req_ids.append(target.req_id)
                sample_positions.append(expanded.positions[base + off])
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=num_reqs,
            sample_rows=sample_rows, sample_req_ids=sample_req_ids,
            sample_positions=sample_positions,
            seq_lens=list(expanded.seq_lens),
            history_end={target.req_id: target.history_end for target in rows})

    # -------- 前向 --------

    def _precompute_context_kv(self) -> None:
        """把 target hidden 翻译成草稿各层的 K/V 并写进缓存（上游 `build_model_inputs_first_pass`）。

        这件事**不能进 CUDA Graph**（上游注释：context 形状每步都变，所以放在 query 图之外）。
        写的是草稿自己的缓存，槽位是本轮 target 行自己的物理槽位——被拒尾部也在里面，
        因为 `seq_lens` 已经把它们的长度减掉了（上游 V1 内核同样是"按 is_ctx 全写"）。
        """
        if self._dflash_context_states is None:
            raise RuntimeError("DFlash 的上下文特征还没准备好（set_inputs_first_pass 没跑过）")
        num_context = self._dflash_num_context
        if self.device != "cpu":
            self.context_positions[:num_context].copy_(
                self.context_positions_cpu[:num_context])
            self.context_slot_mapping[:num_context].copy_(
                self.context_slot_mapping_cpu[:num_context])
        self.model.precompute_and_store_context_kv(
            self._dflash_context_states[:num_context],
            self.context_positions[:num_context],
            self.context_slot_mapping[:num_context])

    def _forward(self, num_tokens: int, num_reqs: int, hidden_states=None):
        self._precompute_context_kv()
        return super()._forward(num_tokens, num_reqs, hidden_states)
