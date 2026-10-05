"""EAGLE / EAGLE3 提议者（需求 063 §2/§3）：`EagleProposer(SpecDecodeBaseProposer)`。

上游 `v1/spec_decode/eagle.py:10-22` 本体只有十几行——差别只在构造时
`pass_hidden_states_to_model=True`，**不是复制一套 draft 循环**：`set_inputs_first_pass()` /
`build_model_inputs_first_pass()` / `_sample_draft_tokens()` / `propose()` 都在基类里共用。
本文件照这个结构做：基类（`draft_model.py::SpecDecodeBaseProposer`）负责工作区、KV、AR 循环，
这里只覆盖**第一遍输入怎么摆**（需求 §3 的行号规则）与**模型怎么加载**。

与普通 draft 的两处输入差异（阶段 A 已逐值对齐上游内核）：

1. **token 逐请求错开一格**：整体左移 + 每条请求的最后一格换成它自己的新 token
   （`query_start_loc[1:] - 1`）。最后一格留着谁由这个下标决定，算错一位就会把下一条请求的
   token 留给自己（`utils.py::eagle_first_pass_input_ids` 里有反证）。
2. **hidden states 不跟着移**：第 i 行的特征还是 target 第 i 行的特征，扩容行（最后一格）用
   这条请求**采样行**的特征——于是配对是 `(h_i, t_{i+1}) → t_{i+2}`。
"""

import torch

from ..outputs import DraftTokenIds
from .draft_model import DraftModelProposer
from .utils import DraftInputRows, FirstPassPlan, expand_eagle_inputs_shifted


class EagleProposer(DraftModelProposer):
    """EAGLE/EAGLE3：draft 额外吃 target 本轮算出来的 hidden states。"""

    pass_hidden_states_to_model = True
    model_returns_tuple = True          # EAGLE3 的 forward 返回 (hidden, prenorm)

    # -------- 模型加载 --------

    def load_model(self) -> None:
        """按 draft 目录的 `architectures` 建 EAGLE3 模型，并把 target 的嵌入接过来。

        与普通 draft 的区别只有两点（需求 §3.5/§3.6）：
          - 用**模型注册表**按 `architectures[0]` 取类（`Eagle3Qwen3ForCausalLM` /
            `LlamaForCausalLMEEagle3` → 本仓库的 `Eagle3ForCausalLM`），不写死成 Qwen3；
          - 检查点里**没有** `embed_tokens` 时与 target 共享（不是"shape 相同就共享"，
            而是"检查点里缺这一份"，对应上游 `_maybe_share_embeddings`）。
        """
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        return self.model

    def share_embeddings(self, target_model) -> None:
        """把 target 的 `embed_tokens` 接到 draft 上（仅当检查点里缺这一份）。"""
        if hasattr(self.model, "share_embeddings"):
            self.model.share_embeddings(target_model)

    # -------- 第一遍输入（EAGLE 对齐） --------

    def set_inputs_first_pass(self, rows: list["TargetRows"], all_token_ids,
                              target_hidden_states=None) -> FirstPassPlan:  # noqa: F821
        """EAGLE 的第一遍：`[shift 后的有效行] + [扩容行] + [被拒占位行]`，positions/特征不移。

        与基类（普通 draft，`shift_input_ids=False`）的差别：
          - `valid_token_ids` 里那个"第一个 token"会被跳过（`expand_eagle_inputs_shifted` 干的）；
          - positions 与 target 逐行相同，扩容行的位置 = 该请求最后一行的位置；
          - 每行配的 hidden 来自**本轮 target**：前 `num_valid-1` 行用对应行的特征，
            扩容行用**采样行**（`start + target_rows - 1`）的特征。

        `target_hidden_states` 由 Runner 给：`{req_id: [target_rows, hidden]}`（本轮这一条请求
        被算过的那些行的特征）。没有它就直接报错——EAGLE 没有特征就跑不了，**不静默退回 token-only**。
        """
        if target_hidden_states is None:
            raise ValueError(
                "EAGLE 提议者需要本轮 target 的 hidden states（target_hidden_states）："
                "它吃的是 (token, 特征) 对，缺特征就退化成普通 draft 了，不能静默降级")
        input_rows = []
        hidden_rows: list[torch.Tensor] = []
        for target in rows:
            tokens = all_token_ids[target.req_id]
            valid = [int(tokens[position])
                     for position in range(target.start, target.start + target.num_valid)]
            input_rows.append(DraftInputRows(valid_token_ids=valid, start=target.start,
                                             next_token_id=target.next_token_id,
                                             num_rejected=target.num_rejected))
            # 多辅助层拼接 → fc 投影（模型自己的 `combine_hidden_states`，不许自己平均/拼接替代）
            hidden = self.model.model.combine_hidden_states(target_hidden_states[target.req_id])
            # 有效行的特征（注意：token 左移了，特征不动 → 第 i 行仍取第 i 行）
            hidden_rows.extend(hidden[index] for index in range(max(len(valid) - 1, 0)))
            # 扩容行 = 采样行的特征（上游 `out_hidden_state_mapping` 把最后一行映射到扩容行）
            hidden_rows.append(hidden[target.target_rows - 1])
            for _ in range(target.num_rejected):
                hidden_rows.append(torch.zeros_like(hidden[0]))

        input_ids, positions, is_rejected, sample_indices = expand_eagle_inputs_shifted(input_rows)
        num_tokens = len(input_ids)
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"EAGLE 第一遍要 {num_tokens} 行，超过输入工作区 {self.max_num_tokens} 行："
                f"input_budget 没兜住，属于控制面/执行面口径不一致（不是模型问题）")

        # 每请求的物理行数 = (num_valid - 1)（shift 掉第一个）+ 1（扩容行）+ num_rejected（占位）。
        # `compute_new_slot_mapping` 的公式是 `query_lens + num_new_tokens`，所以这里传
        # "有效行数 + 被拒行数 - 1" 再让 num_new_tokens=1 补上扩容行，总数正好等于 num_tokens
        # （EAGLE 不做 `extend_all_queries_by_N`：它的扩容行由 shift 省下来的那一格换的）。
        query_lens = [max(target.num_valid + target.num_rejected - 1, 0) for target in rows]
        query_start_loc = [0]
        for length in query_lens:
            query_start_loc.append(query_start_loc[-1] + length + 1)
        seq_lens = [target.start + target.num_valid for target in rows]

        self.input_ids_cpu[:num_tokens] = torch.tensor(input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = torch.tensor(positions, dtype=torch.int64)
        self.hidden_states_cpu[:num_tokens] = torch.stack(hidden_rows).to(
            self.hidden_states_cpu.dtype)
        self.is_rejected_token_mask_cpu[:num_tokens] = torch.tensor(is_rejected, dtype=torch.bool)
        self.query_start_loc_cpu[:len(query_start_loc)] = torch.tensor(query_start_loc,
                                                                      dtype=torch.int64)
        self.seq_lens_cpu[:len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int64)
        # 槽位：扩容行按它自己的位置算；被拒占位行打哨兵（不写 KV）
        from .utils import compute_new_slot_mapping

        self.slot_mapping_cpu[:num_tokens] = compute_new_slot_mapping(
            self.block_table_cpu[:len(rows)], query_lens, self.positions_cpu[:num_tokens],
            self.is_rejected_token_mask_cpu[:num_tokens], self.block_size,
            1, self.max_model_len)
        self._check_valid_positions(rows)
        ready = [index for index, target in enumerate(rows) if target.ready]
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=len(rows),
            sample_rows=[sample_indices[index] for index in ready],
            sample_req_ids=[rows[index].req_id for index in ready],
            history_end={target.req_id: target.history_end for target in rows})
