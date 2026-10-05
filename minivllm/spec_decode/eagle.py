"""EAGLE / EAGLE3 提议者（需求 063 §2/§3）：`EagleProposer(SpecDecodeBaseProposer)`。

上游 `v1/spec_decode/eagle.py:10-22` 本体只有十几行——差别只在构造时
`pass_hidden_states_to_model=True`，**不是复制一套 draft 循环**：`set_inputs_first_pass()` /
`build_model_inputs_first_pass()` / `_sample_draft_tokens()` / `propose()` 都在基类里共用。
本文件照这个结构做：基类（`draft_model.py::SpecDecodeBaseProposer`）负责工作区、KV、AR 循环，
这里只覆盖**第一遍输入怎么摆**（需求 §3 的行号规则）与**模型怎么加载**。

与普通 draft 的两处输入差异（阶段 A 已逐值对齐上游内核）：

1. **token 逐请求错开一格**：整体左移 + 每条请求的最后一格换成它自己的新 token
   （`query_start_loc[1:] - 1`）。最后一格留着谁由这个下标决定，算错一位就会把下一条请求的
   token 留给自己（`minivllm/testing/eagle_inputs_ref.py::eagle_first_pass_input_ids` 里有反证（参考实现，生产路径不调））。
2. **hidden states 不跟着移**：第 i 行的特征还是 target 第 i 行的特征，扩容行（最后一格）用
   这条请求**采样行**的特征——于是配对是 `(h_i, t_{i+1}) → t_{i+2}`。
"""

import torch

from ..outputs import DraftTokenIds
from .draft_model import DraftModelProposer
from .utils import FirstPassPlan, TargetRows


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

    def set_inputs_first_pass(self, rows: list[TargetRows], all_token_ids,
                              target_hidden_states=None,
                              target_token_ids=None,
                              target_positions=None) -> FirstPassPlan:
        """**逐行照抄上游** `set_inputs_first_pass()` 的不扩容分支（`llm_base_proposer.py:846-872`）：

            num_tokens = target_token_ids.shape[0]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]      # 整体左移一格
            self.input_ids[token_indices_to_sample] = next_token_ids     # 每请求最后一格换成新 token
            self._set_positions(num_tokens, target_positions)            # positions **原样**
            self.hidden_states[:num_tokens] = target_hidden_states       # 特征 **原样逐行拷贝**

        关键就是"不重排行"：行数 = 本轮 target 的行数（含被拒草稿那几行），positions 与特征
        都按下标原样对齐，扩容行的特征天然就是它自己那一行的特征——所以上游这里**没有"选哪一行特征"
        这种逻辑**，也就没有对应的代码可抄歪。

        我们之前按 `expand_eagle_inputs_shifted`（内核的**扩容分支**，额外槽位那套）重建了行，
        才引入"位置用紧凑布局、特征取另一行"的混用错误；现在按本函数改回上游口径：
        输入 token/positions 由 Runner 直接给本轮的原始缓冲（含被拒草稿），我们只做左移与打补丁。
        """
        if target_hidden_states is None:
            raise ValueError(
                "EAGLE 提议者需要本轮 target 的 hidden states（target_hidden_states）："
                "它吃的是 (token, 特征) 对，缺特征就退化成普通 draft 了，不能静默降级")
        if target_token_ids is None or target_positions is None:
            raise ValueError(
                "EAGLE 第一遍需要本轮 target 的原始输入（target_token_ids / target_positions）："
                "上游就是在这两份缓冲上做『整体左移 + 打补丁』，本仓库不自己重建行")

        num_tokens = sum(target.target_rows for target in rows)
        tokens = [int(t) for t in target_token_ids[:num_tokens]]
        positions = [int(p) for p in target_positions[:num_tokens]]
        if len(tokens) != num_tokens or len(positions) != num_tokens:
            raise RuntimeError(
                f"本轮 target 的输入行数 {len(tokens)} 与调度快照的 target_rows 之和 "
                f"{num_tokens} 不一致：控制面/执行面口径不一致（不是模型问题）")

        # 1) 整体左移一格（最后一格暂时是脏值，下一步会被覆盖）
        input_ids = list(tokens)
        if num_tokens > 1:
            input_ids[:num_tokens - 1] = tokens[1:]
        # 2) 每条请求的**最后一格**换成这条请求新采出的 token
        token_indices_to_sample: list[int] = []
        cursor = 0
        for target in rows:
            token_indices_to_sample.append(cursor + target.target_rows - 1)
            cursor += target.target_rows
        for index, target in zip(token_indices_to_sample, rows):
            input_ids[index] = target.next_token_id

        # 3) 特征：**逐行原样**（多辅助层先各自投影，再按行拼回来）
        hidden_rows: list[torch.Tensor] = []
        for target, index in zip(rows, token_indices_to_sample):
            hidden = self.model.model.combine_hidden_states(target_hidden_states[target.req_id])
            hidden_rows.extend(hidden[i] for i in range(target.target_rows))
        if len(hidden_rows) != num_tokens:
            raise RuntimeError(
                f"特征行数 {len(hidden_rows)} 与输入行数 {num_tokens} 不一致："
                f"hidden states 必须按本轮 target 的每一行给全（含被拒行）")

        query_start_loc = [0]
        for target in rows:
            query_start_loc.append(query_start_loc[-1] + target.target_rows)
        seq_lens = [target.start + target.target_rows for target in rows]

        self.input_ids_cpu[:num_tokens] = torch.tensor(input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = torch.tensor(positions, dtype=torch.int64)
        self.hidden_states_cpu[:num_tokens] = torch.stack(hidden_rows).to(
            self.hidden_states_cpu.dtype)
        # 上游这条通路**不做被拒掩码**（被拒行的 token 是真的被拒草稿，KV 下一轮由 Scheduler 丢），
        # 所以这里全 False、槽位按各行自己的位置算（`num_new_tokens=0`：不额外加行）
        self.is_rejected_token_mask_cpu[:num_tokens] = False
        self.query_start_loc_cpu[:len(query_start_loc)] = torch.tensor(query_start_loc,
                                                                      dtype=torch.int64)
        self.seq_lens_cpu[:len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int64)
        from .utils import compute_new_slot_mapping

        self.slot_mapping_cpu[:num_tokens] = compute_new_slot_mapping(
            self.block_table_cpu[:len(rows)], [target.target_rows for target in rows],
            self.positions_cpu[:num_tokens], self.is_rejected_token_mask_cpu[:num_tokens],
            self.block_size, 0, self.max_model_len)
        ready = [index for index, target in enumerate(rows) if target.ready]
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=len(rows),
            sample_rows=[token_indices_to_sample[index] for index in ready],
            sample_req_ids=[rows[index].req_id for index in ready],
            history_end={target.req_id: target.history_end for target in rows})
