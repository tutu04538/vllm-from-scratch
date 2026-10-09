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
    """EAGLE/EAGLE3/**MTP**：draft 额外吃 target 本轮算出来的 hidden states。

    MTP 走的就是这个类（上游 `config.use_eagle()` 把 `mtp` 也算进来，Runner 同样建
    `EagleProposer`）：差别只有三处，都由配置/模型自己回答，提议循环一行都不用改——
    **第一遍的 hidden 是哪一份**（EAGLE3：多个辅助层拼起来再 `combine_hidden_states`；
    MTP：target 的**最后一层** hidden，原样）、**模型返回一个还是两个 hidden**（见
    `model_returns_tuple()`）、以及 **draft 配置从哪来**（EAGLE3：用户给的 draft 目录；
    MTP：从 target 配置派生，权重就在 target 的 checkpoint 里）。
    """

    pass_hidden_states_to_model = True

    def model_returns_tuple(self) -> bool:
        """上游 `llm_base_proposer.py:1015-1023` 的规则：

            if method == "mtp": 只有 DeepSeekMTPModel / KimiK3MTPModel 返回两个
            return method not in ("mtp", "draft_model", "dflash")

        本仓库的 MTP 是 Qwen3 家族（上游 `Qwen3NextMTP.forward` **只返回一个** hidden，
        与它的 `compute_logits` 直接把 lm_head 套在那个 hidden 上一致），所以 mtp → False；
        EAGLE/EAGLE3 的 draft 返回 `(hidden, hidden_prenorm)` → True。
        """
        return self.method != "mtp"

    # -------- 模型加载 --------

    def load_model(self) -> None:
        """按 draft 配置的 `architectures` 建 draft 模型（EAGLE3 或 MTP 同一个入口）。

        EAGLE3 与 MTP 的区别只有"配置从哪来"：
          - EAGLE3：用户给的 draft 目录（用**模型注册表**按 `architectures[0]` 取类）；
            检查点里**没有** `embed_tokens` 时与 target 共享（不是"shape 相同就共享"，
            而是"检查点里缺这一份"，对应上游 `_maybe_share_embeddings`）；
          - MTP：`_build_proposer()` 里用 `SpeculativeConfig.derive_mtp_draft_config()` 从
            **target 配置**派生（`architectures=["Qwen3MTPModel"]`、模型目录 = target 目录），
            于是加载器读的是 target 那份 checkpoint，而 MTP 的 `load_weights()` 只挑 spec 层。
        """
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        # 72 关：并行提议要从权重里的 `mask_hidden` 取出 mask 槽位的常量特征
        self._maybe_fill_parallel_drafting_hidden_state()
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

        **特征的形态由方法决定**（上游 `llm_base_proposer.py:532-549`）：

            if self.method in ("eagle3", "dflash"):
                target_hidden_states = self.model.combine_hidden_states(target_hidden_states)

        也就是"只有 EAGLE3 那一族才在提议者里做多层融合投影"；**MTP 与 EAGLE-1 收的是已经
        可用的单份 hidden**（MTP 是 target 最后一层，宽度就是 draft 的 hidden_size），原样写进
        缓冲即可——MTP 自己的 `fc`/`eh_proj` 才是做拼接投影的地方（在模型里，不在提议者里）。
        """
        if self.parallel_drafting:
            # 72 关（P-EAGLE）：左移 + 复用 target 块最后一行放锚点，再补 K−1 个 mask，
            # 一次 forward 出 K 枚（与串行 no-extra-slots 通路的区别就在这块 mask 区）
            return self._parallel_first_pass(rows, target_hidden_states, target_token_ids,
                                             target_positions)
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
        # 2) 把这条请求新采出的 token 打到**最后一枚有效 token 所在的那一格**
        #    （上游 padded 通路的 `token_indices_to_sample`，`prepare_inputs_padded` 的产物：
        #     块起点 + target_rows − 1 − num_rejected；69 关已与上游内核逐值对过）
        #
        # ⚠️ 这里**不能**写成"块的最后一行"（`target_rows − 1`）：被拒的那几行在真历史里不存在，
        # 把锚点（以及第 1 枚草稿的采样行）放在最后一行会让它的**位置与上下文都偏出 rejected 格**，
        # 上下文里装的还是被拒草稿的 KV；症状是 `position != seq_len − 1`（自回归行掉到自己的
        # 上下文之外），不报错、只是第 1 枚草稿质量崩（实测真实 EAGLE3：位置1 接受率 22.5% vs
        # 上游 41.7%）。2026-10-08 复核发现并修正，见 docs/step63_alignment.md §8。
        token_indices_to_sample: list[int] = []
        cursor = 0
        for target in rows:
            token_indices_to_sample.append(
                cursor + target.target_rows - 1 - target.num_rejected)
            cursor += target.target_rows
        for index, target in zip(token_indices_to_sample, rows):
            input_ids[index] = target.next_token_id

        # 3) 特征：**逐行原样**（EAGLE3 先做多层融合投影；MTP 收的就是最终 hidden）
        hidden_rows: list[torch.Tensor] = []
        for target, index in zip(rows, token_indices_to_sample):
            rows_hidden = target_hidden_states[target.req_id]
            if self.method == "eagle3":
                rows_hidden = self.model.model.combine_hidden_states(rows_hidden)
            hidden_rows.extend(rows_hidden[i] for i in range(target.target_rows))
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
            # **采样行自己的 position**（= target 行块的最后一行，位置原样未移）：自回归步从
            # `它 + 1` 开始（上游 `positions = self.positions[token_indices_to_sample]`，之后
            # 每步 +1）。EAGLE 的采样行与 draft 的尾部扩容行**不是同一行**，所以这个起点必须
            # 由布局自己给出，不能在提议循环里反推。
            sample_positions=[positions[token_indices_to_sample[index]] for index in ready],
            seq_lens=list(seq_lens),
            history_end={target.req_id: target.history_end for target in rows})
