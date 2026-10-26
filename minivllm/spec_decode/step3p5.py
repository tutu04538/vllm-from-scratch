"""80F：Step3p5 多 group MTP 提议者（V1）。

对应冻结 `vllm/v1/spec_decode/step3p5.py::Step3p5MTPProposer(EagleProposer)`。

与 Qwen3 MTP 的三处真实差异：

1. **draft 层可以跨 KV group**：每个 group 有自己的 block table / slot mapping /
   metadata；不能拿主 group 的 metadata 广播给所有层。
2. **每个草稿步换 MTP 层/head**：`spec_step_idx % num_mtp_layers` 同时选
   `model.layers[45+i]` 和它自己的 `shared_head`。
3. **每层都可以有自己的 head 数/head_dim/滑窗**：group 映射按 Attention 的实际
   `(head_size, num_kv_heads, sliding_window)` 匹配 target 的 group spec。
"""

from __future__ import annotations

import torch

from ..attention import Attention
from ..spec_decode.eagle import EagleProposer


class Step3p5MTPProposer(EagleProposer):
    """V1 Step3p5 proposer：per-group metadata + 分步 head。"""

    def model_returns_tuple(self) -> bool:
        return False

    # -------- 加载 / KV 分组 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        # target 与 draft 的模块路径会重名（`model.layers.0...`）；metadata 的键必须唯一。
        self.kv_caches = {}
        self._draft_layer_groups = {}
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            module.layer_name = "draft::" + name
            cache = torch.zeros(
                2, self.vllm_config.cache_config.num_gpu_blocks, self.block_size,
                module.num_kv_heads, module.head_size, dtype=dtype, device=self.device)
            module.kv_cache = cache
            self.kv_caches[module.layer_name] = cache
        return self.model

    def set_kv_config(self, kv_cache_config) -> None:
        """把 draft Attention 按形状映射到 target 的 KV group，并开每组的块表工作区。"""
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = kv_cache_config.num_kv_cache_groups
        self.max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        self.max_blocks_per_req = (
            self.max_model_len + self.block_size - 1) // self.block_size
        self._group_block_tables_cpu = {}
        self._group_block_tables = {}
        self._group_slot_mappings_cpu = {}
        self._group_slot_mappings = {}
        for gid in range(self.num_kv_cache_groups):
            self._group_block_tables_cpu[gid] = torch.zeros(
                (self.max_num_reqs, self.max_blocks_per_req), dtype=torch.int64)
            self._group_block_tables[gid] = torch.zeros(
                (self.max_num_reqs, self.max_blocks_per_req), dtype=torch.int64,
                device=self.device)
            self._group_slot_mappings_cpu[gid] = torch.zeros(
                self.max_num_tokens, dtype=torch.int64)
            self._group_slot_mappings[gid] = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=self.device)

        # 每个 draft layer 按真实 Attention 形状匹配 target group。
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            key = (module.head_size, module.num_kv_heads,
                   module.per_layer_sliding_window)
            matches = [
                gid for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                if (group.kv_cache_spec.head_size == key[0]
                    and group.kv_cache_spec.num_kv_heads == key[1]
                    and getattr(group.kv_cache_spec, "sliding_window", None) == key[2])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Step3p5 draft 层 {name!r} 的 KV 形状 {key} 在 target 的 "
                    f"{len(kv_cache_config.kv_cache_groups)} 个 group 中匹配到 "
                    f"{matches}：必须恰好匹配一个 group（否则 metadata 会用错块表）")
            module.kv_cache_group_id = matches[0]
            self._draft_layer_groups[module.layer_name] = matches[0]

    # 兼容 Runner 的 Gemma4 分支接口；Step3p5 不需要绑定 target cache。
    def bind_kv_caches(self, target_layers, kv_caches) -> None:
        if not self.kv_caches:
            raise RuntimeError("Step3p5 的 draft cache 还没分配")
        # 只做一次形状/组检查，不复制 target 的 K/V。
        for name, cache in self.kv_caches.items():
            if cache.shape[0] != 2:
                raise ValueError(f"draft cache {name!r} 形状非法：{tuple(cache.shape)}")

    # -------- 每步输入 --------

    def set_inputs_first_pass(self, rows, all_token_ids, target_hidden_states=None,
                              target_token_ids=None, target_positions=None):
        self._current_rows = list(rows)
        self._spec_step_idx = 0
        plan = super().set_inputs_first_pass(
            rows, all_token_ids, target_hidden_states,
            target_token_ids, target_positions)
        self._prepare_group_block_tables(rows)
        self._prepare_group_slot_mappings(rows, plan.num_tokens)
        return plan

    def _set_autoregressive_inputs(self, pending, drafts, input_batch) -> None:
        super()._set_autoregressive_inputs(pending, drafts, input_batch)
        self._spec_step_idx = max((len(drafts[target.req_id]) for target, _ in pending),
                                  default=0)
        rows = [target for target, _ in pending]
        self._prepare_group_block_tables(rows)
        self._prepare_group_slot_mappings(rows, len(pending))

    def _prepare_group_block_tables(self, rows) -> None:
        """把本轮每个 target 行在每组里的块表拷进工作区。"""
        batch = getattr(self, "_current_input_batch", None)
        if batch is None:
            raise RuntimeError("Step3p5 proposer 缺少 input_batch，无法读 per-group block table")
        for gid in range(self.num_kv_cache_groups):
            table = batch.block_table.group_tables[gid]
            for index, target in enumerate(rows):
                count = table.num_blocks(target.row)
                self._group_block_tables_cpu[gid][index, :count].copy_(
                    table.cpu[target.row, :count].to(torch.int64))
                self._group_block_tables_cpu[gid][index, count:].zero_()

    def _prepare_group_slot_mappings(self, rows, num_tokens: int) -> None:
        """每组按自己的 block table 重算 slot；不能用主 group 的 slot 广播。"""
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(f"Step3p5 slot 行数 {num_tokens} 超过工作区")
        counts = ([1] * len(rows) if num_tokens == len(rows)
                  else [int(target.target_rows) for target in rows])
        if sum(counts) != num_tokens:
            raise RuntimeError(
                f"Step3p5 slot 行数 {num_tokens} 与请求行数 {sum(counts)} 不一致")
        cursor = 0
        for row_index, target in enumerate(rows):
            for offset in range(counts[row_index]):
                token_index = cursor + offset
                position = int(self.positions_cpu[token_index])
                rejected = bool(self.is_rejected_token_mask_cpu[token_index])
                for gid in range(self.num_kv_cache_groups):
                    if rejected or position < 0:
                        slot = -1
                    else:
                        block_idx = position // self.block_size
                        if block_idx >= self.max_blocks_per_req:
                            slot = -1
                        else:
                            block_id = int(
                                self._group_block_tables_cpu[gid][row_index, block_idx])
                            slot = block_id * self.block_size + position % self.block_size
                    self._group_slot_mappings_cpu[gid][token_index] = slot
            cursor += counts[row_index]

    # -------- 前向 --------

    def _upload(self, num_tokens: int, num_reqs: int) -> None:
        super()._upload(num_tokens, num_reqs)
        if self.device == "cpu":
            return
        for gid in range(self.num_kv_cache_groups):
            self._group_block_tables[gid][:num_reqs].copy_(
                self._group_block_tables_cpu[gid][:num_reqs].to(self.device))
            self._group_slot_mappings[gid][:num_tokens].copy_(
                self._group_slot_mappings_cpu[gid][:num_tokens].to(self.device))

    def _forward(self, num_tokens: int, num_reqs: int, hidden_states=None):
        self._upload(num_tokens, num_reqs)
        metadata = {}
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            md = self.metadata_builder.build(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                block_table=self._group_block_tables[gid][:num_reqs],
                slot_mapping=self._group_slot_mappings[gid][:num_tokens],
                num_reqs=num_reqs,
                sliding_window=getattr(group.kv_cache_spec, "sliding_window", None),
            )
            for name, layer_gid in self._draft_layer_groups.items():
                if layer_gid == gid:
                    metadata[name] = md
        if len(metadata) != len(self._draft_layer_groups):
            raise RuntimeError("Step3p5 有 draft Attention 层没有拿到 metadata")
        from ..forward_context import set_forward_context
        with set_forward_context(metadata, num_tokens=num_tokens):
            features = self.hidden_states if hidden_states is None else hidden_states
            return self.model(
                self.input_ids[:num_tokens], self.positions[:num_tokens],
                features[:num_tokens], spec_step_idx=self._spec_step_idx)

    def _compute_draft_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden, spec_step_idx=self._spec_step_idx)

    def propose(self, rows, all_token_ids, input_batch, **kwargs):
        self._current_input_batch = input_batch
        return super().propose(rows, all_token_ids, input_batch, **kwargs)


__all__ = ["Step3p5MTPProposer"]
