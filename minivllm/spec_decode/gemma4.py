"""V1 Gemma4 assistant 提议者（80E.E2；对应冻结 `v1/spec_decode/gemma4.py`）。

V1 和 V2 的 Gemma4 语义相同（Q-only、常量位置、跨宽度投影），但 V1 的提议循环是按
CPU 行块拼输入的，所以这里覆盖三处：

    load_model / setup_shared   加载 assistant、共享 target backbone embedding、建源层映射
    bind_kv_caches              只绑 target cache，不调用 `_allocate_kv_caches()`
    _forward                    按 KV group 建 metadata（sliding/full 各一张块表）
    _set_autoregressive_inputs  位置/seq_len **不前进**（Q-only 读 target 固定上下文）

**不把 V1 接到 V2 的 speculator 上**：需求明确要求两条执行路径独立；共享的只有模型层与
Q-only 映射这些纯逻辑。
"""

from __future__ import annotations

from typing import Any

import torch

from ..attention.layer import Attention
from ..forward_context import set_forward_context
from ..worker.gpu.spec_decode.gemma4.kv_sharing import setup_gemma4_kv_sharing
from .eagle import EagleProposer


class Gemma4Proposer(EagleProposer):
    def model_returns_tuple(self) -> bool:
        return True

    # -------- 加载 / 接线 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self.model = get_model(self.draft_model_config, self.device)
        self.kv_caches = {}
        # 草稿层与 target 的模块路径会重名（`model.layers.0...`），metadata 的键必须唯一。
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                module.layer_name = "draft::" + name
        return self.model

    def _validate_configs(self) -> None:
        """Gemma4 的 hidden/KV 头数本来就可以与 target 不同，跳过基类的同形检查。

        仍需检查词表与 dtype：assistant 的 lm_head 与 target 共享 token 空间；dtype 不同
        会让 Q-only 读 target cache 时被 `bind_shared_kv_cache` 拒绝（这里提前给清晰错误）。
        """
        target = self.vllm_config.model_config
        draft = self.draft_model_config
        if target.get_vocab_size() != draft.get_vocab_size():
            raise ValueError(
                f"Gemma4 assistant 的词表 {draft.get_vocab_size()} 与 target "
                f"{target.get_vocab_size()} 不一致：草稿 token 在 target 空间没有意义")
        if str(target.dtype) != str(draft.dtype):
            raise ValueError(
                f"Gemma4 assistant 与 target 的 dtype 必须一致"
                f"（{draft.dtype} vs {target.dtype}）：Q-only 直接读 target cache")

    def propose(self, rows, all_token_ids, input_batch, **kwargs):
        self._current_input_batch = input_batch
        return super().propose(rows, all_token_ids, input_batch, **kwargs)

    def setup_shared(self, target_model) -> None:
        """在 target 模型已加载、KV 尚未分配时建立嵌入与 Q-only 源层映射。"""
        from ..models.gemma4_mtp import _text_config

        target_names = {
            name for name, module in target_model.named_modules()
            if isinstance(module, Attention)
        }
        target_text = _text_config(self.vllm_config.model_config.hf_config)
        draft_text = _text_config(self.draft_model_config.hf_config)
        setup_gemma4_kv_sharing(
            self.model, target_names, draft_text, target_text)

        target_embed = getattr(getattr(target_model, "model", None),
                               "embed_tokens", None)
        if target_embed is None:
            raise AttributeError(
                "Gemma4 target 没有 `model.embed_tokens`：无法共享 backbone 词嵌入")
        backbone = int(getattr(self.model.model, "backbone_hidden_size", 0))
        if int(target_embed.embedding_dim) != backbone:
            raise ValueError(
                f"target embedding 宽度 {target_embed.embedding_dim} 与 assistant "
                f"backbone_hidden_size={backbone} 不一致")
        del self.model.model.embed_tokens
        self.model.model.embed_tokens = target_embed
        self._setup_centroid_graphs()

    def _setup_centroid_graphs(self) -> None:
        """V1 的 centroid greedy 小图（对应上游 `_setup_centroids_cuda_graphs` 的 V1 版）。

        只对 `use_ordered_embeddings=True` 生效；`get_top_tokens` 不物化完整词表。
        图按 hidden 行数建 1/2/4/.../64 档，超过 64 行退回直接调用（不重捕、不错桶）。
        """
        self._centroid_graphs: dict[int, Any] = {}
        self._centroid_inputs: dict[int, torch.Tensor] = {}
        self._centroid_outputs: dict[int, torch.Tensor] = {}
        masked = getattr(self.model, "masked_embedding", None)
        if masked is None or self.device == "cpu" or not torch.cuda.is_available():
            return
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        for size in (1, 2, 4, 8, 16, 32, 64):
            static_input = torch.zeros(
                size, masked.hidden_size, dtype=dtype, device=self.device)
            for _ in range(3):
                self.model.get_top_tokens(static_input)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self.model.get_top_tokens(static_input)
            self._centroid_graphs[size] = graph
            self._centroid_inputs[size] = static_input
            self._centroid_outputs[size] = static_output

    def _centroid_greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        for size in sorted(self._centroid_graphs):
            if size >= num_tokens:
                self._centroid_inputs[size][:num_tokens].copy_(hidden_states)
                self._centroid_graphs[size].replay()
                return self._centroid_outputs[size][:num_tokens].clone()
        return self.model.get_top_tokens(hidden_states)

    def bind_kv_caches(self, target_layers: dict[str, Attention],
                       kv_caches: dict[str, torch.Tensor]) -> None:
        """把 assistant 的 Q-only 层绑到 target 源层；不分配第二份 cache。"""
        self.kv_caches = {}
        self._draft_layer_groups: dict[str, int] = {}
        dtype = next(p.dtype for p in self.model.parameters() if p.is_floating_point())
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            source = module.kv_sharing_target_layer_name
            if source is None or source not in kv_caches:
                raise KeyError(
                    f"Gemma4 assistant 层 {name!r} 的源层 {source!r} 没有 target cache")
            source_layer = target_layers[source]
            module.bind_shared_kv_cache(
                source_layer, kv_caches[source], draft_dtype=dtype)
            module.kv_cache_group_id = source_layer.kv_cache_group_id
            self.kv_caches[module.layer_name] = kv_caches[source]
            self._draft_layer_groups[module.layer_name] = (
                source_layer.kv_cache_group_id)

    # -------- 第一遍 / 自回归步 --------

    def set_inputs_first_pass(self, rows, all_token_ids, target_hidden_states=None,
                              target_token_ids=None, target_positions=None):
        self._current_rows = list(rows)
        plan = super().set_inputs_first_pass(
            rows, all_token_ids, target_hidden_states,
            target_token_ids, target_positions)
        self._current_rows = list(rows)
        return plan

    def _prepare_group_workspace(self, rows) -> None:
        """把本轮这些 target 行的块表按请求顺序拷进每组的工作区。"""
        if not hasattr(self, "_group_workspaces"):
            self._group_workspaces = {}
        for gid in range(self.num_kv_cache_groups):
            workspace = self._group_workspaces.get(gid)
            if workspace is None:
                workspace = torch.zeros(
                    (self.max_num_reqs, self.max_blocks_per_req),
                    dtype=torch.int64, device=self.device)
                self._group_workspaces[gid] = workspace
            for index, target in enumerate(rows):
                table = self._current_input_batch.block_table.group_tables[gid]
                count = table.num_blocks(target.row)
                workspace[index, :count].copy_(table.cpu[target.row, :count].to(
                    self.device))
                workspace[index, count:].zero_()

    def _forward(self, num_tokens: int, num_reqs: int, hidden_states=None):
        self._upload(num_tokens, num_reqs)
        rows = list(getattr(self, "_current_rows", []))
        if len(rows) != num_reqs:
            raise RuntimeError(
                f"当前行块 {len(rows)} 与 _forward 的请求数 {num_reqs} 不一致")
        self._prepare_group_workspace(rows)
        metadata = {}
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            md = self.metadata_builder.build(
                query_start_loc=self.query_start_loc[:num_reqs + 1],
                seq_lens=self.seq_lens[:num_reqs],
                block_table=self._group_workspaces[gid][:num_reqs],
                slot_mapping=self.slot_mapping[:num_tokens],
                num_reqs=num_reqs,
                sliding_window=getattr(spec, "sliding_window", None),
            )
            for name, layer_gid in self._draft_layer_groups.items():
                if layer_gid == gid:
                    metadata[name] = md
        with set_forward_context(metadata, num_tokens=num_tokens):
            features = self.hidden_states if hidden_states is None else hidden_states
            out = self.model(
                self.input_ids[:num_tokens], self.positions[:num_tokens],
                features[:num_tokens])
        if self.model_returns_tuple():
            for_logits, for_next = out
            return for_logits, for_next
        return out

    def set_kv_config(self, kv_cache_config) -> None:
        self.kv_cache_config = kv_cache_config
        self.num_kv_cache_groups = kv_cache_config.num_kv_cache_groups
        self.max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs
        self.max_blocks_per_req = (
            self.max_model_len + self.block_size - 1) // self.block_size
        self._group_workspaces = {}

    def _sample_draft_tokens(self, hidden, row_refs, input_batch, drafts, probs) -> None:
        """greedy 且 centroid 开启时用 `get_top_tokens`（V1 的稀疏采样分支）。

        只要有任意一行是概率草稿，就整批回退父类的完整 logits 路径：概率草稿需要
        q 的完整列，不能把 sparse top-token 当分布。greedy 的 q 是点质量
        （`draft_probs=None`），所以这里不塞 one-hot 假分布。
        """
        masked = getattr(self.model, "masked_embedding", None)
        all_greedy = all(
            input_batch.sampling_params[input_batch.req_id_to_index[req_id]].is_greedy
            for req_id, _ in row_refs)
        if masked is None or not all_greedy:
            return super()._sample_draft_tokens(
                hidden, row_refs, input_batch, drafts, probs)
        rows = torch.tensor([row for _, row in row_refs],
                            dtype=torch.int64, device=hidden.device)
        tokens = self._centroid_greedy(hidden[rows])
        for index, (req_id, _) in enumerate(row_refs):
            drafts[req_id].append(int(tokens[index]))
            probs[req_id].append(None)

    def _set_autoregressive_inputs(self, pending, drafts, input_batch) -> None:
        """Q-only 的 AR 步：位置/上下文**固定**在采样锚点，不按第 k 枚推进。"""
        num_reqs = len(pending)
        tokens = [drafts[target.req_id][-1] for target, _ in pending]
        positions = [self._ar_anchor_position[target.req_id]
                     for target, _ in pending]
        rows = [target for target, _ in pending]
        self._current_rows = rows
        self.input_ids_cpu[:num_reqs] = torch.tensor(tokens, dtype=torch.int64)
        self.positions_cpu[:num_reqs] = torch.tensor(positions, dtype=torch.int64)
        rows_hidden = [self._ar_hidden[target.req_id] for target, _ in pending]
        self.hidden_states_cpu[:num_reqs] = torch.stack(rows_hidden).to(
            self.hidden_states_cpu.dtype)
        self.slot_mapping_cpu[:num_reqs] = 0
        self.is_rejected_token_mask_cpu[:num_reqs] = False
        self.query_start_loc_cpu[:num_reqs + 1] = torch.arange(
            num_reqs + 1, dtype=torch.int64)
        self.seq_lens_cpu[:num_reqs] = torch.tensor(
            [self._ar_base_seq_len[target.req_id] for target, _ in pending],
            dtype=torch.int64)


__all__ = ["Gemma4Proposer"]
