"""DSpark 的 V2 提议者（对应 vLLM `v1/worker/gpu/spec_decode/dspark/`，需求 077）。

DSpark = **DFlash 的并行主干** + **串行 Markov 纠正**：

    一次 query 前向（沿用 DFlash：上下文 KV 预计算 + 非因果注意力）→ `[num_reqs, K]` 个 base logits
        ↓  从左到右 K 步（每步只是一次 `[B,r] × [r,V]` 的低秩偏置）
    第 i 步的 logits = base_logits[:, i] + markov_bias(markov_embed(上一步采出的 token))
        ↓
    采样 + **把这一步实际用的 logits 写进 `draft_logits[req, i]`**（= 验证器的 q）

**为什么 q 必须是"加过偏置的那一份"**：拒绝采样要求 `q` 是**实际提议分布**。只在采样时加偏置、
验证时用 base logits，接受概率就会算错——分布不再等价于 target 的分布，而且**不会报错**。
所以 `_sample_logits()` 把采样与缓存写成同一份（`gumbel_sample(..., logits_cache=...)`）。

**两种 query 布局**（需求 077 §3.1，由 `hf_config.sample_from_anchor` 决定）：

    True（默认，anchor-as-first）  K 个 query：anchor（= 真实 bonus token）+ K−1 个 noise；
                                  每个位置都预测**下一个** token（`sample_pos = query_pos + 1`）
    False（speculators 格式）      1 + K 个 query，退回 DFlash 的 fill-in 布局：
                                  anchor 是 bonus，只有 mask 行采样、且预测**自己**这个位置

布局差异全部落在共享内核的 `SAMPLE_FROM_ANCHOR` 上，提议者只负责选值。

**缩减词表**：`compute_draft_logits()` 给的是 draft 宽度，Markov 偏置在 draft 空间加；
采样前 scatter 到 target 宽度的 `_draft_scatter_buf`（**只在 load 时全填一次 `-inf`**，
之后每步覆盖 draft→target 那几列），于是 ① 采样用的 id 是 target id（`markov_w1` 按 target id
取行）② 缓存下来的 q 天生是 target 宽度的完整分布 ③ 不存在的列永远是 `-inf`，不可能继承
上一轮的值（需求 077 §3.3）。

**top-k**（`dspark_draft_topk`，只在 Qwen3 DSpark + 全候选约束下放开，见 `config._resolve_dspark`）：
候选只取一次（对 base logits），串行步只对候选算偏置、其余列保持 `-inf`；普通采样器吃这份
截断分布，所以缓存的 q 与采样的 q 仍然是同一份。

**图**：`_generate_draft()` 把主干前向与 K 步串行采样一起录进 FULL 图（需求 077 §3.6）；
K 步之间是**数据依赖**（第 i 步要等第 i−1 步采样结果），所以绝不能改成"各位置独立 argmax"。
"""

from typing import Any

import torch

from .....config import CUDAGraphMode, VllmConfig
from ..dflash.speculator import DFlashSpeculator
from .utils import load_dspark_model


class DSparkSpeculator(DFlashSpeculator):
    """DSpark 的 V2 提议者（上游 `DSparkSpeculator(DFlashSpeculator)`）。"""

    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        draft_hf = self.speculative_config.draft_model_config.hf_config or {}
        # 是否从 anchor 位置开始采样。上游读的是**顶层**字段（speculators 格式的 DSpark 把它
        # 写在顶层；DFlash 的 `dflash_config.sample_from_anchor` 是另一个键，默认 False）。
        # True = anchor-as-first（K 个 query，每行预测下一个 token）；False = 1+K fill-in。
        self.sample_from_anchor = bool(draft_hf.get("sample_from_anchor", True))
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        # 每步的列号（写 `draft_logits[req, step, :]`）；`_anchor_idx` 是每请求 query 块第 0 行
        # 在 `input_buffers.input_ids` 里的下标——**捕获期也要用**，所以是常驻张量而不是现算。
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device)
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )

        # 缩减词表（概率草稿）才会填这两样，见 `load_draft_model()`。
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        # `dspark_draft_topk` 由配置期校验并写进 draft 配置（需求 077 §3.4）。
        self._draft_topk: int | None = draft_hf.get("dspark_draft_topk")

        # 每个草稿位置的接受概率估计（`[max_num_reqs, K]`，fp32）。79 关的自适应验证读它。
        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32)
        # 77 关只在配置层拒绝 True（阈值/预算属 79 关）；这里照上游读值，供单测直接打开
        # 置信度计算路径（`enable_adaptive_verification` 只控制"算不算"，不控制"怎么用"）。
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification)

    # -------- 加载 --------

    def load_draft_model(self, target_model: torch.nn.Module) -> torch.nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        # 缩减词表：概率拒绝采样按 **target id** 索引 draft logits，所以先把 draft→target 的
        # 列映射与一块 target 宽度的散射缓冲备好。
        if self.draft_logits is not None and model.draft_id_to_target_id is not None:
            d2t = model.draft_id_to_target_id
            self._d2t_scatter_index = torch.arange(d2t.shape[0], device=d2t.device) + d2t
            # `-inf` 只填这一次：之后每步的 scatter 覆盖 draft→target 的那几列。**不存在的列
            # 永远是 -inf**，所以"上一轮的旧值"不可能漏进这一轮的 q（需求 077 §3.3）。
            # 与 `draft_logits` 分开分配，避免别名（它要按请求行整体覆盖）。
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size), float("-inf"),
                dtype=self.draft_logits.dtype, device=self.device)
        if self.enable_adaptive_verification and model.model.confidence_head is None:
            raise ValueError(
                "自适应验证需要带 confidence 头的 DSpark checkpoint，而这份没有。"
                "把 `enable_adaptive_verification=false`（或干脆不写）——固定枚数验证不需要它")
        return model

    # -------- 采样（串行 Markov） --------

    def _sample_logits(self, logits: torch.Tensor, idx_map: torch.Tensor,
                       sample_pos: torch.Tensor, step: int) -> torch.Tensor:
        """采一步草稿，并把**这一步实际用的 logits** 缓存成 q（需求 077 §2/§3.2）。

        greedy（`draft_logits is None`）：q 是点质量，draft id 要映回 target 空间。
        probabilistic：q 必须与 target 同宽（验证器按 target 词表步长索引），所以缩减词表
        先 scatter 到 target 宽度；`sample_pos` 是"被预测 token 的位置 Q"，验证侧按**前一个**
        位置取 Gumbel 噪声，所以传 Q−1。
        """
        if self.draft_logits is None:
            return self.model.map_draft_to_target(logits.argmax(dim=-1))

        if self._d2t_scatter_index is not None:
            assert self._draft_scatter_buf is not None
            buf = self._draft_scatter_buf[: logits.shape[0]]
            buf.index_copy_(1, self._d2t_scatter_index, logits.to(buf.dtype))
            logits = buf

        # 延迟 import：CPU 上不装 Triton 也能 import 本模块（与 `sample_draft()` 同款）。
        from ...sample.gumbel import gumbel_sample

        return gumbel_sample(
            logits,
            idx_map,
            self.temperature,
            self.seeds,
            sample_pos - 1,
            apply_temperature=True,
            logits_cache=self.draft_logits,
            logits_cache_col=self._step_cols[step],
            use_fp64=self.use_fp64_gumbel,
        )

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # 每个 (请求, 位置) 的主干输出，按 (req, step) 排布。
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # **draft 词表**的 base logits：Markov 偏置在 draft 空间加，采完再映回 target。
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # 第 0 步的"前一个 token" = anchor 行喂进去的那个 id（真实 bonus token；fill-in 布局
        # 下同样是 bonus）。用预计算的常驻下标读，捕获期也成立。
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            # 下一步的偏置来自**这一步实际选中**的 token —— 这就是"块内依赖"。
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)

    def _sample_sequential_topk(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """只给 base logits 的 top-k 候选加 Markov 偏置（`dspark_draft_topk`）。

        候选**只选一次**（覆盖所有草稿位置）；每步把候选列"原值 + 偏置"散射回那份被整份
        填成 `-inf` 的缓冲，其余列一直是 `-inf`。普通稠密采样与拒绝采样路径原样消费这份
        截断分布——所以缓存的 q 就是截断后的 q，不会出现"采样用截断、验证用全量"。
        """
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        # 复用主干输出当采样器的输入：整份填 -inf 一次，串行循环里只 scatter 候选列。
        base_logits.fill_(float("-inf"))

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed, base_logits[:, i], base_values[:, i], draft_indices[:, i])
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i)
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)

    # -------- 草稿一步（图里录的就是它） --------

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """主干并行前向 → 串行 Markov 采样。两者都在图里（需求 077 §3.6）。"""
        head_hidden = self._run_model(
            num_tokens_padded, attn_metadata, slot_mappings, cudagraph_runtime_mode)
        self._sample_sequential(num_reqs, head_hidden)
