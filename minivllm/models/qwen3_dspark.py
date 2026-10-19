"""DSpark 草稿模型（需求 077）：`Qwen3DSparkModel` / `Qwen3DSparkForCausalLM`。

对应上游 `vllm/model_executor/models/qwen3_dspark.py`。DSpark 的**并联主干与 DFlash 完全一样**
（上下文 KV 预计算 + 非因果 query 前向，见 `qwen3_dflash.py`），它只多两个头：

    markov_head      低秩转移偏置 `markov_w1`（**target** 词表 × r）→ `markov_w2`（**draft** 词表 × r）。
                     提议者从左到右串行采样时，把"上一个已选 token"的 r 维嵌入投成词表大小的
                     偏置加到当前 logits 上 —— 这就是"块内依赖"。
    confidence_head  （可选）每个位置的"被接受概率"估计：`[hidden, markov_embed] → 1`。

**为什么偏置要有"两个词表宽度"**：`markov_w1` 的输入是**已经采出来的草稿 token**（target 空间
的 id，缩减词表时由提议者 scatter 后再采），而偏置要加在**draft 空间的 base logits** 上
（`compute_draft_logits()` 不做 d2t 散射），所以 `markov_w2` 的输出宽 = draft 词表。
两个宽度只有在全词表草稿时才相等。

**本模型不负责**"从左到右怎么采"（那是提议者 `DSparkSpeculator._sample_sequential()` 的事）；
它只提供 `markov_embed / markov_bias / apply_markov_bias_gathered / compute_confidence`
四个可被图捕获的张量算子。缩减词表 → target 的搬运也在提议者里做（`_sample_logits`），
因为"实际采样用的 q"要写进提议者持有的 `draft_logits`。

**本仓库只实现 Qwen3 基座**（上游另有 Gemma4 / K3-MLA / DeepSeek-V4 三族，配置期明确拒绝）。
"""

from typing import Iterable

import torch
from torch import nn

from ..layers import ParallelLMHead
from ..model_loader.auto_weights_loader import AutoWeightsLoader
from .qwen3 import maybe_prefix
from .qwen3_dflash import DFlashForCausalLM, DFlashModel


class DSparkMarkovHead(nn.Module):
    """串行转移偏置头（低秩 `V_tgt × r` / `r × V_draft`），上游同名类。

    `markov_w1[token]` 把**上一个已采出的 token**（target 词表）映成 r 维嵌入，`markov_w2`
    再把它投成 draft 词表大小的偏置，加到当前步的 base logits 上。

    两份权重都不切分（TP=1 下本来也不切）：这个头在**每一个草稿位置**上串行跑一次，
    切分就等于每步加一次 all-reduce 与一次全词表 gather。
    """

    def __init__(self, vocab_size: int, draft_vocab_size: int, markov_rank: int,
                 prefix: str = "") -> None:
        super().__init__()
        # `markov_w1` 用普通 `nn.Embedding`：它按 **target** id 取行，与 `embed_tokens`
        # （draft 词表）不是同一张表，也不参与并行草稿的输入嵌入。
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(draft_vocab_size, markov_rank, bias=False,
                                        prefix=maybe_prefix(prefix, "markov_w2"))

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """`[B]`（target id）→ `[B, r]`。"""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor) -> torch.Tensor:
        """`[B, r]` → `[B, V_draft]`：走词表 GEMM 的收尾（含 `logit_scale`，与 base logits 同口径）。"""
        return logits_processor(self.markov_w2, markov_embed)

    def apply_bias_gathered(self, markov_embed: torch.Tensor, logits: torch.Tensor,
                            values: torch.Tensor, index: torch.Tensor,
                            scale: float = 1.0) -> torch.Tensor:
        """只给 top-k 候选加偏置（需求 077 §3.4 的 `dspark_draft_topk`）。

        调用方已经把 `logits` 整份填成 `-inf`（一次、覆盖所有草稿位置），这里只把候选列的
        "原值 + scale·(w2·embed)" 散射回去，于是普通采样器看到的**就是**截断后的提议分布
        —— q 不需要另一份实现。
        """
        weight = self.markov_w2.weight[index]        # [B, k, r]
        corrected = values.unsqueeze(-1)             # [B, k, 1]
        corrected.baddbmm_(weight, markov_embed.unsqueeze(-1), beta=1.0, alpha=scale)
        return logits.scatter_(1, index, corrected.squeeze(-1))


class DSparkConfidenceHead(nn.Module):
    """接受置信度头（上游同名类）：`[hidden, markov_embed] → 1`，外面套 sigmoid。

    输出是"这个位置的草稿会被 target 接受"的概率估计，供 **79 关**的自适应验证做预算；
    本关只负责**算出来并写对槽位**，不设计阈值。
    """

    def __init__(self, input_dim: int, prefix: str = "", bias: bool = False,
                 with_markov: bool = True) -> None:
        super().__init__()
        self.with_markov = with_markov
        # 上游用 `ReplicatedLinear(..., params_dtype=torch.float32)`；本仓库 TP=1、没有
        # `ReplicatedLinear`，`nn.Linear` 就是它的等价物。**只有参数 dtype 需要补**：
        # 加载器会把整个模型 `.to(bf16)`，而前向里输入是 `.float()` 的 —— 见
        # `Qwen3DSparkForCausalLM.load_weights()` 里把本头升回 fp32 的那一步。
        self.proj = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1) if self.with_markov else hidden
        # 权重 dtype 为准（fp32 时与上游逐位一致；万一没升成 fp32 也不会因 dtype 不匹配崩）
        return self.proj(x.to(self.proj.weight.dtype)).squeeze(-1)


class Qwen3DSparkModel(DFlashModel):
    """DFlash Qwen3 主干 + DSpark 的两个头（上游同名类的本仓库子集）。"""

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__(config, prefix)
        if config.get("markov_rank") is None:
            raise ValueError(
                "DSpark 的配置里没有 `markov_rank`：Markov 头是低秩的 `V × r / r × V`，"
                "缺了 r 就建不出这层权重（这份 checkpoint 不是 DSpark 的）")
        draft_vocab_size = int(config.get("draft_vocab_size") or config["vocab_size"])
        self.markov_head = DSparkMarkovHead(
            int(config["vocab_size"]), draft_vocab_size, int(config["markov_rank"]),
            prefix=maybe_prefix(prefix, "markov_head"))
        self.confidence_head: DSparkConfidenceHead | None = None
        if config.get("enable_confidence_head", False):
            with_markov = bool(config.get("confidence_head_with_markov", False))
            input_dim = int(config["hidden_size"])
            if with_markov:
                input_dim += int(config["markov_rank"])
            self.confidence_head = DSparkConfidenceHead(
                input_dim, prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True, with_markov=with_markov)


class Qwen3DSparkForCausalLM(DFlashForCausalLM):
    """`Qwen3DSparkForCausalLM`：DSpark 草稿的对外接口（上游同名类）。"""

    _model_cls = Qwen3DSparkModel

    # -------- Markov / confidence 的算子（提议者调用；全部可进图） --------

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """**draft 词表宽度**的 logits，不做 d2t 散射。

        为什么不能直接用 `compute_logits()`：Markov 偏置是在 draft 空间加的（`markov_w2` 的
        宽度），散射要到**采完**之后再做——那一步由提议者的 `_sample_logits()` 负责，因为它
        同时要把"实际用的 q"写进 `draft_logits`（需求 077 §3.3）。
        """
        return self.logits_processor(self.lm_head, hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """draft 空间的 id → target 空间的 id（全词表草稿时是恒等映射）。"""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def apply_markov_bias_gathered(self, markov_embed: torch.Tensor, logits: torch.Tensor,
                                   values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed, logits, values, index, self.logits_processor.scale)

    def compute_confidence(self, head_hidden: torch.Tensor,
                           markov_embed: torch.Tensor) -> torch.Tensor:
        """每个草稿位置的接受概率估计（`[num_sample]`，sigmoid 之后）。"""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    # -------- 权重加载 --------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """按上游 `Qwen3DSparkForCausalLM.load_weights` 的规则重排名字。

            `t2d`            跳过（训练用的反向映射，推理只用 d2t）
            `d2t`            → `draft_id_to_target_id`
            其余非 lm_head   → 加 `model.` 前缀（`markov_head.*` / `confidence_head.*` 也在这里）
            `mask_hidden`    拒绝（那是 P-EAGLE 的并行槽位常量向量，DSpark 用 mask token 的嵌入）

        与上游的两处差别（都写进 `docs/step77_alignment.md` §5）：

            上游用 `process_eagle_weight()` 记 `has_own_embed_tokens/has_own_lm_head`，
            本仓库没有那条通道 → 这里直接记成两个属性（`load_dspark_model()` 读它们决定共享）。
            上游的 `ReplicatedLinear(params_dtype=fp32)` 让置信度头天生是 fp32；本仓库先被
            加载器 `.to(bf16)`，所以**加载前**把那个头升回 fp32，权重拷进 fp32 参数不丢精度。
        """
        if self.model.confidence_head is not None:
            self.model.confidence_head.to(torch.float32)

        model_weights: dict[str, torch.Tensor] = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        includes_confidence_head = False
        for name, loaded_weight in weights:
            if "mask_hidden" in name:
                raise ValueError(
                    "DSpark 权重里不该有 `mask_hidden`（那是 P-EAGLE 的并行槽位常量向量）："
                    f"收到 {name!r}。这份 checkpoint 不是 DSpark 的格式")
            # `t2d` 是训练时的反向映射；推理只用 d2t（草稿 id → target id）。
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            if "confidence_head" in name:
                includes_confidence_head = True
            model_weights[name] = loaded_weight

        # 两处"检查点里没有这一份"的判定要留给 `load_dspark_model()`：缺 embed_tokens /
        # lm_head 时与 target 共享，所以这里不能因为它们未加载而报覆盖错误。
        # 内层 `DFlashModel.load_weights()` 也要知道"这一份没有"——否则它的覆盖检查会把
        # `model.embed_tokens.weight` 算成漏加载（外层 skip_substrs 管不到内层）。
        self.model.tie_embeddings = not includes_embed_tokens
        skip_substrs: list[str] = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if self.model.confidence_head is None:
            skip_substrs.append("confidence_head")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        # 与 DFlash 同一条理由：加载器按"第一段名字"分组且只合并相邻同名，排序让同前缀连续。
        loaded = loader.load_weights(sorted(model_weights.items()))

        # `load_dspark_model()` 的共享判据（上游 `_should_share` 的 flag 参数）。
        self.has_own_embed_tokens = includes_embed_tokens
        self.has_own_lm_head = includes_lm_head
        # 融合的上下文 KV 权重只能引用各层参数，必须在权重到位之后建。
        self.model._build_fused_kv_buffers()
        return loaded


# 注册发生在 `models/registry.py`（与 `Qwen3ForCausalLM` / `DFlashForCausalLM` 同一处）：
# 上游注册名是 checkpoint `architectures` 里的字符串 `Qwen3DSparkModel`，类名是
# `Qwen3DSparkForCausalLM`；本仓库照抄这个"注册名 ≠ 类名"的对应关系。
