"""DFlash2 草稿模型（需求 078）：`DFlash2Qwen3ForCausalLM` / `DFlash2Qwen3Model`。

对应上游 `vllm/model_executor/models/qwen3_dflash2.py`（0.28.0 快照）。DFlash2 = DFlash 的
并行主干 + **候选关联选择**，与 DSpark（77 关）的差别是"动态打分"而不是"固定偏置"：

    DSpark    每个位置对**整张词表**加一个低秩转移偏置（`markov_w1/w2`），位置之间独立打分
    DFlash2   每个位置先取词表 top-k 候选（`compute_candidates`），再用**上下文 hidden**
              给"前一个候选 → 当前候选"的每条边打分（`candidate_selector`），然后沿候选图
              做条件 walk（提议者里的 `_selector_walk_kernel`）

三层新增结构（缺一层就不是同一个模型，都能跑、只是草稿静默变差）：

    DFlashGroupedConv    沿**序列维**的因果分组卷积；注意力和 MLP 各挂一个**独立实例**
                         （`attention_conv` / `mlp_conv`），每个实例内部又分 prepare
                         （子层输入侧）/ finish（子层输出侧）两侧。系数 = 静态
                         `base_kernel` + 由 hidden 现算的 `kernel_projection`（所以它不是
                         普通卷积：每 token 的卷积核不同）。`block_size = 1 + K` 让卷积在
                         **每请求的 query 块内**回看，不跨请求。
    CandidateSelector    两张码本 `predecessor_codebook` / `successor_codebook`（`[V, r]`）
                         + `hidden_projection`（hidden → r）。它给 `[B, K, k, k]` 的打分：
                         `score[b,l,p,c] = unary[b,l,c]
                          + Σ_r pred[cand(l-1,p)][r] · hidden[b,l][r] · succ[cand(l,c)][r]`
                         其中 `l=0` 的前驱是 anchor（target 刚采出的 bonus token）。
    `compute_candidates` 只取词表 top-k（`LogitsProcessor.get_top_k_tokens`），并按
                         checkpoint 的 `output_multiplier` / `final_logit_softcapping` 收尾。
                         提议者拿到的 `(candidate_ids, unary_logits)` 是**候选集**，
                         不再是整词表分布。

**本仓库只实现 Qwen3 基座**（上游另有 Laguna/MuseGlimmer 等 DFlash 变体，配置期/注册表都
不认）。**缩减词表（`d2t`）在 DFlash2 上明确拒绝**：`compute_candidates` 用的是 `lm_head`
的 draft 宽度 id，而候选分数写回 `draft_logits` 时要按 target 词表宽度索引；上游这条路径
没有做散射，本仓库不去"顺手补一个看起来能用"的映射（记进 `docs/step78_alignment.md` §5）。
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..layers import ColumnParallelLinear
from ..layers.logits_processor import LogitsProcessor
from .qwen3 import maybe_prefix
from .qwen3_dflash import DFlashDecoderLayer, DFlashForCausalLM, DFlashModel


def _grouped_conv(hidden_states: torch.Tensor, delta: torch.Tensor, base: torch.Tensor,
                  block_size: int, num_groups: int, group_size: int,
                  taps: int) -> torch.Tensor:
    """沿 token 维的因果分组卷积（上游同名函数，逐行等价）。

    每个 token `n` 的输出 = `Σ_{t=0}^{taps-1} coeff[n, t] * hidden[n − t]`，其中
    `coeff[n, t] = base[t] + delta[n, t]`（按 `num_groups × group_size` 分组，各组独立）。

    三个细节缺一不可：

        `block_size`   位置在**每个请求的 query 块内**取模（`n % block_size`）。query 行是
                       `[req0 的 1+K 行, req1 的 1+K 行, …]`，不取模的话请求 1 的第一行会
                       回看到请求 0 的最后几行——不报错，只是把别人的 token 卷进来。
        `position>=tap` 块内前 `tap` 行没有足够的历史，回看要清零（否则会卷到上一块的尾巴；
                       取模后的位置判断与 `F.pad` 的回看方向必须一起用）。
        `F.pad(..., (0,0,0,0,tap,0))` 把序列**往下移** `tap` 行（前面补零），所以
                       `shifted[n] = hidden[n − tap]`——与"因果卷积回看"同向。
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """DFlash2 的分组卷积（上游同名类）。

    `base_kernel` 是**静态**的 `[2, taps, hidden]`：这里的 2 是 **prepare（子层输入侧）/
    finish（子层输出侧）** 两份静态核，不是"注意力侧 / MLP 侧"——注意力和 MLP 是两个
    独立的 `DFlashGroupedConv` 实例（`attention_conv` / `mlp_conv`），各自都有这 2 侧。
    `kernel_projection` 由当前 hidden 现算 `[2, taps, num_groups]` 的**增量**系数；它与
    `base_kernel` 相加后得到这一行、这一 tap、这一组的卷积系数。于是卷积核同时依赖
    "子层/输入输出侧"与"这一行是什么 token"。

    上游用 `ReplicatedLinear`；本仓库 TP=1、没有这个类，`ColumnParallelLinear`（`return_bias=False`）
    在 TP=1 下就是同一个算子（形状、权重名、加载路径都一样）。
    """

    def __init__(self, hidden_size: int, taps: int, group_size: int, block_size: int,
                 prefix: str = "") -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} 必须整除 hidden_size={hidden_size}"
                "（否则 unflatten 分不出 (num_groups, group_size)）")
        if taps < 1:
            raise ValueError(f"conv_kernel_size（taps）必须 >= 1，收到 {taps}")
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=torch.float32), requires_grad=False)
        self.kernel_projection = ColumnParallelLinear(
            hidden_size, 2 * taps * self.num_groups, bias=False, return_bias=False,
            prefix=maybe_prefix(prefix, "kernel_projection"))

    def _convolve(self, hidden_states: torch.Tensor, delta: torch.Tensor,
                  side: int) -> torch.Tensor:
        return _grouped_conv(hidden_states, delta, self.base_kernel[side], self.block_size,
                             self.num_groups, self.group_size, self.taps)

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """注意力/MLP **之前**：用输入 hidden 一次算出两侧系数，返回 side 0 的卷积结果
        （side 1 的系数留给 `finish`）。"""
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups)
        return (self._convolve(hidden_states, coefficients[:, 0], 0),
                coefficients[:, 1])

    def finish(self, hidden_states: torch.Tensor,
               coefficients: torch.Tensor) -> torch.Tensor:
        """注意力/MLP **之后**：用 `prepare` 交回的 side 1 系数再卷一次。"""
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashDecoderLayer):
    """DFlash2 的解码器层（上游 `DFlash2Qwen3DecoderLayer`）。

    与 DFlash 层的唯一差别是注意力/MLP 两侧各多了一个分组卷积：

        hidden → input_layernorm → attention_conv.prepare → self_attn
               → attention_conv.finish → residual → post_attention_layernorm
               → mlp_conv.prepare → mlp → mlp_conv.finish → residual

    卷积的 `block_size = 1 + num_speculative_tokens`（每请求的 query 行数）——**必须**与
    提议者的 query 布局一致；不一致时卷积会跨请求或漏掉本块的行，草稿静默变差。
    """

    def __init__(self, config: dict, prefix: str = "", causal: bool = True) -> None:
        super().__init__(config, prefix=prefix, causal=causal)
        draft_config = config.get("dflash_config") or {}
        num_speculative_tokens = config.get("num_speculative_tokens")
        if num_speculative_tokens is None:
            raise ValueError(
                "DFlash2 的配置里没有 `num_speculative_tokens`：分组卷积的 `block_size` 是"
                "每请求的 query 行数 `1 + K`，缺 K 就建不出正确的卷积窗口。"
                "（这份值由 `SpeculativeConfig._resolve_dflash()` 写进 draft hf_config，"
                "模型构造只拿得到 hf_config；上游是从 `vllm_config.speculative_config` 读的）")
        conv_args = dict(
            hidden_size=config["hidden_size"],
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # 每请求的 query token 数 = bonus + K 个 mask（与 DFlash 的 `1 + K` 布局同口径）
            block_size=1 + int(num_speculative_tokens),
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv"))
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv"))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(predecessor_table: torch.Tensor, successor_table: torch.Tensor,
                 candidate_ids: torch.Tensor, unary_logits: torch.Tensor,
                 hidden: torch.Tensor, anchor_token_ids: torch.Tensor,
                 top_k: int) -> torch.Tensor:
    """候选边的三元打分（上游同名函数）。

    返回 `[B, K, k, k]`：`[b, l, p, c]` = "第 `l` 步、前驱候选是 `p`、当前候选是 `c`"
    的分数。三项分别是：

        unary        当前候选自己的 logit（与路径无关的部分）
        predecessor  前驱 token 的码本行（第 0 步的前驱是 anchor，不是候选）
        hidden       当前位置的 hidden 投影（**这一项才让打分随上下文变**）
        successor    当前候选的码本行

    与 DSpark 的对比：DSpark 的偏置 `W2·W1[prev]` 与位置无关；这里同一个 (prev, cand)
    在不同位置可以得不同分，而且只在 k 个候选之间打分。注意 `predecessor_ids` 是
    `[anchor, cand(l−1)]` 拼接后整体查表：`l=0` 的 k 条边共用 anchor，`l>=1` 的边来自
    上一位置的候选（上游同一个 `torch.cat`）。
    """
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


class CandidateSelector(nn.Module):
    """候选关联打分器（上游同名类）。

    两张码本按 **token id** 取行（`[V, rank]`），第三个因子是当前位置的 hidden 投影。
    打分是双线性的：`pred[p] · (hidden · succ[c])`——注意 `hidden` 被乘进 pred 那一侧，
    所以它同时调制"这条边在当前上下文里有多合理"。
    """

    def __init__(self, hidden_size: int, vocab_size: int, rank: int, top_k: int,
                 prefix: str = "") -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=torch.float32), requires_grad=False)
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=torch.float32), requires_grad=False)
        self.hidden_projection = ColumnParallelLinear(
            hidden_size, rank, bias=False, return_bias=False,
            prefix=maybe_prefix(prefix, "hidden_projection"))

    def forward(self, candidate_ids: torch.Tensor, unary_logits: torch.Tensor,
                hidden_states: torch.Tensor, anchor_token_ids: torch.Tensor
                ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(self.predecessor_codebook, self.successor_codebook,
                            candidate_ids, unary_logits, hidden, anchor_token_ids,
                            self.top_k)


class DFlash2Qwen3Model(DFlashModel):
    """DFlash 主干 + 候选选择器（上游同名类）。"""

    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__(config, prefix)
        draft_config = config.get("dflash_config") or {}
        self.input_embedding_scale = float(draft_config.get("input_embedding_scale", 1.0))
        for key in ("selector_rank", "selector_top_k"):
            if draft_config.get(key) is None:
                raise ValueError(
                    f"DFlash2 的 `dflash_config` 里没有 `{key}`：候选选择器的码本秩与 top-k "
                    "是 checkpoint 训练时就定死的，缺了不能猜（这份 checkpoint 不是 DFlash2 的）")
        self.candidate_selector = CandidateSelector(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            rank=int(draft_config["selector_rank"]),
            top_k=int(draft_config["selector_top_k"]),
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """输入嵌入先乘 checkpoint 的 `input_embedding_scale`（上游同名覆盖）。"""
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashForCausalLM):
    """`DFlash2Qwen3ForCausalLM`：DFlash2 草稿的对外接口（上游同名类）。"""

    _model_cls = DFlash2Qwen3Model

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__(config, prefix)
        draft_config = config.get("dflash_config") or {}
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        # 候选的 unary logits 用**自己的**收尾（checkpoint 的 output_multiplier /
        # final_logit_softcapping），不是 base 类那份 `logits_processor`（scale=logit_scale）：
        # 两者服务不同的头（整词表草稿 / 候选打分），上游同样是两个 `LogitsProcessor`。
        self.candidate_logits_processor = LogitsProcessor(
            self.config["vocab_size"],
            scale=float(draft_config.get("output_multiplier", 1.0)),
            soft_cap=softcap if softcap > 0 else None,
        )

    def compute_candidates(self, hidden_states: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """`[N, hidden]` → `(candidate_ids [N, k], unary_logits [N, k])`（上游同名方法）。

        `k = candidate_selector.top_k`（checkpoint 字段），**不是**采样器的 top-k 参数：
        它决定候选图每层有几个节点，采样只在这 k 个节点的边上走。
        """
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.candidate_selector.top_k)

    def get_candidate_top_k(self) -> int:
        return self.model.candidate_selector.top_k


# 注册发生在 `models/registry.py`：上游注册名是 checkpoint `architectures` 里的字符串
# `DFlash2DraftModel`（`registry.py:628`），类名是 `DFlash2Qwen3ForCausalLM`。
