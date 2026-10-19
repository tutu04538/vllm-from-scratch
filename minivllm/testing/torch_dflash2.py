"""**参考实现**（只给测试）：DFlash2 的三个新算子 + 解码器层的逐元素公式。

需求 078 要求"模型 forward / selector / logits / verifier 四层差分"。生产路径里的三个新
算子都是**向量化**写法（`unflatten` / `einsum` / 广播），一眼看不出跟公式是否一致；这里用
**显式循环**把同一份公式重写一遍，于是"公式理解错"与"向量化写错"能分开。

    reference_grouped_conv   沿 token 维的因果分组卷积（`n%block_size` 与 `position>=tap`）
    reference_score_edges    `[B,K,k,k]` 的三元边分（前驱码本 · hidden · 后继码本）
    reference_candidates     top-k 候选 + `output_multiplier` / `final_logit_softcapping`
    reference_greedy_walk    温度 0 的条件 walk（argmax；概率 walk 的噪声流无法在 Torch 里
                             与 Triton 逐位复现，反过来用**上游内核**做 oracle）
    reference_decoder_layer  一层的 norm/conv/attention 占位/MLP/残差顺序（attention 由调用方
                             注入，测试里用同一个无参 stub，从而只比 DFlash2 新增的接线）

**边界**：这些函数只给测试；生产包不 import 它（与 `torch_dflash.py` / `torch_dspark.py`
同一条约定）。
"""

import torch
import torch.nn.functional as F


def reference_grouped_conv(hidden_states: torch.Tensor, delta: torch.Tensor,
                           base: torch.Tensor, block_size: int, num_groups: int,
                           group_size: int, taps: int) -> torch.Tensor:
    """因果分组卷积的逐 token 定义。

    `out[n] = Σ_t (base[t] + delta[n,t]) * hidden[n−t]`，仅在 `n−t >= 0` 且
    `n % block_size >= t` 时累加（后者保证不跨请求的 query 块回看）。
    """
    num_tokens = hidden_states.shape[0]
    hidden = hidden_states.view(num_tokens, num_groups, group_size)
    coefficients = base.view(taps, num_groups, group_size).unsqueeze(0) \
        + delta.unsqueeze(-1)
    out = torch.zeros_like(hidden)
    for n in range(num_tokens):
        position = n % block_size
        for tap in range(taps):
            if n - tap < 0 or position < tap:
                continue
            out[n] = out[n] + coefficients[n, tap] * hidden[n - tap]
    return out.reshape(num_tokens, num_groups * group_size)


def reference_score_edges(predecessor_table: torch.Tensor, successor_table: torch.Tensor,
                          candidate_ids: torch.Tensor, unary_logits: torch.Tensor,
                          hidden: torch.Tensor, anchor_token_ids: torch.Tensor,
                          top_k: int) -> torch.Tensor:
    """`[B,K,k,k]` 边分的逐元素定义（`l=0` 的 p 全部指向 anchor）。"""
    batch, steps, k = candidate_ids.shape
    rank = hidden.shape[-1]
    scores = torch.empty(batch, steps, k, k, dtype=unary_logits.dtype,
                         device=unary_logits.device)
    for b in range(batch):
        for step in range(steps):
            for p in range(k):
                # 第 0 步的前驱是 anchor；之后是上一位置的第 p 个候选（上游同一个 expand）
                pred_id = (anchor_token_ids[b] if step == 0
                           else candidate_ids[b, step - 1, p])
                for c in range(k):
                    value = unary_logits[b, step, c]
                    succ_id = candidate_ids[b, step, c]
                    for r in range(rank):
                        value = value + (predecessor_table[pred_id, r]
                                         * hidden[b, step, r]
                                         * successor_table[succ_id, r])
                    scores[b, step, p, c] = value
    return scores


def reference_candidates(hidden_states: torch.Tensor, lm_head_weight: torch.Tensor,
                         k: int, scale: float = 1.0,
                         soft_cap: float | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """`compute_candidates()` 的参考公式（顺序与上游 `get_top_k_tokens` 一致）。"""
    logits = F.linear(hidden_states, lm_head_weight)
    values, ids = torch.topk(logits, k, dim=-1)
    values = values.float()
    if scale != 1.0:
        values = values * scale
    if soft_cap is not None:
        values = torch.tanh(values / soft_cap) * soft_cap
    return ids, values


def reference_greedy_walk(scores: torch.Tensor, candidate_ids: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """温度 0 的条件 walk：每步在"上一步选中的前驱"那一片里取 argmax。

    返回 `(tokens [B,K], realized_scores [B,K,k])`。`realized_scores[b,l]` 是**实际走到**
    的那一片 `[k]` 分数（= 生产内核写进 `_selector_scores` 的东西）。
    """
    batch, steps, k, _ = scores.shape
    selected = torch.zeros(batch, steps, dtype=candidate_ids.dtype,
                           device=candidate_ids.device)
    realized = torch.zeros(batch, steps, k, dtype=scores.dtype, device=scores.device)
    previous = torch.zeros(batch, dtype=torch.long, device=scores.device)
    rows = torch.arange(batch, device=scores.device)
    for step in range(steps):
        edge_scores = scores[rows, step, previous, :]
        index = edge_scores.argmax(dim=-1)
        selected[:, step] = candidate_ids[rows, step, index]
        realized[:, step] = edge_scores
        previous = index
    return selected, realized


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """与 `RMSNorm.forward` 同一条数值路径（fp32 方差、回原 dtype 再乘权重）。"""
    dtype = x.dtype
    x32 = x.to(torch.float32)
    out = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.to(dtype) * weight


def _reference_mlp(mlp, hidden_states: torch.Tensor) -> torch.Tensor:
    # 本仓库的 Qwen2MLP 用打包的 `gate_up_proj`（`[gate; up]` 沿输出维拼接）；
    # 参考实现按它在 checkpoint 里的两段拆开，不与生产路径共用 `SiluAndMul`。
    gate_up = F.linear(hidden_states, mlp.gate_up_proj.weight)
    intermediate = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :intermediate], gate_up[..., intermediate:]
    return F.linear(F.silu(gate) * up, mlp.down_proj.weight)


def reference_decoder_layer(layer, attention, positions: torch.Tensor,
                            hidden_states: torch.Tensor,
                            residual: torch.Tensor | None):
    """一层 DFlash2 decoder 的显式公式（attention 由调用方注入）。

    生产层的注意力/MLP 本身沿用 DFlash（76 关已与上游差分过）；这里把**新增的接线**写死：
    两侧卷积的 `prepare/finish` 位置、系数复用、残差流顺序。`attention(hidden)` 在测试里是
    同一个无参 stub，于是两边的唯一变量就是卷积/归一化/残差的排列。
    """
    eps = layer.input_layernorm.variance_epsilon
    residual_eps = layer.post_attention_layernorm.variance_epsilon
    if residual is None:
        residual = hidden_states
        normed = _rms_norm(hidden_states, layer.input_layernorm.weight, eps)
    else:
        residual = hidden_states + residual
        normed = _rms_norm(residual, layer.input_layernorm.weight, eps)

    def conv(module, states, coefficients, side):
        return reference_grouped_conv(
            states, coefficients, module.base_kernel[side], module.block_size,
            module.num_groups, module.group_size, module.taps)

    num_tokens = normed.shape[0]
    projection = F.linear(normed, layer.attention_conv.kernel_projection.weight)
    coefficients = projection.view(num_tokens, 2, layer.attention_conv.taps,
                                   layer.attention_conv.num_groups)
    prepared = conv(layer.attention_conv, normed, coefficients[:, 0], 0)
    attended = attention(prepared, positions)
    hidden_states = conv(layer.attention_conv, attended, coefficients[:, 1], 1)

    residual = hidden_states + residual
    normed = _rms_norm(residual, layer.post_attention_layernorm.weight, residual_eps)
    projection = F.linear(normed, layer.mlp_conv.kernel_projection.weight)
    coefficients = projection.view(num_tokens, 2, layer.mlp_conv.taps,
                                   layer.mlp_conv.num_groups)
    prepared = conv(layer.mlp_conv, normed, coefficients[:, 0], 0)
    activated = _reference_mlp(layer.mlp, prepared)
    hidden_states = conv(layer.mlp_conv, activated, coefficients[:, 1], 1)
    return hidden_states, residual
