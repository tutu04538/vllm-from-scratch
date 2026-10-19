"""DFlash 的纯 Torch 参考实现（**只给测试用**；生产路径不 import）。

它存在的理由：DFlash 的两条"不报错但会错"的路径没法靠读代码发现——

    1. **融合的上下文 KV 预计算**：生产实现把所有层的 K/V 投影拼成一个大 GEMM、K-norm 分组做、
       RoPE 跨层一次做。数学上等价于"逐层各算各的"，但布局/归一化/RoPE 顺序错一格都不报错。
    2. **非因果 query 注意力**：mask query 是"槽位"，同块互相可见；把它当普通因果 LM 跑，
       输出是另一个模型的输出（草稿质量崩，但没有任何异常）。

所以这里用最笨的写法（逐层、逐请求、显式 mask）复算同一份数学，测试拿它当冻结参考：

    reference_context_kv(model, context_states, context_positions) → [num_layers, 2, ctx, nkv, hd]
    reference_forward(model, context_states, context_positions, query_ids, query_positions, causal)

参考实现**直接复用模型自己的参数**（`qkv_proj` / `k_norm` / `rotary_emb` / MLP），不复制权重，
所以它验的是"生产路径的融合与缓存布局"，不是"两套独立实现的数值巧合"。
"""

import torch


def _layer_qkv(layer, hidden: torch.Tensor, positions: torch.Tensor):
    """一层的 q/k/v（含 q/k norm 与 RoPE），与模型 forward 里的同一步骤逐字相同。"""
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(hidden)
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.view(-1, attn.num_heads, attn.head_dim)).view(q.shape)
    k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim)).view(k.shape)
    q, k = attn.rotary_emb(positions, q, k)
    return q, k, v


def _layer_kv(layer, context_states: torch.Tensor, context_positions: torch.Tensor):
    """一层的上下文 K/V（**只用 K/V 那两路投影**，不含 q：上下文行不产生 query）。"""
    attn = layer.self_attn
    qkv, _ = attn.qkv_proj(context_states)
    _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim)).view(k.shape)
    _, k = attn.rotary_emb(context_positions, k, k)   # 第二个返回值就是旋转后的 K
    return (k.view(-1, attn.num_kv_heads, attn.head_dim),
            v.view(-1, attn.num_kv_heads, attn.head_dim))


def reference_context_kv(model, context_states: torch.Tensor,
                         context_positions: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """逐层复算上下文 K/V（输入 = 已过 `fc`、未过 `hidden_norm` 的特征）。

    生产实现先对**所有层共用**的 hidden 做一次 `hidden_norm`，再乘各层的融合 KV 权重；
    这里退化成逐层：`hidden_norm` 一次 + 每层的 `qkv_proj` 的 K/V 段 + `k_norm` + RoPE。
    """
    inner = model.model
    with torch.no_grad():
        normed = (context_states * torch.rsqrt(
            context_states.pow(2).mean(-1, keepdim=True) + inner.rms_norm_eps)
            * inner.hidden_norm.weight)
        return [_layer_kv(layer, normed, context_positions) for layer in inner.layers]


def _gqa_expand(tensor: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return tensor
    return tensor.repeat_interleave(num_heads // num_kv_heads, dim=1)


def reference_forward(model, context_states: torch.Tensor, context_positions: torch.Tensor,
                      query_ids: torch.Tensor, query_positions: torch.Tensor,
                      causal: bool) -> torch.Tensor:
    """完整的 query 前向参考：上下文 KV 来自 `reference_context_kv()`，query 走显式注意力。"""
    inner = model.model
    context_kv = reference_context_kv(model, context_states, context_positions)
    with torch.no_grad():
        hidden = inner.embed_input_ids(query_ids)
        residual = None
        for layer_index, layer in enumerate(inner.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            q, k, v = _layer_qkv(layer, hidden, query_positions)
            ctx_k, ctx_v = context_kv[layer_index]
            keys = torch.cat([ctx_k, k.view(-1, attn.num_kv_heads, attn.head_dim)], dim=0)
            values = torch.cat([ctx_v, v.view(-1, attn.num_kv_heads, attn.head_dim)], dim=0)
            keys = _gqa_expand(keys, attn.num_heads, attn.num_kv_heads)
            values = _gqa_expand(values, attn.num_heads, attn.num_kv_heads)
            num_query, heads, head_dim = q.view(-1, attn.num_heads, attn.head_dim).shape
            qh = q.view(num_query, heads, head_dim).transpose(0, 1).float()
            scores = torch.matmul(qh, keys.transpose(0, 1).float().transpose(-1, -2)) * (
                head_dim ** -0.5)
            if causal:
                num_context = keys.shape[0] - num_query
                mask = torch.ones(num_query, keys.shape[0], dtype=torch.bool,
                                  device=scores.device).tril(
                    diagonal=num_context)
                scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(torch.float32).min)
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, values.transpose(0, 1).float())
            out = out.transpose(0, 1).reshape(num_query, heads * head_dim).to(q.dtype)
            hidden = attn.o_proj(out)[0]
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
        hidden, _ = inner.norm(hidden, residual)
        return hidden
