"""教学版 learned top-k 稀疏注意力（无真实权重；不冒充 DeepSeek DSA/MLA）。

这个文件只回答一个问题：

    如果每一层先用一个小的 indexer 选出「这条 query 值得看的 top-k 历史位置」，
    然后只在这些位置上做 attention，模型该怎么组织、因果性/掩码怎么保证？

与真实模型的关系：

- **不是** Step3p5。Step3p5 是 full/sliding 固定窗口 + MoE，没有 learned indexer；
- **不是** DeepSeek DSA/MLA 的逐位复刻。这里没有 MLA 低秩 KV、没有 lightning
  indexer 的训练目标、没有 paged KV cache；
- 它是 `minivllm/testing/` 下的教学模型：随机初始化即可前向/生成，用来观察
  indexer 选中了哪些位置、稀疏 mask 是否因果、top-k 是否退化成 dense。

`SparseLM.forward` 可选返回每层 `selected`，形状 `[B, T, S]` 的 bool：
`selected[b, i, j] == True` 表示第 i 个 query 在第 j 个 key 上做了 attention。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SparseAttentionConfig:
    vocab_size: int = 64
    hidden_size: int = 64
    intermediate_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 16
    max_seq_len: int = 64
    sparse_top_k: int = 8
    index_heads: int = 2
    index_dim: int = 8
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads 必须是 num_kv_heads 的整数倍")
        if self.sparse_top_k < 0:
            raise ValueError("sparse_top_k 不能为负")
        if self.sparse_top_k > self.max_seq_len:
            raise ValueError("sparse_top_k 不能大于 max_seq_len")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32.to(x.dtype) * self.weight)


class SparseIndexer(nn.Module):
    """小的 learned indexer：给每个 query 的每个历史 key 打分并选 top-k。

    打分（简化版 lightning indexer）：

        score[t, s] = sum_h weight[t, h] * (q_index[t, h] · k_index[s, h])

    真实 DeepSeek 的 indexer 还有 ReLU/训练目标/多 head 归一化等；这里只保留
    「用 learned 小投影选位置」这一层，方便观察稀疏 mask。
    """

    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.num_heads = config.index_heads
        self.head_dim = config.index_dim
        self.q_proj = nn.Linear(config.hidden_size,
                                config.index_heads * config.index_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                                config.index_heads * config.index_dim, bias=False)
        self.w_proj = nn.Linear(config.hidden_size, config.index_heads, bias=False)
        self.scale = 1.0 / math.sqrt(config.index_dim)

    def forward(self, hidden_states: torch.Tensor,
                valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 `(scores, selected)`，形状 `[B,T,S]`。

        `valid_mask` 是 `[B,S]` bool；`scores` 里无效位置是 `-inf`。
        """
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B,H,T,S]
        scores_h = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = torch.sigmoid(self.w_proj(hidden_states))  # [B,T,H]
        scores = (scores_h * weights.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
        scores = scores.masked_fill(~valid_mask[:, None, :], float("-inf"))
        return scores, scores


class SparseAttention(nn.Module):
    """QKV attention + learned top-k key selection（教学版）。"""

    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_groups = config.num_heads // config.num_kv_heads
        qkv_out = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_out, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim,
                                config.hidden_size, bias=False)
        self.indexer = SparseIndexer(config)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _rope(self, q: torch.Tensor, k: torch.Tensor,
              positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对 [B,T,H,D] 的 q/k 做 RoPE（教学版，不缓存 cos/sin）。"""
        head_dim = q.shape[-1]
        inv_freq = 1.0 / (
            self.config.rope_theta ** (
                torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
                / head_dim))
        freqs = positions.float()[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B,T,D]
        cos = emb.cos()[:, :, None, :]
        sin = emb.sin()[:, :, None, :]
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                top_k: int | None = None,
                return_selected: bool = False):
        batch, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(torch.bool)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool,
                            device=hidden_states.device).tril()
        valid = attention_mask[:, None, :] & causal[None, :, :]  # [B,T,S]

        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        positions = torch.arange(seq_len, device=hidden_states.device)[None, :].expand(
            batch, -1)
        q, k = self._rope(q, k, positions)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(self.kv_groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.kv_groups, dim=1)

        index_scores, _ = self.indexer(hidden_states, attention_mask)
        index_scores = index_scores.masked_fill(~valid, float("-inf"))
        if top_k is None:
            selected = valid
        else:
            k_eff = min(int(top_k), seq_len)
            if k_eff <= 0:
                selected = torch.zeros_like(valid)
            else:
                top_values, top_indices = torch.topk(index_scores, k_eff, dim=-1)
                selected = torch.zeros_like(valid)
                selected.scatter_(-1, top_indices, top_values != float("-inf"))
                selected = selected & valid
        sparse_mask = selected
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~sparse_mask[:, None, :, :], float("-inf"))
        any_valid = sparse_mask.any(dim=-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(any_valid[:, None, :, :], probs,
                            torch.zeros_like(probs))
        context = torch.matmul(probs, v).transpose(1, 2).reshape(
            batch, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(context)
        if return_selected:
            return out, selected
        return out


class SparseMLP(nn.Module):
    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class SparseDecoderLayer(nn.Module):
    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps)
        self.self_attn = SparseAttention(config)
        self.mlp = SparseMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor,
                top_k: int | None, return_selected: bool):
        attn_out, selected = self.self_attn(
            self.input_layernorm(x), attention_mask, top_k, return_selected=True)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        if return_selected:
            return x, selected
        return x


class SparseLM(nn.Module):
    """随机初始化的教学语言模型；不加载任何 checkpoint。"""

    def __init__(self, config: SparseAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            SparseDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                top_k: int | None = None,
                return_selected: bool = False):
        x = self.embed_tokens(input_ids)
        selected_layers = []
        for layer in self.layers:
            x, selected = layer(x, attention_mask, top_k, return_selected=True)
            if return_selected:
                selected_layers.append(selected)
        x = self.norm(x)
        logits = self.lm_head(x)
        if return_selected:
            return logits, selected_layers
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8,
                 top_k: int | None = None) -> torch.Tensor:
        """greedy 生成；只用于教学观察，不声称质量。"""
        tokens = input_ids
        for _ in range(max_new_tokens):
            logits = self(tokens, top_k=top_k)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
