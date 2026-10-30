"""DFlash 草稿模型（需求 076）：`DFlashQwen3ForCausalLM` / `DFlashQwen3Model`。

对应上游 `vllm/model_executor/models/qwen3_dflash.py`（0.28.0 快照）。它与普通 Qwen3 的
**三处**本质区别，缺一条就不是同一个模型（都能跑、只是草稿静默变差）：

1. **上下文 K/V 不是 forward 算的，而是"翻译"出来的**：本项目/上游的 target 每步把
   若干辅助层的 hidden states 交出来，`combine_hidden_states()` 用 `fc` 把它投到草稿的
   hidden 维，`precompute_and_store_context_kv()` 再做一次**跨层融合**的 K/V 投影
   （所有层的 KV 权重拼成一个大矩阵，一次 GEMM）+ K-norm + RoPE，直接写进草稿自己的 KV 缓存。
   于是上下文那些 token **一行都不进草稿模型**——这是 DFlash 与 PARD/EAGLE 的根本区别
   （PARD/EAGLE 的草稿要自己把前缀跑一遍）。
2. **注意力可以是非因果的**：每层的因果性由 `dflash_config.causal` / 顶层 `is_causal` 决定，
   默认非因果（"标准"DFlash checkpoint 就是这样）。mask query 是"槽位"，训练时同块的槽位
   互相可见；把它当普通因果 LM 跑，输出是另一个模型的输出。
3. **query 行 = 1 个 anchor（bonus）+ K 个 mask**：anchor 只贡献 K/V、不采样；采样只发生在
   mask 行上，且每个 mask 预测**它自己那个位置**（不是下一个位置）。

权重名（真实 DFlash checkpoint）：`embed_tokens.*`、`layers.N.*`（q/k/v → qkv_proj、
gate/up → gate_up_proj）、`norm.*`、`fc.*`（aux 特征投影）、`hidden_norm.*`、`lm_head.*`、
`d2t`（缩小词表 → target 词表的映射）。上游还有 `midlayer.*` / `encoder.*` 两种历史命名，
名字映射照抄上游的 `WeightsMapper`。

**本仓库不实现**（配置期明确拒绝，见 `SpeculativeConfig._resolve_dflash()`）：
`sample_from_anchor=True`（DSpark，77 关）、混合滑窗/全注意力（`use_swa`/`layer_types`，
需要多 KV group）、独立的 `mask_embedding.pt`（上游按需下载；本仓库的加载器只读本地
safetensors，遇到它会明确报错而不是静默忽略）。每张图默认非因果、anchor 不采样。
"""

from typing import Iterable

import torch
from torch import nn

from ..attention import Attention
from ..layers import (ParallelLMHead, QKVParallelLinear, RMSNorm, RowParallelLinear,
                      VocabParallelEmbedding)
from ..layers.logits_processor import LogitsProcessor
from ..layers.rotary_embedding import get_rope
from ..model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .qwen3 import Qwen2MLP, maybe_prefix
from .registry import register_model

_SLIDING_ATTENTION = "sliding_attention"


def _dflash_layer_causal(config: dict, layer_idx: int) -> bool:
    """这一层是不是因果注意力（上游 `qwen3_dflash._dflash_layer_causal` 的子集）。

    取值顺序与上游一致：顶层 `is_causal` → `dflash_config.causal` → 默认 **False**（非因果）。
    上游还有一条 `layer_types[layer_idx] == "sliding_attention"` 的兜底（滑窗层因果），
    本仓库不支持混合滑窗（配置期已拒绝 `layer_types`/`use_swa`），所以没有那一支。
    """
    is_causal = config.get("is_causal")
    if is_causal is not None:
        return bool(is_causal)
    dflash_config = config.get("dflash_config") or {}
    if dflash_config.get("causal") is not None:
        return bool(dflash_config["causal"])
    return False


def _aux_layer_ids(config: dict) -> list[int] | None:
    """草稿要拼接的 target 辅助层（用于算 `fc` 的输入宽度）。

    取值链与 `SpeculativeConfig.eagle_aux_hidden_state_layers()` 相同：
    `eagle_aux_hidden_state_layer_ids` → `dflash_config.target_layer_ids + 1`。
    两者都没有时返回 None（= 不吃 aux，直接用 target 最后一层，与 EAGLE-1 同款）。
    """
    layers = config.get("eagle_aux_hidden_state_layer_ids")
    if not layers:
        dflash_config = config.get("dflash_config") or {}
        target_layer_ids = dflash_config.get("target_layer_ids")
        layers = [int(index) + 1 for index in target_layer_ids] if target_layer_ids else None
    return [int(index) for index in layers] if layers else None


def dflash_mask_token_id(config: dict) -> int | None:
    """并行槽位（mask / noise）占位符的 id：与 `SpeculativeConfig._dflash_mask_token_id()`
    同一条取值链，**多认一个顶层 `mask_token_id`**。

    为什么模型也要认顶层：speculators 格式的 DSpark checkpoint（`Qwen3DSparkModel`）把
    `mask_token_id` 写在**顶层**、根本没有 `dflash_config`（`update_dspark()` 的输出）。
    模型侧只把它当"配置自洽性检查"（独立的 mask embedding 权重本仓库不支持，遇到会拒绝），
    但配置期校验与内核填行都要同一条取值链——两处不一致时，一个能过、另一个在启动后才炸。
    """
    dflash_config = config.get("dflash_config") or {}
    for value in (dflash_config.get("mask_token_id"), config.get("mask_token_id"),
                  config.get("dspark_noise_token_id"), config.get("pard_token"),
                  config.get("ptd_token_id")):
        if value is not None:
            return int(value)
    return None


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """按最后一维做 RMSNorm，**数值路径与 `layers/layernorm.py::RMSNorm.forward` 逐字相同**。

    为什么要一个独立函数（而不是直接用 `RMSNorm` 模块）：
    **分组 K-norm** 是"一次调用、按层用不同权重"——输入 `[L, num_ctx, nkv, hd]`、权重
    `[L, 1, 1, hd]` 广播到每层；而 `RMSNorm` 是"一个模块 = 一个权重"，表达不了"按层选权重"。
    换成一个函数、把权重当参数传，公式不变、调用只多一个参数。

    为什么不用 for 循环逐层调各层的 `k_norm` 模块：那会把"一次跨层归一化"拆成 L 次调用
    （L 次 kernel 启动），而这段预计算的设计目标就是少 ops（它不在 CUDA Graph 里，每步都跑）。

    **方差必须升 fp32 算**（与 `RMSNorm.forward` 一致、也与 vLLM 的 fused kernel 一致）：
    在 bf16 里先 `pow(2).mean()` 会把方差舍到 bf16，实测与 fp32 方差相比输出相对误差
    ~1.2%——对草稿的上下文 KV 来说这不是噪声级差异。
    """
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(variance + eps)).to(input_dtype) * weight


class DFlashAttention(nn.Module):
    """DFlash 的注意力层（上游 `DFlashQwen3Attention`）。

    与 Qwen3Attention 的差别只有两处：**因果性来自配置**（存成 `self.causal`，由提议者
    在建注意力元数据时传给后端），以及**上下文 K/V 由外部预计算写入**——所以 `forward()`
    只处理 query 行（它照样写自己的 K/V，位置是 anchor 与 mask 的位置）。
    """

    def __init__(self, config: dict, prefix: str, causal: bool) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.causal = causal

        self.qkv_proj = QKVParallelLinear(hidden_size, head_dim, num_heads, num_kv_heads,
                                          bias=config.get("attention_bias", False),
                                          prefix=maybe_prefix(prefix, "qkv_proj"))
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size,
                                        bias=config.get("attention_bias", False),
                                        prefix=maybe_prefix(prefix, "o_proj"))
        rope_parameters = dict(config.get("rope_parameters") or {})
        rope_parameters.setdefault(
            "rope_theta", config.get("rope_theta", 10000.0))
        self.rotary_emb = get_rope(
            head_size=head_dim,
            max_position=config["max_position_embeddings"],
            is_neox_style=True,
            rope_parameters=rope_parameters)
        self.attn = Attention(num_heads=num_heads, head_size=head_dim,
                              num_kv_heads=num_kv_heads,
                              layer_name=maybe_prefix(prefix, "attn"))
        eps = config.get("rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.qkv_proj.kv_size, self.qkv_proj.kv_size],
                            dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class DFlashDecoderLayer(nn.Module):
    """一层解码器：pre-norm + 注意力 + pre-norm + MLP（与 Qwen3 的层同构）。"""

    def __init__(self, config: dict, prefix: str, causal: bool) -> None:
        super().__init__()
        eps = config.get("rms_norm_eps", 1e-6)
        self.self_attn = DFlashAttention(config, prefix=maybe_prefix(prefix, "self_attn"),
                                         causal=causal)
        self.mlp = Qwen2MLP(config, prefix=maybe_prefix(prefix, "mlp"))
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=eps)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashModel(nn.Module):
    """`DFlashQwen3Model`：上下文 KV 预计算 + query-only 的 forward。"""

    # 内层解码器类：DFlash 用 DFlashDecoderLayer，DFlash2（78 关）换成带分组卷积的子类
    # （上游同一个位置叫 `decoder_layer_cls`，由 `DFlash2Qwen3Model` 覆盖）。
    decoder_layer_cls = DFlashDecoderLayer

    # 名字映射照抄上游（`midlayer.*` 是历史 checkpoint 的层名；`encoder.*` 是
    # MiMo 系 checkpoint 的 aux 编码器命名）。
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            "midlayer.": "layers.0.",
            "encoder.output_norm_enc.": "hidden_norm.",
            "encoder.fc.": "fc.",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        dflash_config = config.get("dflash_config") or {}
        self.mask_token_id = dflash_mask_token_id(config)
        if self.mask_token_id is None:
            raise ValueError(
                "DFlash 的配置里没有 mask token：mask/并行槽位要填这个训练时约定的占位符"
                "（取值链见 `dflash_mask_token_id()`；换一个不会报错，只会让草稿全部基于"
                "模型没见过的输入）")

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleList([
            self.decoder_layer_cls(
                config, prefix=maybe_prefix(prefix, f"layers.{index}"),
                causal=_dflash_layer_causal(config, index))
            for index in range(config["num_hidden_layers"])
        ])
        # 是否吃 target 的**多个辅助层**特征（上游 `use_aux_hidden_state`，默认 True）。
        # False = 直接用 target 最后一层 hidden（与 EAGLE-1 同款），这时没有 `fc`。
        self.use_aux_hidden_state = bool(dflash_config.get("use_aux_hidden_state", True))
        self.fc: RowParallelLinear | None = None
        if self.use_aux_hidden_state:
            aux_layers = _aux_layer_ids(config)
            if not aux_layers:
                raise ValueError(
                    "DFlash 需要 `dflash_config.target_layer_ids`（或 "
                    "`eagle_aux_hidden_state_layer_ids`）来确定 `fc` 的输入宽度："
                    "这几层是草稿读的 target 特征，缺了就没法建投影")
            target_hidden_size = int(config.get("target_hidden_size", self.hidden_size))
            self.fc = RowParallelLinear(target_hidden_size * len(aux_layers),
                                        self.hidden_size, bias=False,
                                        prefix=maybe_prefix(prefix, "fc"))
        self.hidden_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # 检查点里**没有** `embed_tokens.*`（词嵌入与 target 共享）→ 本模型的加载器跳过它。
        # 默认 False；DSpark 的 `load_weights()` 会在"没有这一份"时置 True，随后
        # `load_dspark_model()` 把 target 的模块挂上来（上游 `_should_share` 那条路）。
        self.tie_embeddings = False
        # 上下文 KV 预计算用的融合权重；`load_weights()` 之后才填（与上游同位置）。
        self._num_attn_layers = len(self.layers)
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]
        self._fused_kv_weight: torch.Tensor | None = None
        self._fused_kv_bias: torch.Tensor | None = None
        self._k_norm_weights: torch.Tensor | None = None
        self._hidden_norm_weight: torch.Tensor | None = None

    # -------- 上下文 KV 预计算（DFlash 的核心） --------

    def _build_fused_kv_buffers(self) -> None:
        """把所有注意力层的 KV 投影权重拼成一份（上游 `_build_fused_kv_buffers`）。

        为什么要拼：上下文行**不过模型**，它们的 K/V 只能从 hidden 一次性投出来。每层单独
        调用一次 GEMM 会变成 L 次小 kernel（L 通常 1~4），拼成 `[L*2*kv_size, hidden]`
        之后是**一次** GEMM（上游注释：minimize the number of torch ops）。代价是加载后
        多一份权重副本（bf16 下 = 2 × L × kv_size × hidden 字节）。
        """
        attn_layers = [layer.self_attn for layer in self.layers]
        attn0 = attn_layers[0]
        for attn in attn_layers[1:]:
            # 融合的前提：所有层的 KV 规格与 RoPE 相同（上游同款断言）。不同就必须分开算。
            if (attn.kv_size, attn.head_dim, attn.num_kv_heads) != (
                    attn0.kv_size, attn0.head_dim, attn0.num_kv_heads):
                raise ValueError("DFlash 的融合 KV 预计算要求所有层的 KV 规格一致")
            if (attn.rotary_emb.head_size, attn.rotary_emb.rotary_dim) != (
                    attn0.rotary_emb.head_size, attn0.rotary_emb.rotary_dim):
                raise ValueError("DFlash 的融合 KV 预计算要求所有层的 RoPE 参数一致")

        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_kv_weight = torch.cat(
            [attn.qkv_proj.weight[attn.q_size:] for attn in attn_layers], dim=0).contiguous()
        if attn0.qkv_proj.bias is not None:
            self._fused_kv_bias = torch.cat(
                [attn.qkv_proj.bias[attn.q_size:] for attn in attn_layers],
                dim=0).contiguous()
        else:
            self._fused_kv_bias = None
        # K-norm 的权重按层堆成 [L, head_dim]，配 `[L, num_ctx, nkv, hd]` 的广播。
        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in attn_layers], dim=0).contiguous()

    def precompute_and_store_context_kv(self, context_states: torch.Tensor,
                                        context_positions: torch.Tensor,
                                        context_slot_mapping: torch.Tensor | None = None) -> None:
        """把 target 的 hidden（已过 `fc`）翻译成草稿各层的 K/V，并写进缓存。

        五步，顺序不能换（上游同序）：

            1. `hidden_norm`（RMSNorm）—— 与 query 行用的是同一套归一化
            2. **一次**跨层 GEMM：[num_ctx, hidden] × [L*2*kv_size, hidden]^T
            3. 布局重排成层级优先 `[L, num_ctx, nkv, hd]`（一次 contiguous 拷贝）
            4. 分组 K-norm（每层用自己的权重）+ 跨层一次性 RoPE（位置按层重复）
            5. 每层按槽位写缓存（`do_kv_cache_update`，跳过哨兵槽位）

        `context_slot_mapping=None` 只在热身/显存剖析时出现（上游 dummy_run 同款）：
        只算不写，避免用假块表把真实缓存写脏。
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        num_ctx = context_states.shape[0]
        if num_ctx == 0:
            return
        layers = self._num_attn_layers
        attn0 = self._attn_layers[0]
        num_kv_heads, head_dim = attn0.num_kv_heads, attn0.head_size
        kv_size = num_kv_heads * head_dim

        normed = _rms_norm(context_states, self._hidden_norm_weight, self.rms_norm_eps)
        all_kv_flat = torch.nn.functional.linear(
            normed, self._fused_kv_weight, self._fused_kv_bias)
        # [num_ctx, L, 2, nkv, hd] → [2, L, num_ctx, nkv, hd]
        all_kv = all_kv_flat.view(num_ctx, layers, 2, num_kv_heads, head_dim).permute(
            2, 1, 0, 3, 4).contiguous()
        all_k, all_v = all_kv[0], all_kv[1]

        # 分组 K-norm：[L, num_ctx, nkv, hd] 的最后一维用 `[L, 1, 1, hd]` 的权重
        k_normed = _rms_norm(all_k, self._k_norm_weights.view(layers, 1, 1, head_dim),
                             self.rms_norm_eps)
        # 跨层一次性 RoPE：位置按层重复（层优先布局 ⇒ positions.repeat(L) 正好对齐）
        k_flat = k_normed.view(layers * num_ctx, kv_size)
        k_flat = self.layers[0].self_attn.rotary_emb.apply_to_key(
            context_positions.repeat(layers), k_flat)
        k_final = k_flat.view(layers, num_ctx, num_kv_heads, head_dim)

        if context_slot_mapping is None:
            return
        for layer_index in range(layers):
            attention = self._attn_layers[layer_index]
            attention.impl.do_kv_cache_update(
                attention, k_final[layer_index], all_v[layer_index],
                attention.kv_cache, context_slot_mapping)

    # -------- 前向（只有 query 行） --------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights) -> set[str]:
        skip = ["embed_tokens."] if self.tie_embeddings else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class DFlashForCausalLM(nn.Module):
    """`DFlashQwen3ForCausalLM`：模型的对外接口（combine / logits / 加载）。"""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    # 内层模型类（DSpark 换上带 Markov / confidence 头的子类，其余流程完全复用）。
    _model_cls = DFlashModel

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = self._model_cls(config, prefix=maybe_prefix(prefix, "model"))
        draft_vocab_size = config.get("draft_vocab_size") or config["vocab_size"]
        self.draft_vocab_size = int(draft_vocab_size)
        self.lm_head = ParallelLMHead(self.draft_vocab_size, config["hidden_size"],
                                      prefix=maybe_prefix(prefix, "lm_head"))
        target_vocab_size = int(config["vocab_size"])
        if self.draft_vocab_size != target_vocab_size:
            # 缩小词表 → target 词表的偏移映射（与 EAGLE3 的 d2t 同一件事，但**不是** TLI：
            # 这里是同一套 tokenizer 的缩小词表，67 关的 TLI 是两套 tokenizer 的交集）。
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.draft_vocab_size, dtype=torch.long), requires_grad=False)
        else:
            self.register_parameter("draft_id_to_target_id", None)
        # 词表 GEMM 的收尾（上游 `DFlashQwen3ForCausalLM` 同款）：`logit_scale` 是少数
        # checkpoint 会带的缩放（真实 DFlash/DSpark 配置里叫 `logit_scale`，缺省 1.0）。
        # 77 关把 DSpark 的 markov 偏置也按同一个 scale 加，所以两者必须共用这一份。
        logit_scale = float(config.get("logit_scale", 1.0) or 1.0)
        self.logits_processor = LogitsProcessor(self.draft_vocab_size, scale=logit_scale)

    # -------- 对外接口（提议者/引擎调用） --------

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model(input_ids, positions)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """把 target 的（多辅助层拼接后的）hidden 投到草稿宽度（上游同名方法）。

        宽度对不上时**当场报错**（上游同款）：通常意味着 draft 的 `target_layer_ids` 指向了
        target 没有的层，或者 target 没开辅助层输出。不报错的话，`fc` 会拿错位的特征做投影
        ——草稿能出、只是全错。
        """
        if not self.model.use_aux_hidden_state:
            return hidden_states
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlash 草稿期望 {expected} 维拼接后的 aux 特征，收到 "
                f"{hidden_states.shape[-1]} 维。通常是 draft 的 `target_layer_ids` 指向了 "
                f"target 里不存在的层（draft/target 不配套）")
        projected, _ = self.model.fc(hidden_states)
        return projected

    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mapping=None) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """词表 GEMM；缩小词表时 scatter 回 target 宽度（其余位置 `-inf`）。

        与 EAGLE3 的 `compute_logits()` 同一条规则：提议者拿到 logits 就 argmax，那个 id 会
        被当成**草稿 token** 交给 target 验证，所以它必须是 **target 空间**的 id。
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(self.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full((logits.shape[0], self.config["vocab_size"]),
                                     float("-inf"))
        logits_new[:, targets] = logits
        return logits_new

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    # -------- 权重加载 --------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """按上游的规则重排名字，再交给 `AutoWeightsLoader`。

        规则（上游 `DFlashQwen3ForCausalLM.load_weights` 逐条）：

            `t2d`           跳过（反向映射，推理不用）
            `d2t`           → `draft_id_to_target_id`
            其余非 lm_head  → 加 `model.` 前缀（`model.` 这一组交给 `DFlashModel`）

        三条**明确拒绝**，都是"静默错"的来源：
            `mask_hidden`    DFlash 用 mask token 的 embedding，权重里不该有它（上游 assert）
            未知名字          加载器的覆盖检查会当场报错（不会静默跳过）
        """
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if "mask_hidden" in name:
                raise ValueError(
                    "DFlash 权重里不该有 `mask_hidden`（那是 P-EAGLE 的并行槽位常量向量）："
                    f"收到 {name!r}。这份 checkpoint 不是 DFlash 的格式")
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight

        skip_substrs: list[str] = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        # **必须按名字排序**：加载器按"第一段名字"分组，而 `itertools.groupby` 只合并
        # **相邻**的同一前缀。checkpoint 里 `norm.weight` 可能排在 `lm_head.weight` 之后，
        # 于是 `model.*` 会被拆成不连续的两组——第二组（只有 norm）单独进 `DFlashModel`，
        # 第一组的覆盖检查会立刻报"norm.weight 没加载"。排序让同一前缀连续，语义不变。
        loaded = loader.load_weights(sorted(model_weights.items()))
        # 融合权重只能在权重都到位之后建（它引用各层的 qkv_proj/k_norm）
        self.model._build_fused_kv_buffers()
        return loaded


register_model("DFlashQwen3ForCausalLM")(DFlashForCausalLM)
