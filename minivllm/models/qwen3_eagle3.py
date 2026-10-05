"""EAGLE3 draft 模型（需求 063 §2/§3.2）：`Eagle3Qwen3ForCausalLM` / Llama 风格的同构实现。

对应上游 `vllm/model_executor/models/qwen3_eagle3.py`（L282-442）与 `llama_eagle3.py`；两者数学相同，
只差 q/k norm（Qwen3 有、Llama 没有），所以本文件用一个实现 + `use_qk_norm` 开关覆盖，注册两个名字。

forward 语义（**不能替换成普通 Qwen3**）：

    h = combine_hidden_states(aux_hidden_states)        # 多层特征拼接 → fc 投影到 hidden
    layer0:  x = cat([input_layernorm(embeds), hidden_norm(h)])   ← 第一层吃 2*hidden
    layer1+ (真实 EAGLE3 只有 1 层): 普通 pre-norm
    hidden_states, hidden_prenorm = norm(x)             # 返回**两个**：lm_head 用前者，下一步 draft 用后者
    → return (hidden_states, hidden_prenorm)            # 上游 `model_returns_tuple()` 拿到的 tuple

权重名（真实 checkpoint，见 docs/results.json → step63.models）：`fc.weight`、`midlayer.*`（映射到
`layers.0.*`，q/k/v → qkv_proj、gate/up → gate_up_proj）、`norm.weight`、`lm_head.weight`（draft 词表）、
`d2t`/`t2d`（异构词表映射，67 关主题）。没有 `embed_tokens.*` → 与 target 共享 embedding。
"""

import torch
from torch import nn

from ..layers import ParallelLMHead, QKVParallelLinear, RMSNorm, RowParallelLinear, VocabParallelEmbedding
from ..layers.rotary_embedding import RotaryEmbedding
from ..attention import Attention
from ..model_loader.auto_weights_loader import AutoWeightsLoader, WeightsMapper
from .qwen3 import Qwen2MLP, maybe_prefix
from .registry import register_model


class Eagle3Attention(nn.Module):
    """EAGLE 的注意力：`qkv_input_size` 可以是 `2*hidden`（第一层拼接了 embedding），q/k norm 可选。"""

    def __init__(self, config: dict, prefix: str, qkv_input_size: int, use_qk_norm: bool) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.qkv_proj = QKVParallelLinear(qkv_input_size, head_dim, num_heads, num_kv_heads,
                                          bias=config.get("attention_bias", False),
                                          prefix=maybe_prefix(prefix, "qkv_proj"))
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False,
                                        prefix=maybe_prefix(prefix, "o_proj"))
        rope_parameters = config.get("rope_parameters") or {}
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim, rotary_dim=head_dim,
            max_position_embeddings=config["max_position_embeddings"],
            base=rope_parameters.get("rope_theta", config.get("rope_theta", 10000.0)))
        self.attn = Attention(num_heads=num_heads, head_size=head_dim,
                              num_kv_heads=num_kv_heads, layer_name=maybe_prefix(prefix, "attn"))
        eps = config.get("rms_norm_eps", 1e-6)
        # q/k norm 只在 Qwen3 风格的头里（真实 EAGLE3-Llama checkpoint 的权重里没有 q_norm/k_norm）
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=eps)
            self.k_norm = RMSNorm(head_dim, eps=eps)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.qkv_proj.kv_size, self.qkv_proj.kv_size], dim=-1)
        if self.q_norm is not None:
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class Eagle3DecoderLayer(nn.Module):
    """一层 EAGLE3 解码层；`layer_idx == 0` 时把 embedding 与（投影后的）target 特征拼起来。"""

    def __init__(self, config: dict, prefix: str, layer_idx: int, use_qk_norm: bool,
                 norm_before_residual: bool = False) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        qkv_input_size = 2 * hidden_size if layer_idx == 0 else hidden_size
        self.layer_idx = layer_idx
        self.norm_before_residual = norm_before_residual
        self.self_attn = Eagle3Attention(config, prefix=maybe_prefix(prefix, "self_attn"),
                                         qkv_input_size=qkv_input_size, use_qk_norm=use_qk_norm)
        self.mlp = Qwen2MLP(config, prefix=maybe_prefix(prefix, "mlp"))
        eps = config.get("rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.hidden_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, positions: torch.Tensor, embeds: torch.Tensor, hidden_states: torch.Tensor,
                residual: torch.Tensor | None):
        if self.layer_idx == 0:
            # 上游两条分支只差"先做 hidden_norm 还是先记 residual"（norm_before_residual）
            embeds = self.input_layernorm(embeds)
            if self.norm_before_residual:
                hidden_states = self.hidden_norm(hidden_states)
                residual = hidden_states
            else:
                residual = hidden_states
                hidden_states = self.hidden_norm(hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Eagle3Model(nn.Module):
    """EAGLE3 的"模型"部分：`combine_hidden_states` + 若干解码层 + 末尾 norm。"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        })

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        eagle_config = config.get("eagle_config") or {}
        # 检查点里没有 `embed_tokens.*`（EAGLE 与 target 共享词表嵌入）→ 加载时跳过它
        self.tie_embeddings = False
        self.use_aux_hidden_state = bool(
            eagle_config.get("use_aux_hidden_state",
                             config.get("use_aux_hidden_state", True)))
        self.norm_before_fc = bool(eagle_config.get("norm_before_fc",
                                                    config.get("norm_before_fc", False)))
        self.norm_output = bool(eagle_config.get("norm_output",
                                                 config.get("norm_output", False)))
        self.norm_before_residual = bool(eagle_config.get("norm_before_residual",
                                                          config.get("norm_before_residual",
                                                                     False)))
        # 多个辅助层的特征拼接后投影：fc 的输入维度 = target_hidden_size × 辅助层数。
        # 层数优先取显式配置，其次由 `fc_input_size` 反推（真实 checkpoint 的 fc 是 (hidden, 3*hidden)）。
        aux_ids = (eagle_config.get("eagle_aux_hidden_state_layer_ids")
                   or config.get("eagle_aux_hidden_state_layer_ids"))
        self.target_hidden_size = int(config.get("target_hidden_size", self.hidden_size))
        num_aux = config.get("num_aux_layers", config.get("num_aux_hidden_states"))
        if num_aux is None and config.get("fc_input_size"):
            num_aux = int(config["fc_input_size"]) // self.target_hidden_size
        if num_aux is None and aux_ids:
            num_aux = len(aux_ids)
        self.num_aux_layers = int(num_aux if num_aux else 3)

        use_qk_norm = config.get("model_type") in ("qwen3", "qwen3_moe") or bool(
            config.get("use_qk_norm", False))
        self.embed_tokens = VocabParallelEmbedding(config["vocab_size"], self.hidden_size,
                                                   prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleList([
            Eagle3DecoderLayer(config, prefix=maybe_prefix(prefix, f"layers.{index}"),
                               layer_idx=index, use_qk_norm=use_qk_norm,
                               norm_before_residual=self.norm_before_residual)
            for index in range(config.get("num_hidden_layers", 1))
        ])
        self.norm = RMSNorm(self.hidden_size, eps=config.get("rms_norm_eps", 1e-6))

        if self.use_aux_hidden_state:
            fc_input_size = self.target_hidden_size * self.num_aux_layers
            self.fc = nn.Linear(fc_input_size, self.hidden_size, bias=False)
            self.input_norm = (RMSNorm(fc_input_size, eps=config.get("rms_norm_eps", 1e-6))
                               if self.norm_before_fc else None)
            if config.get("fc_norm", False):
                self.fc_norm = nn.ModuleList([
                    RMSNorm(self.target_hidden_size, eps=config.get("rms_norm_eps", 1e-6))
                    for _ in range(self.num_aux_layers)])
            else:
                self.fc_norm = None
        else:
            self.fc = None
            self.input_norm = self.fc_norm = None

    # -------- 特征融合（上游同名方法）--------

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """把 target 的**多个辅助层**特征融合成 draft 的 hidden 大小。

        上游 `qwen3_eagle3.py:369-389` 逐条对应：`input_norm`（可选，整段）→ `fc_norm`（可选，
        逐块）→ `fc`。**不允许**用"平均/直接拼接"替代这个投影（需求 063 §3.2）。
        """
        if not self.use_aux_hidden_state:
            return hidden_states
        if self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(self.num_aux_layers, dim=-1)
            hidden_states = torch.cat([norm(chunk)
                                       for norm, chunk in zip(self.fc_norm, chunks)], dim=-1)
        return self.fc(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                hidden_states: torch.Tensor, input_embeds: torch.Tensor | None = None):
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, input_embeds, hidden_states, residual)
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        # 上游：`aux_output = hidden_states if norm_output else hidden_prenorm`
        return hidden_states, (hidden_states if self.norm_output else hidden_prenorm)

    def load_weights(self, weights) -> set[str]:
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        skip = ["embed_tokens."] if self.tie_embeddings else None
        return AutoWeightsLoader(self, skip_prefixes=skip).load_weights(
            weights, mapper=self.hf_to_vllm_mapper)


@register_model("Eagle3Qwen3ForCausalLM")
@register_model("Eagle3LlamaForCausalLM")
@register_model("LlamaForCausalLMEagle3")
class Eagle3ForCausalLM(nn.Module):
    """EAGLE3 的顶层：`model` + draft 词表的 `lm_head`（+ 异构词表映射 `d2t`/`t2d`）。"""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = Eagle3Model(config, prefix=maybe_prefix(prefix, "model"))
        draft_vocab_size = int(config.get("draft_vocab_size", config["vocab_size"]))
        self.draft_vocab_size = draft_vocab_size
        self.target_vocab_size = int(config["vocab_size"])
        self.lm_head = ParallelLMHead(draft_vocab_size, config["hidden_size"],
                                      prefix=maybe_prefix(prefix, "lm_head"))
        # 异构词表（TLI）：d2t = draft id → target id；t2d 标出 target 词表里哪些 id 属于 draft 空间。
        # 本关只用它把**采样出来的 draft id** 映射回 target id（完整 TLI 采样空间语义属 67 关）。
        self.d2t = None
        self.t2d = None

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states)

    def model_returns_tuple(self) -> bool:
        return True

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def map_draft_ids_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """draft 词表 id → target 词表 id（没有 d2t 时是恒等映射）。"""
        if self.d2t is None:
            return draft_ids
        return self.d2t.to(draft_ids.device)[draft_ids]

    def load_weights(self, weights) -> set[str]:
        """权重名映射 + 跳过/接收 `d2t`/`t2d`。

        真实 checkpoint 的 `fc.weight`/`midlayer.*`/`norm.weight` 由 `AutoWeightsLoader` +
        `Eagle3Model.hf_to_vllm_mapper` 路由；`d2t`/`t2d` 是 buffer（不是参数），单独收；
        `lm_head.weight` 是 draft 词表那一份，必须真的加载（不能被 tie 跳过）。
        """
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        buffers = {}
        model_weights, head_weights = [], []
        has_embed = False
        for name, tensor in weights:
            if name in ("d2t", "t2d"):
                buffers[name] = tensor
                continue
            # 检查点把 draft 的 fc/norm 放在**顶层**、那唯一一层叫 `midlayer.*`；
            # 模型里它们在 `model.` 下（上游的权重名映射同样要做这一步）。
            if name.startswith("midlayer."):
                name = "model.layers.0." + name[len("midlayer."):]
            elif name.startswith(("fc.", "norm.")):
                name = "model." + name
            if name.startswith("model.embed_tokens."):
                has_embed = True
            if name.startswith("model."):
                model_weights.append((name[len("model."):], tensor))
            else:
                head_weights.append((name, tensor))

        # `embed_tokens` 不在检查点里 → 与 target 共享（上游 `_maybe_share_embeddings` 同款条件：
        # 不是"shape 相同就共享"，而是"检查点里没有这一份"）
        self.model.tie_embeddings = not has_embed
        loaded = set()
        if model_weights:
            loaded |= {f"model.{name}" for name in self.model.load_weights(iter(model_weights))}
        if head_weights:
            loader = AutoWeightsLoader(self, skip_prefixes=["model."] if model_weights else None)
            loaded |= loader.load_weights(iter(head_weights))
        if "d2t" in buffers:
            self.d2t = buffers["d2t"].to(torch.int64)
        if "t2d" in buffers:
            self.t2d = buffers["t2d"].to(torch.bool)
        return loaded | set(buffers)

    def share_embeddings(self, target_model) -> None:
        """与 target 共享词表嵌入（上游 `_maybe_share_embeddings`）：**只在检查点缺这一份时**做。"""
        if not getattr(self.model, "tie_embeddings", False):
            return
        self.model.embed_tokens = target_model.embed_input_ids.__self__ \
            if hasattr(target_model.embed_input_ids, "__self__") else target_model.model.embed_tokens
