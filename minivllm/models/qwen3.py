"""Qwen3 模型（对应 vLLM `model_executor/models/qwen3.py` 与 `qwen2.py` 的子集）。

类组织照搬 vLLM：

    Qwen3ForCausalLM
      model: Qwen3Model
        embed_tokens      VocabParallelEmbedding
        layers[]          Qwen3DecoderLayer
            input_layernorm / post_attention_layernorm   RMSNorm
            self_attn     Qwen3Attention（qkv_proj / q_norm / k_norm / rotary_emb / attn / o_proj）
            mlp           Qwen2MLP（gate_up_proj / act_fn / down_proj）
        norm              RMSNorm
      lm_head             ParallelLMHead

**forward 的签名里没有 Request / KV 池 / sampling_params**（198 §6）：`forward(input_ids, positions)
-> hidden_states`。为什么先给 hidden 再单独 `compute_logits()`？因为 hidden 还能给别的路径用
（投机、pooling），而 LM head 是词表大小的一次 GEMM——只在需要采样的行上做。

Qwen3 相对 Qwen2 的两处区别（都在这里实现）：**q/k 各有一个 RMSNorm**（作用在 head 维上，
不是 hidden 维），以及 **GQA**（q 头数 ≠ kv 头数，所以不能三等分 qkv 投影）。
"""

import torch
from torch import nn

from ..attention import Attention
from ..layers import (MergedColumnParallelLinear, ParallelLMHead, QKVParallelLinear,
                      RMSNorm, RotaryEmbedding, RowParallelLinear, SiluAndMul,
                      VocabParallelEmbedding)
from ..model_loader.auto_weights_loader import WeightsMapper


def maybe_prefix(prefix: str, name: str) -> str:
    """拼名字前缀：空前缀时不留下开头的点（对应 vLLM `vllm/utils/torch_utils.py::maybe_prefix`）。

    为什么在意一个点：`Attention.layer_name` 就是模块路径，它必须与 `named_modules()` 报出来的
    路径**逐字相同**——Runner 靠这个对应关系绑 KV、挂 metadata。写成 `f"{prefix}.attn"` 时，
    顶层 prefix 是空串，层名就变成 `.model.layers.0...`，多一个点，对应关系当场断掉。
    """
    return name if not prefix else f"{prefix}.{name}"


class Qwen2MLP(nn.Module):
    """Qwen3 复用 Qwen2 的 MLP（vLLM 也是这么复用的：Qwen3Model 继承 Qwen2Model）。"""

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError(f"只支持 silu 激活，收到 {config.get('hidden_act')!r}")
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size],
            prefix=maybe_prefix(prefix, "gate_up_proj"))
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False,
                                           prefix=maybe_prefix(prefix, "down_proj"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden_size // num_heads)
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim

        self.qkv_proj = QKVParallelLinear(hidden_size, head_dim, num_heads, num_kv_heads,
                                          bias=config.get("attention_bias", False),
                                          prefix=maybe_prefix(prefix, "qkv_proj"))
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False,
                                        prefix=maybe_prefix(prefix, "o_proj"))
        rope_parameters = config.get("rope_parameters") or {}
        self.rotary_emb = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=config["max_position_embeddings"],
            base=rope_parameters.get("rope_theta", config.get("rope_theta", 10000.0)),
        )
        self.attn = Attention(num_heads=num_heads, head_size=head_dim,
                              num_kv_heads=num_kv_heads, layer_name=maybe_prefix(prefix, "attn"))
        # Qwen3 特有的 q/k norm：作用在**每个 head 的 head_dim 维**上
        eps = config.get("rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.qkv_proj.kv_size, self.qkv_proj.kv_size],
                            dim=-1)
        # 按 head 切开做 q/k norm（这就是 Qwen3 与 Qwen2 的第一处区别）
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """一层解码器。`forward(positions, hidden_states, residual) -> (hidden_states, residual)`
    ——残差沿两条链传下去，不在层内提前相加（对应 vLLM 的融合 add-RMSNorm 约定）。

    **第一层要特判 `residual is None`**（vLLM 的 `Qwen2DecoderLayer` 也是这样）：残差流的起点
    就是 embedding 的输出，此时"融合相加"没东西可加，只能走 `RMSNorm` 的单参数分支——
    它返回的是**裸张量**而不是 `(out, residual)`，直接解包会报 "too many values to unpack"。
    """

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        eps = config.get("rms_norm_eps", 1e-6)
        self.self_attn = Qwen3Attention(config, prefix=maybe_prefix(prefix, "self_attn"))
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


class Qwen3Model(nn.Module):
    # 检查点名 → 模型名的打包映射。**放在这里而不是加载器里**：只有模型自己知道它的
    # qkv_proj 里装的是 q/k/v（vLLM 也是把 hf_to_vllm_mapper 挂在 Qwen2Model 上）。
    # 外层 Qwen3ForCausalLM 的加载器把 `model.*` 这一组交给本类的 load_weights()，
    # 映射就在这一层生效——所以 `lm_head.*` 不经过这里（它在外面就被跳过了）。
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_stacked={
        # 权重名后缀: (目标参数名后缀, 第几段)
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    })

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.embed_tokens = VocabParallelEmbedding(config["vocab_size"],
                                                   config["hidden_size"],
                                                   prefix=maybe_prefix(prefix, "embed_tokens"))
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, prefix=maybe_prefix(prefix, f"layers.{index}"))
            for index in range(config["num_hidden_layers"])
        ])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        # 63 关（EAGLE3）：要输出的辅助层编号。空 tuple = 不采集（默认，零开销）。
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers) -> None:
        """上游 `SupportsEagle3.set_aux_hidden_state_layers`：指定哪些层顺带输出 hidden states。"""
        self.aux_hidden_state_layers = tuple(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        """上游默认值 `interfaces.py:1580-1601`：`(2, num_layers // 2, num_layers - 3)`。

        为什么是这三层：低/中/高三段各取一层做特征融合（EAGLE3 的多层融合）；
        真实 draft 的 `fc.weight` 形状 `(hidden, 3*hidden)` 正好对上这个层数。
        """
        num_layers = len(self.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                capture_aux: bool = False):
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        aux_hidden_states: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if capture_aux and index in self.aux_hidden_state_layers:
                # 上游 `interfaces.py:1506`：采的是**残差流**（hidden + residual），不是裸 hidden
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture_aux:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights) -> set[str]:
        """自己的权重自己路由：`layers.*` / `norm.*` / `embed_tokens.*` 交给 AutoWeightsLoader，
        打包映射由 `hf_to_vllm_mapper` 生效。名字是**相对本模块**的（顶层加载器剥掉了 `model.`）。
        """
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3ForCausalLM(nn.Module):
    # 打包映射：检查点里的分离权重 → 模型里的打包参数（AutoWeightsLoader 用它生成 shard_id）
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config["vocab_size"], config["hidden_size"],
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.tie_word_embeddings = bool(config.get("tie_word_embeddings", False))
        if self.tie_word_embeddings:
            # 共享同一个 Parameter（vLLM 在 __init__ 里就这么做）。**必须在加载权重之前完成**：
            # 之后 lm_head.weight 就是 embed_tokens.weight，检查点里那份 lm_head.weight 要跳过。
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

    # -------- 计算接口（198 §6）--------

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                capture_aux: bool = False):
        return self.model(input_ids, positions, capture_aux=capture_aux)

    def set_aux_hidden_state_layers(self, layers) -> None:
        self.model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.model.get_eagle3_default_aux_hidden_state_layers()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """只对**需要采样的行**调用（调用方先按行号选 hidden_states）——词表 GEMM 是最贵的一步。"""
        return self.lm_head(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # -------- 权重加载 --------

    def load_weights(self, weights) -> set[str]:
        """顶层加载器：只做委派 + 决定**跳过什么**。

        为什么要 `skip_prefixes=["lm_head."]`：Qwen3-1.7B 的检查点里**同时**有
        `model.embed_tokens.weight` 和 `lm_head.weight`（内容相同），而模型里这两者是同一个
        Parameter（见 `__init__` 的 tie）。不跳过的话，第二份会把第一份原地再覆盖一遍——
        数值上无害，但"哪份是真相"就说不清了；跳过之后只有一条来源。

        名字路由（q/k/v → qkv_proj）不在这里，而在 `Qwen3Model.load_weights`：`model.` 这一组
        会被 AutoWeightsLoader 整组交给子模块，映射写在与参数最近的那一层。
        """
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["lm_head."] if self.tie_word_embeddings else None)
        return loader.load_weights(weights)
