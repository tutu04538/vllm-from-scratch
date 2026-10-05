"""原生 MTP 模型（对应 vLLM `model_executor/models/qwen3_next_mtp.py`）。

**MTP 是什么**：训练时让 target 模型"一次想好几个 token"而多长出来的一层预测头。权重就在
**target 自己的 checkpoint 里**（不是另开一个小模型），所以 65 关要解决的是"怎么从同一个文件里
只挑出这一层、改名、与 target 共享词表，并把它接进通用的迭代提议循环"。

    Qwen3MTP                        顶层：predictor + 自己的 lm_head + compute_logits(spec_step_idx)
      model: Qwen3MultiTokenPredictor
        embed_tokens              词嵌入（检查点里有一份共享的）
        pre_fc_norm_embedding     对 embedding 做 RMSNorm
        pre_fc_norm_hidden        对上一枚的 hidden 做 RMSNorm
        fc                        `2H → H` 的拼接投影（ColumnParallelLinear，与上游同款）
        layers[]                  MTP 层（一层 = target 的 decoder layer），按 spec_step_idx 选
        norm                      最终 RMSNorm

**每一步吃什么**（这是 MTP 与"连调两次 lm_head"的本质区别）：

    hidden_t = norm( block( fc( [ norm_emb(embed(token_t)) ‖ norm_hidden(hidden_{t-1}) ] ) ) )

`token_t` 是**上一枚草稿**、`hidden_{t-1}` 是**上一步算出的 hidden**——所以第 2 枚草稿真的条件在
第 1 枚之上（不是复读同一个 argmax）。第一遍的 `token/hidden` 来自 target 本轮的输入与最后一层
hidden（与 EAGLE 的 `(h_i, t_{i+1}) → t_{i+2}` 配对同构）。

**与上游的差异**（逐条见 docs/step65_alignment.md §3）：

1. 上游这一步是 `Qwen3NextDecoderLayer`（**混合注意力**：GatedDeltaNet + full attention）；
   本仓库只有稠密 Qwen3，所以 `layers[]` 用 `Qwen3DecoderLayer`。因此注册名用**我们自己的**
   `Qwen3MTPModel`，不冒充上游的 `Qwen3NextMTP`——真 Qwen3-Next checkpoint 的
   `linear_attn.*` 权重在这里会**明确报"没有这个参数"**，不会静默跑出一个错的模型。
2. 上游的 `Qwen3NextMTP` 只返回**一个** hidden（`model_returns_tuple() == False`），
   `compute_logits()` 直接在它上面过 lm_head；DeepSeek/Kimi 那一族的 MTP 返回
   `(pre_norm, post_norm)` 两个（`model_returns_tuple() == True`）。本仓库实现的是前者
   （与所选模型源码一致），后者在别名表里登记为未适配。
3. 上游 `spec_step_idx` 在通用 V1 提议路径里恒为 0（只有 step3p5 的专用提议者会递进）；
   本仓库照抄，所以 `num_nextn_predict_layers > 1` 时通用路径复用**第 0 个**模块——
   多模块的调度/状态行为属 80 关。
4. 上游加载时还会 `maybe_fuse_shared_experts`（MoE 的 shared expert 融合）与 MoE 专家参数映射；
   稠密 Qwen3 没有 MoE，这一步不存在。
"""

import torch
from torch import nn

from ..layers import (ColumnParallelLinear, ParallelLMHead, RMSNorm, VocabParallelEmbedding)
from .qwen3 import Qwen3DecoderLayer, Qwen3Model, maybe_prefix
from .registry import register_model
from .utils import get_spec_layer_idx_from_weight_name

# 绝对层号命名那一派（DeepSeek/GLM/MiMo...）里，哪些名字是"胶水"（属于 spec 模块本身、
# 要在改写时提到顶层），其余都算 block（self_attn/mlp/两个 layernorm）→ 改成相对层号。
# **必须按"名字段"前缀匹配，不能用 `in`**：`"norm" in "input_layernorm.weight"` 会命中，
# 于是 block 的输入归一化会被错误地提到顶层（上游的名单是 enorm/hnorm/eh_proj 这种天然不冲突的
# 名字，所以它敢用 `in`；我们的名字里 `norm` 是别人的子串，见 docs §3）。
SPEC_LAYER_GLUE_PREFIXES = ("embed_tokens.", "fc.", "pre_fc_norm_embedding.",
                            "pre_fc_norm_hidden.", "norm.")


class Qwen3MultiTokenPredictor(nn.Module):
    """MTP 的 predictor（对应上游 `Qwen3NextMultiTokenPredictor`）。

    与上游同名同职责：embedding、两侧归一化、拼接投影、MTP 层、最终归一化。
    """

    # 与 target 同一套打包映射（q/k/v → qkv_proj、gate/up → gate_up_proj）。
    # 上游也是复用 `Qwen3NextModel.hf_to_vllm_mapper`。
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.vocab_size = int(config["vocab_size"])
        # MTP 层的**绝对**层号起点：target 有多少层，spec 层就从那之后开始编号
        self.mtp_start_layer_idx = int(config["num_hidden_layers"])
        self.num_mtp_layers = int(config.get("num_nextn_predict_layers") or 1)
        hidden_size = int(config["hidden_size"])
        eps = config.get("rms_norm_eps", 1e-6)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, hidden_size, prefix=maybe_prefix(prefix, "embed_tokens"))
        # 上游：`ColumnParallelLinear(hidden * 2, hidden, gather_output=True, bias=False,
        # return_bias=False)`；TP=1 下 gather_output 无意义，本仓库的这个类没有这个参数
        self.fc = ColumnParallelLinear(
            hidden_size * 2, hidden_size, bias=False, return_bias=False,
            prefix=maybe_prefix(prefix, "fc"))
        # 相对层号：`layers.{i}` 对应绝对层号 `num_hidden_layers + i`
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, prefix=maybe_prefix(prefix, f"layers.{index}"))
            for index in range(self.num_mtp_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def load_weights(self, weights) -> set[str]:
        """自己这一组权重自己路由（上游 `Qwen3NextMultiTokenPredictor.load_weights` 同款）。

        打包映射写在**这一层**（与 `Qwen3Model.load_weights` 同一个理由：只有模型自己知道
        `qkv_proj` 里装的是 q/k/v / `gate_up_proj` 里是 gate/up），顶层加载器只负责"哪些名字要丢"。
        """
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def forward(self, input_ids, positions, hidden_states, spec_step_idx: int = 0):
        """上游 `Qwen3NextMultiTokenPredictor.forward()` 的逐行对应（顺序一致）。

        `hidden_states` 是**上一枚**的 hidden（第一遍来自 target，之后来自上一步的返回值）；
        `input_ids` 是**上一枚 token**（第一遍是 target 本轮左移一格后的输入）。
        """
        inputs_embeds = self.embed_input_ids(input_ids)
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        # 拼接顺序是 [embedding ‖ hidden]（上游 `torch.cat([inputs_embeds, hidden_states], dim=-1)`）
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        # 层选择：`spec_step_idx % num_mtp_layers`（上游同款；通用路径 spec_step_idx 恒为 0）
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        hidden_states, residual = mtp_layer(positions=positions,
                                            hidden_states=hidden_states, residual=None)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_model("Qwen3MTPModel")
class Qwen3MTP(nn.Module):
    """MTP 顶层（对应上游 `Qwen3NextMTP`）：predictor + 词表头。

    它与 target **共享** `embed_tokens`/`lm_head` 的**权重内容**（检查点里各有一份，加载时都吃），
    但不是同一个 Parameter 对象（上游也是各自建层、各自加载）。
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MultiTokenPredictor(config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(int(config["vocab_size"]), int(config["hidden_size"]),
                                      prefix=maybe_prefix(prefix, "lm_head"))

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, hidden_states, spec_step_idx: int = 0):
        return self.model(input_ids, positions, hidden_states, spec_step_idx=spec_step_idx)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        """上游 `Qwen3NextMTP.compute_logits()` 的签名（`spec_step_idx` 保留）。

        上游在它上面套 `LogitsProcessor`（那是 TP/词表分片与 logits 后处理的位置）；本仓库
        TP=1、无后处理，所以就是 `lm_head(hidden)`（与 `Qwen3ForCausalLM.compute_logits` 同款）。
        """
        return self.lm_head(hidden_states)

    # -------- 权重名识别与改写（对应上游的 remap_weight_names / _rewrite_spec_layer_name） --------

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """把"绝对层号"命名的 spec 权重改成 draft 模型内部的名字（上游
        `DeepSeekMTP._rewrite_spec_layer_name()` 的等价物）。

        - 胶水权重（`fc` / `pre_fc_norm_*` / `norm` / `embed_tokens`）→ 提到 `model.` 下；
        - block 权重（`self_attn` / `mlp` / 两个 layernorm）→ 层号改成**相对**层号。

        上游保留绝对层号（它的 `layers` 字典就用绝对层号做键），本仓库的模块路径是相对的
        （`model.layers.0.*`），所以这里多一次减法；映射关系与"胶水提顶层"的分法一致。
        """
        local = spec_layer - self.model.mtp_start_layer_idx
        prefix = f"model.layers.{spec_layer}."
        rest = name[len(prefix):]
        if rest.startswith(SPEC_LAYER_GLUE_PREFIXES):
            return name.replace(prefix, "model.")
        return name.replace(prefix, f"model.layers.{local}.")

    def _spec_weights(self, weights):
        """只留 spec 层与共享词表权重，其余（target 自己的那些层）整份丢掉。

        上游 `Qwen3NextMTP.load_weights()` 里的 `remap_weight_names()` 就是这段：

            if name.startswith("mtp."):     name = name.replace("mtp.", "model.")
            elif not any(key in name for key in ("embed_tokens", "lm_head")): continue

        本仓库在此基础上多支持"绝对层号"命名（DeepSeek 那一派的 `model.layers.{N+i}.*`）：
        两派命名走同一个入口，识别规则来自上游 `get_spec_layer_idx_from_weight_name()`。
        丢掉 target 层是**必须**的：`AutoWeightsLoader` 对未知名字会报错，而 target 的层在这里
        既不是参数也不该被加载。
        """
        for name, weight in weights:
            if name.startswith("mtp."):
                yield name.replace("mtp.", "model.", 1), weight
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                yield self._rewrite_spec_layer_name(spec_layer, name), weight
                continue
            if "embed_tokens" in name or "lm_head" in name:
                # 共享权重：`model.embed_tokens.weight` / `lm_head.weight` 原样交给加载器
                yield name, weight
                continue
            # 其余都是 target 自己的层：与 MTP 无关，丢掉（不计入"未加载参数"）
            continue

    def load_weights(self, weights) -> set[str]:
        from ..model_loader.auto_weights_loader import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(self._spec_weights(weights))
