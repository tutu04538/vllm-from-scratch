"""模型（对应 vLLM `model_executor/models/`）：Qwen3/EAGLE/DFlash/DSpark/DFlash2 等
已有模型，以及 80E.E1 的 Gemma4 assistant 模型层（Q-only、跨宽度投影）。

`models/registry.py` 是"模型名 → 类"的查表（vLLM 用 `ModelRegistry` + 装饰器注册，本关一个
字典就够，但要显式存在：加载器不该自己做 if/elif）。
"""

from .extract_hidden_states import (CacheOnlyAttentionBackend, CacheOnlyAttentionLayer,
                                    CacheOnlyAttentionMetadata, ExtractHiddenStatesModel)
from .gemma4_mtp import (Gemma4MLP, Gemma4MTP, Gemma4MTPAttention,
                         Gemma4MTPDecoderLayer, Gemma4MTPMaskedEmbedder,
                         Gemma4MultiTokenPredictor)
from .gemma4 import Gemma4Attention, Gemma4DecoderLayer, Gemma4ForCausalLM, Gemma4TextModel
from .medusa import Medusa, ResidualBlock
from .qwen3 import Qwen2MLP, Qwen3Attention, Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from .qwen3_mtp import Qwen3MTP, Qwen3MultiTokenPredictor
from .step3p5 import (Step3p5Attention, Step3p5DecoderLayer, Step3p5ForCausalLM,
                      Step3p5MLP, Step3p5MoE, Step3p5Model, Step3p5MTP,
                      Step3p5MTPLayer, Step3p5MultiTokenPredictor, Step3p5SharedHead)
from .qwen3_eagle3 import Eagle3Attention, Eagle3DecoderLayer, Eagle3ForCausalLM, Eagle3Model
from .registry import get_model_class, register_model

__all__ = ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention", "Qwen2MLP",
           "Eagle3ForCausalLM", "Eagle3Model", "Eagle3DecoderLayer", "Eagle3Attention",
           "ExtractHiddenStatesModel", "CacheOnlyAttentionLayer", "CacheOnlyAttentionBackend",
           "CacheOnlyAttentionMetadata",
           "Qwen3MTP", "Qwen3MultiTokenPredictor",
           "Step3p5ForCausalLM", "Step3p5Model", "Step3p5DecoderLayer", "Step3p5Attention",
           "Step3p5MLP", "Step3p5MoE", "Step3p5MTP", "Step3p5MTPLayer",
           "Step3p5MultiTokenPredictor", "Step3p5SharedHead",
           "Gemma4MTP", "Gemma4MultiTokenPredictor", "Gemma4MTPDecoderLayer",
           "Gemma4MTPAttention", "Gemma4MTPMaskedEmbedder", "Gemma4MLP",
           "Gemma4ForCausalLM", "Gemma4TextModel", "Gemma4DecoderLayer",
           "Gemma4Attention",
           "Medusa", "ResidualBlock",
           "get_model_class", "register_model"]
