"""模型（对应 vLLM `model_executor/models/`）。本关只有 Qwen3（复用 Qwen2 的 MLP 与堆叠逻辑）。

`models/registry.py` 是"模型名 → 类"的查表（vLLM 用 `ModelRegistry` + 装饰器注册，本关一个
字典就够，但要显式存在：加载器不该自己做 if/elif）。
"""

from .extract_hidden_states import (CacheOnlyAttentionBackend, CacheOnlyAttentionLayer,
                                    CacheOnlyAttentionMetadata, ExtractHiddenStatesModel)
from .qwen3 import Qwen2MLP, Qwen3Attention, Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from .qwen3_eagle3 import Eagle3Attention, Eagle3DecoderLayer, Eagle3ForCausalLM, Eagle3Model
from .registry import get_model_class, register_model

__all__ = ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention", "Qwen2MLP",
           "Eagle3ForCausalLM", "Eagle3Model", "Eagle3DecoderLayer", "Eagle3Attention",
           "ExtractHiddenStatesModel", "CacheOnlyAttentionLayer", "CacheOnlyAttentionBackend",
           "CacheOnlyAttentionMetadata",
           "get_model_class", "register_model"]
