"""模型层（对应 vLLM `model_executor/layers/`）：只实现本关用到的子集，**TP=1、无通信、无量化**。

    linear.py             QKVParallelLinear / MergedColumnParallelLinear / ColumnParallelLinear
                          / RowParallelLinear / VocabParallelEmbedding / ParallelLMHead
                          （带各自的 weight_loader）
    layernorm.py          RMSNorm（融合残差相加的签名）
    rotary_embedding.py   get_rope / RotaryEmbedding / Gemma4RotaryEmbedding
                          （共享紧凑 cos/sin 表，绝对位置）
    ops/rotary_embedding.py 融合原地 Q/K 旋转（CUDA Triton + Torch 参考）
    ops/layernorm.py    融合 RMSNorm / add-RMSNorm（CUDA Triton + Torch 参考）
    ops/activation.py   融合 GeGLU / SwiGLU（CUDA Triton + Torch 参考）
    activation.py         GeluAndMul / SiluAndMul（gate|up 合并激活）
    logits_processor.py   词表 GEMM 的收尾：截断词表切片 + logit 缩放（66 关的 Medusa 用）
"""

from .activation import (GeluAndMul, SiluAndMul, SwigluStepAndMul,
                         get_act_and_mul_fn)
from .layernorm import GemmaRMSNorm, RMSNorm
from .linear import (ColumnParallelLinear, MergedColumnParallelLinear, ParallelLMHead,
                     QKVParallelLinear, RowParallelLinear, VocabParallelEmbedding)
from .logits_processor import LogitsProcessor
from .rotary_embedding import (Gemma4RotaryEmbedding, RotaryEmbedding,
                              get_rope, get_rope_cache_stats)

__all__ = ["QKVParallelLinear", "MergedColumnParallelLinear", "ColumnParallelLinear",
           "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead", "RMSNorm",
           "RotaryEmbedding", "Gemma4RotaryEmbedding", "get_rope",
           "get_rope_cache_stats", "GeluAndMul", "SiluAndMul",
           "get_act_and_mul_fn", "LogitsProcessor", "GemmaRMSNorm",
           "SwigluStepAndMul"]
