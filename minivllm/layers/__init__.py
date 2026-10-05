"""模型层（对应 vLLM `model_executor/layers/`）：只实现本关用到的子集，**TP=1、无通信、无量化**。

    linear.py             QKVParallelLinear / MergedColumnParallelLinear / ColumnParallelLinear
                          / RowParallelLinear / VocabParallelEmbedding / ParallelLMHead
                          （带各自的 weight_loader）
    layernorm.py          RMSNorm（融合残差相加的签名）
    rotary_embedding.py   RotaryEmbedding（预计算 cos/sin 表，绝对位置）
    activation.py         SiluAndMul（gate|up 合并激活）
"""

from .activation import SiluAndMul
from .layernorm import RMSNorm
from .linear import (ColumnParallelLinear, MergedColumnParallelLinear, ParallelLMHead,
                     QKVParallelLinear, RowParallelLinear, VocabParallelEmbedding)
from .rotary_embedding import RotaryEmbedding

__all__ = ["QKVParallelLinear", "MergedColumnParallelLinear", "ColumnParallelLinear",
           "RowParallelLinear", "VocabParallelEmbedding", "ParallelLMHead", "RMSNorm",
           "RotaryEmbedding", "SiluAndMul"]
