"""第 48 关：KV 缓存与调度的行为不变重构。

代码按模块拆开，依赖方向是单向的：

    formats/  外部目录 -> 内部配置与权重名（只认 json 和 safetensors）
    cache     分块 KV 池（按需分配）、前缀缓存、LRU
    attention 分页 attention：逐行 kernel 与分块 kernel 两条路径 + 元数据缓冲
    norm      融合 RMSNorm 的 Triton kernel
    rope      融合 RoPE 的 Triton kernel
    model     RoPE / RMSNorm / DecoderLayer（QKV 与 gate/up 各一次 GEMM）/ TinyCausalLM
    sampler   采样
    scheduler 本轮跑谁、跑几个 token
    engine    Engine：把上面这些装起来，并提供目录加载入口

上层只依赖下层，model 不认请求、scheduler 不认模型。
"""

from .attention import (AttentionMetadata, _paged_attention_kernel,
                        _tiled_paged_attention_kernel, paged_attention,
                        tiled_paged_attention)
from .cache import CacheConfig, InfeasibleRequest, KVCachePool, SequenceConfig, _stable_hash
from .engine import (COMPATIBLE_FORMAT_VERSIONS, FORMAT_VERSION,
                     MODEL_CONFIG_NAME, MODEL_DTYPE, MODEL_TYPE, MODEL_WEIGHTS_NAME,
                     Engine, build_model_from_config, load_model_config,
                     load_model_weights)
from .formats import GENERATION_CONFIG_NAME, native as native_format
from .formats import qwen3 as qwen3_format
from .formats.native import save_model
from .model import (DEFAULT_EOS_TOKEN_IDS, DecoderLayer, DummyModel, RMSNorm, RotaryEmbedding,
                    TinyCausalLM, _rotate_half)
from .norm import rms_norm
from .rope import rope
from .sampling import SamplingParams, SamplingState, TorchSampler, apply_penalties
from .sampler import Sampler
from .scheduler import Scheduler

__all__ = [
    "Engine", "TinyCausalLM", "DecoderLayer", "RMSNorm", "RotaryEmbedding", "DummyModel",
    "AttentionMetadata", "paged_attention", "tiled_paged_attention", "KVCachePool", "CacheConfig", "SequenceConfig", "InfeasibleRequest",
    "Sampler", "Scheduler", "save_model", "load_model_config", "load_model_weights",
    "build_model_from_config", "native_format", "qwen3_format",
    "DEFAULT_EOS_TOKEN_IDS", "GENERATION_CONFIG_NAME", "rms_norm", "rope",
    "SamplingParams", "SamplingState", "TorchSampler", "apply_penalties",
    "FORMAT_VERSION", "COMPATIBLE_FORMAT_VERSIONS", "MODEL_TYPE", "MODEL_DTYPE",
    "MODEL_CONFIG_NAME", "MODEL_WEIGHTS_NAME",
]
