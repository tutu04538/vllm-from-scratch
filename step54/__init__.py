"""第 54 关：随机采样投机解码与拒绝修正。

代码按模块拆开，依赖方向是单向的：

    formats/   外部目录 -> 内部配置与权重名（只认 json 和 safetensors）
    request    请求自身的状态：历史、输出、块表、惩罚计数
    cache      分块 KV 池（按需分配）、前缀缓存、LRU
    attention  分页 attention：逐行 kernel 与分块 kernel 两条路径 + 元数据缓冲
    norm       融合 RMSNorm 的 Triton kernel
    rope       融合 RoPE 的 Triton kernel
    model      RoPE / RMSNorm / DecoderLayer（QKV 与 gate/up 各一次 GEMM）/ TinyCausalLM
    sampling   采样原语与后端：参数/状态/三种惩罚/温度与 top-k,p/目标分布、TorchSampler
    speculative  投机解码的纯函数：n-gram 提议 + 贪心/随机两种验证
    validation 配置与后端的组合校验（构造阶段就报错）
    loading    模型装配与目录加载
    sample_loop  采样执行层：行映射 -> 三条路径 -> 验证与回滚 -> 交回提交
    scheduler  本轮跑谁、跑几个 token
    engine     Engine：把上面这些装起来，并按轮编排（调度 -> forward -> 采样）

上层只依赖下层，model 不认请求、scheduler 不认模型，
sample_loop 也不认 Engine 这个类型（依赖按参数传进去）。
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
from .sampling import (SamplingParams, SamplingState, TorchSampler, apply_penalties,
                       row_distribution)
from .sample_loop import SampleRuntime
from .scheduler import Scheduler
from .speculative import (DraftVerification, propose_ngram, residual_probs,
                          verify_drafts, verify_drafts_random)

__all__ = [
    "Engine", "TinyCausalLM", "DecoderLayer", "RMSNorm", "RotaryEmbedding", "DummyModel",
    "AttentionMetadata", "paged_attention", "tiled_paged_attention", "KVCachePool", "CacheConfig", "SequenceConfig", "InfeasibleRequest",
    "Scheduler", "SampleRuntime", "save_model", "load_model_config", "load_model_weights",
    "build_model_from_config", "native_format", "qwen3_format",
    "DEFAULT_EOS_TOKEN_IDS", "GENERATION_CONFIG_NAME", "rms_norm", "rope",
    "SamplingParams", "SamplingState", "TorchSampler", "apply_penalties", "row_distribution",
    "propose_ngram", "verify_drafts", "verify_drafts_random", "residual_probs",
    "DraftVerification",
    "FORMAT_VERSION", "COMPATIBLE_FORMAT_VERSIONS", "MODEL_TYPE", "MODEL_DTYPE",
    "MODEL_CONFIG_NAME", "MODEL_WEIGHTS_NAME",
]
