"""第五十七关：对齐 vLLM V1 架构的文本生成子集（57A 竖直骨架 + 57B 真实模型）。

目标不是再写一个推理框架，而是**做一个能逐层映射到本机 vLLM 的、可运行的文本生成子集**：
每一层都要能回答"谁拥有这份状态、谁可以改它、模块之间传什么、对应 vLLM 哪个类、去掉它会
出现什么问题"。

模块（与 vLLM 的对应关系写在各自文件头）：

    config.py                 ModelConfig / CacheConfig / SchedulerConfig / DeviceConfig / VllmConfig
    sampling_params.py        请求级采样参数（不可复制到执行侧的活状态不在这里）
    request.py                Request / RequestStatus（Scheduler 持有的请求状态）
    outputs.py                三层协议：EngineCoreRequest / ModelRunnerOutput / EngineCoreOutput / RequestOutput
    core/kv_cache_manager.py  KV 控制面：块归谁（不含模型计算与 GPU 写入）
    core/sched/output.py      调度数据包：SchedulerOutput / NewRequestData / CachedRequestData
    core/sched/request_queue.py  等待队列（FCFS / priority）
    core/sched/utils.py       停止判定 check_stop
    core/sched/scheduler.py   统一预算调度 + 进度维护 + 结果提交
    engine/core.py            一轮编排：调度 → 执行 → 采样 → 更新
    engine/core_client.py     EngineCoreClient 协议与同进程实现
    engine/llm_engine.py      用户侧入口
    engine/output_processor.py 增量结果 → 用户可见 RequestOutput
    executor/uniproc_executor.py 单进程执行部署
    worker/worker.py          执行端运行环境（设备、模型装载、转发）
    worker/gpu_model_runner.py 一轮怎么跑：镜像 → 输入打包 → 模型 → 采样 → 记账（57B）
    worker/gpu_input_batch.py 批状态缓冲：请求 ↔ batch 行（定长缓冲，原地写）
    worker/block_table.py     块表镜像与 slot_mapping
    model_loader/*            权重读取、打包路由、覆盖检查（57B）
    models/qwen3.py           Qwen3（GQA + q/k norm + RoPE，对应 vLLM 的 qwen2/qwen3）
    layers/* / attention/*    并行线性层、RMSNorm、RoPE、Attention 边界与教学后端（57B）
    sample/sampler.py         最小采样器：贪心 / 温度随机（57D 才做完整采样）
    testing/fake_runner.py    **只给测试**的脚本化 Runner

依赖方向单向：

    config / sampling_params / request / outputs
        → core/kv_cache_manager → core/sched
        → layers / attention / models / model_loader
        → worker / executor
        → engine

**明确不做**（各自属于后面的段落，都在代码里用"明确报错"或注释标出，不假装已完成）：

- KV 前缀缓存与抢占恢复（57C）、完整采样与停止（57D）、投机（57E）、异步/多进程/指标；
- `CacheConfig.enable_prefix_caching=True` 在构造时就拒绝（见 config.py）；
- 本包 **只支持 TP=1**（`layers/linear.py` 名字叫 Parallel 是为了源码映射，没有通信）；
  权重只读本地 safetensors（单文件或带 index 的分片），不做 HF hub 下载。
"""

from .config import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                     SpeculativeConfig, VllmConfig)
from .engine import EngineCore, InprocClient, LLMEngine, OutputProcessor
from .outputs import (EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason,
                      ModelRunnerOutput, RequestOutput)
from .executor import UniProcExecutor
from .model_loader import get_model
from .models import Qwen3ForCausalLM
from .request import Request, RequestStatus
from .sample import Sampler
from .sampling_params import SamplingParams
from .worker import GPUModelRunner, Worker

__all__ = [
    "LLMEngine", "EngineCore", "InprocClient", "OutputProcessor",
    "UniProcExecutor", "Worker", "GPUModelRunner",
    "Request", "RequestStatus", "SamplingParams",
    "EngineCoreRequest", "EngineCoreOutput", "EngineCoreOutputs", "ModelRunnerOutput",
    "RequestOutput", "FinishReason",
    "VllmConfig", "ModelConfig", "CacheConfig", "SchedulerConfig", "DeviceConfig",
    "SpeculativeConfig",
    "get_model", "Qwen3ForCausalLM", "Sampler",
]
