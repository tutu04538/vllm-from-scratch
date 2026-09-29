"""第五十七关（57A）：对齐 vLLM V1 架构的**竖直骨架**。

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
    worker/worker.py          执行端运行环境（57A 没有真实模型）
    testing/fake_runner.py    **只给测试**的脚本化 Runner

依赖方向单向：

    config / sampling_params / request / outputs
        → core/kv_cache_manager → core/sched
        → executor / worker
        → engine

**57A 明确不做**（各自属于后面的段落，且都在代码里用"明确报错"或注释标出，不假装已完成）：

- 真实模型加载与 GPU 执行（57B）、KV 物理存储与前缀缓存（57C）、
  抢占与恢复（57C）、投机（57E）、采样与惩罚（57E）、异步/多进程/指标；
- `CacheConfig.enable_prefix_caching=True` 在构造时就拒绝（见 config.py）。
"""

from .config import (CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig,
                     SpeculativeConfig, VllmConfig)
from .engine import EngineCore, InprocClient, LLMEngine, OutputProcessor
from .outputs import (EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest, FinishReason,
                      ModelRunnerOutput, RequestOutput)
from .executor import UniProcExecutor
from .request import Request, RequestStatus
from .sampling_params import SamplingParams
from .worker import Worker

__all__ = [
    "LLMEngine", "EngineCore", "InprocClient", "OutputProcessor",
    "UniProcExecutor", "Worker",
    "Request", "RequestStatus", "SamplingParams",
    "EngineCoreRequest", "EngineCoreOutput", "EngineCoreOutputs", "ModelRunnerOutput",
    "RequestOutput", "FinishReason",
    "VllmConfig", "ModelConfig", "CacheConfig", "SchedulerConfig", "DeviceConfig",
    "SpeculativeConfig",
]
