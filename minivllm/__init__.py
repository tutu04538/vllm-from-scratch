"""minivllm：对齐 vLLM V1 架构的文本生成子集（原第五十七关，含 204/205 验收修复）。

仓库里**唯一**的实现。旧的 `stepNN/` 代码目录已删除（历史在 git 与 `docs/` 里），后续改动
只在这个包上做。包名不叫 `vllm` 是刻意的：对照脚本要在同一个进程里 `import vllm`
（site-packages 的真 vLLM），同名会静默遮蔽。

目标不是再写一个推理框架，而是**做一个能逐层映射到本机 vLLM 的、可运行的文本生成子集**：
每一层都要能回答"谁拥有这份状态、谁可以改它、模块之间传什么、对应 vLLM 哪个类、去掉它会
出现什么问题"。

模块（与 vLLM 的对应关系写在各自文件头）：

    config.py                 ModelConfig / CacheConfig / SchedulerConfig / DeviceConfig / VllmConfig
    sampling_params.py        请求级采样参数（不可复制到执行侧的活状态不在这里）
    request.py                Request / RequestStatus（Scheduler 持有的请求状态）
    outputs.py                三层协议：EngineCoreRequest / ModelRunnerOutput / EngineCoreOutput / RequestOutput
    core/kv_cache_manager.py  KV 控制面入口：命中查询、分配、发布、释放
    core/kv_cache_coordinator.py  KV group 这一层（本关只有一组，退化实现）
    core/single_type_kv_cache_manager.py  请求 → 逻辑块：核算容量、发布完整块、查命中
    core/block_pool.py        物理块池：引用计数、空闲队列（LRU 淘汰序）、hash 索引
    core/kv_cache_utils.py    块元数据、空闲双向链表、链式块 hash
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
    sample/metadata.py        按行组织的采样参数（Sampler 不接 Request）
    sample/sampler.py         顺序：惩罚 → 约束 → greedy/random 分流
    sample/rejection_sampler.py  投机验证：Triton 批量拒绝采样（59）
    sample/ops/*              三种惩罚、top-k/top-p 筛选与指数竞赛抽样
    spec_decode/metadata.py   投机验证的索引（两个坐标系）
    testing/torch_rejection_sampler.py  **只给测试**的 Torch 参考版验证
    spec_decode/ngram_proposer.py / draft_model.py  两种提议者（历史匹配 / 小模型）
    testing/fake_runner.py    **只给测试**的脚本化 Runner

依赖方向单向：

    config / sampling_params / request / outputs
        → core/kv_cache_manager → core/sched
        → layers / attention / models / model_loader
        → worker / executor
        → engine

**明确不做**（各自属于后面的段落，都在代码里用"明确报错"或注释标出，不假装已完成）：

- 异步/多进程/指标、logprobs、EAGLE/MTP；
- 本包 **只支持 TP=1**（`layers/linear.py` 名字叫 Parallel 是为了源码映射，没有通信）；
  权重只读本地 safetensors（单文件或带 index 的分片），不做 HF hub 下载；
- 只有一个 KV group（`core/kv_cache_coordinator.py` 是单组实现，不假装支持混合 KV）；
- 前缀缓存的**发布时机**与 vLLM 不同（本关在结果处理之后发布，见 `core/kv_cache_manager.py`
  与 docs/step57c_kv_and_prefix.md 的差异账本）。
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
