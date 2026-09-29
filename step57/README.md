# step57：对齐 vLLM V1 架构的文本生成子集

第五十七关换方向：不再自己发明协议，而是做一个**能逐层映射到本机 vLLM（0.28.0）的、可运行的
文本生成子集**。每一层都要能回答：谁拥有这份状态？谁可以改它？模块之间传什么（而不是偷偷共享
什么）？对应 vLLM 哪个类、哪个方法？省略了哪些条件？

**当前进度：57A（竖直骨架）**。设计与差异账本见
[`docs/step57a_skeleton.md`](../docs/step57a_skeleton.md)。

| 层 | 文件 | 对应 vLLM |
|---|---|---|
| 配置 | `config.py` / `sampling_params.py` | `vllm/config/*`、`vllm/sampling_params.py` |
| 请求状态 | `request.py` | `v1/request.py` |
| 协议 | `outputs.py`、`core/sched/output.py` | `v1/engine/__init__.py`、`v1/outputs.py`、`v1/core/sched/output.py` |
| KV 控制面 | `core/kv_cache_manager.py` | `v1/core/kv_cache_manager.py` |
| 调度 | `core/sched/{scheduler,request_queue,utils}.py` | `v1/core/sched/*` |
| 编排 | `engine/{core,core_client,output_processor,llm_engine}.py` | `v1/engine/*` |
| 执行部署 | `executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` |
| 执行端 | `worker/worker.py` | `v1/worker/gpu_worker.py` |
| 测试替身 | `testing/fake_runner.py` | 无（只给测试） |

## 怎么用（57A 还没有真实模型）

```python
from step57 import (CacheConfig, LLMEngine, ModelConfig, SchedulerConfig, SamplingParams,
                    UniProcExecutor, VllmConfig, Worker)
from step57.testing.fake_runner import FakeRunner

config = VllmConfig(model_config=ModelConfig(model="dummy", max_model_len=64),
                    cache_config=CacheConfig(block_size=4, num_gpu_blocks=4),
                    scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2))
runner = FakeRunner(tokens={"r1": [11, 12]})          # 只给测试用；生产路径不会退回假执行
engine = LLMEngine(config, UniProcExecutor(config, Worker(config, model_runner=runner)))
engine.add_request("r1", [10, 11, 12], SamplingParams(max_tokens=2, eos_token_id=99))
while engine.has_unfinished_requests():
    for out in engine.step():
        print(out.request_id, out.token_ids, out.finished)
```

## 57A 明确不做

真实模型与 GPU 执行（57B）、KV 物理存储与前缀缓存（57C）、抢占与恢复（57C）、采样与惩罚、
投机（57E）、异步与多进程、指标。这些在代码里都用"明确报错"或注释标出，不假装已完成。

## 验证

```bash
python benchmarks/check_step57_engine_protocol.py    # 23 项：协议与数据契约
python benchmarks/check_step57_request_progress.py   # 29 项：请求进度与停止判定
python benchmarks/check_step57_scheduler_basic.py    # 25 项：统一预算调度
```
