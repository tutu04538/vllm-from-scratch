# minivllm：对齐 vLLM V1 架构的文本生成子集

这是仓库里**唯一的实现**（原 `step57/`，含 204/205 两轮验收修复）。旧的 `stepNN/` 代码目录已经
删除，历史记录留在 `docs/` 与 git 历史里；后续改动只在这个包上做。

它不自己发明协议，而是做一个**能逐层映射到本机 vLLM（0.28.0）的、可运行的文本生成子集**。
每一层都要能回答：谁拥有这份状态？谁可以改它？模块之间传什么（而不是偷偷共享什么）？
对应 vLLM 哪个类、哪个方法？省略了哪些条件？

包名不叫 `vllm` 是刻意的：`benchmarks/compare_step57_vllm.py` 等对照脚本要在**同一个进程**里
`import vllm`（site-packages 的真 vLLM），同名会静默遮蔽。

设计与差异账本见
[`docs/step57a_skeleton.md`](../docs/step57a_skeleton.md)（骨架）、
[`docs/step57b_real_model.md`](../docs/step57b_real_model.md)（模型、loader、Attention、Runner）、
[`docs/step57c_kv_and_prefix.md`](../docs/step57c_kv_and_prefix.md)（块池、前缀缓存、抢占恢复）、
[`docs/step57d_sampling_and_stop.md`](../docs/step57d_sampling_and_stop.md)（采样、惩罚、停止、增量输出）、
[`docs/step57e_speculative.md`](../docs/step57e_speculative.md)（投机验证、草稿时序、draft 模型）；
57F 的两篇对照见 [`docs/step57_architecture.md`](../docs/step57_architecture.md)（与真实 vLLM 的
结构/数值对照）与 [`docs/step57_alignment.md`](../docs/step57_alignment.md)（差异账本）。

| 层 | 文件 | 对应 vLLM |
|---|---|---|
| 配置 | `config.py` / `sampling_params.py` | `vllm/config/*`、`vllm/sampling_params.py` |
| 请求状态 | `request.py` | `v1/request.py` |
| 协议 | `outputs.py`、`core/sched/output.py` | `v1/engine/__init__.py`、`v1/outputs.py`、`v1/core/sched/output.py` |
| KV 控制面 | `core/kv_cache_manager.py` → `core/kv_cache_coordinator.py` → `core/single_type_kv_cache_manager.py` → `core/block_pool.py` → `core/kv_cache_utils.py` | `v1/core/` 下同名文件（这条链一层一个问题） |
| 调度 | `core/sched/{scheduler,request_queue,utils}.py` | `v1/core/sched/*` |
| 编排 | `engine/{core,core_client,output_processor,llm_engine}.py` | `v1/engine/*` |
| 执行部署 | `executor/uniproc_executor.py` | `v1/executor/uniproc_executor.py` |
| 执行端 | `worker/worker.py` | `v1/worker/gpu_worker.py` |
| 一轮怎么跑 | `worker/gpu_model_runner.py` | `v1/worker/gpu_model_runner.py` |
| 批状态缓冲 | `worker/gpu_input_batch.py`、`worker/block_table.py` | 同名文件 |
| 权重加载 | `model_loader/*` | `model_executor/model_loader/*` + `models/utils.py` |
| 模型 | `models/qwen3.py` | `model_executor/models/{qwen2,qwen3}.py` |
| 层与注意力 | `layers/*`、`attention/*` | `model_executor/layers/*`、`attention/*` |
| 采样 | `sample/{metadata,sampler}.py`、`sample/ops/*` | `v1/sample/{metadata,sampler}.py`、`v1/sample/ops/*` |
| 投机 | `spec_decode/{metadata,rejection_sampler,ngram_proposer,draft_model}.py` | `v1/spec_decode/*`、`v1/sample/rejection_sampler.py` |
| 测试替身 | `testing/fake_runner.py` | 无（只给测试） |

## 怎么用（本地模型短生成）

```bash
python minivllm/demo.py                                     # 默认 models/Qwen3-1.7B + 默认问题
python minivllm/demo.py --max-new-tokens 64 --trace "问题"    # --trace 打印第一轮喂给模型的数字
```

`--trace` 那段输出是本关的核心证据之一：**协议只给了 `num_scheduled_tokens`，其它数字
（input_ids / positions / slot_mapping / 块表）全是执行侧自己算的**。

## 怎么用（注入假 Runner，不装真实模型）

给 `Worker` 注入 `testing.FakeRunner` 就不装真实模型，协议/调度用例靠它跑得又快又确定。

```python
from minivllm import (CacheConfig, LLMEngine, ModelConfig, SchedulerConfig,
                     SamplingParams,
                    UniProcExecutor, VllmConfig, Worker)
from minivllm.testing.fake_runner import FakeRunner

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

## 调度器可打印的轨迹（57C 交付物）

```bash
python minivllm/demo.py --scheduler-trace "同样的问题" "同样的问题"
```

采样参数（57D）：`--temperature/--top-k/--top-p/--seed`、三种惩罚、`--min-tokens`、`--ignore-eos`；
默认贪心、不开任何筛选。

```text
step scheduled                    hits           preempted    running            waiting        free cached
   1 {'q0': 19}                                               ['q0']             ['q1']           10      0
   2 {'q0': 1, 'q1': 3}           {'q1': 16}                  ['q0', 'q1']       []                9      1
```

读法：第 1 轮只排得下 q0 的 prompt；第 2 轮 q1 **命中 16 个 token**（用小预算逼它晚一轮进来），
所以只需要再算 3 个。抢占会出现在 `preempted` 那一列，块占用看最后两列。不运行模型也能看懂调度器
在做什么，`check_step57_preemption.py` 里也是靠它做断言。

## 明确不做

EAGLE/MTP、异步与多进程、指标、logprobs、KV 连接器、多 KV group。
另外两条容易误以为已经具备的能力：

- **只支持 TP=1**：`layers/linear.py` 里的名字（`QKVParallelLinear` 等）是为了与 vLLM 源码
  一一对应，**没有实现通信**，改 `tp_size` 不会跑起来；
- **只读本地 safetensors**：单文件或带 index 的分片都行，不做 HF hub 下载、不支持 `.bin`/量化。

## 验证

```bash
python benchmarks/check_step57_engine_protocol.py    # 23 项：协议与数据契约
python benchmarks/check_step57_request_progress.py   # 29 项：请求进度与停止判定
python benchmarks/check_step57_scheduler_basic.py    # 25 项：统一预算调度
python benchmarks/check_step57_runner_inputs.py      # 30 项：输入打包（198 §4 逐值）、批状态、入口边界
python benchmarks/check_step57_weight_loading.py     # 25 项：权重读取/打包路由/覆盖检查/tied embedding
python benchmarks/check_step57_model_logits.py       # 18 项：GQA+qk norm+RoPE 参考对照、HF 对照、三种切分
python benchmarks/check_step57_block_pool.py         # 28 项：空闲队列/引用计数/同 hash 多块/发布/失败原子性
python benchmarks/check_step57_prefix_cache.py       # 25 项：hash 链、发布边界、命中粒度、共享不覆写、开关一致
python benchmarks/check_step57_preemption.py         # 23 项：victim 选择、计划撤销与预算退回、恢复整表替换、端到端一致
python benchmarks/check_step57_sampler.py            # 29 项：混批分流、min_tokens 屏蔽、三种惩罚、top-k/p 边界、分布统计
python benchmarks/check_step57_stop_and_outputs.py   # 20 项：五条停止规则、min_tokens 两处职责、增量输出、seed 可复现
python benchmarks/check_step57_spec_metadata.py      # 13 项：两个坐标系的索引数学（vLLM 算例逐值）
python benchmarks/check_step57_rejection_sampler.py  # 22 项：greedy/random 验证、恢复分布、CPU 公式与统计对照
python benchmarks/check_step57_spec_lifecycle.py     # 12 项：提议与采用的时序、K 裁剪、进度回退、抢占清草稿
python benchmarks/check_step57_draft_model.py        # 32 项：draft 规格校验、KV 独立、端到端、逻辑上限与请求生命周期（205 回归）
```

## 与真实 vLLM 的对照（需要 GPU）

```bash
# WSL2 必须带这两个环境变量（否则 vLLM 引擎起不来，见 docs/step57_architecture.md §4）
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/compare_step57_vllm.py            # logits / 增量位置 / 拒绝采样 / 端到端
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/check_step57_vllm_boundaries.py   # 两条边界在真实 vLLM 上核实
python benchmarks/trace_step57.py                       # 一条覆盖五种事件的状态轨迹（CPU）
```

数值对照用的外部参照是 **transformers 的 Qwen3**（同一份 tiny 权重）与一份**按公式手写**的
注意力参考实现；权重与 tokenizer 用本机 `fixtures/step30_qwen3` 与 `models/Qwen3-1.7B`
（都已存在，不下载新模型）。
