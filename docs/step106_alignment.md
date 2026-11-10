# 106 PP 层切分与 IntermediateTensors

## 冻结源码映射与所有权

| 本项目 | 冻结 vLLM 0.28.0 | 行为 |
|---|---|---|
| `models/utils.py::get_pp_indices/make_layers` | `distributed/utils.py::get_pp_indices`、`model_executor/models/utils.py::make_layers` | 余数从末阶段之前向左分配，支持 `VLLM_PP_LAYER_PARTITION`；使用占位层保留全局编号 |
| `models/utils.py::PPMissingLayer/is_pp_missing_parameter` | `model_executor/models/utils.py` 同名入口 | 非本地层、embedding、norm/head 无参数；加载流跳过缺层，但本地覆盖与 packed shard 检查不放宽 |
| `sequence.py::IntermediateTensors` | `sequence.py::IntermediateTensors` | 阶段交接 `hidden_states` 与 `residual`，支持取键、切片、复制结构 |
| `models/qwen3.py::Qwen3Model` | `model_executor/models/qwen3.py`、`qwen2.py::Qwen2Model` | 首阶段 embedding；逐个本地全局编号层；非末阶段交接 hidden/residual；末阶段 norm |
| `models/qwen3.py::Qwen3ForCausalLM` | `qwen2.py::Qwen2ForCausalLM` | 仅末阶段 LM head；tied 时末阶段亦拥有 embedding 并与 head 共享；非本地权重跳过 |

Scheduler 继续拥有请求与逻辑 KV；阶段仅拥有本地权重与 attention。当前 Worker 仍拒绝 PP 多进程模型，107 才负责跨进程传输与反馈。

## 布局与验收

40 层 PP2 使用 `[0,20)` 和 `[20,40)`。边界 `hidden_states`、`residual` 均为 `[T,H]`；保留原 dtype。以 CPU 上顺序调用实际 Qwen3 attention 的 tiny 模型为单测，不代表多进程、真实 14B、GPU PP 或流水线并发。

独立测试覆盖 4/5 层 PP2/PP4、冻结 helper 的默认及自定义分层差分、逐边界 hidden/residual 与最终 hidden/logits 对照、残差污染负例、本地缺权重、非本地跳过、PP1 与 tied embedding。结果与回归命令记在 `docs/results.json`。

分层 helper 对负层数、非正 PP 大小和越界 rank 增加显式参数检查；合法配置沿用冻结分层算法。层数少于阶段数时仅验证分层 helper 与冻结实现相同，空阶段的实际张量传输与完整执行尚未验收。

独立验收为 15 passed。首次 58—88 回归为 1287 passed、2 failed、23 skipped：包 hash 未及时同步已修正；另一项 structured outputs 位图错误单项重跑通过。更新 hash 后完整复跑为 1288 passed、1 failed、23 skipped，同一 structured outputs 用例再次失败。

定位对照：项目与安装版上游 `StructuredOutputsWorker` 均可复现初始化流清零与拷贝流写入的竞争；在测试探针中让拷贝流等待初始化流后，两者各 5 次均通过，未等待时分别 4/5、5/5 失败。安装版上游文件与冻结文件的 SHA256 相同（`8d00e766d8d38d87ef16cccd0661d71833e523e1dc8fee5aeab251536b57180c`）。最小探针和疑似上游问题记录在私有 `vllm_bugs/structured_outputs_stream_probe.py` 与 `vllm_bugs/UPSTREAM_SUSPECTED_BUGS.md`；生产路径未在 106 中变更。这个对照支持 stream 竞争是回归失败原因，但不宣称已证明每次实际失败的内部时序。

089—105 回归为 205 passed、1 failed：103 的 8B TP1 上游参考启动时 GPU 0 可用 10.78 GiB，不足其请求的 20.34 GiB，本项目模型尚未运行。日志保留于 `benchmarks/results/step106/`，不能将该次失败记作通过。GPU 被其他用户任务占用，未中断其任务。106 的模型切分与独立验收完成，因历史回归与共享显存条件仍有未关闭项，暂不标记完全关闭。
