# 106：真实 EAGLE3 单卡身份与逐轮差分

状态：106 功能及单卡专门验收已完成；全量回归中有历史结构化输出 stream 问题和 GPU0 显存不足的失败，暂不将整个关卡标记为完全关闭。107 可基于已实测的单卡草稿协议继续，但这些回归异常仍须单独保留。

## 冻结源码与职责

| 项目路径 | 冻结 vLLM 0.28.0 | 职责、时序与布局 |
|---|---|---|
| `models/qwen3_eagle3.py::Eagle3Model` | `model_executor/models/llama_eagle3.py::LlamaModel` | 草稿模型身份来自配置；第一层拼接 embedding 和草稿 hidden，隐藏维度 H，特征沿最后一维排列 |
| `Eagle3ForCausalLM.combine_hidden_states` | `Eagle3LlamaForCausalLM.combine_hidden_states` | target 多层 `[T,H]` 特征按冻结顺序拼接为 `[T,3H]`，投影为 `[T,H]` |
| `Eagle3ForCausalLM.compute_logits` | `Eagle3LlamaForCausalLM.compute_logits` | 草稿 logits `[T,32000]` 先生成，再按 `arange+d2t` 置入 `[T,151936]`，无效列为 `-inf` |
| `worker/gpu/spec_decode/eagle/utils.py::load_eagle_model` | 同名冻结 `load_eagle_model/_should_share` | 按参数存在性和值共享 embedding/head；target 与 draft KV 独立 |
| `worker/gpu/spec_decode/autoregressive/speculator.py` | 同名冻结类 `propose` | 第一遍使用 target 真正输入与辅助层特征；逐步返回草稿，下一轮拒绝位置重算 |

Scheduler 唯一拥有 Request 与逻辑提交位置；Runner/Worker 持有本地缓冲、模型和设备 KV。`K=2` 时草稿提出 d0/d1，验证后按每请求有效位置修正，预留或被拒行不算已计算。

## 已确认身份（090 历史记录；106 尚待独立前向验收）

8B/14B 草稿均为 `LlamaForCausalLMEagle3`，draft vocab 32000、target vocab 151936；两份 checkpoint 均为 15 张量、缺 target embedding，权重转换保持原 dtype。14B 草稿原始权重为 FP16，不能冒充 BF16 源文件。

## 实际诊断（2026-11-11）

- GPU0 被其他用户进程占用，8B 测量显式使用 GPU1（UUID `deb35ff6-c14d-e1d9-5b24-8a7e4eb1bbd9`）。冻结上游 K1/2/4 各运行三遍、32 token、128 输入；`verify_oracle` 校验 342 个冻结源码文件。参考 trace 与仅针对重复运行的统计位于本地 `benchmarks/results/step106/reference_k*_repeat_contract.json`；slot 地址因物理块复用可能不同，不能称为固定 block table 的状态验收。
- 在项目首次真实 K2 诊断中，最终 token 一致但候选不同；首个分歧是 target 辅助特征。冻结 Qwen2Model 在 embedding 处记录边界 0，每个 decoder 层后按 `idx+1` 记录；项目旧代码在第 `idx` 层后按 `idx` 记录，错取后一层。按冻结源码修正后，K1/2/4 正常生成 trace 的特征合并、草稿层输出、映射后 logits、提议候选均逐值一致，最大误差为 0；K2 180 个事件，真实发生拒绝。K0 32 token 及首轮 prefill/前三轮 decode 的 target hidden/logits/第 0 层有效 KV 逐值一致。
- `benchmarks/step106_reference_fp32.py` 在查看 mini 结果前，使用参考第一行和真实 draft 权重做了 FC 与 logits 的局部 FP32 重放；它只约束指定行，不可冒充整模型或其他行的数值契约。所有 trace 输出另有冻结参考重复契约；本次实际逐值相等，未使用更宽阈值豁免差异。
- K2 初始化后按唯一 parameter storage 去重：两端 target 均为 16,381,470,720 bytes；草稿与 target 共享 1,244,659,712 bytes。上游草稿参数去重后为 2,043,963,392 bytes（增量 799,303,680），项目为 2,043,707,392 bytes（增量 799,047,680）。两者相差 256,000 bytes，正好是 32000 个 int64 d2t：冻结上游将其注册为 Parameter，项目把它作为普通 tensor，不能把这份映射实际内存忽略。项目还持有 bool t2d；这两个数字仅是参数 storage，非实际总增量显存。
- V2 中同 shape 的独立 embedding 是否共享按冻结 `_should_share` 的值相等规则处理；真实 checkpoint embedding 共享、缩减 LM head 独立。V1 的单卡共享路径未在本关重构。旧 `test_first_pass_inputs_follow_eagle_alignment` 的 tiny 草稿 seed 由 0 调整为 2，以继续覆盖“接受后拒绝”这一必要状态；逐行断言未降低。
- 106 独立 pytest：30 passed，包含 K1/2/4 的每个接受前缀，在 tiny V2 引擎强制候选后逐项验证接受/拒绝数、下一轮起始位置及最终输出；另在实际 GPU 上与冻结拒绝采样器逐值差分。8B K0 目标输出及 prefill/前三次 decode 的首层 KV、hidden、logits 相等。K1/2/4 的正常生成与固定历史分别核对 target 真实输入、draft 输入、QKV、attention、每层 hidden、映射前后 logits、候选、采样计数及首层 target 和 draft 有效 KV；这不是全部 target 层 KV 的覆盖。K1/K2/K4 固定历史分别为 240/390/690 个 trace 事件、570/690/930 项状态检查，全部逐值相等，保存 32 token。自然拒绝后的下一轮包含在这些运行中。
- 14B 只加载 draft，以固定特征执行 FC 与映射 logits；参考重复、局部 FP32 测量后与项目输出逐值相同。未执行 14B target 或 14B draft attention。验收驱动 `CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/check_step106_acceptance.py --case all --timeout 1800 --output benchmarks/results/step106/acceptance_final.json`：33 子步骤通过；后续独立重比对修正了固定历史的结果标签，结果仍通过。
- 受影响目录首次回归 123 passed / 2 failed / 10 skipped：旧 step63 样本不再产生所需接受/拒绝组合，已调整确定性 seed 后单项重跑通过；step73 structured-output stream 竞争为先前上游同源已记录问题，生产路径未修改。58—88 完整回归 1288 passed / 1 failed / 23 skipped，失败仍为该 step73；089—105 回归 205 passed / 1 failed，103 上游参考在 GPU0 初始化时仅有 10.75 GiB 可用，小于其要求的 20.34 GiB，本项目模型尚未运行。两个失败均不计通过；23 个历史 skip 不计验收。
- 在 GPU1 单独重跑该 103 真正 8B 用例：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/step103/test_step103_e2e.py::test_real_8b_tp1_32_token_generation_without_prior_103_benchmark -q -ra`，1 passed；这不抹去完整回归中的 GPU0 失败。
