# Step 118：W4 分组表示与 packing（实现中）

本关核对 4-bit 逻辑数值、AWQ/GPTQ checkpoint 打包和 zero point 语义；测试参考不进入生产推理，尚不实现量化 GEMM。

编码前映射：R118.01/03/05/07 在测试参考 `tests/step118/step118_w4_reference.py` 定义输入轴、group、位与有效区域；R118.02 在同一测试模块提供慢速解包和 FP32 反量化；R118.04 对照冻结上游 `auto_awq._convert_awq_to_standard_format`；R118.06 区分 AWQ 明示 zero 和 GPTQ 文件编码，127 的 kernel offset 不提前实现；R118.08 的手算点积和边界样本在 `tests/step118/`。`benchmarks/benchmark_step118_w4_format.py` 只做离线检查；性能延后。

## 实现与语义

仅新增测试参考 `tests/step118/step118_w4_reference.py`：用 Python 整数屏蔽 32 位再解包，避免有符号右移污染；按标准 bit 位 0,4,…,28 打包，按冻结 AWQ 的逆序恢复逻辑列，转换后的 qweight 为 `[K/8,N]`、qzeros 为 `[N/8,G]`。FP32 参考以 scale 为反量化乘数，按 `(q-z)*scale` 计算，group 沿逻辑 K，显式裁掉不属于有效区域的 padding；不用于生产推理。

标准手算固定 q=`[0,1,7,8,15,2,3,4]` 得到 `0x432F8710`，s=0.25、z=8 的结果分别为 `[-2,-1.75,-0.25,0,1.75,-1.5,-1.25,-1]`；x=`[1,2,0,-1,1,0,0,2]` 的点积为 -5.75。8 个元素仅是打包小例，不把实际 group 128 改为 8。AWQ 物理字 `0x76543210` 独立预期标准解包为 `[0,1,2,3,4,5,6,7]`，AWQ 位序解包为 `[0,4,1,5,2,6,3,7]`。覆盖全 0、全 15、最高位 bit31、所有 nibble、两组不同 scale、zero 0/15、有效区域以外 padding 改值以及盲目 +1 的 AWQ 负例。

AWQ 的文件 qzeros、转换后的逻辑 zero 和 FP32 反量化参考均按冻结转换对照。GPTQ 只核验文件 qweight 与 qzeros 的原始 packed 位值；本关没有声称文件 qzeros 已是 kernel 偏置或已完成 offset 转换，具体 GPTQ offset 留给 127。两者不共用未经验证的 +1。该测试没有推导新的量化算法。

## 冻结上游与真实文件 CPU 差分

独立子进程 `tests/step118/step118_frozen_probe.py` 对同一原始输入调用安装的 vLLM `pack_rows`、`pack_cols`、`awq_pack`、`unpack_quantized_values_into_int32` 和 `_convert_awq_to_standard_format`，比较前逐字节核对安装源码与冻结文件：`auto_awq.py` SHA256 `af09b7bff7a8d2d500b9df649efd5570b32a762032cf711d63383d1f0f01cf55`；`utils/quant_utils.py` 为 `b8cd5aa5aacaa7658b5f5a51d5b9766df80832904281b24e429d03ee0f9eb0ad`。上游 Parameter 构造需要 TP 进程组，因此 CPU 隔离探针仅把上游 rank/world-size 查询替换成 0/1；被比较的上游转换与位运算函数未替换，不涉及 GPU 或 NCCL。

命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python benchmarks/benchmark_step118_w4_format.py --check --output benchmarks/results/step118/public_cpu_check.json`。四份 114 公开 checkpoint 固定 revision，从首层 q_proj 抽取逻辑 K=0:256、N=0:16，覆盖两个 group；真实块无 padding。每份先对照 114 manifest 核实 index hash 和涉及分片的完整 LFS SHA256。结果文件 SHA256 `7c35883f2c31228ac950dec6918ed6e62ce26ceb7f40a344168faa7a4655fe8a`；文件路径和各 shard hash 在该结果登记。

| checkpoint | 原始 qweight shape / stride | 原始 qzeros shape / stride | scales shape / stride / dtype | 对照结果 |
|---|---|---|---|---|
| Qwen3-8B AWQ | `[256,2] / [512,1]` | `[2,2] / [512,1]` | `[2,16] / [4096,1]` BF16 | packed qweight/qzeros、q、z、FP32 反量化一致 |
| Qwen3-14B AWQ | `[256,2] / [640,1]` | `[2,2] / [640,1]` | `[2,16] / [5120,1]` BF16 | 同上 |
| Qwen3-8B GPTQ | `[32,16] / [4096,1]` | `[2,2] / [512,1]` | `[2,16] / [4096,1]` FP16 | q、文件 zero 原始位值一致 |
| Qwen3-14B GPTQ | `[32,16] / [5120,1]` | `[2,2] / [640,1]` | `[2,16] / [5120,1]` FP16 | 同上 |

AWQ 转换后小块 qweight 是 `[32,16]`、qzeros 是 `[2,2]`；GPU kernel 的实际计算、Marlin 重排及全模型加载不在此证据内。公开文件来源的 BF16 精确 revision 和校准集仍未知，抽层差分不代表质量通过。

## 测试与未完成事项

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/step118 -q -ra`：6 passed、0 failed、0 skipped。
- 上述新增第六项手算位序负例之前，同环境执行 `pytest tests/step115 tests/step116 tests/step117 tests/step118 -q -ra`：54 passed、0 failed、0 skipped；后续单独跑 118 六项。
- 按用户要求未做 GPU 正确性、完整历史回归或性能测试；`--trace` 明确返回 not_run。117 前置尚未完整验收，本关的真实 GPU、TP/Graph 和后续 kernel 对照不能由 CPU 位运算替代。

本次未修改生产包；当前 `minivllm` 包 SHA256 仍为 `d22c6eb37917618b05c3f3857b8fa630eba48a6bd2080ab704e87911f2b6f90d`。118 的 CPU 格式参考及限定抽层对照已完成，完整关卡验收仍待相关前置与后续 GPU 证据。
