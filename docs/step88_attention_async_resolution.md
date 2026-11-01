# Step88：Triton paged attention 异步加载问题的定位与优化

日期：2026-11-01。问题来源：`docs/step88_attention_open_problem.md`。旧生产基线：`bb1380d`（其中 kernel 与开放问题提交一致）。

## 0. 先看结论

**问题已经定位，不需要修改 Triton 编译器，不需要 CUDA 扩展，也不需要大面积移植 vLLM。**

关键漏项在 kernel 的参数声明，不在 `tl.load` 的外观：

```diff
- stride_k_cache_3: tl.int64,
+ stride_k_cache_3: tl.constexpr,
- stride_v_cache_3: tl.int64,
+ stride_v_cache_3: tl.constexpr,
```

这是 K/V **最后一维的 stride**。上游这两个参数本来就是 `tl.constexpr`；mini 声明为运行时 `tl.int64`，让编译器失去了“相邻 head 元素在内存中连续”的证明。

只改这两行，保持旧 TILE64/BLOCK_M8 不动，已经能让真实 K/V 出现 async copy，输出不变。但它不是最终最快方案：旧 TILE64 workaround 需要额外的 shared memory、subtile 布局转换。最终只对 **BF16/FP16 全注意力**、head≤256 使用 TILE32/BLOCK_M16；head512 保留 BLOCK_M8。SWA 和 FP32 保持原 tile/query 分组，原因见 §5.4。

**同轮、同输入、Graph-batched CUDA event 实测：**

| B=1，q_len=3，全注意力 | 旧 mini | 优化后 mini | vLLM | 相对旧 mini 加速 |
|---|---:|---:|---:|---:|
| D=256，ctx128 | 9.87 μs | 7.95 μs | 7.54 μs | 1.24× |
| D=256，ctx1024 | 49.88 μs | 41.13 μs | 40.07 μs | 1.21× |
| D=256，ctx2048 | 95.67 μs | 79.02 μs | 77.05 μs | 1.21× |
| D=512，ctx1024 | 95.01 μs | 57.97 μs | 61.53 μs | 1.64× |

22 个 BF16 case 的新旧 mini 输出全部逐元素相同；vLLM 能运行的 20 个 case 也与旧 mini 完全相同。另两个 D512+SWA case 上游超出本 GPU shared-memory 上限，明确记为不可运行，不算通过。

以上是热缓存单 kernel 测量，不是模型 tok/s；没有证明“所有模型/硬件都快这么多”。**D256/SWA512 的 B1/B4 case 分别退化约 8%/12%**，没有把它们从表中删掉，完整数据见附录。真实模型结果单独列于 §8。

## 1. 先建立一个简单认知

### 1.1 stride 是什么？

固定 K 的布局：`[physical_block, token_in_block, kv_head, head_element]`。

如果最后一维 stride=1，同一个 head 的元素地址是：

```text
元素编号：0  1  2  3  4 ...
元素地址：A A+1 A+2 A+3 A+4 ...
```

stride=2 则是 `A, A+2, A+4...`。这里按元素计地址；BF16 每个元素占 2 字节。

编译器知道 stride=1，就能安排一个线程连续搬多个元素。只知道“运行时会传来一个整数”，就不能保证连续，哪怕这一次实际传进来的确实是 1。

### 1.2 `constexpr` 没有把所有输入强行当成连续

调用点仍传 `k.stride(3)` / `v.stride(3)`，没有改成常量数字 1。

`constexpr` 表示**按实际 stride 值专门编译**：stride=1 的版本可以证明连续；stride=2 的版本仍用真实的 2 计算地址，不伪造连续性。新增测试特意用了非单位内层 stride 和带间隙的页布局。

这里“运行时参数”不等于它每轮都会变：Python 明明已经拿到了整数 1，但 `tl.int64` 让本次生成的 IR 仍保留这个 GPU kernel 参数。`constexpr` 把布局值纳入特化。也不要反过来把请求 ID、实际 seq_len 等全部设为 constexpr，造成无谓的重复编译。

### 1.3 async copy 解决什么？

原路径大致是 `global → registers → layout conversion/shared → dot`。

异步流水线可以将下一 tile 的 `global → shared` 搬运与当前 tile 的计算交错安排。不是取消内存访问，也不是有了 `cp.async` 就必然更快；shared 占用、寄存器、tile 形状和并行度仍会限制收益。

## 2. 哪个 pass 做了决定？

实际读取本机 Triton 3.7.1 的 `backends/nvidia/compiler.py::make_ttgir`，并用 `MLIR_ENABLE_DUMP=1` 保存本机每个 pass 的 IR。C++ 判定对照官方 v3.7.1 tag，不使用滚动 main 推测本机行为。

本机 `compiler.py`、`runtime/jit.py` 还与官方 tag 做了 SHA256 比较，两份文件逐字节相同；记录在 `compiler_python_identity.json`。C++ 源码用于解释，实际编译行为仍以本机 pass dump 为证据。

### 2.1 第一个语义分叉：TTIR 中的动态乘法

旧 mini 在 Coalesce 之前仍包含这个数据依赖，以下保留操作含义、简化 SSA 名：

```text
stride_k_cache_3 : i64                    # 仍是 kernel 参数
offset = arange(0, 256) * stride_k_cache_3 # arith.muli
k_ptr = page_base + offset
```

constexpr=1 版本则化简为：

```text
k_ptr = page_base + arange(0, 256)
```

IR 中 `stride_k_cache_3`、`stride_v_cache_3` 从运行时函数签名消失，动态乘法也消失。这是“load 源码看起来一样，编译器看到的信息不一样”的直接证据。

保存在 `compiler_evidence/base.ttir` 的实际操作是：

```mlir
%k_ptrs_89 = tt.splat %stride_k_cache_3 : i64 -> tensor<1x256xi64>
%k_ptrs_90 = arith.muli %k_ptrs_88, %k_ptrs_89 : tensor<1x256xi64>
```

`stride.ttir` 中这两条最后维 stride 操作消失。原始 SSA 编号随编译版本变化，诊断应跟踪输入依赖，而不是硬搜某个临时编号。


### 2.2 AxisInfo：什么情况下连续性能够保留？

`lib/Analysis/AxisInfo.cpp::MulIOpAxisInfoVisitor::getContiguity`，本次源码约 396–414 行：乘法的一侧能证明为常量 1，才保留另一侧的 contiguity；否则这个乘法的连续性按 1 处理。

因此运行时整数 1 与编译期常量 1 在此不是同一份证明。[Triton 3.7.1 AxisInfo 源码](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Analysis/AxisInfo.cpp)

### 2.3 Coalesce：第一次 load layout 分叉

调用链：

```text
TritonGPUCoalesce / tritongpu-coalesce
  → buildCoalescedEncoding（CoalesceUtils.cpp）
  → getNumElementsPerThread（Transforms/Utility.cpp）
```

它根据指针的 contiguity、divisibility、shape 和线程数量选择布局；不是根据“这是 attention 的 K”硬编码一个漂亮 layout。

`getNumElementsPerThread` 的核心约束包括可证明对齐、连续元素数，以及 128 bit 的向量宽度上限。BF16 在本例可达到 8 元素/线程；未知 stride 使连续性退到 1。[Coalesce](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)、[CoalesceUtils](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp)、[Utility](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/Utility.cpp)

本机 pass dump 中，相同 `[64,256]` 的 K/V load 在 Coalesce 后：

| 只差两处参数声明 | load layout 的 sizePerThread |
|---|---|
| 运行时 stride | `[1,1]` |
| constexpr stride=1 | `[1,8]` |

mini 的 K/V 都按 `[TILE,HEAD]` 读，所以都是 `[1,8]`。上游 K 按 `[HEAD,TILE]` 读时显示 `[8,1]`，不必强行把两者的 layout 文本做成一模一样。

### 2.4 AssignLatencies：为什么原来没资格进入 pipeline？

`Pipeliner/AssignLatencies.cpp::AssignLoadLatencies::isPipeliningBeneficial` 会调用 `canBeConvertedToAsyncLoad`。后者位于 `PipeliningUtility.cpp` 约 380–406 行，结合可连续向量宽度、mask alignment、元素位宽，要求至少 32 bit。

```text
旧 BF16：1 个连续元素 × 16 bit = 16 bit < 32 → 不满足
新 BF16：可向量化的多元素访问 → 满足至少 32 bit 的门槛
```

另外还检查使用者的 shared encoding 是否兼容、loop 等条件；不是只要超过 32 bit 就无条件 pipeline。[AssignLatencies](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp)、[PipeliningUtility](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp)

“每线程标量 load”不等于“整个 warp 的访存完全不能合并”。本例决定 async 资格的是编译器能证明的**每线程连续搬运宽度**。增加 stages、删 mask 都不能凭空补上缺失的 stride=1 证明。

实际 dump：旧 K/V `tt.load` 没有 latency 标记；两行修改后 block-table、K、V 的相关 load 获得 `tt.latency = 1`。

### 2.5 Schedule / Pipeline：最终生成 async copy

`ScheduleLoops` 安排 load 与计算的阶段。`Pipeline` 的 lower-loop 路径再检查 register→shared 的 copy 宽度等条件，将选中的 load 转成 `ttg.async_copy_global_to_local`，并生成相应同步操作。`LowerLoops.cpp` 约 450–486 行也明确检查至少 4 字节的 copy。[LowerLoops](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp)

**判定链是：编译期 stride → AxisInfo → Coalesce → load latency/资格 → schedule/pipeline。** Remove-layout-conversions 会影响最终 IR，但不是本例连续性丢失的起点。

## 3. 最小因果实验，而不是继续枚举开关

固定 q_len=3、ctx1024、GQA=8、D=256、block16、BF16。每次只改变明确的一组因素：

| 版本 | stride | TILE / BLOCK_M | regs / spills | shared bytes | async copy / BF16 copy | ctx1024 |
|---|---|---|---|---:|---|---:|
| 旧生产 | runtime | 64 / 8 | 255 / 0 | 20,480 | 0 / 0 | 49.88 μs |
| 仅两行声明 | constexpr | 64 / 8 | 199 / 0 | 70,400 | 7 / 4 | 50.43 μs |
| 最终方案 | constexpr | 32 / 16 | 155 / 0 | 42,112 | 7 / 4 | 41.13 μs |
| 上游对照 | constexpr | 32 / 16 | 158 / 0 | 42,240 | 10 / 4 | 40.07 μs |

async 数是 IR 中的静态操作计数，不是运行时执行次数。BF16 指针的四处 copy 对应 K/V 的流水线展开位置；不是把 block-table 的 int32 copy 当成 K/V 成功。

仅两行修改的实验，直接否定了“高层 Triton 做不到、必须手写 cp.async”的推断。与此同时它也说明旧 TILE64 workaround 在新布局下不能继续当成最优配置。

最终 D256 距上游的单 kernel 差距约为 3%，不是严格相等；D512 本轮 mini 更快，但不外推为整个引擎优于 vLLM。

## 4. 实际生产 patch

只修改 `minivllm/attention/ops/triton_unified_attention.py`：

1. K/V 最后维 stride 改为 `tl.constexpr`，保持调用接口传真实 stride。
2. 2D BF16/FP16 **全注意力**、head≤256：TILE32，BLOCK_M16；GQA=8 时 BLOCK_Q=2，相邻两个 query 可共享一次 KV 扫描。
3. head512 保持 BLOCK_M8；不盲目扩大 accumulator，控制寄存器压力。
4. 3D 的 TILE16/原行分组不变，不更改 2D/3D 选择条件。
5. FP32 保持原 TILE/BLOCK_M 和 IEEE dot；SWA 也保持原 TILE/BLOCK_M，避免改变边界 subtile 的浮点缩放历史。
6. 更新过时注释，删除“改变 K/V 转置就是 async 根因”的误导性归因；不改概率算法、mask 或 cache 所有权。

没有更改 Triton 安装，没有引入 CUDA 扩展，没有 import vLLM kernel 到生产，也没有新造 TMA/descriptor 路径。

## 5. 数值与支持性边界

### 5.1 已验证的逐位不变

- 22 个 BF16 场景：q_len=1/2/3/4/8、ctx128–2048、B=1/2/4/8、D256/512、full/SWA，全部新旧 mini `max_abs == 0`。
- 新增回归包括随机物理块映射、packed 页间 stride、最后维 stride=2、FP16、FP32 D16/32/64、SWA。
- 检查真正的 BF16 K/V async IR、PTX `cp.async`，并验证动态 stride 从 TTIR 函数签名中移除。
- 既有 step86/87 负责独立 attention 数值对照、共享 KV、3D、Graph 等端到端契约；新旧代码对照不是独立数学证明。

### 5.2 为什么 FP32 没一起切换 BLOCK_M16？

试验中 FP32 D64 的 TILE32/BLOCK_M16 与旧算法产生约 `1.19e-7` 的差异，不满足本任务的逐元素零差门槛。因此没有把这个候选合入 FP32 路径。

另外，**旧生产自身**在本机某些 FP32 大 head case 无法 launch：D128/TILE64 申请约 135,424 B shared，D256 约 270,592 B，超过 SM120 此设备的 101,376 B 上限；D512 也观察到资源超限。本次没有把这件事偷偷“修成宽容差通过”，也没有声称真实 D256/512 FP32 已完成零差对照。

这是原有的 FP32 大 head 支持性问题，后续若扩展需单独设计资源参数及数值契约。当前优化目标是实际 BF16 Gemma4 路径。

### 5.3 两个上游 case 不可运行

本机 vLLM 的 D512+SWA512（B1/B4）2D kernel 申请 115,968 B shared，超出此 GPU 上限。benchmark 保留 `unsupported` 及错误原文，不删除 case、不算 pass。

### 5.4 真实模型暴露了随机 microbenchmark 遗漏的 SWA 边界

第一次候选把 BF16 SWA 也切到 TILE32/BLOCK_M16。它通过了 22-case 随机输入和原有回归，**但真实 Gemma4 128-token 输出在第 109 个 token 与旧 mini 分歧**。这个候选已撤回，不能把当时的 176.18 tok/s 当成无损结果。

定位方法不是继续比较最后一段文本：截取真实 attention 的 Q/K/V/块表，同一份输入分别运行旧 kernel、只改 stride 的 kernel、改 TILE32 的 kernel。

- 差异集中在 D256、SWA512、q_len=128 的 prefill 调用；只改 stride 时为零差。
- 捕获输入中，TILE32 候选与旧值有 BF16 差异；用单位尺度随机 Q/K 也能复现，不需要保存真实权重。
- 大方差随机 Q/K 往往很快出现较大的正最大值，掩盖了此问题；真实归一化 Q/K 的数值分布不同。

**一个可手算的边界：**窗口大小 512，query 的绝对位置为 574，则第一个可见 key 为 `574 - 512 + 1 = 63`。

| 方案 | 开始扫描的位置 | 第一次 softmax 更新 |
|---|---:|---|
| 旧 TILE64，内部两个 32 宽 subtile | 0 | key 0–31 全被 mask，`_softmax_step` 把 `M=-inf` 钳到 0 |
| 直接 TILE32 | 32 | key 32–63 中只有 63 有效，直接按它的 score 更新 M |

如果第一个有效 score 为负，例如 -2.56，旧路径的 M 已经是 0，新路径可以是 -2.56。实数数学下缩放相消后 attention 仍等价，但这里 **P 会先转为 BF16 再参与 dot**；`exp(-2.56)` 的 BF16 舍入与 `1` 的舍入不一样，后面再缩放无法保证逐位相同。

因此，旧注释“两个 32 subtile 就一定与 TILE32 完全相同”过于绝对：它忽略了窗口左边界处的全 mask subtile。已修正注释；没有通过放宽容差掩盖差异。

最终保留 SWA 的旧 tile/query 分组，只给它启用 stride 特化。新增 4 个回归：heads=4/8 × 归一化随机 Q/K/全负 score，固定 context512、qlen128、window512，要求与旧实现 `torch.equal`。最终真实层差分与完整生成也重新验证，见 §8。

代价是 D256/SWA512 的 kernel 本轮没有提速，反而退化。观察到其 shared 占用增加；是否由 occupancy/流水线开销主导，还需要独立剖析，本文不把推测当成定论。后续可专门优化 SWA，但要显式处理上述边界，不能只改 tile 就宣称无损。

## 6. 测量方法修正

旧诊断在两个 CUDA event 之间用 Python 循环反复调用 wrapper；GPU 如果先做完，会等待 CPU 提交。那个时间可以混入提交空隙，不能无条件叫纯 kernel latency。

新 `benchmarks/check_step88_async_stride.py`：

- 所有候选用同一份 Q/K/V/块表；基线从本仓库 `git show bb1380d:...` 加载到忽略目录，仅用于诊断。
- 延续既有 microbenchmark 的 `scale=D**-0.5`；真实 Gemma4 的 `scale=1.0` 由新增回归与真实模型测试覆盖，不能把随机输入 microbenchmark 当成真实层数值分布。
- 计时前完成编译和预热；图中录入 100 次相同 kernel 调用，每轮重放，除以 100。
- 7 轮、交替候选顺序；保存每轮值和中位数，不挑最高速度。
- 这是 warm-cache CUDA-Graph-batched microbenchmark；prefill/模型/CPU 调度、真实多层 cache 压力不在其中。
- 通过拦截本次 launch 的返回对象记录 CompiledKernel，不再拿 `device_caches` 第一个 entry 误报 D512 寄存器数。

例如不能沿用 D256 的“零 spill”结论描述 D512：本次精确捕获 D512/full 变体得到旧 28 spills、新 4 spills。不要用缓存中第一个编译对象的属性代替所有 specialization。

## 7. 怎样复现？

```bash
cd /home/user/proj/vllm-from-scratch
export PATH=/home/user/anaconda3/envs/vllm-omni-dev/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 22-case 四方对照：旧生产、仅 constexpr、最终方案、vLLM
python benchmarks/check_step88_async_stride.py \
  --output-dir benchmarks/results/step88/async_stride/final

# 最小 IR 因果差分：下面两次仅差 stride 声明
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python \
  benchmarks/check_step88_async_stride.py --variant baseline \
  --cases qlen3_ctx1024 --calls 1 --rounds 1 2>/tmp/base_passes.mlir
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python \
  benchmarks/check_step88_async_stride.py --variant constexpr_only \
  --cases qlen3_ctx1024 --calls 1 --rounds 1 2>/tmp/const_passes.mlir

# 同一真实模型、同样的输入预算与 128 输出 token；分别运行，不并行占 GPU
python benchmarks/check_step88_async_model.py --variant baseline
python benchmarks/check_step88_async_model.py --variant optimized

# 同一真实 Q/K/V 的逐层差分；包含 warmup/prefill/decode，不用于计时
python benchmarks/check_step88_async_model.py --eager --compare-attention \
  --max-new-tokens 8 --repeats 1

python -m pytest tests/step88/test_attention_async_stride.py -q
python -m pytest tests/step86 tests/step87 tests/step88 -q
python -m pytest $(printf 'tests/step%s ' $(seq 58 88)) -q
```

注意：单 variant 诊断不含跨候选数值对照；它用于 dump，不代替完整 benchmark。指定不同 `--output-dir` 可以避免覆盖自己上一轮的 summary。

IR/原始测量在 `benchmarks/results/step88/async_stride/`，被 Git 忽略；`compiler_evidence/` 保存 pass 前后截取、完整 dump 的 gzip、官方 v3.7.1 源码副本及 SHA256。代码与小型结果摘要跟踪，编译器大产物不提交。

## 8. 最终验收记录

- 新增 `tests/step88/test_attention_async_stride.py`：**14 passed**。
- `tests/step86 tests/step87 tests/step88`：**82 passed**，无失败/skip。
- `tests/step58`–`step88`：最终 **1311 passed**（221.87 秒），无失败/skip；152 条已有框架/配置 warnings。
- 首次全量为 1306 passed + 1 failed：唯一失败是 `step81` 的“当前包哈希匹配结果记录”检查；刷新包摘要后完整重跑通过，没有降低测试断言。
- 22-case microbenchmark 初版跑了两轮，最终 SWA 保护版又完整跑了一轮；最终 mini 新旧 22/22 零差，上游可运行 20/20 零差、2 项 shared-memory 超限。上表采用 `async_stride/final/summary.json`；目录根部的前两轮数据属于已撤回 SWA 改 tile 的候选，不代表最终生产版本。
- `py_compile` 与 `git diff --check` 通过。

### 8.1 真实 Gemma4：数值验收和性能结论分开

配置：本机固定 revision 的 target+assistant，BF16、B1、K2、1024 prompt token、128 committed output token、temperature=0、ignore_eos=True、prefix off、FULL decode Graph、KV 144 MiB。每个变体独立进程，预热后重复 3 次；计时包含本次请求的 prefill 与生成，不是纯 decode kernel。

**最终数值结果：**

- 对真实 eager 路径的全部 **946 次** attention 调用，用相同实际 Q/K/V/块表分别跑新旧 kernel：`different_calls=0`，`max_abs=0`。覆盖 target/assistant、prefill/decode、D256 SWA 与 D512 full；这是 8-token 验证任务加预热的调用总数。
- Graph 的 128-token 任务：新旧每轮完整 token 序列相同，3/3 重复一致；双方均为 47 轮 draft、94 个候选、80 个接受，接受率 `80/94 = 85.11%`。
- 本轮没有重新跑完整的 B1/B4 × K0/K2 × eager/Graph 20-case 模型矩阵，不从空的 identity 列表推导“全矩阵通过”。

**真实 tok/s：两轮 A/B 都记录，不挑有利的一轮。**

| 测量 | 旧 mini 中位数 [min,max] | 最终 mini 中位数 [min,max] | new/old |
|---|---:|---:|---:|
| 第一轮，旧→新 | 159.42 [158.69,159.50] | 175.09 [170.75,176.59] | 1.098× |
| 复测，新→旧（跟踪脚本） | 175.80 [160.49,178.36] | 175.44 [174.39,176.58] | 0.998× |

复测旧 mini 的速度明显上升，因此**不能确认整模型有稳定约 10% 的收益**：最后一组中位数基本持平。没有锁定整机功耗/CPU 频率与后台环境，尚未定位这部分波动的唯一原因。不能拿第一轮的 159 tok/s 当成永久固定基线。

同批 vLLM 参考为 **172.48 tok/s**（169.52–173.22），但与最终 mini 在第 **112** 个输出 token 已分歧，draft/接受统计也不同（48 轮、96 候选、81 接受）。旧 mini 同样存在这项跨引擎差异；本次修复没有新增它。这个值只能作为量级参照，不能据此宣称本项目已全面超过 vLLM。

为什么 kernel 提速不能直接换算为模型提速？真实 Gemma4 的 D256 层是 SWA，不是本文重点测量的 D256/full；SWA 的数值保护保留了旧 tile，而其 async 版本在部分 shape 还有退化。另外还有 D512 full、prefill、线性层、草稿执行、采样和 CPU 等开销。整模型应看新的逐层 trace 和稳定重复，而不是只乘单 kernel 的 1.21×。

最终原始记录：`async_stride/model/{baseline,optimized,optimized_opcheck}.json`。第一轮在 `baseline_e2e.json` / `final_e2e.json`；早期未保护 SWA 的 `e2e/` 候选数据保留用于解释失败，不用作最终版本的无损依据。


## 9. 对开放问题的逐项回答

1. **哪个 pass？** 第一处 load layout 差异在 Coalesce；更早的原因是 TTIR 的动态 stride 乘法使 AxisInfo 丢掉连续性。随后 AssignLatencies 的 32-bit 门槛过滤了旧 BF16 load。
2. **为什么复刻 shape 仍失败？** TILE/BLOCK_M/warp 都不等于参数的编译期属性；动态 `offs_d * stride` 没消失，换 shape 也无法证明邻接元素连续。
3. **有无高层方法？** 有，传入真实 stride 并声明 `tl.constexpr` 即可。不需要不成立的 layout hint，更不需要跳到 TMA/手写 cp.async。
4. **最小 patch 与收益？** 两行是根因修复；配套 TILE32/BLOCK_M 调整才让 D256 明显提速。保留 FP32/SWA 的旧计算布局，不对所有 dtype/shape 作未经验证的零误差承诺。

下一步不是再枚举二十个开关。先用相同签名、相同 stride、同一编译变体比较 IR；确认编译器能证明哪些事实，再决定是否要调 tile 或并行策略。

## 附录 A：最终 22-case 完整 kernel 数据

单位 μs/kernel；7 轮中位数；`old/new > 1` 才表示加速。所有 case 都固定走 2D，所以 `seg4/8/16/32` 名称不代表此次启用 split-K 3D。

| case | old mini | final mini | vLLM | old/new |
|---|---:|---:|---:|---:|
| qlen1_ctx1024 | 50.14 | 40.96 | 40.47 | 1.22× |
| qlen2_ctx1024 | 50.08 | 41.04 | 40.39 | 1.22× |
| qlen4_ctx1024 | 50.19 | 41.31 | 40.26 | 1.21× |
| qlen8_ctx1024 | 50.63 | 41.38 | 40.56 | 1.22× |
| qlen3_ctx128 | 9.87 | 7.95 | 7.54 | 1.24× |
| qlen3_ctx256 | 15.54 | 12.43 | 12.14 | 1.25× |
| qlen3_ctx512 | 27.04 | 21.92 | 21.39 | 1.23× |
| qlen3_ctx1024 | 49.88 | 41.13 | 40.07 | 1.21× |
| qlen3_ctx2048 | 95.67 | 79.02 | 77.05 | 1.21× |
| qlen3_b2_ctx1024 | 50.49 | 41.60 | 40.23 | 1.21× |
| qlen3_b4_ctx1024 | 50.84 | 41.64 | 40.97 | 1.22× |
| qlen3_b8_ctx1024 | 50.76 | 42.20 | 40.83 | 1.20× |
| qlen3_hd256_win-1 | 49.95 | 41.25 | 40.02 | 1.21× |
| qlen3_hd256_win512 | 27.52 | 29.68 | 24.53 | 0.93× |
| qlen3_hd512_win-1 | 95.01 | 57.97 | 61.53 | 1.64× |
| qlen3_hd512_win512 | 52.26 | 33.81 | 资源超限 | 1.55× |
| qlen3_seg4 | 49.98 | 41.31 | 39.73 | 1.21× |
| qlen3_seg8 | 50.00 | 41.01 | 40.02 | 1.22× |
| qlen3_seg16 | 50.04 | 41.20 | 39.98 | 1.21× |
| qlen3_seg32 | 50.58 | 41.47 | 40.13 | 1.22× |
| qlen3_hd512_win512_b4 | 52.63 | 34.60 | 资源超限 | 1.52× |
| qlen3_hd256_win512_b4 | 28.15 | 31.41 | 24.98 | 0.90× |

## 附录 B：改动清单与未完成边界

| 文件 | 职责 |
|---|---|
| `minivllm/attention/ops/triton_unified_attention.py` | 两处 stride 特化、full-attention launch 参数、FP32/SWA 保护及注释 |
| `tests/step88/test_attention_async_stride.py` | 新旧逐位对照、非连续布局、真实 BF16 async IR、SWA 边界回归 |
| `benchmarks/check_step88_async_stride.py` | 同输入四方 kernel 对照、逐变体 IR/PTX、Graph burst 计时 |
| `benchmarks/check_step88_async_model.py` | 隔离进程新旧真实模型 A/B；实际 attention 输入逐调用差分 |
| `docs/results.json` | 机器可读摘要、源码 hash、命令、数值及性能边界；同步当前包 hash |
| 本文、`docs/README.md`、`minivllm/README.md` | 完整说明与索引；历史开放问题原样保留 |

未声称完成：所有 GPU/shape/dtype 的通用最优、所有 BF16 输入的数学逐位保证、大 head FP32 资源问题、SWA 最快参数、全矩阵模型性能、与 vLLM 完整生成逐 token 恒等。量化/并行/编译器改造不在本次范围。

## 附录 C：本次环境与源码身份

- GPU：NVIDIA GeForce RTX 5090 Laptop GPU，compute capability [12, 0]。
- PyTorch：`2.13.0+cu130`；Triton：`3.7.1`；vLLM：`0.28.0`。
- 对照旧生产提交：`bb1380d`。
- 最终 kernel 文件 SHA256：`82202dd0873c09dfc9d81fbc2b89acc43ec23d5b4afcfdd43db2d5770013a4bd`。
- 最终 `minivllm/*.py` 包摘要（递归 177 文件）：`f28d24752ce90b3c713075fb62cbc85c4f1a251002e81de17001262093bfa4a6`。
- 包摘要算法：按相对路径排序，逐文件 SHA256，拼接 `路径\0哈希\n` 后再 SHA256。
