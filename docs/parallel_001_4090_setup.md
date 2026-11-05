# 并行开发 001：4090×4 服务器环境

日期：2026-11-05。目的：把现有项目和学习资料迁到四卡服务器，准备能真正运行 CUDA/Triton/NCCL 的开发环境。**本次只搭环境，没有实现 TP/PP，也没有把四张卡自动合成一张 96 GiB 的卡。**

## 1. 日常怎么用

```bash
ssh 4090X4-NSSL
cd ~/infra/vllm-from-scratch
source scripts/activate-4090.sh

python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'

# 单进程验收：每张卡的 BF16 matmul、Triton、CUDA Graph，及默认 C++ UVA
python benchmarks/check_parallel_environment.py --single \
  --output benchmarks/results/parallel_setup/single_gpu.json

# 四进程，每进程绑定一张卡；包含通信结果的数值断言，不是只初始化一下
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 \
  benchmarks/check_parallel_environment.py \
  --output benchmarks/results/parallel_setup/nccl_4gpu.json

# 普通回归先固定用一张卡，避免把测试进程和多卡实验混在一起
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/step73 tests/step86 -q
```

`activate-4090.sh` 要在仓库根目录 **source**，不要用 `bash scripts/activate-4090.sh`，否则环境变量只留在子进程里。脚本不更改系统驱动，不全局关闭 NCCL P2P，也不改其他项目环境。

## 2. 仓库和需求资料在哪里

| 内容 | 服务器位置 / 状态 |
|---|---|
| 仓库 | `~/infra/vllm-from-scratch` |
| origin | `git@github.com:tutu04538/vllm-from-scratch.git`，SSH 读取已验证 |
| 分支 | `master` |
| 私有学习资料 | `learning_notes/14_vllm_from_scratch/` |
| 完整投机需求 | 上述目录的 `投机解码完整需求/` |
| 单卡优化需求 | 上述目录的 `单卡执行优化需求/` |
| 冻结参考源码、校验脚本 | 两个需求子目录内的 `参考资料/`，随资料一起复制 |
| Python 环境 | 仓库内 `.venv/` |
| 私有 CUDA 编译工具链 | `~/infra/cuda-12.9/` |
| 安装 / 测试原始日志 | `benchmarks/results/parallel_setup/` |

原有 **270 篇 Markdown** 按原目录结构复制，逐文件 SHA256 对账；不是只复制最后几关。参考资料也复制了，没有携带历史 profiler 大产物。原始本机文档仍保留，同时在本机 `vllm-from-scratch/learning_notes/` 保存一份。

`.gitignore` 新增 `/learning_notes/`；`.venv/`、`AGENTS.md` 和 benchmark 结果原本就被忽略。**忽略不等于 Git 会同步**：代码通过 Git 更新，后续私有需求需另行 `rsync/scp`。服务器也放置了本地 `AGENTS.md`，将当前仓库和需求目录指向服务器路径。

## 3. 为什么不直接复用原来的环境

检查到的已有环境包括：

- `infra/flash-attention-lab/.venv`：Python 3.10，Torch 2.5.1+cu121，Triton 3.1。
- conda `torch`：Torch 2.5.1+cu121；`nano-vllm`：Torch 2.4；`treegrpo4090`：vLLM 0.6.3。

它们与本项目的 vLLM 0.28 / Triton 3.7.1 基线差距较大，所以没有原地升级这些环境，而是创建独立 `.venv`。

另一处差异在驱动：服务器为 **535.309.01**，`nvidia-smi` 显示 CUDA 12.2；本机环境使用 CUDA 13。这个显示值不是“已安装的 nvcc 版本”。检查发现服务器默认 `nvcc` 实际为 11.5，直接用它编译新 Torch 的扩展也不合适。

处理方式：

1. Python 包使用 **CUDA 12.9 构建的 Torch 和 vLLM**，保持项目主要 Python 包版本，不升级系统驱动。
2. vLLM **显式安装官方 `+cu129` wheel**；仅指定 Torch 的 CUDA index，不保证 vLLM wheel 也跟着变成 cu129。
3. 安装独立 CUDA 12.9 编译工具链，用于项目默认 C++ UVA 扩展；不覆盖 `/usr/bin/nvcc`。
4. `TRITON_PTXAS_PATH` 指向这套工具链，避免 Triton 使用另一套 CUDA 13 汇编器；`CUDA_INC_PATH` 指向 conda 的 `targets/x86_64-linux/include`。

NVIDIA 的 CUDA 12.x minor-version compatibility 覆盖 525 及更新驱动，但有新特性/PTX 等限制，**不能只看到版本表就宣布所有操作可用**。这里仍以 GPU 运算、内核编译、CUDA Graph、UVA 和 NCCL 的实际测试为准。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

官方 vLLM 0.28 发布页确实提供独立 `+cu129` wheel；这不是修改已安装源码或用旧版 vLLM 冒充新版本。[官方发布文件](https://github.com/vllm-project/vllm/releases/expanded_assets/v0.28.0)

## 4. 重建环境的命令

下列命令用于新目录重建；已配置好的服务器直接 source 即可，不必反复安装。

```bash
cd ~/infra/vllm-from-scratch
bash scripts/setup-4090.sh
source scripts/activate-4090.sh
```

脚本分阶段处理，避免安装过程中遇到的三个问题：

- **Python 头文件**：系统 Python 3.12.13 缺 `Python.h`，会导致 Triton 的 launcher 和 Arctic C++ 编译失败。最终使用 **uv managed Python 3.12.13**（包含开发头文件），不是给系统目录乱加软链接。
- **下载速度**：PyTorch R2 源在本服务器很慢；本次从官方 `download.pytorch.org` 分段下载 Torch/Triton，并与官方 index 中的 SHA256 对账。缓存放在 `~/infra/.wheels/`；脚本有缓存时先校验再使用，没有时走正常官方安装。
- **隔离构建**：Arctic 声明的构建依赖包含 Torch 和 protobuf 5.29.6。将这些依赖直接装进 vLLM 运行环境会冲突。脚本先装运行依赖，再通过独立构建环境安装 Arctic；`--build-constraint` 固定构建依赖，并复用已校验的 Torch/Triton wheel，避免在隔离环境重复慢速下载。运行环境仍使用 vLLM 需要的 protobuf 6。

主要依赖在 `requirements/4090-dev.txt`，隔离构建约束在 `requirements/4090-build.txt`；完整运行依赖锁定在 `requirements/4090-dev.lock.txt`，安装脚本优先使用它；测试记录见下节。不要在服务器上直接安装本机完整 freeze：其中包含 CUDA 13 组合以及其他项目的依赖。

## 5. 实测验收

环境核心验证已通过，**项目全量回归尚不能认定稳定通过**。本次没有为消除失败而修改生产代码、禁用 Graph 或降低断言。

| 检查 | 实际结果 |
|---|---|
| 依赖一致性 | `uv pip check`：200 个包无依赖冲突；安装脚本重复执行成功 |
| 四张卡分别运行 | BF16 matmul、Triton kernel、CUDA Graph replay：4/4 通过 |
| 默认 UVA | C++ 扩展编译、CPU 修改后 GPU 别名读取：通过 |
| 上游 CUDA 扩展 | vLLM RMSNorm 数值断言：通过 |
| 四进程 NCCL | all_reduce / all_gather / reduce_scatter / broadcast / 环形 send-recv：4/4 rank 数值通过 |
| 本机迁移相关单测 | step62：37 passed，24 warnings，10.19 s |
| 服务器全量第 1 次 | 1287 passed，1 failed，23 skipped，209.36 s；唯一失败为缺失 assistant config.json |
| 服务器全量第 2 次 | 1275 passed，13 failed，23 skipped，101.31 s；CUDA Graph 分配器断言 |
| Graph 文件单独复测 | step69/test_spec_cudagraph.py：19 passed，1 warning，3.37 s |

实际版本：Python 3.12.13、Torch 2.13.0+cu129、Triton 3.7.1、vLLM 0.28.0+cu129、Transformers 5.14.1、Arctic 0.3.0、NCCL 2.29.7、nvcc 12.9.86。

全量命令（bash/zsh 均可）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  $(printf 'tests/step%s ' $(seq 58 88)) -q -rs
```

#### 缺少真实模型资料

第一次运行缺少 `models/gemma-4-E2B-it-assistant/config.json`。随后从本机复制了该 **配置文件**，SHA256 为 `7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9`；对应失败不再出现。**没有复制模型权重或 tokenizer**。23 个 skipped 是既有真实模型测试因本地资料不足而跳过，不能算通过；后续选定并行学习模型后再准备权重。

#### 尚未关闭的问题：长进程 CUDA Graph 回归

第二次全量运行首先在 `test_polluted_padding_buffer_does_not_change_valid_output` 的第二个 engine 捕获图时出现：

```text
CUDACachingAllocator.cpp:3332
it->second->use_count > 0 INTERNAL ASSERT FAILED
```

栈落在 `minivllm/compilation/cuda_graph.py` 的 `torch.cuda.graph(..., pool=pool)`，本进程使用共享 graph pool；后续 step69 / step71 共 13 项失败均报同一断言。第一次全量没有这些失败，整个 step69 文件单独运行也全部通过，说明不能简单归因于“4090 不支持 Graph”。

**根因未确认**：共享池/图的生命周期、PyTorch 分配器行为及 CUDA 环境差异仍需隔离验证，不把猜测当作修复结论。本次限定为环境迁移，不顺带改动已有引擎。原始三类日志分别为 `full_pytest_initial.log`、`full_pytest.log` / `.xml`、`graph_isolated.log`，保存在 `benchmarks/results/parallel_setup/`。

第三次重新启动进程、重复全量运行仍得到 **1275 passed / 13 failed / 23 skipped（111.29 s）**，同一组 Graph 相关用例触发相同断言，日志 `full_pytest_repeat.log` / `.xml`。因此此次交付是“基础环境与四卡通信已验证，项目长进程 Graph 回归问题待解决”，不是一次全绿验收。


### 测试路径的可移植性修正

迁移时发现 `tests/step62/test_custom_proposer.py` 将上游源码写死在本机 `/home/user/anaconda3/...`。改为通过当前环境的 `vllm.__file__` 定位，同时断言基础版本是 0.28.0；原有源码和功能断言均保留，没有为了服务器跑绿而删测试或放宽容差。生产 `minivllm/` 未改动。

## 6. 四卡并行前要知道的硬件事实

四张卡均为 RTX 4090，每卡约 24 GiB。拓扑报告中没有 NVLink：

- GPU 0/1 在 NUMA 0，GPU 2/3 在 NUMA 1。
- 0↔1、2↔3 为 `NODE`；跨组为 `SYS`，要经过 CPU 间互联。
- GPU 间通信不是免费操作，后续学习 TP/PP 时需要测量通信开销。

NCCL 可以根据设备和拓扑选择传输路径；不应为了“跑通”永久写死 `NCCL_P2P_DISABLE=1` 等开关。本次通信测试的数值通过，也不等价于已经完成 TP 推理或达到最优通信带宽。

实测 `torch.cuda.can_device_access_peer` 的所有非对角项均为 false；NCCL 日志仍显示经 `SHM/direct/direct` 完成通信。不存在 CUDA peer access 不等于 NCCL 不可用。
