# step36 附：从 step35 移除 beam search 与 Triton 采样 kernel

- 对应代码：`step35/`（**就地修改**，未新建包）
- 删除前的存档提交：`31c8ba8`（"保存第 33–36 关实现、vLLM 对照基准与剖析结果"）
- 删除前的包摘要 SHA256：`3f260b0caed96d4e7936ad1b3ec201a9db278d26b67671a4491890965201d3b0`（15 个 .py）
- 删除后的包摘要 SHA256：`a00a46d92b79d18a1af832b271e593bcd91ed34053af29438342826869cd93f1`（13 个 .py，见 §4.2 复核方式）

## 0. 需求大概

beam search 与 Triton 采样 kernel 这两项目前不列入研究主线，要求从 `step35/` 中删除。

背景是前两关已经把主线收敛清楚：

- `136` 把「Beam COW、更复杂的采样」列为**选修**，不算当前完成条件。
- `133` 把 beam COW 从第三十六关的任务里移出，主线改为「先测清与 vLLM 的差距」。
- 本仓库 `docs/step36_benchmark_gap.md` 的结论是：下一个该动的地方是 **attention**，
  sampler 在最大差距（`prefill_c8`）里根本不占 GPU 时间。

所以保留这两个模块只会让主线的代码量与实际研究方向脱节。

**要删的**：`step35/beam.py`（293 行）、`step35/triton_sampling.py`（172 行）。
**要留的**：采样策略本身（`sampling.py`：`SamplingParams` / `SamplingState` /
`apply_penalties` / `TorchSampler`）。删除的是「另一条挑 token 的执行路径」和
「另一种生成策略」，不是采样能力。

## 1. 改动内容

| 文件 | 变化 |
|---|---|
| `step35/beam.py` | **删除**（293 行） |
| `step35/triton_sampling.py` | **删除**（172 行） |
| `step35/engine.py` | 删 `make_sampler()`；删 `Engine.beam_search()`；删 `sampler_backend` 参数与合法性校验（`__init__` / `_init_runtime` / `from_model_dir` 三处签名）；采样器固定为 `TorchSampler()` |
| `step35/step35.py` | 删 `--sampler-backend`、`--beam-width`、`--length-penalty`；`--mode` 取值由 `{greedy,random,beam}` 收为 `{greedy,random}`；删 beam 分支；删随之变成死代码的 `_or_default()`；`enable_prefix_caching` 不再带 `and args.mode != "beam"` 条件 |
| `step35/sampling.py` | 模块 docstring 由「三条路径」改为「两条路径」 |
| `docs/step35_sampling_beam_triton.md` | **原文保留不改写**，开头加删除说明 |
| `docs/step36_benchmark_gap.md` | 加后续变更说明（基准数字对应删除前的指纹，仍然有效） |

`step35/` 从 15 个 .py / 2283 行降到 13 个 .py / 1753 行。

## 2. 设计要点

### 2.1 为什么就地改而不是新建 step36/

本项目的惯例是每一步留一个冻结副本（`step30/` → … → `step35/`），按惯例本该复制出
`step36/` 再删。这次没有这么做，原因和代价都写清楚：

- 这两个模块**不在任何执行路径上**。`beam.py` 只被 `Engine.beam_search()` 引用，
  `triton_sampling.py` 只被 `make_sampler()` 引用，两者都不参与普通生成。
- 删掉它们不改变任何一次普通 forward 的数值，也不改变 attention / 模型 / 调度 / KV。
  换句话说，**「冻结 step35 作为对照」的价值在这两个模块上是零**。
- 代价是：包摘要失效、`docs/step36_benchmark_gap.md` 的基线指纹变成历史值。

因为 `step35/` 当时**未被 git 跟踪**（`?? step35/`），直接删会不可恢复，所以先做了
一次存档提交 `31c8ba8`，把删除前的完整状态固定下来。要回到删除前，`git checkout 31c8ba8 -- step35/`。

### 2.2 删除的边界划在哪里

三个判断，都遵循「只删需求点名的两样」：

- **`sampling.py` 全部保留。** 温度、top-k、top-p、三类惩罚、请求级随机状态，
  是普通采样的组成部分，不是「Triton sampler」。删掉 `triton_sampling.py` 只是
  换回 Torch 挑 token，采样语义一个字没变。
- **`--length-penalty` 一并删除。** 它只被 beam 分支消费（`beam.py` 的
  `_final_score`），没有第二个使用者；留着会变成一个「传了没反应」的哑参数。
- **`_or_default()` 一并删除。** 它是为 beam 分支里「显式传 0 不能被 `or` 吞掉」
  而加的。普通路径用的是 `if value is not None: request[key] = value`，本来就
  正确处理了显式 0，所以 beam 一走它就没人调用了。**这里特意确认过**：删它不是
  删掉一个保护，而是删掉一个重复的保护。

### 2.3 `self.sampler_backend` 属性没有保留

`engine.sampler_backend` 只是 `step35.py` 打印用（"采样后端：triton"）。现在只有一条
路径，留一个恒为 `"torch"` 的属性属于自造状态。改成直接打印 `engine.sampler.name`
（`TorchSampler` 类自带的），信息量相同且不会和实际执行路径不一致。

## 3. 验证

### 3.1 功能

| 检查 | 结果 |
|---|---|
| `import step35` | OK |
| `Engine.beam_search` 已不存在 | `hasattr(...) == False` |
| `from_model_dir` 不再含 `sampler_backend` | 源码检查通过 |
| `step35/*.py` 残留引用 `beam` / `triton_sampling` / `TritonSampler` / `sampler_backend` | 仅剩 `engine.py` 一句注释（说明移除原因），无代码引用 |
| `--help` | `--mode {greedy,random}`，三个已删参数不再出现 |
| greedy 实跑（真实 Qwen3-0.6B，BF16 + Triton attention + CUDA） | 正常，输出「你好！有什么可以帮助你的吗？😊」，13 prompt + 9 生成，0.64s |
| **EOS 正常终止** | 通过——`--max-new-tokens 12` 实际只生成 9 个就停了，说明停止路径没被删坏 |
| random 模式（temperature 0.8 / top-k 20 / top-p 0.9 / repetition 1.1 / seed 42） | 正常，0.74s |
| `benchmarks/bench_step36_vllm_compare.py`（以 step35 为基线） | 正常，`short_c1` 0.1597s，输出长度断言通过 |
| `benchmarks/check_step36_capacity_gap.py` | 正常，容量缺口复现结果不变 |

### 3.2 验收脚本影响（实测，非推测）

验收方在另一仓库共有 10 个 `verify_step35_*.py`，全部实跑：

**仍然全过（8 个，355/355）**

| 脚本 | 结果 |
|---|---|
| `verify_step35_contract.py` | 96 / 96 |
| `verify_step35_external.py` | 53 / 53 |
| `verify_step35_io_contract.py` | 45 / 45 |
| `verify_step35_norm.py` | 56 / 56 |
| `verify_step35_selection.py` | 32 / 32 |
| `verify_step35_features.py` | 29 / 29 |
| `verify_step35_precision.py` | 23 / 23 |
| `verify_step35_qwen3.py` | 21 / 21 |

**导入即失败（2 个，98 项）**

| 脚本 | 失败原因 |
|---|---|
| `verify_step35_sampling_beam.py` | 第 12 行 `from step35.triton_sampling import TritonSampler,_pick_batch` → `ModuleNotFoundError` |
| `verify_step35_fixed_boundaries.py` | 第 9 行同样的导入 → `ModuleNotFoundError` |

这是本次删除的**已知且已被接受**的代价：这两个脚本测的正是被删掉的功能，模块没了，
导入就断了，属于预期结果而非回归。删除前它们分别是 81/81 和其余部分，合计 98 项；
453 − 355 = 98 与之一致。

**注意**：这两个脚本在验收方仓库里，本仓库无权修改。如果后续希望它们恢复，
需要验收方把导入改成可选、或按删除后的接口重写；在那边处理之前，这两个脚本会一直
在导入阶段失败。

### 3.3 这次删除没有重跑的性能验证

删掉的代码不在执行路径上，但**没有重跑六个测点矩阵**。单测点抽查（`short_c1`，
`--warmup 1 --reps 2`）得到 0.1597s，与删除前的 0.1761s（3 进程 × 5 次的正式中位）
不同。两次的预热/重复次数不一样，**不能据此说删完变快了**，差异应在噪声量级内。
需要正式结论就得重跑 `benchmarks/run_step36_matrix.py`。

## 4. 接口变化与遗留

### 4.1 移除的公开接口

| 接口 | 类型 | 替代 |
|---|---|---|
| `step35.beam.BeamSearch` 及其 `BeamCandidate` / `_top_tokens` / `_copy_kv` / `_reparent` | 模块 | 无 |
| `step35.triton_sampling.TritonSampler` / `_pick_batch` / 两个 argmax kernel / gumbel kernel | 模块 | `sampling.TorchSampler` |
| `Engine.beam_search(...)` | 方法 | 无 |
| `Engine(..., sampler_backend=...)`、`Engine.from_model_dir(..., sampler_backend=...)` | 关键字参数 | 已移除；传了会 `TypeError` |
| `engine.sampler_backend` | 属性 | `engine.sampler.name` |
| `--sampler-backend` / `--beam-width` / `--length-penalty` | CLI | 已移除；传了 `argparse` 报错 |
| `--mode beam` | CLI 取值 | 已移除；`--mode` 只剩 `{greedy, random}` |

**保留不动**：`SamplingParams`、`SamplingState`、`apply_penalties`、
`TorchSampler`（含 `select` 与 `select_batch`）、`step35.sampler.Sampler`。

### 4.2 包摘要复核方式

删除后重算：

```bash
python -c "
import hashlib,pathlib
files=sorted(pathlib.Path('step35').rglob('*.py'))
lines=[f'{p} {hashlib.sha256(p.read_bytes()).hexdigest()}' for p in files]
print(len(files), hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest())
"
```

### 4.3 遗留

1. **`verify_step35_fixed_boundaries.py` 的两个失败与本次删除无关的部分没有单独验证。**
   它整份脚本在导入阶段就挂了，所以它测的其他边界项（不只是 Triton 采样那几条）
   这次**没有得到验证**。如果那些项仍然重要，需要把导入改成可选后重跑。
2. **`docs/README.md` 的 step35 索引行加了「后两者已移除」的括注**，正文按
   「保留原文」要求未动。索引是导航，指向的是文档当前内容与代码现状，所以标注了；
   `step35_sampling_beam_triton.md` 正文则完整保留当时的实现与验证过程。
3. **删除后未重跑六个测点基准**（见 §3.3）。
4. 若要恢复，`git checkout 31c8ba8 -- step35/` 可回到删除前；两个模块的完整实现
   与验证记录都在那个提交和 `docs/step35_sampling_beam_triton.md` 里。
