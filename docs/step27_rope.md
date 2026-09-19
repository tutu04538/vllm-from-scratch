# step27：RoPE 与缓存位置一致性

- 对应代码：`step27/step27.py`（新增，未提交）
- 源码 SHA256：`5330e6aed1dd50fc08f909dc47f14ec7331721f40e77e3cd737a38f77b265be9`
- 基线：`step26/step26.py`（SHA256 `004c6594…`）原样保留、未修改

## 0. 需求大概

去掉 learned position embedding，把位置信息改成**每层对 Q/K 做 RoPE 旋转**，并且保证分块 prefill、共享前缀、CUDA Graph 下位置始终正确。

```text
旧：token embedding + position embedding → decoder 层
新：token embedding → decoder 层
                        norm1 → QKV 投影 → 按逻辑位置旋转 Q/K（V 不动）
                                        → 写入本层 KV → attention
```

保留多层、GQA、FP32 和随机权重；不加真实权重、tokenizer、Q/K norm；**不恢复**已删除的完整模型旧入口。顺手删掉闲置的 `KVCachePool.append_batch()`。

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `_rotate_half` / `RotaryEmbedding` | 新增：前后半段配对；初始化时算好 `cos/sin` 表，按逻辑位置查表 |
| `TinyCausalLM.__init__` | 删掉 `position_embedding`，换成 `self.rotary`；新增 `rope_theta`；校验 `head_dim` 为正偶数（`head_dim=0` 也要拒绝）、theta 为正 |
| `TinyCausalLM._input_embeds` | 只剩 token embedding |
| `TinyCausalLM._layer_qkv` | 新增 `positions` 参数，投影后旋转 Q/K，V 不旋转 |
| `TinyCausalLM._attention_out` / `gpu_forward` / `_torch_forward` | 往下传 positions（各层共用同一组逻辑位置） |
| `TinyCausalLM._prepare_inputs` | **去掉 clamp**，位置上下界都用 Python 整数查（负位置也要拒绝）；所有检查提到改状态之前 |
| `AttentionMetadata.validate` | 新增：只做容量检查、不碰缓冲；`fill()` 内部也调它 |
| `KVCachePool.append_batch` | 删除（已无调用方） |
| `Engine.__init__` | 新增 `rope_theta=10000.0` |

## 2. 设计要点

### 2.1 配对方式：前后半段

`head_dim=8` 时配对是 `(0,4) (1,5) (2,6) (3,7)`，不是相邻的 `(0,1)`。

```python
def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

out = x * cos + _rotate_half(x) * sin
```

展开一对 `(a, b) = (x[i], x[i+half])`：

```text
a_new = a*cos - b*sin
b_new = b*cos + a*sin
```

与需求 §3 的公式一致。`cos/sin` 由 `freqs = outer(positions, inv_freq)` 查表得到，其中 `inv_freq[i] = theta ** (-2i/head_dim)`，再把 `[head_dim/2]` 沿最后一维拼两份以对齐 `rotate_half` 的布局。

rotary 是正交变换，所以每对的模长保持不变——这也是验证里的一条。

### 2.2 位置：逻辑位置，不是打包行号

```text
三个请求旧长度 [3,0,0]，本轮追加 [1,3,1]
packed 行号    [0,1,2,3,4]
正确的位置     [3,0,1,2,0]
```

位置在 `_prepare_inputs` 里用**写入前的 `cache.length`** 逐请求生成，写进 `position_buffer`。各层共用同一份 `positions`，但各自旋转自己投影出的 Q/K。

物理块号完全不参与位置计算——缓存换一块物理存储，数学结果不变（测试里用不同提交顺序制造了不同的物理块分配，结果一致）。

### 2.3 只旋转一次：缓存里存旋转后的 K

```text
新 token：先旋转 K，再写进它的 slot
历史 token：从缓存直接读，不再旋转
```

V 全程不旋转。这样 attention kernel 一行都不用改——它照旧只负责读 KV 和算 attention，online softmax 也没动。

实测：跑完多轮后，最早写入的那几个位置的 K **逐位未变**；缓存里的 K 等于"投影后旋转"的结果，也等于"未旋转"的比较为 False；V 等于未旋转的 V。

### 2.4 位置上下界都查，而且不留下副作用

第 26 关是 `torch.clamp(position_ids, max=max_seq_len-1)`，本关去掉 clamp，改成区间检查 `0 <= position < max_seq_len`。

**两个界都要查，而且用 Python 整数查**——不必先建 GPU position Tensor 再取回主机来判断：

```python
for _cache, count in zip(past_kv, num_scheduled_tokens):
    if count == 0:
        continue
    start = _cache.length
    if start < 0:
        raise ValueError(f"请求的缓存长度 {start} 为负，位置必须从 0 开始")
    if start + count > self.max_seq_len:
        raise ValueError(f"请求的位置区间 [{start}, {start + count}) 超出 max_seq_len=...")
```

`start + count > max_seq_len` 等价于"最大位置 `start+count-1 >= max_seq_len`"，但用的是整数比较，不碰 Tensor。

**所有检查都必须在推进长度、写 KV、上传元数据之前**。查代码时还发现一个既有问题：元数据容量检查原来在 `AttentionMetadata.fill()` 里，而 `fill()` 发生在长度推进之后——容量超限时长度已经被推进了。这次把容量检查抽成 `AttentionMetadata.validate()`，在 `_prepare_inputs` 里统一前置，三条错误路径都不再改动状态。

实测：

```text
位置越界（max_seq_len=5、block_size=4）:
  报错: 请求的位置区间 [7, 9) 超出 max_seq_len=8 的支持上限
  cache.length 未推进

负位置（注入 length=-1）:
  cpu/cuda × count=1/2 四种组合全部报错，
  cache.length 未推进、位置缓冲未改、元数据缓冲未改
```

注意一个几何关系：`Kmax = ceil(max_seq_len / block_size)`，而块表长度是 `ceil((prompt+max_new-1)/block_size)`。两者都从 `max_seq_len` 推出来，所以**位置越界只在 block_size ≥ 2 时可达**——block_size=1 时容量检查一定先触发。

### 2.5 Graph

- `cos/sin` 表是 `register_buffer(..., persistent=False)`：会跟着 `.to(device)` 走，但**不进 `state_dict`**（它是查表数据，不是可学参数）；
- 位置来自固定的 `position_buffer`，地址稳定，图内 `cos_table[positions]` 每次 replay 读到的都是新内容；
- 捕获/预热不推进 Python 状态——和上一关一样，长度只前进一次。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 旋转公式本身

- **与逐对公式对照**：参考实现直接照 `a*cos-b*sin` 逐对循环算，不调用模型里的实现。`theta = 10000 / 500 / 1000000` 三档，最大误差 `1.2e-07 ~ 2.4e-07`。
- **位置 0 不旋转**：输出与输入逐位相同。
- **非零位置确实变了**：位置 1、5 都与原向量不同。
- **旋转是正交的**：前后半段配对后每对的模长保持不变。

### 3.2 缓存内容

- 缓存 K == 旋转后的 K：True；== 未旋转的 K：False；缓存 V == 未旋转的 V：True。
- 多轮之后老位置的 K **逐位未变**（不会被再次旋转）。

### 3.3 数值一致

- **1/2/3 层 × MHA/GQA/MQA × torch/triton**：9 种配置 × 2 后端 = **36 条逐 token 对照，不一致 0**。
- **逐步逐层 KV**：12 组场景、**85 次逐层核对，不符 0**。
- **整段 prefill vs 分块 prefill（预算 1/3/16）vs 后续 decode**：三者输出一致，且都与独立参考一致。
- **提交顺序 A,B / B,A / C,A,B**（制造不同的物理块分配）：各自都与独立参考一致。

### 3.4 位置与缓存边界

- **prefix 命中**：B 命中 2 个块、`cache.length = 11`，从命中长度之后继续；输出与"完整 prompt 直接算"一致（不从 0 重来）。
- **Graph**：21 组随机场景，eager 与 graph 逐轮返回**不一致 0**——同 N 改变请求与位置时读到的是新值。
- **越界**：位置到 `max_seq_len` 时明确报错，`cache.length` 停在最后一次成功推进处，没有缓存条目被写入。
- **负位置**：注入 `length=-1`，cpu/cuda × count=1/2 四种组合都先报错，长度不推进、位置缓冲与元数据缓冲都不被修改。
- **大上限**：`max_seq_len=128`、prompt 40 个 token（位置 0..39），输出与独立参考一致。旧版会把 `position>=32` clamp 到 31，这里不再 clamp。

### 3.4.1 验收方脚本

`验收记录/tools/verify_step27_contract.py`（96 项）在补齐上面两处校验后 **96/96 通过**。第一次验收是 91/96，5 项失败分别是：负位置两种输入 × CPU/GPU 两条准备路径共 4 项，`head_dim=0` 配置 1 项。

### 3.5 既有回归

- prefix cache 开关输出对照 300 组：不一致 0。
- 索引双射 + 无引用泄漏：300 组场景 / 60027 次 step 逐步检查通过。

### 3.6 计时（粗测，同机同配置，三轮取最快）

8 请求 × prompt 500 + 生成 24，`d_model=64`、`intermediate_size=256`：

| 层数 | 后端 | 旧：位置 embedding | 新：RoPE |
|---:|---|---:|---:|
| 1 | torch | 1517.9 ms | 1075.5 ms |
| 1 | triton | 114.6 ms | 164.5 ms |
| 1 | triton+graph | 101.0 ms | 111.7 ms |
| 2 | torch | 2073.2 ms | 2648.6 ms |
| 2 | triton | 179.3 ms | 219.2 ms |
| 2 | triton+graph | 114.4 ms | 135.0 ms |

两种位置编码的参数量和算子数不同，**不能读成"RoPE 更快"或"更慢"**：这张表里 RoPE 有快有慢，差异在测量噪声量级。正式测量由验收方按同配置进行。

## 4. 接口变化与遗留

- `Engine(...)` / `TinyCausalLM(...)` 新增 `rope_theta=10000.0`。
- `TinyCausalLM` 新增 `self.rotary` 与 `self.rope_theta`；**删除 `position_embedding`**（`state_dict` 少一项）。
- `AttentionMetadata.validate(past_kv, num_scheduled_tokens)` 新增；`fill()` 内部也会调它。
- `KVCachePool.append_batch()` **已删除**（token 写入一直走 `_write_kv` 的 `index_copy_`）。
- `head_dim` 必须是**正**偶数：`head_dim=13`（奇数）与 `head_dim=0` 都在模型初始化时明确拒绝，不再拖到建 KV 池时才因零元素 reshape 抛 `RuntimeError`。
- 遗留：只支持固定的普通 RoPE（`theta` 可配），没有 NTK / 线性缩放等长上下文扩展。
- 遗留：`cos/sin` 表在初始化时按 `max_seq_len` 预计算，上限固定。
- 遗留：`from torch import ceil` 与 `from more_itertools import last` 仍未使用，是前几关带下来的。
- 未提交。
