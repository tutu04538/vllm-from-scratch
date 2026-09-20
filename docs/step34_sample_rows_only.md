# step34：只为需要采样的位置计算 logits

- 对应代码：`step34/`（新增，未提交）
- 包摘要 SHA256：`79366b0eae15b89db418b2c793eec3d1b686f0749615dab25cee39cb7c50391c`
  （`sha256("\n".join(sorted("路径 sha256(文件)")) + "\n")`，只算 `step34/**/*.py`）
- 基线：`step33/` 原样保留、未修改

## 0. 需求大概

原来的调用顺序是：

```text
全部 N 个 token 的 hidden → 最终 RMSNorm → lm_head：[N, vocab_size] → 采样只挑几行
```

**很多行费力算出了整张词表的分数，随后直接被丢弃。** 真实模型 `vocab_size=151936`，BF16 下 `[16, 151936]` 的 logits 约 4.64 MiB；只要 3 行的话是 0.87 MiB。

本关改成：**先挑出需要采样的 hidden，再做最终 norm 和 lm_head。** attention、MLP 和 KV 仍然处理全部本轮输入——中间 chunk 的 token 后面还要靠它们的 KV。

## 1. 改动内容

| 文件 | 内容 |
|---|---|
| `step34/engine.py` | 新增 `_sample_plan()`：从 `scheduled_items` 一次整理出「模型要的行号」与「对应的请求」；`_sample()` 不再按原始行号二次索引 |
| `step34/model.py` | `_forward_append(..., sample_rows=None)`；新增 `_final_logits()`；新增固定地址的 `sample_buffer`；Graph 改按 `(N, M, 是否挑选)` 分键 |

## 2. 设计要点

### 2.1 一次整理出「行号」和「请求」

原来 `last_rows` 是在 `_sample()` 里现算的，那时 logits 已经算完了。现在提到 forward 之前，**同一份 `scheduled_items` 同时产出两样东西**：

```python
rows   = [offset - 1 for 每个 can_sample 的 item]      # 给模型：打包行号
picked = [同一批 item]                                 # 给采样：结果写回哪个请求
```

`offset` 就是该请求片段末行的行号（累加完它的 token 数再减一）。两者的顺序天然一致，**不需要在模型前后各推一套对应关系**。

模型只拿 `rows`，不认识请求对象、scheduler 或 `can_sample`——它收到的就是「这几行要算 logits」。

### 2.2 挑行在前，norm/lm_head 在后

```python
def _final_logits(self, hidden):
    if not self.sample_rows_given:
        return self.lm_head(self.norm(hidden))          # 调试路径：全部行
    if self.num_samples == 0:
        return hidden.new_empty((0, self.vocab_size))   # 没人要采样
    picked = hidden.index_select(0, self.sample_buffer[:self.num_samples])
    return self.lm_head(self.norm(picked))              # 只对 M 行做 norm 和 lm_head
```

关键在顺序：**先 `index_select` 挑出 M 行，再对 M 行做最终 norm 和 lm_head**。如果写成「先算完 `[N, vocab]` 再切片」，那些被丢掉的行照样花了算力和显存——那不是本关要的。

`sample_rows=None` 保留为明确的「全部行」调试入口，用于数值对照；**正常 Engine 路径一律传真实行号**，走的都是精简版。

### 2.3 M = 0

多个请求都还在中间 prefill chunk 时，一轮里没有任何 `can_sample`：

- **仍然执行模型主体**，写入 KV、推进缓存长度（否则后续 token 会缺 KV）；
- 不运行最终 norm、不运行 lm_head、不调用 sampler；
- 返回 `[0, vocab_size]` 的空 logits。**不是把这次 step 整体跳过**——`scheduler.post_step()` 照常走。

### 2.4 Graph 要按 (N, M) 分键，行号还得能换

输入 token 数 `N` 不再是唯一的形状决定因素：同样 `N = 9`，这一轮 `M = 2`、下一轮 `M = 1`，**lm_head 的输入形状和输出形状都变了**，一张图盖不住

```python
graph_key = (num_tokens, self.num_samples, self.sample_rows_given)
```

第三个分量是必要的：`sample_rows=None`（全部行、不索引）和「给了 N 个行号」在 `M == N` 时形状相同但计算不同，不能共用一张图。

**同一个 `(N, M)` 下，行号的值可能不同**（比如 N=4、M=1，这次采第 3 行、下次采第 0 行）。所以行号放在一个固定地址的 `sample_buffer` 里，`_prepare_inputs()` 每轮先写进去，replay 时读的是新内容——和 `input_buffer` / `position_buffer` / `slot_buffer` 是同一套做法，不会让 replay 一直用首次捕获的行号。

### 2.5 采样不再二次索引

```python
def _sample(self, logits, picked):
    if not picked:
        return None
    if logits.shape[0] != len(picked):
        raise RuntimeError(...)
    output_ids = self.sampler.sample(logits.float())
    for item, output_id in zip(picked, output_ids):
        item["request"].output_ids.append(output_id.item())
```

模型返回的两行已经**按 `picked` 的顺序**排好，直接采样后逐条写回。原来那句 `logits[rows, :]` 必须去掉：旧行号 7 在 `[2, vocab_size]` 的第一维上根本不存在了。

`logits.shape[0] != len(picked)` 这条断言留着——它对不上就说明行号与请求的对应关系错位，会**静默地把 token 写到错误的请求上**，值得花一次比较来挡。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，TF32 关闭。

### 3.1 lm_head 实际拿到几行

用 `register_forward_pre_hook` 直接记录 `lm_head` 的输入形状，与 `_sample_plan` 报出的 M 逐轮对比。

小模型（3 条请求，4 步）：

| 本轮 M | lm_head 输入行数 | 采样的请求 |
|---:|---:|---|
| 1 | 1 | A |
| 3 | 3 | A,B,C |
| 3 | 3 | A,B,C |
| 2 | 2 | A,B |

**每一步都相等**，不是「配置上支持、实际还走全量」。

真实模型（19 个 prompt token + 31 个生成 token，31 步）：`N = [19,1,1,...]`，`M = [1,1,1,...]`，`lm_head` 输入行数序列与 M 完全相同，共调用 31 次。

三请求混合负载（预算 8，28 步）：

```
每步 N = [8,8,8,8,8,8,6,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1]
每步 M = [0,0,1,1,2,2,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,1,1]
lm_head 实际调用 26 次，行数序列 = [1,1,2,2,3,3,...,1,1]
去掉 M=0 后与 M 序列完全一致 = True
```

### 3.2 M = 0 的行为

两条长 prompt、预算 4 的小模型：`每步 M = [0, 0, 1, 1, 1]`。前两步没有任何请求可采样，`lm_head` **一次都没被调用**（总调用 4 次，而步数是 5），但两步都正常写入了 KV、推进了长度，后面 3 步正常生成并收尾。

### 3.3 精简路径与全量调试路径一致

同一份权重、同一批请求：一边走正常精简路径，另一边每步用 `sample_rows=None` 走全量路径再自己挑行采样：

```
精简: {'A': (6, 10, 8, 6), 'B': (10, 8, 6)}
全量: {'A': (6, 10, 8, 6), 'B': (10, 8, 6)}
一致 = True
```

### 3.4 Graph

| 检查 | 结果 |
|---|---|
| N=4 分别以 M=1/2/0 调用 | 捕获 3 张图，key = `[(4,0,True), (4,1,True), (4,2,True)]` |
| 输出形状 | `(2,11)` / `(1,11)` / `(0,11)`，M=0 的 dtype/device 正常 |
| **同一 (4,1,True) 下换行号**：先采第 3 行、再采第 0 行 | logits 不同（说明 replay 用的是新行号，不是首次捕获的） |
| 这两行分别与全量路径的第 3 / 第 0 行比 | 都对得上 |

### 3.5 回归

- 小模型 `tiny_gqa` / `tiny_mqa` × norm{torch,triton} × attention{torch,triton} × graph：**12 组全部与 step33 逐位一致**。
- 分块 prefill 预算 1 / 3 / 8、prefix cache（命中 2 块）、零预算请求，都与 step33 相同。
- 真实模型 FP32：norm{torch,triton} × graph{on,off} 四种组合**互相一致**，且与官方参考的 31 个 token 逐位一致。BF16 四种组合除已知的 BF16 舍入敏感性外无新差异。
- 行号校验：越界 `[4]`、负数 `[-1]`、行数超过 N 都明确报 `ValueError`，且在改状态之前拦下。

### 3.6 省下多少行（真实模型，`lm_head` 输入行数统计）

| 负载 | 步数 | ΣN | ΣM | 省下 |
|---|---:|---:|---:|---:|
| 短 prompt + 短生成 | 8 | 20 | 8 | **60%** |
| 长 prompt + 短生成 | 8 | 32 | 8 | **75%** |
| 三条不同长度（含 prefill 混合） | 24 | 101 | 56 | **45%** |

**prefill 越长、生成越短，省得越多**——正是这一关针对的场景。纯 decode 时通常 N = M，省不到。

### 3.7 对验收脚本的影响（重要）

`_forward_append` 多了参数、`model.graphs` 换了键，**上一版验收脚本会大面积报错，但全部是测试侧签名问题，不是行为回归**：

| 脚本 | 对 step33 | 对 step34 | 失败原因 |
|---|---:|---:|---|
| `verify_step29_contract.py` | 96/96 | 61/96 | 25× `wrapped()` 不收 `sample_rows` + 10× `graphs[N]` 键 |
| `verify_step29_io_contract.py` | 45/45 | 36/45 | 同上 |
| `verify_step29_qwen3.py` | 21/21 | 8/21 | 13× `wrapped()` 不收 `sample_rows` |
| `verify_step33_norm.py` | 56/56 | 38/56 | 18× `wrapped()` 不收 `sample_rows` |

四个脚本的 harness 都是这个形状：

```python
def wrapped(ids, counts, caches, pool):
    ...
    out = f(ids, counts, caches, pool)      # ← 4 个参数写死
```

引擎现在调的是 `_forward_append(..., sample_rows=rows)`，于是 mock 直接抛 `TypeError`。另外 `KeyError: 5` / `KeyError: 3` 来自 `model.graphs[n]`，键现在是 `(n, M, given)`。

两处改动都是本关需求明确要求的（「可以扩展 `_forward_append()` 的参数」「Graph 不能再只考虑 N，按 (N,M) 区分图」），所以旧 harness 需要跟着改一行签名。**行为本身没有回归**：§3.5 的 12 组小模型对比与真实模型 31 个 token 都能证明。

## 4. 接口变化与遗留

- `TinyCausalLM._forward_append(..., sample_rows=None)`：给定时只返回这些行的 logits `[M, vocab_size]`；`None` 时返回全部行 `[N, vocab_size]`（调试用）。
- 新增属性 `model.num_samples`、`model.sample_rows_given`；新增固定缓冲 `model.sample_buffer`。
- `model.graphs` / `model.graph_outputs` 的键从 `N` 变为 `(N, M, sample_rows_given)`。
- 新增 `Engine._sample_plan(scheduled_items) -> (rows, picked)`；`Engine._sample(logits, picked)` 的第二个参数由 `scheduled_items` 换成 `picked`。
- 没有改调度策略、KV、attention 或 norm 后端；`config.json` 与权重格式不变。
- 遗留：没有做 padding bucket、动态图编译或图缓存淘汰——`(N, M)` 组合变多会让捕获的图数量增加（每个新组合一张）。
- 遗留：纯 decode 负载下 `N == M`，本关不带来收益。
- 未提交。
