# step26：多层 decoder 与逐层 KV 缓存

- 对应代码：`step26/step26.py`（新增，未提交）
- 源码 SHA256：`004c6594de8e3503c11a7899a51a51314d9c53c7c0ce3f7fa5d2b69686a1c2d2`
- 基线：`step25/step25.py`（SHA256 `8ab7b18e…`）原样保留、未修改

## 0. 需求大概

前面所有关卡里模型都是"embedding → 一次 attention → o_proj → lm_head"。本关换成真正的堆叠结构：**同一批 token 连续经过多个 decoder 层**，每层补齐归一化、残差和前馈网络，并且**每层有自己独立的 KV 缓存**。

保留 learned position embedding、FP32、随机权重和现有 Engine；不引入 RoPE、量化或多 GPU。

```text
打包 token IDs
    ↓
token embedding + position embedding     只做一次
    ↓
decoder 第 0 层                           使用第 0 层 KV
    ↓
decoder 第 1 层                           使用第 1 层 KV
    ↓
最终 RMSNorm
    ↓
lm_head                                  只做一次
```

## 1. 改动内容

| 位置 | 改动 |
|---|---|
| `RMSNorm` | 新增：`x / sqrt(mean(x², 最后一维) + eps) * weight`，逐 token，无 bias |
| `DecoderLayer` | 新增：两个 norm、Q/K/V/O、SwiGLU 三投影，`forward` 负责 `h + MLP(RMSNorm_2(h))` |
| `TinyCausalLM.__init__` | 新增 `num_layers` / `intermediate_size` / `rms_norm_eps` 与校验；投影移到层里；新增最终 `norm` |
| `TinyCausalLM._input_embeds` | 新增：embedding 只做一次 |
| `TinyCausalLM._layer_qkv` | 由 `_embeds_qkv` 改成按层投影 |
| `TinyCausalLM._attention_out` | 新增：一层里的 Attention 段（QKV → 写本层 KV → kernel → o_proj） |
| `TinyCausalLM.gpu_forward` / `_torch_forward` | 改成逐层循环；删掉 `_heads_to_logits` |
| `KVCachePool.__init__` | 新增 `num_layers`（关键字，默认 1）；池形状 `[L, blocks, block_size, kv_heads, head_dim]` |
| `KVCachePool.block_view` / `gather` | 新增 `layer_idx`（默认 0） |
| `Engine.__init__` | 新增 `num_layers=2` / `intermediate_size=64` / `rms_norm_eps=1e-6`；池带上层数 |

## 2. 设计要点

### 2.1 一层的顺序

```text
h = x + Attention(RMSNorm_1(x))
y = h + MLP(RMSNorm_2(h))
```

第二个残差加的是 **`h`**，不是最初的 `x`。`Attention` 含 QKV 投影、GQA、分页 KV 读写、head 拼接和 `o_proj`，**不含 lm_head**。

### 2.2 RMSNorm

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)      # 只对最后一维，不跨 token
return x * torch.rsqrt(variance + self.eps) * self.weight
```

- `weight` 是可学习的 `[d_model]`，初始全 1，不加 bias；
- 不减均值（这是和 LayerNorm 的区别）；
- `eps=1e-6` 显式指定，不依赖不同实现的默认值；
- 每层两个 + 模型最后还有一个，共 `2*num_layers + 1` 个。

### 2.3 SwiGLU

```python
MLP(z) = down_proj(silu(gate_proj(z)) * up_proj(z))
```

三个无 bias 投影：`gate`/`up` 是 `d_model → intermediate_size`，`down` 是 `intermediate_size → d_model`。`*` 是两个 `[N, intermediate]` 的**逐元素**相乘。

MLP 部分抽成 `DecoderLayer.forward`，Torch 路径和 GPU 路径共用。

### 2.4 层维进 KV 池

```text
k_cache / v_cache:  [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
每层的扁平视图:      k_flat[layer]  ->  [num_blocks * block_size, num_kv_heads, head_dim]
```

一个 slot 仍是**一个 token 的缓存位置**，只是现在有 `num_layers` 份互不相同的数据。

- 传给已有 kernel 的只是 `k_cache[layer_idx]` 这个四维视图，kernel 一行没改；
- **块表、slot mapping、有效长度、引用计数全层共用一套**——分配一个物理块号，意味着这个块号在所有层上都留给该请求；
- prefix cache 的 key 是按 token 块算的，命中时所有层的 KV 都已有效；释放和 LRU 淘汰同样全层一起生效。

最容易犯的错是长度：

```text
原 length = 7，本轮追加 2 个 token，模型 2 层
正确：两层都写逻辑位置 7、8，最后 length = 9
错误：每层各加一次，最后 length = 11
```

实现上靠"图外统一准备"保证：`_prepare_inputs` 只推进一次长度、只填一次元数据，各层都读同一份 `slot_buffer` 与 `AttentionMetadata`。实测：两层模型调度 6 个 token 后 `cache.length == 6`。

### 2.5 `_heads_to_logits` 拆开

上一关这个方法把 `o_proj` 和 `lm_head` 连在一起。现在 `o_proj` 属于层内的 Attention 段，只有整个模型最后才输出 logits：

```text
层输出  →  hidden [N, d_model]     （不再投影到词表）
最终    →  norm → lm_head → [N, vocab]
```

### 2.6 Triton 与 Graph

- 每层一次分页 attention kernel，两层就是两次调用——**不要求跨层融合**；
- 图内扩成 `embedding → 所有层 → 最终 norm → lm_head`，图外仍是准备、状态推进、调度、采样；
- 层数在 Engine 生命周期内固定，所以仍按 N 缓存图；各层 KV 存储地址稳定。

## 3. 验证

conda `vllm-omni-dev`，RTX 5090 Laptop，torch 2.13.0+cu130，Triton 3.7.1，FP32。

### 3.1 配置校验

```text
层数 0: 层数必须为正，收到 num_layers=0
中间维度 0: intermediate_size 必须为正，收到 0
eps 0: rms_norm_eps 必须为正，收到 0.0
```

### 3.2 与独立参考一致

参考实现逐层做完整历史的 causal attention、残差与 SwiGLU，返回 logits 和**每层的 K/V**。

- **1/2/3 层 × MHA/GQA/MQA × torch/triton**：9 种配置 × 2 后端 = **36 条逐 token 对照，不一致 0**。
- **每层 KV**：15 组场景逐步核对，**108 次逐层检查，不符 0**——两层没有互相覆盖。
- **每层确实不同**：`gather(cache, 0)` 与 `gather(cache, 1)` 的 K 不相等；两个层切片的 `untyped_storage().data_ptr()` 不同。
- **一次 prefill vs 分块 prefill**（预算 1 / 3 / 16）：输出 token 完全相同，**每层 KV 也逐值一致**，且都与参考一致。不是只比最终 argmax。

### 3.3 执行与缓存

- **三后端**（torch / triton / triton+graph）：26 组随机场景（1/2/3 层、MHA/GQA/MQA、块大小 1–4、池 4–8 块、缓存在 2/3 场景开启），**逐轮返回不一致 0**。
- **Graph**：大→小→大 batch 下捕获序列 `[8, 2, 1]`，**每个 N 只捕获一次**，之后复用。
- **长度只推进一次**：两层模型、本轮调度 6 个 token → `cache.length == 6`（不是 12），且这一轮包含了首次捕获与 3 次预热。
- **共享前缀**：A 完成后 B 命中 1 个块，**第 0 层和第 1 层命中位置的 KV 都与独立参考一致**。
- **prefix cache 开关**：34 组场景，开/关的回调与结果不一致 0。
- **索引不变量**：100 组场景逐步检查 `block_hash` 与 `block_to_hash` 互为双射，结束时引用计数全零。

### 3.4 KV 字节数随层数增长

块大小 4、8 块 = 32 个 slot、`num_kv_heads=2`、`head_dim=8`、FP32：

| 层数 | 池形状 | K+V |
|---:|---|---:|
| 1 | `(1, 8, 4, 2, 8)` | 4096 字节 |
| 2 | `(2, 8, 4, 2, 8)` | 8192 字节 |
| 3 | `(3, 8, 4, 2, 8)` | 12288 字节 |

参数结构：3 层模型共 31 个参数张量，其中 `layers.*` 占 27（每层 9 个：两个 norm + Q/K/V/O + gate/up/down）；`state_dict` 存读往返一致，两层权重不同，norm 权重初始为全 1。

### 3.5 计时（粗测，同机同数据，三轮取最快）

完整 Engine：8 请求 × prompt 500 + 生成 24，`d_model=64`、`intermediate_size=256`、`num_kv_heads=2`：

| 层数 | torch | triton | triton+graph | KV 字节 |
|---:|---:|---:|---:|---:|
| 1 | 1494.9 ms | 146.4 ms | 113.0 ms | 1572864 |
| 2 | 2289.3 ms | 125.3 ms | 139.8 ms | 3145728 |
| 3 | 3667.1 ms | 249.1 ms | 139.3 ms | 4718592 |

torch 参考路径随层数明显变慢（每层都有 per-request × per-head 的 Python 循环）；triton 与 graph 的数字在这台机器上抖动较大（2 层比 1 层还快，是 GPU 频率波动），**不能从这张表读出"层数不影响耗时"**。正式测量由验收方按同配置同权重进行。

## 4. 接口变化与遗留

- `Engine(...)` 新增 `num_layers=2`、`intermediate_size=64`、`rms_norm_eps=1e-6`；`TinyCausalLM(...)` 同样。
- `KVCachePool(..., num_layers=1)`：新增关键字参数，**旧的位置参数调用仍然可用**。
- `KVCachePool.block_view(cache, logical_block, count, layer_idx=0)`、`gather(cache, layer_idx=0)` 新增可选参数。
- **参数键变了**：`q_proj.weight` → `layers.0.q_proj.weight` 等；新增 `norm.weight` 与各层 `norm1/norm2`。旧 checkpoint 不能直接加载。
- 删除 `TinyCausalLM._heads_to_logits` 与 `_embeds_qkv`（职责已拆到 `_attention_out` / `_layer_qkv` / 最终的 `norm + lm_head`）。
- 遗留：仍是 learned position embedding，没有 RoPE / RMSNorm 融合 / FFN 中间维的 tensor parallel。
- 遗留：每层一次 attention kernel 调用，没有跨层融合。
- 遗留：Torch 参考路径是 per-request × per-head × per-layer 的三层 Python 循环，只作对照。
- 未提交。

## 附：写独立参考时的两个坑

留给验收方写参考实现时参考：

1. **必须复刻 Engine 的提前结束规则**。引擎在 `output_ids[-1] == 4` 时就结束（第 4 关定下的 EOS 行为），参考的生成循环如果不知道这条，会多生成若干个 token，看起来像"引擎少生成了"。本次实现验证时踩过一次。
2. **逐层参考里 hidden 要往下传**。第 `l` 层的 K/V 必须用第 `l-1` 层的输出算，不能拿最初的 embedding 去比第 1 层——否则会在第 1 层报"不一致"，而实际上是参考写错了。
