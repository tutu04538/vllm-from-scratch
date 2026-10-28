# 教学版 learned top-k 稀疏注意力

> **定位：学习材料，不是生产模型，也不是 Step3p5 / DeepSeek DSA 的逐位复刻。**
> 随机初始化，不加载任何 checkpoint。

## 1. 它解决什么

普通 attention 每个 query 看所有历史 key。稀疏注意力先用一个小的 **indexer**
给历史位置打分，只保留 top-k 个，再在这些位置上做 attention。

```text
hidden
  ├─ indexer q/k/weights
  │     → score[t, s]
  │     → causal + valid mask
  │     → top-k 选位置
  │
  └─ 主 attention 的 q/k/v
        → 只在 selected 位置上 softmax
        → output
```

## 2. 代码

```text
minivllm/testing/sparse_attention.py
  SparseAttentionConfig
  SparseIndexer
  SparseAttention
  SparseDecoderLayer
  SparseLM
```

`SparseLM.forward(..., return_selected=True)` 返回：

```python
logits, selected_layers
```

`selected_layers[l][b, i, j] == True` 表示第 l 层、第 b 条样本、第 i 个 query
是否看了第 j 个 key。

## 3. 关键不变量

1. **因果**：query i 只能选 `j <= i`；
2. **有效**：padding key 永远不被选；
3. **top-k**：每条 query 最多 `min(k, i+1)` 个 key；
4. **退化**：`top_k >= seq_len` 时必须与 dense 完全相同；
5. **无 NaN**：`top_k=0` / 全掩行输出 0，不产生 NaN。

## 4. 运行

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step80/test_sparse_attention_teaching.py -q   # 5 passed
```

也可以直接看选中位置：

```python
import torch
from minivllm.testing.sparse_attention import SparseAttentionConfig, SparseLM

torch.manual_seed(0)
model = SparseLM(SparseAttentionConfig(vocab_size=32, hidden_size=32,
                                       intermediate_size=64, num_layers=2,
                                       num_heads=4, num_kv_heads=2, head_dim=8,
                                       max_seq_len=16, sparse_top_k=2))
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
logits, selected = model(ids, top_k=2, return_selected=True)
print(selected[0][0].int())   # [T, T] bool -> 0/1
```

## 5. 与真实模型的关系

| | 本教学模型 | Step3p5 | DeepSeek DSA |
|---|---|---|---|
| 选位置 | learned indexer | 固定 sliding/full | learned lightning indexer |
| 权重 | 随机 | 真实 | 真实 |
| MLA / 低秩 KV | 无 | 无（GQA） | 有 |
| paged KV cache | 无 | 有 | 有 |
| 目标 | 观察 mask/因果/top-k | 真实模型对齐 | 真实稀疏模型 |

本文件不声称任何真实模型的数值/质量对齐。
