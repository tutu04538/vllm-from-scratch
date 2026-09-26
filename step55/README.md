# step55：draft model、双 KV 与一般拒绝采样

第五十四关把投机的**算法**（拒绝采样、纠正分布、逐行惩罚历史）做完了，但草稿来自
n-gram 重复历史。这一关换成**第二个模型**：Qwen3-1.7B 做 target、Qwen3-0.6B 做 draft，
两套 KV 各按自己的结构建，提议分布是**一般 q**（不再是一枚 one-hot），验证用
`min(1, p/q)` 接受、`max(p-q, 0)` 纠正。

不承诺加速：本关只把机制做完整、做对。设计与验证见
[`docs/step55_draft_model.md`](../docs/step55_draft_model.md)。

## 怎么用

```python
# 两个模型各读自己的目录；结构可以完全不同
engine = Engine.from_model_dir(
    "models/Qwen3-1.7B", draft_model_dir="/path/to/Qwen3-0.6B",
    speculative_mode="draft_model", draft_num_kv_blocks=48,
    device="cuda", attention_backend="torch", dtype=torch.bfloat16)
```

命令行（主运行）：

```bash
python step55/step55.py --model-dir models/Qwen3-1.7B \
    --draft-model-dir /path/to/Qwen3-0.6B --speculative draft_model \
    --draft-num-kv-blocks 48 --device cuda --dtype bfloat16 --backend torch \
    --max-new-tokens 32 "问题一" "问题二"
```

## 三件容易写错的事

1. **两套 KV 的对齐**：全接受时 draft 恰好**少 1**（最后一枚被接受的草稿还没进过 draft
   模型），下一轮提议前必须先补算；被拒时 draft 多出来的部分要夹回 target 的真实边界。
   `align()` 与 `catch_up()` 是这两条规则的唯一实现。
2. **提议真的从 q 抽**：不能 argmax 提议却拿 softmax 的 q 去验证——那样接受概率是假的。
   q 是**抽出那一枚时用的分布**（含温度、过滤、逐行临时惩罚历史），原样交给验证层。
3. **两池容量的原子性**：计划阶段只读地问，缩草稿时两个池子都问；draft 不够只缩草稿，
   绝不为可选草稿抢占 target 的其他请求。实际草稿比预留少时，多预留的 target 块由回滚
   归还，draft 那边由 `align()` 夹回——各有一条归还路径。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step55_draft_kv.py`：双 KV 轨迹 / 对齐 / 边界 / 回退 / 抢占 / 前缀命中 | **39 项全通过** |
| `benchmarks/check_step55_rejection.py`：一般 p/q 拒绝采样（含统计检验） | **48 项全通过** |
| `benchmarks/check_step55_loading.py`：分片权重 + 双目录加载 | **12 项全通过** |
| `benchmarks/check_step55_real_qwen3.py`：真实 1.7B + 0.6B 端到端 | **9 项全通过** |
| `benchmarks/diff_step54_step55.py`：投机关闭时与 step54 逐步对照 | **88 项全通过** |
| 第五十二~五十四关的四个回归套件（speculative / batch / random / engine / combinations） | 57 / 35 / 22 / 55 / 17 项全过 |

最硬的两条：**两套 KV 的有效部分与「该模型对已提交前缀单独重算」逐张量一致**（并配了
「故意改坏一个位置立刻失败」的对照），以及**真实模型 greedy 下投机与普通贪心逐 token 相同**。

## 遗留

不做 GPU rejection kernel、不做异步采样、draft 池不做前缀共享（恢复全靠补算）、
不做流式低峰值加载、不承诺加速。详见 `docs/step55_draft_model.md` §9。
