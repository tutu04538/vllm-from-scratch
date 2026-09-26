# step54：随机采样投机解码与拒绝修正

从 `step53/` 复制。`step53/` 原样保留，未改。

## 结论

第五十三关的投机只判断「草稿 == 目标模型 argmax」，把 argmax 当成了唯一正确答案。
本关实现 n-gram 提议下的**拒绝采样**，让 `temperature / top-k / top-p / 惩罚项` 都生效：

```python
verify_drafts_random(draft_ids, row_probs, eos, remaining, draw_uniform, draw_token)
```

n-gram 是确定性提议，所以 `q(d)=1`、接受概率就是 `p[d]`：

```text
逐位置抽 u ∈ [0,1)：u < p[d] 就接受这枚草稿
首次拒绝 -> 从「挖掉 d 的纠正分布」抽一枚，本轮结束
全部接受 -> 从最后一行的分布抽 bonus
```

**纠正分布不能省**：`p=[0.6,0.3,0.1]`、草稿 1 时，挖掉再归一化才让最终分布还原成 p；
拒绝后仍从原 p 重抽会把 token1 抬到 0.51、token0 压到 0.42——验收就是拿经验分布对 p。

三件容易写错的事：

1. **逐行历史**：行 j 看到的生成历史是「真实生成 + 前 j 枚草稿」，所以要一份**临时计数**；
   真实 `sampling_state` 与 `all_token_ids` 在提交之前一个字不动。
2. **RNG 归请求**：抢占不重置也不消耗它；`p[d] ∈ {0,1}` 时不抽 uniform；终止之后一个
   随机数都不再消耗。
3. **快路径不能被破坏**：贪心且无惩罚的投机项仍走第五十三关的整批 argmax；只有带惩罚的
   贪心与随机采样才走逐行分布。

## 模块划分

`engine.py` 只做装配与编排（210 行）：装运行时、`step()` 走「调度 → 一次 forward → 采样」、
以及唯一提交入口 `_commit_tokens()`。三块内容各归一个模块：

| 模块 | 装什么 |
|---|---|
| `sample_loop.py` | `SampleRuntime`：行映射、三条采样路径、验证与 KV 回滚 |
| `validation.py` | 配置与后端的组合校验（构造阶段报错） |
| `loading.py` | 模型装配与目录加载 |

用**组合**不用继承——依据是 vLLM（`Sampler` / `RejectionSampler` 都是独立类、被
`gpu_model_runner` 持有为属性）。详见
[`docs/step54_random_speculative.md`](../docs/step54_random_speculative.md) §8。

## 明确限制

`speculative_mode="ngram"` 只剩两条**实现方式**决定的硬约束：`attention_backend="torch"`
（拒绝采样是逐请求的 Torch 参考循环，不做 Triton kernel）、`use_cuda_graph=False`
（采样要在图外逐行做设备同步）。

其余都放开并验过：`max_num_seqs`（第五十三关）、采样参数 / `priority` / 前缀缓存（本关）。
`check_step54_combinations.py` 专门验组合：priority 下的名额抢占与容量抢占、前缀缓存下的
命中与**已发布块 KV 不被回滚污染**、以及三条一起跑。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step54_rejection.py`：验证层（分支 / 概率 / 惩罚） | **29 项全通过** |
| `benchmarks/check_step54_random.py`：引擎状态（临时计数、RNG、重算计数、混批） | **21 项全通过** |
| `benchmarks/check_step54_speculative.py` / `check_step54_batch.py` / `check_step54_engine.py` | 53 / 35 / 51 项全过（第五十二、五十三关整套用例） |
| `benchmarks/diff_step53_step54.py`：投机关闭时与 step53 逐步对照 | **88 项全通过**（未放宽字段） |
| `benchmarks/check_step54_combinations.py`：priority / 前缀缓存与投机组合 | **17 项全通过** |
| 随机压测 500 组（随机/惩罚/不同 K、紧池子、动态到达） | 0 崩溃 0 活锁，计数与输出始终一致 |
| CUDA FP32 / BF16 | 同批随机投机 + 带惩罚的贪心都能跑完 |

概率正确性用统计检验钉住：`p=[0.6,0.3,0.1]` 20 万次采样得到 `[0.5997, 0.3004, 0.0999]`
（5σ 内），并且**明显偏离**「拒绝后仍从原 p 抽」的错分布 `[0.42,0.51,0.07]`。

「逐行历史」有一条**判别性**用例：固定 logits + 频率惩罚下，正确实现输出
`[5,9,5,9,5,9]`，「所有行都用真实计数」的错误实现输出 `[5,9,5,9,5,5]`——差异是先跑出来
再写成断言的。

复现：

```bash
python benchmarks/check_step54_rejection.py
python benchmarks/check_step54_random.py
python benchmarks/diff_step53_step54.py
```

完整说明（含一轮的具体数字、临时/真实计数的边界、RNG 约定、
以及「为什么分布相同不等于同 seed 文本相同」）见
[`docs/step54_random_speculative.md`](../docs/step54_random_speculative.md)。

**本关不承诺加速**：没有吞吐结论。
