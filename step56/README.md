# step56：GPU 批量拒绝采样与单次结果回传

第五十五关把投机的**算法**做完整了：draft model（Qwen3-0.6B）提议、target（Qwen3-1.7B）
验证，两套 KV 各按自己的结构建，提议分布是一般 q，验证用 `min(1, p/q)` 接受、`max(p-q, 0)`
纠正。但**验证的执行方式**还是「一条请求取一批标量回 CPU、Python 里决定、再取下一条」：
target 早就整批 forward，CPU 却还在为每条请求反复等 GPU。

这一关把决策搬进 GPU：**一次批量验证 + 一次集中回传**，随机数改成 counter-based
（随机数 = f(请求 seed, 事件编号, 事件类型, token 下标)）。判据不是「跑得快」，而是与同一套
算法的 CPU oracle **逐位一致**，外加定向观测「`verify_batch()` 内没有任何回传」。
设计与验证见 [`docs/step56_gpu_rejection.md`](../docs/step56_gpu_rejection.md)。

不承诺加速：本关只把机制做完整、做对。

## 怎么用

```python
# 两个模型各读自己的目录；结构可以完全不同
engine = Engine.from_model_dir(
    "models/Qwen3-1.7B", draft_model_dir="/path/to/Qwen3-0.6B",
    speculative_mode="draft_model", draft_num_kv_blocks=48,
    device="cuda", attention_backend="torch", dtype=torch.bfloat16,
    rejection_backend="triton")   # 默认 "torch" = 第五十五关的参考路径
```

命令行（主运行）：

```bash
python step56/step56.py --model-dir models/Qwen3-1.7B \
    --draft-model-dir /path/to/Qwen3-0.6B --speculative draft_model \
    --draft-num-kv-blocks 48 --device cuda --dtype bfloat16 --backend torch \
    --rejection-backend triton --max-new-tokens 32 "问题一" "问题二"
```

`rejection_backend` 在引擎创建时**二选一固定**，运行中不切：

- `"torch"`（默认）：逐请求调用既有的 `verify_drafts()` / `verify_drafts_random()`，
  行为与第五十五关一字不改，CPU 也能跑；
- `"triton"`：CUDA 批量验证。不支持的组合在构造时明确拒绝（CPU 设备、不开投机、
  未知后端名），**绝不悄悄退回参考路径还报告 GPU 路径成功**。

## 三件容易写错的事（第五十六关新增）

1. **随机数只由事件编号决定**。一轮里「必接受 / 必拒绝」的位置不抽随机数、接受终止
   token 之后的随机事件一个都不消费——所以计数器的增量只能来自**真的发生**的事件。
   预抽 K 个 uniform 的做法在这里是错的，见 docs §2.2。
2. **结果只有一次回传**。`verify_batch()` 里不许出现 `.item()` / `.cpu()` / `.tolist()`，
   也不许按 `kind` 筛子集（掩码索引会把形状从设备拷回主机，是一次隐式同步）。整批结论在
   `materialize_results()` 里一次 `.cpu().tolist()` 拿回来。
3. **整批先检查再提交**。triton 后端把非法输入（`q[d] = 0`、纠正分布整行为零）编码成结果
   张量里的错误码，`SampleRuntime` 必须**整批检查完**才提交，不能「提交了半批再报错」。

## 第五十五关就定下、这一关不能破的三条

1. **两套 KV 的对齐**：全接受时 draft 恰好**少 1**（最后一枚被接受的草稿还没进过 draft
   模型），下一轮提议前必须先补算；被拒时 draft 多出来的部分要夹回 target 的真实边界。
   `align()` 与 `catch_up()` 是这两条规则的唯一实现。
2. **提议真的从 q 抽**：不能 argmax 提议却拿 softmax 的 q 去验证——那样接受概率是假的。
   q 是**抽出那一枚时用的分布**（含温度、过滤、逐行临时惩罚历史），原样交给验证层。
3. **两池容量的原子性**：计划阶段只读地问，缩草稿时两个池子都问；draft 不够只缩草稿，
   绝不为可选草稿抢占 target 的其他请求。实际草稿比预留少时，多预留的 target 块由回滚
   归还，draft 那边由 `align()` 夹回——各有一条归还路径。

另外，草稿**不能**「当确定性提议」（接受 `p(d)` + 挖掉 `d`）：那样分布仍然无偏，但接受率
被砍——0.30 vs 0.60，`p == q` 时 0.46 vs 1.00。收益全在接受率上，所以不接受这个简化。
推导见 `docs/step55_draft_model.md` §1.5。

## 怎么证明它是对的

| 检查 | 结果 |
|---|---|
| `benchmarks/check_step56_gpu_rejection.py`：与 CPU oracle 逐位对照（18 条用例 × 2 个事件起点）/ ragged 批 / 抽样内核逐事件对照 / 分布与流隔离 / 定向观测 / 后端组合 | **39 项全通过** |
| `benchmarks/check_step56_rejection_rng.py`：counter RNG 的 CPU/GPU 逐位一致（2 万事件）、均匀性、流隔离 | **9 项全通过** |
| `benchmarks/check_step56_real_qwen3.py`：真实 1.7B + 0.6B 端到端（含 triton 后端） | **15 项全通过** |
| 从第五十五关移植的回归套件（speculative / batch / rejection / random / engine / combinations / draft_kv / loading） | 316 项全通过 |
| `benchmarks/diff_step55_step56.py`：投机关闭时与 step55 逐步对照 | **88 项全通过** |

最硬的三条：**与 CPU oracle 的逐位一致**（同一个 counter RNG、同一个指数竞赛）、
**抽样内核 512 个事件逐事件相同**（权重故意不归一化）、**真实模型 greedy 下 triton 后端与
普通贪心逐 token 相同**。另配两条反证：去掉被拒位置的 `clamp` 会触发 CUDA device-side
assert；抽样内核误用 `[0,1)` 的 float 会让抽出来的 token 全错。

## 遗留

- 每步仍有一次 O(1) 的元数据上传（约 16 次小 H2D 拷贝，与请求数无关），没做打包/pinned；
- **只覆盖「验证」**：普通（非投机）采样仍用 `sampling_state.generator`；
- draft 提议阶段的同步（逐行 `distribution()` 的 Python 循环）没动；
- 不做异步采样、draft 池不做前缀共享、不做流式低峰值加载、不承诺加速。

详见 `docs/step56_gpu_rejection.md` §7.2。
