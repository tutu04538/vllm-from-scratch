# 第 81 关对齐记录（E2 所需子集）

> **状态：部分实现，不是 81 完成。** 本文件只记录为 80E.E2 打通所必需的多 KV
> group / sliding+full / 异构 head_dim / Q-only 源层基础。Mamba 混合状态、抢占三时刻
> 矩阵、prefix cache 的滑窗块回收、KV transfer/PD、多模态 M-RoPE 仍未实现；不能据此
> 声称第 81 关通过。

## 1. 已落地

| 模块 | 本项目 | 冻结上游 | 状态 |
|---|---|---|---|
| KV spec/group/config | `minivllm/core/kv_cache_interface.py` | `v1/kv_cache_interface.py` | FullAttentionSpec / SlidingWindowSpec / KVCacheGroup / KVCacheConfig；dtype 必须是 torch.dtype |
| 单类型 manager | `minivllm/core/single_type_kv_cache_manager.py` | `v1/core/single_type_kv_cache_manager.py` | FullAttention + SlidingWindow；组块大小可大于 hash 块，用 stride 取链式 hash |
| 多组协调 | `minivllm/core/kv_cache_coordinator.py::HybridKVCacheCoordinator` | 同名 Hybrid | 逐组查命中 → 取共同长度 → 按共同长度裁块；固定点收敛 |
| 前缀边界 | 同上 | `HybridKVCacheCoordinator.find_longest_cache_hit` | 一个组缺块时共同命中 0，不允许另一组单独误命中 |
| V2 执行 | `worker/gpu/model_runner.py` | V2 `prepare_attn` / `init_attn_backend` | `BlockTables` 多组块表、逐组 slot mapping、逐层 metadata、Q-only 源组映射 |
| V1 执行 | `worker/block_table.py`、`worker/gpu_model_runner.py` | V1 `GPUModelRunner` | `BlockTable` 多组包装、逐组槽位/metadata；多组图明确拒绝 |
| 滑窗执行 | `attention/backends/torch_sdpa.py` | 各后端 sliding mask | 绝对位置下界 `q - window + 1`；eager/固定形状/varlen 三条路径一致 |
| Q-only | `attention/layer.py`、`core/...`、Runner | Gemma4 YOCO / assistant | 只读源层 cache；组 id 跟随源层 |

## 2. 验收证据

`tests/step81/test_spec_kv_groups.py`（4 passed）：

1. 组块大小 4/8、hash 粒度 min、16 token 请求分配 `[4,2]` 块、共同命中 8；
2. 删掉 sliding 组第一块 hash 后共同命中 0（无跨组误命中）；
3. 容量不足时原子失败，两组块表都不留半块；
4. 滑窗 spec 缺 window 明确报错。

`tests/step80/test_gemma4_real_integration.py` 的 target/组相关用例另外验证：
真实 assistant 的 4 层分别映射到 sliding/full 源组、`head_size` 256/512、
target Q-only 层绑同一 cache 对象、`propose()` 前后 target cache 不变。

## 3. 未完成 / 明确边界

1. **Mamba/混合状态**：没有 `MambaSpec`/`MambaHybridManager`，没有状态张量
   prepare/update/rollback；拒绝而不是错算。
2. **滑窗块回收**：`SlidingWindowManager` 保留全历史，仅 attention 掩码；没有
   `remove_skipped_blocks` + null block 占位，显存不按窗口回收。
3. **不同 group block_size 的执行**：控制面可分配/命中；Runner 执行层仍要求所有组
   相同 block_size（Gemma4 满足）。逐组不同块大小的 kernel/图未接。
4. **抢占三时刻**：已排计划/执行未回/返回未提交的恢复、stale output、共享块引用计数
   矩阵未做。
5. **prefix cache 的 cache-only hidden**：Gemma4 路径不依赖 cache-only hidden；64 关
   的 extract 路径与本关多组未合并验收。
6. **KV transfer/PD**：未接；不自造传输协议。
7. **多模态/inputs_embeds/M-RoPE/XD-RoPE**：Gemma4 文本路径已接；多模态未接。
8. **生命周期矩阵**：EOS/stop/max_tokens/cancel/slot 复用只做了 E2 的 request-id
   复用用例，不是 81 要求的全矩阵。

## 4. 命令

```bash
PY=/home/user/anaconda3/envs/vllm-omni-dev/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest \
  tests/step81/test_spec_kv_groups.py -q   # 4 passed
```

全量回归与包摘要见 `docs/results.json` 的 `step81` 记录。
