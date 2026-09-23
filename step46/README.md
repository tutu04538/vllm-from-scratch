# step46：抢占恢复与前缀缓存整合

从 `step45/` 复制。`step45/` 原样保留，未改。

## 结论

第四十四、四十五关里 `preemption_mode="recompute"` 强制关掉 prefix cache，恢复时从 0 重算。
本关把两者接上：**恢复时先复用仍然在缓存里的完整块，只重算剩下的历史**。

`recompute` + `enable_prefix_caching=True` 现在是合法组合，`Engine` 没有新参数。

## 三个长度必须分清

| 长度 | 含义 |
|---|---|
| 已知 token | `all_token_ids = prompt_ids + output_ids` |
| 已计算 token | **已经进过模型并写进 KV** 的数量 = `cache.length`（唯一真相） |
| 已缓存完整块 | 整块被 `cache.length` 覆盖、KV 确实写进了物理块 |

`prompt=3, block_size=4` 的实测轨迹：

```text
准入后（还没算）        已知 3  已计算 0  已发布 0
prompt 算完、刚采样 y0  已知 4  已计算 3  已发布 0   ← 刚采样的 y0 还没进模型，不发布
y0 真的进过模型之后      已知 5  已计算 4  已发布 1   ← 块 [p0,p1,p2,y0] 现在可命中
```

发布入口统一成 `publish_computed_blocks()`：覆盖范围完全由 `cache.length` 决定，
hash 链按 `all_token_ids` 的块顺序算，所以生成 token 落进的块也有稳定身份，
而**刚采样、尚未计算的 token 绝不可能伪装成命中**。

## 实测（功能正确性，41 项全通过）

```text
pool 8 块、block_size 4、两条 prompt 6 / 输出上限 12
prefix 开：抢占 1 次，复用 12 token，实际重算 3
prefix 关：抢占 1 次，实际重算 15
```

- 正命中恢复：`cache.length > 0` 起步，实际重算 token 少于关闭 prefix 时，输出仍与独占运行一致
- 零命中 / 淘汰后：正确从 0 重算；闲置块被 LRU 淘汰后 hash 不再指向已复用的物理块
- 边界：历史 8 / 7 / 9 的命中都严格小于历史长度且是整块倍数；始终留一个 token 进模型
- 共享：后到请求复用同一批物理块，两条都完成后引用计数归零
- 容量不足：最坏**逻辑**块数超池仍明确拒绝——命中的块也占物理块，不能用命中放宽
- 阻塞者规则保留：prefix 开/关都是 4 次抢占、20 次阻塞跳过、完成序 `A,B,C,D,E`
- 旧三种配置（`None` + prefix 开/关、`recompute` + prefix 关）输出与 step45 一致

复现：`benchmarks/check_step46_resume_prefix.py`。

**本关不做性能测试**（按 `02_每次更新后的性能测量约定`）。重算 token 数减少是功能指标，
不是吞吐结论——复用会带来额外的缓存占用，收益与代价等专题收尾再统一测。

完整说明见 [`docs/step46_resume_prefix_cache.md`](../docs/step46_resume_prefix_cache.md)。
