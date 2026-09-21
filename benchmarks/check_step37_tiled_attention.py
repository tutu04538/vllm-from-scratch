"""第三十七关正确性检查：分块 kernel 对旧 kernel 与独立 dense 参考。

    python benchmarks/check_step37_tiled_attention.py

三个参考各自回答不同的问题：

    dense      用 PyTorch 直接展开算，不碰分页、不碰 Triton——独立参考
    旧 kernel  同一份打包输入、同一份分页 KV——只换 attention 实现的对照
    新 kernel  被测对象

覆盖需求点名的边界：
  - 计算块 BLOCK_N 跨多个物理块（物理 block_size=4，BLOCK_N=64）
  - 物理块号故意打乱，不连续
  - 尾块（tile 内行数不是 BLOCK_M 的整数倍）
  - 不同历史长度、不同请求长度、多请求打包
  - GQA（q_heads > kv_heads）
  - 历史后追加 chunk（不是从位置 0 开始的完整 prompt）
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import step37 as m
from step37.attention import BLOCK_M, paged_attention, tiled_paged_attention

DEVICE = "cuda"
BLOCK_SIZE = 4          # 物理块很小，保证一个计算块横跨很多物理块
NUM_KV_HEADS = 2
NUM_Q_HEADS = 6         # GROUP_SIZE = 3
HEAD_DIM = 16
DTYPE = torch.bfloat16


class FakeCache:
    """fill() 只要求 .length 和 .block_table，不需要真的分配"""

    def __init__(self, length, block_table):
        self.length = length
        self.block_table = block_table


def build_case(reqs, seed=0):
    """按「每请求 (已有历史长度, 本轮追加 token 数)」造一份分页 KV 与打包 query。

    物理块号打乱后分配，保证 block_table 不连续——拿首块地址连续读会算错。

    注意 seq_lens 用的是**追加之后**的总长度（history + count），
    与真实路径一致：Engine 里 `_cache.length += count` 发生在 fill() 之前。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_reqs = len(reqs)
    total_blocks = sum((hist + c + BLOCK_SIZE - 1) // BLOCK_SIZE for hist, c in reqs)
    pool = m.KVCachePool(BLOCK_SIZE, total_blocks, NUM_KV_HEADS, HEAD_DIM, DEVICE,
                         enable_prefix_caching=False, num_layers=1, dtype=DTYPE)
    order = torch.randperm(total_blocks, generator=g).tolist()

    specs = []
    logical_k, logical_v, logical_q, logical_pos = [], [], [], []
    cursor = 0
    for hist, count in reqs:
        L = hist + count            # 追加之后的总长度
        nblk = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
        table = sorted(order[cursor:cursor + nblk])
        cursor += nblk
        specs.append(FakeCache(L, table))

        # 逻辑视图（不看分页）：每个请求自己的 K/V 序列
        k_r = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, generator=g).to(DTYPE)
        v_r = torch.randn(L, NUM_KV_HEADS, HEAD_DIM, generator=g).to(DTYPE)
        logical_k.append(k_r)
        logical_v.append(v_r)

        # 写进分页池的实际位置：逻辑位置 t -> 块 table[t // bs]、块内 table[t % bs]
        for t in range(L):
            blk, off = table[t // BLOCK_SIZE], t % BLOCK_SIZE
            pool.k_cache[0, blk, off, :, :] = k_r[t]
            pool.v_cache[0, blk, off, :, :] = v_r[t]

        q_r = torch.randn(count, NUM_Q_HEADS, HEAD_DIM, generator=g).to(DTYPE)
        logical_q.append(q_r)
        logical_pos.append(list(range(hist, L)))     # 本轮 query 的真实位置

    return pool, specs, (logical_k, logical_v, logical_q, logical_pos)


def dense_reference(logical, num_reqs):
    """独立参考：不分页、不 Triton，直接展开算因果 softmax attention。"""
    lk, lv, lq, lpos = logical
    outs = []
    scale = 1.0 / (HEAD_DIM ** 0.5)
    for r in range(num_reqs):
        k, v, q, pos = lk[r].to(DEVICE), lv[r].to(DEVICE), lq[r].to(DEVICE), lpos[r]
        group = NUM_Q_HEADS // NUM_KV_HEADS
        per_head = []
        for h in range(NUM_Q_HEADS):
            kh = h // group
            kk = k[:, kh, :].float()                    # [L, D]
            vv = v[:, kh, :].float()
            qq = q[:, h, :].float()                     # [count, D]
            scores = qq @ kk.T * scale                  # [count, L]
            idx = torch.arange(k.shape[0], device=DEVICE)
            causal = idx[None, :] <= torch.tensor(pos, device=DEVICE)[:, None]
            scores = scores.masked_fill(~causal, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            per_head.append(p @ vv)
        outs.append(torch.stack(per_head, dim=1))       # [count, H, D]
    return torch.cat(outs, dim=0)


def main():
    torch.manual_seed(0)
    # (已有历史, 本轮追加)：覆盖尾块、多 tile、历史后追加 chunk、整除边界
    reqs = [
        (17, 20),    # 追加 20 < BLOCK_M，单个 tile
        (5, 100),    # 追加 100 > BLOCK_M，同一请求多个 tile
        (0, 64),     # 从位置 0 开始的完整 prompt，整除边界
        (0, 1),      # 单 token
        (33, 31),    # 历史长、追加少，尾块
        (50, 3),     # 历史远长于追加
    ]
    pool, specs, logical = build_case(reqs)
    counts = [c for _, c in reqs]
    num_tokens = sum(counts)
    num_reqs = len(reqs)

    meta = m.AttentionMetadata(max_num_seqs=num_reqs, max_num_query_tokens=num_tokens,
                               max_blocks_per_request=64, device=torch.device(DEVICE))
    meta.fill(specs, counts, build_tiles=True)
    meta.upload()

    # 打包 query：和请求同序
    q = torch.cat(logical[2], dim=0).to(DEVICE)

    with torch.no_grad():
        old = paged_attention(q, pool.k_cache[0], pool.v_cache[0],
                              meta.gpu_block_tables, meta.gpu_seq_lens,
                              meta.gpu_token_to_req, meta.gpu_query_pos,
                              NUM_Q_HEADS // NUM_KV_HEADS)
        new = tiled_paged_attention(q, pool.k_cache[0], pool.v_cache[0],
                                    meta.gpu_block_tables, meta.gpu_seq_lens,
                                    meta.gpu_query_pos,
                                    meta.gpu_tile_req, meta.gpu_tile_q_start,
                                    meta.gpu_tile_q_count, meta.gpu_num_tiles,
                                    meta.tile_capacity, NUM_Q_HEADS // NUM_KV_HEADS)
        ref = dense_reference(logical, num_reqs)

    # 误差判据的先验推导（不是看到结果再放宽）：
    #   两个 kernel 的输入 Q/K/V 都是 bf16，输出也要存回 bf16。
    #   参考用 fp32 算同一批 bf16 输入，所以「旧 kernel vs 参考」的差主要就是
    #   **bf16 输出存储的量化台阶**：最大元素量级 × 2^-9。
    #   分块 kernel 多一步「P 转成 KV 精度再进 tl.dot」的舍入，其贡献与输出量化同量级，
    #   因此上限取旧 kernel 误差的 2 倍。
    bf16_step = (ref.float().abs().max().item()) * 2 ** -9

    def report(name, got):
        got_f = got.float()
        rms = ref.float().pow(2).mean().sqrt()
        max_abs = (got_f - ref.float()).abs().max().item()
        rel = (max_abs / rms.item()) if rms > 0 else float("nan")
        finite = bool(torch.isfinite(got_f).all())
        print(f"  {name:<10} 有限值={finite}  最大绝对差={max_abs:.6f}  "
              f"（相对参考 RMS {rel:.2e}，bf16 台阶约 {bf16_step:.6f}）")
        return max_abs, finite

    print(f"配置：block_size={BLOCK_SIZE} BLOCK_M={BLOCK_M} BLOCK_N=64 head_dim={HEAD_DIM} "
          f"q_heads={NUM_Q_HEADS} kv_heads={NUM_KV_HEADS} dtype={DTYPE}")
    print(f"请求 (历史, 追加)：{reqs}  共 {num_tokens} 行 query，"
          f"{meta.cpu_num_tiles[0].item()} 个 tile")
    print()
    a_old, f_old = report("旧 kernel", old)
    a_new, f_new = report("新 kernel", new)

    # 新旧直接对照：同一份输入、同一份分页 KV，只换实现
    diff = (new.float() - old.float()).abs().max().item()
    ratio = (a_new / a_old) if a_old > 0 else float("inf")
    print(f"\n  新旧直接对照最大绝对差 = {diff:.6f}")
    print(f"  新/旧 误差比 = {ratio:.2f}（上限 2.0，来自 P 量化的额外舍入）")

    ok = f_old and f_new and ratio <= 2.0
    print(f"\n结果：{'通过' if ok else '失败'}"
          f"（有限值 + 误差不超过旧 kernel 的 2 倍）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
