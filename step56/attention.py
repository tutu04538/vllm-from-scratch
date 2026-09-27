"""分页 attention：Triton 算子 + 元数据缓冲。

这一层只认「打包好的一维 query」和「固定容量的元数据」，不认请求、不认调度。

两条执行路径：

    旧 kernel（逐行）   一个 program = 一行 query 的一个 head。
                        纯 decode 走这里；FP32、混合批次也回退到这里。
    新 kernel（分块）   一个 program = **同一请求**的一小组 query 的一个 head，
                        QK 与 PV 用 tl.dot 走矩阵乘法单元，一批 K/V 供多行 query 复用。
                        只在 BF16 且本轮每条请求都追加多个 token 时使用。

分块范围：BLOCK_M 行 query × BLOCK_N 个 key。
**BLOCK_N 与物理 block_size 不是一个东西**：计算块可能横跨多个物理块，
所以每个 key 都要各自查 block_table 换算物理地址，不能拿到首块地址后连续读。
"""

import math

import torch

import triton
import triton.language as tl

# 分块尺寸。第一版固定，不做自动调优。
BLOCK_M = 32
BLOCK_N = 64
# tl.dot 的 K 维（这里就是 head_dim）至少要 16；小的 head_dim 靠补零到 16 再算
MIN_DOT_DIM = 16


@triton.jit
def _paged_attention_kernel(
    q_ptr, out_ptr, k_pool_ptr, v_pool_ptr,
    block_tables_ptr, seq_lens_ptr, token_to_req_ptr, query_pos_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btn, stride_btb,
    stride_on, stride_oh, stride_od,
    SM_SCALE: tl.constexpr, S: tl.constexpr, S_POW2: tl.constexpr,
    D: tl.constexpr, D_POW2: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    # 一个 program 负责一个 query 的一个 head：按逻辑块读它对应的 KV head

    row = tl.program_id(0)                       # 打包 query 行号
    q_head = tl.program_id(1)                    # query head 编号
    kv_head = q_head // GROUP_SIZE               # 连续分组

    req = tl.load(token_to_req_ptr + row)
    qpos = tl.load(query_pos_ptr + row)
    seq_len = tl.load(seq_lens_ptr + req)

    offs_d = tl.arange(0, D_POW2)
    mask_d = offs_d < D
    offs_s = tl.arange(0, S_POW2)   # 计算范围可能比物理块大小宽，多出来的要屏蔽

    # Q/K/V 从池里读出来是运行精度（可能是 BF16）。这里显式转 FP32 再算：
    # 点积、max、exp、分母和加权和全程 FP32，最后存回时再回到池的精度。
    # 不能依赖「BF16 乘完再转 FP32」——那是先舍入再扩宽，与先扩宽再乘不是一回事。
    q = tl.load(q_ptr + row * stride_qn + q_head * stride_qh + offs_d * stride_qd,
                mask=mask_d, other=0.0).to(tl.float32)

    m_i = float("-inf")
    z_i = 0.0
    acc = tl.zeros([D_POW2], dtype=tl.float32)

    # 只遍历有效历史覆盖的逻辑块；预留但未写入的块不参与
    for blk in range(0, tl.cdiv(seq_len, S)):
        phys = tl.load(block_tables_ptr + req * stride_btn + blk * stride_btb)
        # offs_s < S：补宽出来的位置不属于本块
        # offs_s < seq_len - blk * S：尾块只读有效部分
        # blk * S + offs_s <= qpos：query 不能看到未来的 key
        mask_kv = (offs_s < S) & (offs_s < seq_len - blk * S) & ((blk * S + offs_s) <= qpos)

        kv_offsets = phys * stride_kb + offs_s[:, None] * stride_ks \
            + kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(k_pool_ptr + kv_offsets, mask=mask_kv[:, None] & mask_d[None, :],
                    other=0.0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(mask_kv, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        scale = tl.exp(m_i - m_new)
        p = tl.where(mask_kv, tl.exp(score - m_new), 0.0)

        v_offsets = phys * stride_vb + offs_s[:, None] * stride_vs \
            + kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_pool_ptr + v_offsets, mask=mask_kv[:, None] & mask_d[None, :],
                    other=0.0).to(tl.float32)

        z_i = scale * z_i + tl.sum(p, axis=0)
        acc = scale * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # 写回池/输出的精度：FP32 模式是空操作，BF16 模式下在这里做唯一一次舍入
    tl.store(out_ptr + row * stride_on + q_head * stride_oh + offs_d * stride_od,
             (acc / z_i).to(out_ptr.dtype.element_ty), mask=mask_d)


@triton.jit
def _tiled_paged_attention_kernel(
    q_ptr, out_ptr, k_pool_ptr, v_pool_ptr,
    block_tables_ptr, seq_lens_ptr, query_pos_ptr,
    num_tiles_ptr, tile_req_ptr, tile_q_start_ptr, tile_q_count_ptr,
    stride_qn, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btn, stride_btb,
    stride_on, stride_oh, stride_od,
    SM_SCALE: tl.constexpr, S: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D: tl.constexpr, D_POW2: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    # 一个 program 负责**同一请求**的一小组 query 的一个 head。
    # grid 大小是固定的（TILE_CAP, num_q_heads），实际有多少个 tile 由 GPU 上的
    # num_tiles 决定，多出来的 program 直接退出——这样同一张图能重放不同的请求划分。

    tile = tl.program_id(0)
    num_tiles = tl.load(num_tiles_ptr)
    if tile < num_tiles:
        q_head = tl.program_id(1)
        kv_head = q_head // GROUP_SIZE

        # tile 的归属与范围全部来自 GPU 元数据，图内不依赖任何 Python 侧切分
        req = tl.load(tile_req_ptr + tile)
        q_start = tl.load(tile_q_start_ptr + tile)      # 该 tile 首行的打包行号
        q_count = tl.load(tile_q_count_ptr + tile)      # 该 tile 的有效行数（<= BLOCK_M）
        seq_len = tl.load(seq_lens_ptr + req)
        # 首行的真实位置从 query_pos 读，不用 tile 内行号代替
        q_pos0 = tl.load(query_pos_ptr + q_start)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D_POW2)
        mask_d = offs_d < D
        valid_m = offs_m < q_count
        # padding 行（尾块多出来的槽）没有真实位置。给它一个确定的可见位置，
        # 避免整行被屏蔽时出现 -inf - -inf = NaN；这些行的输出不会被写回。
        q_pos = tl.where(valid_m, q_pos0 + offs_m, 0)

        rows = q_start + offs_m
        q = tl.load(q_ptr + rows[:, None] * stride_qn + q_head * stride_qh
                    + offs_d[None, :] * stride_qd,
                    mask=valid_m[:, None] & mask_d[None, :], other=0.0)

        # 状态从「一行一个标量」扩成「一行一个分量」，每行各算各的 online softmax
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        z_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)

        for blk in range(0, tl.cdiv(seq_len, BLOCK_N)):
            log_n = blk * BLOCK_N + offs_n          # 每个 key 的逻辑位置
            valid_n = log_n < seq_len

            # 计算块可能横跨多个物理块，逐个 key 换算：逻辑位置 -> 块号/块内偏移 -> 物理块
            blk_idx = log_n // S
            blk_off = log_n % S
            phys = tl.load(block_tables_ptr + req * stride_btn + blk_idx * stride_btb,
                           mask=valid_n, other=0)

            kv_mask = valid_n[:, None] & mask_d[None, :]
            k = tl.load(k_pool_ptr + phys[:, None] * stride_kb + blk_off[:, None] * stride_ks
                        + kv_head * stride_kh + offs_d[None, :] * stride_kd,
                        mask=kv_mask, other=0.0)

            # [BLOCK_M, D] @ [D, BLOCK_N] -> [BLOCK_M, BLOCK_N]，每行是自己的分数
            qk = tl.dot(q, tl.trans(k)) * SM_SCALE

            # 因果掩码用**真实位置**：第 i 行 query 的可见上界是 q_pos[i]，
            # 不是 tile 内的行号 i
            keep = valid_n[None, :] & (log_n[None, :] <= q_pos[:, None])
            qk = tl.where(keep, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            z_i = z_i * alpha + tl.sum(p, axis=1)

            v = tl.load(v_pool_ptr + phys[:, None] * stride_vb + blk_off[:, None] * stride_vs
                        + kv_head * stride_vh + offs_d[None, :] * stride_vd,
                        mask=kv_mask, other=0.0)
            # P 转成 KV 的运行精度才能进矩阵乘法单元，这一步会引入额外舍入：
            # 不要求与旧 kernel 逐位相同，但要单独做数值对照
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / z_i[:, None]
        tl.store(out_ptr + rows[:, None] * stride_on + q_head * stride_oh
                 + offs_d[None, :] * stride_od,
                 out.to(out_ptr.dtype.element_ty),
                 mask=valid_m[:, None] & mask_d[None, :])


def tiled_paged_attention(q, k_pool, v_pool, block_tables, seq_lens, query_pos,
                          tile_req, tile_q_start, tile_q_count, num_tiles, tile_capacity,
                          group_size):
    # 整批请求一次启动，grid 固定；实际做多少由 num_tiles（GPU 上）决定。
    # out 与 q 同形状、同行序，调用方不用关心打包行怎么划分。
    _, num_q_heads, head_dim = q.shape
    block_size = k_pool.shape[1]
    out = torch.empty_like(q)
    _tiled_paged_attention_kernel[(tile_capacity, num_q_heads)](
        q, out, k_pool, v_pool,
        block_tables, seq_lens, query_pos,
        num_tiles, tile_req, tile_q_start, tile_q_count,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SM_SCALE=1.0 / math.sqrt(head_dim), S=block_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D=head_dim, D_POW2=max(MIN_DOT_DIM, triton.next_power_of_2(head_dim)),
        GROUP_SIZE=group_size,
    )
    return out


def paged_attention(q, k_pool, v_pool, block_tables, seq_lens, token_to_req, query_pos, group_size):
    # 整批请求一次启动；q[N, num_q_heads, head_dim] -> out 同形状，行序不变
    num_rows, num_q_heads, head_dim = q.shape
    block_size = k_pool.shape[1]
    out = torch.empty_like(q)

    _paged_attention_kernel[(num_rows, num_q_heads)](
        q, out, k_pool, v_pool,
        block_tables, seq_lens, token_to_req, query_pos,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SM_SCALE=1.0 / math.sqrt(head_dim), S=block_size,
        S_POW2=triton.next_power_of_2(block_size), D=head_dim,
        D_POW2=triton.next_power_of_2(head_dim), GROUP_SIZE=group_size,
    )
    return out


class AttentionMetadata:
    # attention 元数据的固定容量缓冲区：CPU 一份、GPU 一份，运行期间只改内容不换存储。
    # 所有区域是同一段连续 int32 存储的不同视图，一次 copy_ 整段上传。

    def __init__(self, max_num_seqs, max_num_query_tokens, max_blocks_per_request, device):
        self.max_num_seqs = max_num_seqs
        self.max_num_query_tokens = max_num_query_tokens
        self.max_blocks_per_request = max_blocks_per_request
        self.device = device

        # 分块 kernel 的 tile 容量上界：
        # 每个请求至少占一个 tile，所以 = ceil(总 query 数上限 / BLOCK_M) + 请求数上限。
        # 这个上界与「本轮怎么切分」无关，于是 grid 大小固定，同一张图能重放不同划分。
        self.tile_capacity = triton.cdiv(max_num_query_tokens, BLOCK_M) + max_num_seqs

        size = (max_num_seqs * (1 + max_blocks_per_request)
                + 2 * max_num_query_tokens
                + 1 + 3 * self.tile_capacity)
        self.cpu_buffer = torch.zeros(size, dtype=torch.int32)
        self.gpu_buffer = torch.zeros(size, dtype=torch.int32, device=device)

        # 区域起止：seq_lens | block_tables | token_to_req | query_pos
        #           | num_tiles | tile_req | tile_q_start | tile_q_count
        end_seq_lens = max_num_seqs
        end_block_tables = end_seq_lens + max_num_seqs * max_blocks_per_request
        end_token_to_req = end_block_tables + max_num_query_tokens
        end_query_pos = end_token_to_req + max_num_query_tokens
        end_num_tiles = end_query_pos + 1
        end_tile_req = end_num_tiles + self.tile_capacity
        end_tile_q_start = end_tile_req + self.tile_capacity

        self.cpu_seq_lens = self.cpu_buffer[:end_seq_lens]
        self.cpu_block_tables = self.cpu_buffer[end_seq_lens:end_block_tables].view(max_num_seqs, max_blocks_per_request)
        self.cpu_token_to_req = self.cpu_buffer[end_block_tables:end_token_to_req]
        self.cpu_query_pos = self.cpu_buffer[end_token_to_req:end_query_pos]
        self.cpu_num_tiles = self.cpu_buffer[end_query_pos:end_num_tiles]
        self.cpu_tile_req = self.cpu_buffer[end_num_tiles:end_tile_req]
        self.cpu_tile_q_start = self.cpu_buffer[end_tile_req:end_tile_q_start]
        self.cpu_tile_q_count = self.cpu_buffer[end_tile_q_start:]

        self.gpu_seq_lens = self.gpu_buffer[:end_seq_lens]
        self.gpu_block_tables = self.gpu_buffer[end_seq_lens:end_block_tables].view(max_num_seqs, max_blocks_per_request)
        self.gpu_token_to_req = self.gpu_buffer[end_block_tables:end_token_to_req]
        self.gpu_query_pos = self.gpu_buffer[end_token_to_req:end_query_pos]
        self.gpu_num_tiles = self.gpu_buffer[end_query_pos:end_num_tiles]
        self.gpu_tile_req = self.gpu_buffer[end_num_tiles:end_tile_req]
        self.gpu_tile_q_start = self.gpu_buffer[end_tile_req:end_tile_q_start]
        self.gpu_tile_q_count = self.gpu_buffer[end_tile_q_start:]

    @staticmethod
    def count_tiles(num_scheduled_tokens):
        # 每个请求各自按 BLOCK_M 切，tile **不跨请求**：
        # 请求 A 的 20 行 + 请求 B 的 50 行，不能把打包前 32 行当成 A 的 tile。
        return sum(triton.cdiv(count, BLOCK_M)
                   for count in num_scheduled_tokens if count > 0)

    def validate(self, past_kv, num_scheduled_tokens, build_tiles=False):
        # 只做容量检查，不碰任何缓冲；调用方要在推进长度之前先调它
        num_requests = len(past_kv)
        num_tokens = sum(num_scheduled_tokens)

        if num_requests > self.max_num_seqs:
            raise ValueError(f"本轮请求数 {num_requests} 超过元数据缓冲容量 {self.max_num_seqs}"
                             f"（由 max_num_seqs 决定）")
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过元数据缓冲容量 {self.max_num_query_tokens}"
                             f"（由 max_num_batched_tokens 决定）")
        if build_tiles:
            num_tiles = self.count_tiles(num_scheduled_tokens)
            if num_tiles > self.tile_capacity:
                raise ValueError(f"本轮 tile 数 {num_tiles} 超过容量 {self.tile_capacity}"
                                 f"（由 max_num_batched_tokens 与 max_num_seqs 决定）")
        for _cache in past_kv:
            if len(_cache.block_table) > self.max_blocks_per_request:
                raise ValueError(
                    f"请求 {_cache.block_table} 的块表长度 {len(_cache.block_table)} 超过元数据缓冲容量 "
                    f"{self.max_blocks_per_request}（由 ceil(max_seq_len / block_size) 决定）")

    def fill(self, past_kv, num_scheduled_tokens, build_tiles=False):
        # 只用本轮真实用量覆盖对应的有效区域；尾部残留旧数据不参与计算。
        # build_tiles=False 时 num_tiles 置 0：这一轮的分块 kernel 一个 program 都不干活。
        self.validate(past_kv, num_scheduled_tokens, build_tiles)
        num_tokens = sum(num_scheduled_tokens)

        token_to_req = []
        query_pos = []
        tile_req = []
        tile_q_start = []
        tile_q_count = []
        row_offset = 0
        for req_idx, (_cache, count) in enumerate(zip(past_kv, num_scheduled_tokens)):
            block_table = _cache.block_table
            self.cpu_seq_lens[req_idx] = _cache.length
            self.cpu_block_tables[req_idx, :len(block_table)] = torch.tensor(block_table, dtype=torch.int32)
            token_to_req.extend([req_idx] * count)
            query_pos.extend(range(_cache.length - count, _cache.length))
            if build_tiles:
                for start in range(0, count, BLOCK_M):
                    tile_req.append(req_idx)
                    tile_q_start.append(row_offset + start)
                    tile_q_count.append(min(BLOCK_M, count - start))
            row_offset += count

        self.cpu_token_to_req[:num_tokens] = torch.tensor(token_to_req, dtype=torch.int32)
        self.cpu_query_pos[:num_tokens] = torch.tensor(query_pos, dtype=torch.int32)
        self.cpu_num_tiles[0] = len(tile_req)
        if tile_req:
            self.cpu_tile_req[:len(tile_req)] = torch.tensor(tile_req, dtype=torch.int32)
            self.cpu_tile_q_start[:len(tile_req)] = torch.tensor(tile_q_start, dtype=torch.int32)
            self.cpu_tile_q_count[:len(tile_req)] = torch.tensor(tile_q_count, dtype=torch.int32)
        return len(past_kv), num_tokens

    def upload(self):
        # 整段一次 H2D；non_blocking=False，不做 pinned memory
        self.gpu_buffer.copy_(self.cpu_buffer)
