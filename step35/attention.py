"""分页 attention：Triton 算子 + 元数据缓冲。

这一层只认「打包好的一维 query」和「固定容量的元数据」，不认请求、不认调度。
"""

import math

import torch

import triton
import triton.language as tl


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
    # 四个区域是同一段连续 int32 存储的不同视图，一次 copy_ 整段上传。

    def __init__(self, max_num_seqs, max_num_query_tokens, max_blocks_per_request, device):
        self.max_num_seqs = max_num_seqs
        self.max_num_query_tokens = max_num_query_tokens
        self.max_blocks_per_request = max_blocks_per_request
        self.device = device

        size = max_num_seqs * (1 + max_blocks_per_request) + 2 * max_num_query_tokens
        self.cpu_buffer = torch.zeros(size, dtype=torch.int32)
        self.gpu_buffer = torch.zeros(size, dtype=torch.int32, device=device)

        # 区域起止：seq_lens | block_tables | token_to_req | query_pos
        end_seq_lens = max_num_seqs
        end_block_tables = end_seq_lens + max_num_seqs * max_blocks_per_request
        end_token_to_req = end_block_tables + max_num_query_tokens

        self.cpu_seq_lens = self.cpu_buffer[:end_seq_lens]
        self.cpu_block_tables = self.cpu_buffer[end_seq_lens:end_block_tables].view(max_num_seqs, max_blocks_per_request)
        self.cpu_token_to_req = self.cpu_buffer[end_block_tables:end_token_to_req]
        self.cpu_query_pos = self.cpu_buffer[end_token_to_req:]

        self.gpu_seq_lens = self.gpu_buffer[:end_seq_lens]
        self.gpu_block_tables = self.gpu_buffer[end_seq_lens:end_block_tables].view(max_num_seqs, max_blocks_per_request)
        self.gpu_token_to_req = self.gpu_buffer[end_block_tables:end_token_to_req]
        self.gpu_query_pos = self.gpu_buffer[end_token_to_req:]

    def validate(self, past_kv, num_scheduled_tokens):
        # 只做容量检查，不碰任何缓冲；调用方要在推进长度之前先调它
        num_requests = len(past_kv)
        num_tokens = sum(num_scheduled_tokens)

        if num_requests > self.max_num_seqs:
            raise ValueError(f"本轮请求数 {num_requests} 超过元数据缓冲容量 {self.max_num_seqs}"
                             f"（由 max_num_seqs 决定）")
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过元数据缓冲容量 {self.max_num_query_tokens}"
                             f"（由 max_num_batched_tokens 决定）")
        for _cache in past_kv:
            if len(_cache.block_table) > self.max_blocks_per_request:
                raise ValueError(
                    f"请求 {_cache.block_table} 的块表长度 {len(_cache.block_table)} 超过元数据缓冲容量 "
                    f"{self.max_blocks_per_request}（由 ceil(max_seq_len / block_size) 决定）")

    def fill(self, past_kv, num_scheduled_tokens):
        # 只用本轮真实用量覆盖对应的有效区域；尾部残留旧数据不参与计算
        self.validate(past_kv, num_scheduled_tokens)
        num_tokens = sum(num_scheduled_tokens)

        token_to_req = []
        query_pos = []
        for req_idx, (_cache, count) in enumerate(zip(past_kv, num_scheduled_tokens)):
            block_table = _cache.block_table
            self.cpu_seq_lens[req_idx] = _cache.length
            self.cpu_block_tables[req_idx, :len(block_table)] = torch.tensor(block_table, dtype=torch.int32)
            token_to_req.extend([req_idx] * count)
            query_pos.extend(range(_cache.length - count, _cache.length))

        self.cpu_token_to_req[:num_tokens] = torch.tensor(token_to_req, dtype=torch.int32)
        self.cpu_query_pos[:num_tokens] = torch.tensor(query_pos, dtype=torch.int32)
        return len(past_kv), num_tokens

    def upload(self):
        # 整段一次 H2D；non_blocking=False，不做 pinned memory
        self.gpu_buffer.copy_(self.cpu_buffer)
