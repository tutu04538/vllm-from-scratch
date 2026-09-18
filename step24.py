'''
Prefill and decode
'''

from dataclasses import dataclass
import math

from more_itertools import last
import torch
from torch import ceil, nn

import hashlib
import json

import triton
import triton.language as tl


def _stable_hash(previous_hash: bytes, block: tuple[int, ...]) -> bytes:
    # token ID 不限于 0~255，先编码成文本，再交给 sha256
    data = json.dumps((previous_hash.hex(), block)).encode("utf-8")
    return hashlib.sha256(data).digest()


@triton.jit
def _paged_attention_kernel(
    q_ptr, out_ptr, k_pool_ptr, v_pool_ptr,
    block_tables_ptr, seq_lens_ptr, token_to_req_ptr, query_pos_ptr,
    stride_qn, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_btn, stride_btb,
    stride_on, stride_od,
    SM_SCALE: tl.constexpr, S: tl.constexpr, S_POW2: tl.constexpr,
    D: tl.constexpr, D_POW2: tl.constexpr,
):
    # 一个 program 负责一个 query：按逻辑块读 KV，online softmax 合并，写回 out[行]

    row = tl.program_id(0)
    req = tl.load(token_to_req_ptr + row)
    qpos = tl.load(query_pos_ptr + row)
    seq_len = tl.load(seq_lens_ptr + req)

    offs_d = tl.arange(0, D_POW2)
    mask_d = offs_d < D
    offs_s = tl.arange(0, S_POW2)   # 计算范围可能比物理块大小宽，多出来的要屏蔽

    q = tl.load(q_ptr + row * stride_qn + offs_d * stride_qd, mask=mask_d, other=0.0)

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

        kv_offsets = phys * stride_kb + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_pool_ptr + kv_offsets, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)
        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(mask_kv, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        scale = tl.exp(m_i - m_new)
        p = tl.where(mask_kv, tl.exp(score - m_new), 0.0)

        v_offsets = phys * stride_vb + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_pool_ptr + v_offsets, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)

        z_i = scale * z_i + tl.sum(p, axis=0)
        acc = scale * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(out_ptr + row * stride_on + offs_d * stride_od, acc / z_i, mask=mask_d)


def paged_attention(q, k_pool, v_pool, block_tables, seq_lens, token_to_req, query_pos):
    # 整批请求一次启动；q[N,D] -> out[N,D]，行顺序与 q 相同
    num_rows, d_model = q.shape
    _, block_size, _ = k_pool.shape
    out = torch.empty_like(q)

    _paged_attention_kernel[(num_rows,)](
        q, out, k_pool, v_pool,
        block_tables, seq_lens, token_to_req, query_pos,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1),
        SM_SCALE=1.0 / math.sqrt(d_model), S=block_size, D=d_model,
        S_POW2=triton.next_power_of_2(block_size), D_POW2=triton.next_power_of_2(d_model),
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

    def fill(self, past_kv, num_scheduled_tokens):
        # 只用本轮真实用量覆盖对应的有效区域；尾部残留旧数据不参与计算
        num_requests = len(past_kv)
        num_tokens = sum(num_scheduled_tokens)

        if num_requests > self.max_num_seqs:
            raise ValueError(f"本轮请求数 {num_requests} 超过元数据缓冲容量 {self.max_num_seqs}"
                             f"（由 max_num_seqs 决定）")
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过元数据缓冲容量 {self.max_num_query_tokens}"
                             f"（由 max_num_batched_tokens 决定）")

        token_to_req = []
        query_pos = []
        for req_idx, (_cache, count) in enumerate(zip(past_kv, num_scheduled_tokens)):
            block_table = _cache.block_table
            if len(block_table) > self.max_blocks_per_request:
                raise ValueError(
                    f"请求 {_cache.block_table} 的块表长度 {len(block_table)} 超过元数据缓冲容量 "
                    f"{self.max_blocks_per_request}（由 ceil(max_seq_len / block_size) 决定）")

            self.cpu_seq_lens[req_idx] = _cache.length
            self.cpu_block_tables[req_idx, :len(block_table)] = torch.tensor(block_table, dtype=torch.int32)
            token_to_req.extend([req_idx] * count)
            query_pos.extend(range(_cache.length - count, _cache.length))

        self.cpu_token_to_req[:num_tokens] = torch.tensor(token_to_req, dtype=torch.int32)
        self.cpu_query_pos[:num_tokens] = torch.tensor(query_pos, dtype=torch.int32)
        return num_requests, num_tokens

    def upload(self):
        # 整段一次 H2D；non_blocking=False，不做 pinned memory
        self.gpu_buffer.copy_(self.cpu_buffer)


@dataclass
class CacheConfig:
    block_table: list[int] = None # List of block indices in the KV cache
    length: int = 0


class SequenceConfig:
    def __init__(self, request_id, prompt_ids, max_new_tokens, block_size):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.cache = CacheConfig()
        self.block_size = block_size  # Size of each block in the KV cache
        self.block_hashes = []  # 本请求已确定的前缀块 hash 链，命中时从缓存里的前缀接上

    @property
    def prefill_len(self):
        return max(len(self.prompt_ids) - self.cache.length, 0)


class KVCachePool:

    def __init__(self, block_size, num_kv_blocks, d_model, device, enable_prefix_caching=True):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.d_model = d_model
        self.device = device
        self.k_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        self.v_cache = torch.zeros(num_kv_blocks, block_size, d_model, device=device)
        # 前两维看成一排 token 槽位，与底层存储共享，不是副本
        self.k_flat = self.k_cache.view(-1, d_model)
        self.v_flat = self.v_cache.view(-1, d_model)
        self.block_usage = [0] * self.num_kv_blocks  # 引用该块的活动请求数
        self.enable_prefix_caching = enable_prefix_caching
        self.block_hash = {}  # 前缀 hash -> 该块物理块编号
        self.block_to_hash = {}  # 物理块编号 -> 仍保留它的缓存条目 hash
        self.block_last_used = [0] * self.num_kv_blocks  # LRU 序号
        self.lru_seq = 0

    def _free_block_indices(self):
        # 真正空闲：没有活动请求引用，也没有被前缀缓存保留
        return [i for i in range(self.num_kv_blocks)
                if self.block_usage[i] == 0 and i not in self.block_to_hash]

    def _mark_used(self, block_idx):
        self.lru_seq += 1
        self.block_last_used[block_idx] = self.lru_seq

    def _evict_block(self, block_idx):
        # 先删除 key 与物理块的关联，之后这个块才能被重新分配
        del self.block_hash[self.block_to_hash.pop(block_idx)]

    def allocate_block(self, seq: SequenceConfig):
        # 先借用命中的前缀块，再补齐私有块；失败时返回 False，且不留下任何副作用

        total_blocks_needed = math.ceil((len(seq.prompt_ids) + seq.max_new_tokens - 1) / self.block_size)

        matched_blocks, matched_hashes = [], []
        if self.enable_prefix_caching:
            # 至少留下最后一个 prompt token 重新计算：本关不缓存 logits
            matched_blocks, matched_hashes = self.find_matched_prefix_blocks(seq.prompt_ids[:-1])
        new_blocks_needed = total_blocks_needed - len(matched_blocks)

        free_blocks = self._free_block_indices()
        evict_blocks = []
        if len(free_blocks) < new_blocks_needed:
            # 只能淘汰闲置缓存（活动引用为 0），且不能淘汰本次要借用的命中块
            idle_cached = [i for i in range(self.num_kv_blocks)
                           if self.block_usage[i] == 0
                           and i in self.block_to_hash
                           and i not in matched_blocks]
            idle_cached.sort(key=lambda i: self.block_last_used[i])
            if len(free_blocks) + len(idle_cached) < new_blocks_needed:
                return False
            evict_blocks = idle_cached[:new_blocks_needed - len(free_blocks)]

        # 容量已经确认足够，从这里开始改动状态
        for block_idx in matched_blocks:
            self.block_usage[block_idx] += 1
            self._mark_used(block_idx)
        for block_idx in evict_blocks:
            self._evict_block(block_idx)
        new_blocks = (free_blocks + evict_blocks)[:new_blocks_needed]
        for block_idx in new_blocks:
            self.block_usage[block_idx] = 1

        seq.cache.block_table = matched_blocks + new_blocks
        seq.cache.length = len(matched_blocks) * self.block_size
        seq.block_hashes = list(matched_hashes)
        return True

    def deallocate_block(self, seq: SequenceConfig):
        # 只释放本请求持有的全部活动引用；已登记的缓存条目继续保留为闲置缓存

        for block_idx in seq.cache.block_table:
            self.block_usage[block_idx] -= 1

    def publish_completed_prompt_blocks(self, seq: SequenceConfig):
        # 登记请求已经写完 KV 的完整 prompt 块；已有 key 保留原条目，私有块随请求释放

        full_blocks = min(len(seq.prompt_ids), seq.cache.length) // self.block_size

        for i in range(len(seq.block_hashes), full_blocks):
            block = tuple(seq.prompt_ids[i * self.block_size:(i + 1) * self.block_size])
            previous_hash = seq.block_hashes[-1] if seq.block_hashes else b""
            hash_value = _stable_hash(previous_hash, block)
            seq.block_hashes.append(hash_value)

            if hash_value in self.block_hash:
                continue

            block_idx = seq.cache.block_table[i]
            self.block_hash[hash_value] = block_idx
            self.block_to_hash[block_idx] = hash_value
            self._mark_used(block_idx)

    def _slots_of_range(self, block_table, start, count):
        # 请求内逻辑位置 [start, start+count) 对应的物理槽位：块编号 * block_size + 块内偏移
        positions = torch.arange(start, start + count, device=self.device)
        blocks = torch.tensor(block_table, device=self.device, dtype=torch.long)
        return blocks[positions // self.block_size] * self.block_size + positions % self.block_size

    def build_slot_mapping(self, caches, counts):
        # 本轮打包输入中第 i 个 token 的 K/V 应写到哪个 slot；用写入前的 length 算地址
        return torch.cat([self._slots_of_range(cache.block_table, cache.length, count)
                          for cache, count in zip(caches, counts)])

    def append_batch(self, caches, counts, new_k, new_v):
        # 整批写入本轮真实 token，K/V 各一次批量索引写入；写完再各自增加 length
        slot_mapping = self.build_slot_mapping(caches, counts)

        self.k_flat.index_copy_(0, slot_mapping, new_k)
        self.v_flat.index_copy_(0, slot_mapping, new_v)

        for cache, count in zip(caches, counts):
            cache.length += count

    def block_view(self, cache: CacheConfig, logical_block, count):
        # 直接给出池里这个物理块的有效切片；仍是池存储的视图，不复制整条历史
        block_idx = cache.block_table[logical_block]
        return self.k_cache[block_idx][:count], self.v_cache[block_idx][:count]

    def gather(self, cache: CacheConfig):
        # 按请求逻辑位置 0..length-1 一次选出 K 和 V，按逻辑顺序返回
        # 保留作参考/调试；attention 路径不再调用它

        if cache.length == 0:
            return self.k_cache.new_empty((0, self.d_model)), self.v_cache.new_empty((0, self.d_model))

        slots = self._slots_of_range(cache.block_table, 0, cache.length)
        return self.k_flat.index_select(0, slots), self.v_flat.index_select(0, slots)


    def find_matched_prefix_blocks(self, prompt_ids):

        # 从第一块开始连续匹配，遇到缺失就停止；返回 (物理块, hash) 两个列表
        matched_blocks = []
        matched_hashes = []
        block_num = len(prompt_ids) // self.block_size
        current_hash = b""
        for i in range(0, block_num):
            block = tuple(prompt_ids[i * self.block_size:(i + 1) * self.block_size])
            current_hash = _stable_hash(current_hash, block)
            if current_hash not in self.block_hash:
                break
            matched_blocks.append(self.block_hash[current_hash])
            matched_hashes.append(current_hash)

        return matched_blocks, matched_hashes


class Sampler:
    
    def __init__(self):
        pass
    
    def sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)


class DummyModel:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 5
        self.next_token_logits = torch.tensor([[0, 10, 0, 0, 0],
                                               [0, 0, 10, 0, 0],
                                               [0, 0, 0, 10, 0],
                                               [0, 0, 0, 0, 10],
                                               [0, 0, 0, 0, 10]], device=self.device, dtype=torch.float32)
        
    def forward(self, last_token_ids):
        # last_token_ids: Tensor of shape (batch_size,)
        # last_token_ids = last_token_ids.to(device=self.device, dtype=torch.long)
        return self.next_token_logits[last_token_ids]


class TinyCausalLM(nn.Module):
    
    def __init__(self, vocab_size=5, d_model=8, max_seq_len=32, device=None, attention_backend="torch",
                 attention_metadata=None, max_num_query_tokens=None, use_cuda_graph=False):
        super().__init__()

        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_backend = attention_backend
        self.attention_metadata = attention_metadata
        self.max_num_query_tokens = max_num_query_tokens
        self.use_cuda_graph = use_cuda_graph
        self.graphs = {}          # N -> CUDAGraph
        self.graph_outputs = {}   # N -> 该图的 logits 输出（存储会被后续 replay 复用）

        if max_num_query_tokens is not None:
            # 固定容量输入缓冲：地址不变，每轮只改内容
            self.input_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)
            self.position_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)
            self.slot_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        self.position_embedding = nn.Embedding(max_seq_len, d_model).to(self.device)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False).to(self.device)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device)
        
    
    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len).to(self.device)
        position_embeds = self.position_embedding(position_ids)
        
        inputs_embeds = token_embeds + position_embeds
        
        q = self.q_proj(inputs_embeds)
        k = self.k_proj(inputs_embeds)
        v = self.v_proj(inputs_embeds)
        
        score = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        mask = torch.triu(torch.ones((seq_len, seq_len), device=input_ids.device), diagonal=1)
        score = score.masked_fill(mask == 1, float('-inf'))
        weights = torch.softmax(score, dim=-1)
        
        out = torch.matmul(weights, v)
        
        logits = self.lm_head(out)
        
        past_kv = [(_k, _v) for _k, _v in zip(k, v)]
        
        return logits, past_kv
    
    def _prepare_inputs(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool):
        # 图外：算地址、推进 Python 状态、把本轮输入填进固定缓冲
        # input_ids: (N,) 只有真实 token；past_kv 与 num_scheduled_tokens 同序

        # 位置和 slot 都用写入前的 length 算
        position_ids = torch.cat([
            torch.arange(_cache.length, _cache.length + num_tokens, device=self.device)
            for _cache, num_tokens in zip(past_kv, num_scheduled_tokens)
        ])
        position_ids = torch.clamp(position_ids, max=self.max_seq_len - 1)
        slot_mapping = kv_cache_pool.build_slot_mapping(past_kv, num_scheduled_tokens)

        # Python 状态在这里且只在这里前进一次；warmup/capture/replay 都不再改它
        for _cache, num_tokens in zip(past_kv, num_scheduled_tokens):
            _cache.length += num_tokens

        num_tokens = sum(num_scheduled_tokens)
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过固定输入缓冲容量 "
                             f"{self.max_num_query_tokens}（由 max_num_batched_tokens 决定）")

        self.input_buffer[:num_tokens].copy_(input_ids)
        self.position_buffer[:num_tokens].copy_(position_ids)
        self.slot_buffer[:num_tokens].copy_(slot_mapping)

        if self.attention_metadata is not None:
            # seq_lens 用写完 KV 之后的长度
            self.attention_metadata.fill(past_kv, num_scheduled_tokens)
            self.attention_metadata.upload()

        return num_tokens

    def _forward_append(self, input_ids: torch.Tensor, num_scheduled_tokens: list[int], past_kv: list[CacheConfig], kv_cache_pool: KVCachePool):
        # 返回 logits (N, vocab_size)，行顺序与 input_ids 相同

        num_tokens = self._prepare_inputs(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool)

        if self.attention_backend != "triton":
            return self._torch_forward(num_tokens, num_scheduled_tokens, past_kv, kv_cache_pool)

        if not self.use_cuda_graph:
            return self.gpu_forward(num_tokens, kv_cache_pool)

        graph = self.graphs.get(num_tokens)
        if graph is None:
            graph = self._capture_graph(num_tokens, kv_cache_pool)
        graph.replay()
        return self.graph_outputs[num_tokens]

    def _embeds_qkv(self, num_tokens):
        # 固定缓冲的前 num_tokens 行 -> 一次投影
        inputs_embeds = self.token_embedding(self.input_buffer[:num_tokens]) + \
            self.position_embedding(self.position_buffer[:num_tokens])
        return self.q_proj(inputs_embeds), self.k_proj(inputs_embeds), self.v_proj(inputs_embeds)

    def _write_kv(self, num_tokens, k, v, kv_cache_pool):
        # 纯 GPU 写：只按 slot 原位写，不碰 Python 状态
        slot_mapping = self.slot_buffer[:num_tokens]
        kv_cache_pool.k_flat.index_copy_(0, slot_mapping, k)
        kv_cache_pool.v_flat.index_copy_(0, slot_mapping, v)

    def gpu_forward(self, num_tokens, kv_cache_pool):
        # 图内：只做 GPU 运算，读写固定地址的缓冲区
        q, k, v = self._embeds_qkv(num_tokens)
        self._write_kv(num_tokens, k, v, kv_cache_pool)

        metadata = self.attention_metadata
        # 传整段容量视图：kernel 由 grid=(N,) 和 seq_len 驱动，只读有效区域
        out = paged_attention(
            q,
            kv_cache_pool.k_cache,
            kv_cache_pool.v_cache,
            metadata.gpu_block_tables,
            metadata.gpu_seq_lens,
            metadata.gpu_token_to_req,
            metadata.gpu_query_pos,
        )
        return self.lm_head(out)

    def _torch_forward(self, num_tokens, num_scheduled_tokens, past_kv, kv_cache_pool):
        # 同设备参考路径：与 gpu_forward 共用准备和 KV 写入，只换 attention 算法
        q, k, v = self._embeds_qkv(num_tokens)
        self._write_kv(num_tokens, k, v, kv_cache_pool)

        logits_list = []
        offset = 0
        for _cache, count in zip(past_kv, num_scheduled_tokens):
            query_positions = torch.arange(_cache.length - count, _cache.length, device=self.device)
            out = self.block_attention(q[offset:offset + count], _cache, kv_cache_pool, query_positions)
            logits_list.append(self.lm_head(out))
            offset += count

        return torch.cat(logits_list, dim=0)

    def _capture_graph(self, num_tokens, kv_cache_pool):
        # 预热用同一批真实输入反复算：只往本轮该写的 slot 写同样的 KV，Python 状态不动
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.gpu_forward(num_tokens, kv_cache_pool)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.gpu_forward(num_tokens, kv_cache_pool)

        self.graphs[num_tokens] = graph
        self.graph_outputs[num_tokens] = output
        return graph

    def block_attention(self, q, cache: CacheConfig, kv_cache_pool: KVCachePool, query_positions):
        # 直接按逻辑块读 KV，用 online softmax 把各块结果合并成全局 attention
        # q: (Q, D)，Q 是本请求本轮的 query 数；query_positions: (Q,)

        device = q.device
        block_size = kv_cache_pool.block_size
        num_blocks = -(-cache.length // block_size)  # ceil

        # 每个 query 的累计状态：已见最大分数、未归一化权重和、加权 value 和
        m = torch.full((q.shape[0], 1), float('-inf'), device=device)
        z = torch.zeros((q.shape[0], 1), device=device)
        u = torch.zeros((q.shape[0], self.d_model), device=device)

        for logical_block in range(num_blocks):
            block_start = logical_block * block_size
            count = min(block_size, cache.length - block_start)  # 尾块只读有效部分
            k_block, v_block = kv_cache_pool.block_view(cache, logical_block, count)  # (count, D)

            key_positions = torch.arange(block_start, block_start + count, device=device)
            score = torch.matmul(q, k_block.transpose(-1, -2)) / (self.d_model ** 0.5)  # (Q, count)
            score = score.masked_fill(key_positions.unsqueeze(0) > query_positions.unsqueeze(-1), float('-inf'))

            # 全被 mask 的行：block_max 是 -inf，m_new 保持 m，scale=1、p 全 0，该块贡献零
            block_max = score.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, block_max)
            scale = torch.exp(m - m_new)   # 旧结果换到新基准
            p = torch.exp(score - m_new)   # (Q, count)，未归一化

            z = scale * z + p.sum(dim=-1, keepdim=True)
            u = scale * u + torch.matmul(p, v_block)
            m = m_new

        return u / z


class Scheduler:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4, block_size=4, enable_prefix_caching=True, on_finished=None, kv_cache_pool: KVCachePool=None):
        
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running : list[SequenceConfig] = []
        self.waiting : list[SequenceConfig] = []
        self.step_done = []
        self.enable_prefix_caching = enable_prefix_caching
        self.block_size = block_size
        self.on_finished = on_finished
        self.num_scheduled_tokens = []
        self.scheduled_items = []  # 本轮计划：prefill 与 decode 合成一份
        self.kv_cache_pool = kv_cache_pool

    def add_request(self, request):
        seq = SequenceConfig(request["request_id"], request["prompt_ids"], request["max_new_tokens"], self.block_size)
        self.waiting.append(seq)
    
    def has_unfinished_requests(self):
        return len(self.running) + len(self.waiting) > 0
    
    def schedule(self):
        # Fill running with waiting sequences if there's space
        self.scheduled_items = []
        self.step_done = []
        
        for seq in self.waiting:
            if seq.max_new_tokens == 0:
                # Zero budget requests are processed immediately
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": []})
                self.step_done.append({"request_id": seq.request_id, "output_ids": []})
        
        self.waiting = [seq for seq in self.waiting if seq.max_new_tokens > 0]
        # Limit the number of batched tokens
        while len(self.running) < self.max_num_seqs and self.waiting:
            next_seq = self.waiting[0]
            if self.kv_cache_pool.allocate_block(next_seq):
                self.waiting.pop(0)
                self.running.append(next_seq)
            else:
                break  # No more blocks available, cannot schedule more sequences
        
        decode_req = [req for req in self.running if req.prefill_len == 0]
        decode_token_budget = len(decode_req)
        assert decode_token_budget <= self.max_num_batched_tokens, "Decode token budget exceeds max_num_batched_tokens"
        prefill_token_budget = self.max_num_batched_tokens - decode_token_budget

        for seq in self.running:
            if seq.prefill_len > 0:
                if prefill_token_budget == 0:
                    continue

                num_scheduled_tokens = min(seq.prefill_len, prefill_token_budget)
                prefill_token_budget -= num_scheduled_tokens
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": seq.prompt_ids[seq.cache.length:seq.cache.length + num_scheduled_tokens],
                    "num_scheduled_tokens": num_scheduled_tokens,
                    "can_sample": num_scheduled_tokens == seq.prefill_len
                })
            else:
                self.scheduled_items.append({
                    "request": seq,
                    "input_ids": [seq.output_ids[-1]],
                    "num_scheduled_tokens": 1,
                    "can_sample": True
                })

        return self.scheduled_items
    
    def post_step(self):

        for seq in self.running:

            if seq.prefill_len > 0:
                continue  # Prefill not finished yet

            if seq.output_ids[-1] == 4 or len(seq.output_ids) >= seq.max_new_tokens:
                if self.enable_prefix_caching:
                    # 先把可复用的完整 prompt 块登记为缓存，再释放本请求的活动引用
                    self.kv_cache_pool.publish_completed_prompt_blocks(seq)
                if self.on_finished:
                    self.on_finished({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.step_done.append({"request_id": seq.request_id, "output_ids": seq.output_ids})
                self.kv_cache_pool.deallocate_block(seq)
                seq.cache = None  # Reset past_kv for completed sequences

        self.running = [seq for seq in self.running if len(seq.output_ids) < seq.max_new_tokens and (not seq.output_ids or seq.output_ids[-1] != 4)]

class Engine:
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True, device=None, attention_backend="torch", use_cuda_graph=False):
        if attention_backend not in ("torch", "triton"):
            raise ValueError(f"未知的 attention_backend: {attention_backend!r}，可选 'torch' 或 'triton'")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if attention_backend == "triton" and device.type != "cuda":
            raise ValueError(f"attention_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
        if use_cuda_graph and (device.type != "cuda" or attention_backend != "triton"):
            raise ValueError(f"use_cuda_graph=True 只支持 CUDA + Triton，当前 device={device.type}、"
                             f"attention_backend={attention_backend!r}")

        # 容量按引擎配置一次分配，与首次出现的 batch 大小无关
        attention_metadata = None
        if attention_backend == "triton":
            attention_metadata = AttentionMetadata(
                max_num_seqs=max_num_seqs,
                max_num_query_tokens=max_num_batched_tokens,
                max_blocks_per_request=math.ceil(max_seq_len / block_size),
                device=device,
            )

        self.model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len,
                                  device=device, attention_backend=attention_backend,
                                  attention_metadata=attention_metadata,
                                  max_num_query_tokens=max_num_batched_tokens,
                                  use_cuda_graph=use_cuda_graph)
        self.model.eval()
        self.sampler = Sampler()
        self.device = self.model.device
        self.attention_backend = attention_backend
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, d_model, self.model.device, self.enable_prefix_caching)
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool)
        
    def add_request(self, request):
        self.scheduler.add_request(request)
    
    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()
    
    def _sample(self, logits, scheduled_items):
        # 就绪请求取自己片段 [start:end) 的最后一行 logits，行与请求从同一份计划里对应

        last_rows = []
        offset = 0
        for i, item in enumerate(scheduled_items):
            offset += item["num_scheduled_tokens"]
            if item["can_sample"]:
                last_rows.append((i, offset - 1))

        if not last_rows:
            return None

        rows = torch.tensor([row for _, row in last_rows], device=logits.device, dtype=torch.long)
        output_ids = self.sampler.sample(logits[rows, :])
        for (i, _), output_id in zip(last_rows, output_ids):
            scheduled_items[i]["request"].output_ids.append(output_id.item())
        return None

    def step(self):

        with torch.inference_mode():

            self.scheduler.schedule()

            if not self.scheduler.has_unfinished_requests():
                return self.scheduler.step_done

            scheduled_items = self.scheduler.scheduled_items

            if scheduled_items:
                # 本轮所有真实 token 拼成一维，prefill 与 decode 共用一次模型调用
                # 先在 CPU 组装，再由图外的准备步骤一次写进固定 GPU 缓冲
                input_ids = torch.tensor([token for item in scheduled_items for token in item["input_ids"]], dtype=torch.long)
                num_scheduled_tokens = [item["num_scheduled_tokens"] for item in scheduled_items]
                past_kv = [item["request"].cache for item in scheduled_items]

                logits = self.model._forward_append(input_ids, num_scheduled_tokens, past_kv, self.kv_cache_pool)
                self._sample(logits, scheduled_items)

            self.scheduler.post_step()

        return self.scheduler.step_done


if __name__ == "__main__":
    def on_finished(result):
        print("完成通知：", result)

    engine = Engine(max_num_seqs=2, vocab_size=100, d_model=8, max_seq_len=32, on_finished=on_finished)

    # 先收到 A，只让它执行一轮，不要在这里把 A 跑到结束。
    print("提交 A B C")
    engine.add_request({"request_id": "A", "prompt_ids": [0, 1, 2, 3, 4], "max_new_tokens": 4})
    engine.add_request({"request_id": "B", "prompt_ids": [3], "max_new_tokens": 2})
    engine.add_request({"request_id": "C", "prompt_ids": [0, 1], "max_new_tokens": 2})
    
    round_id = 0
    while engine.has_unfinished_requests():
        print(f"step {round_id} 返回：", engine.step())
        round_id += 1

    print("所有请求完成，当前还有未完成请求吗？", engine.has_unfinished_requests())
