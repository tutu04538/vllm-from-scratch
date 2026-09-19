'''
Prefill and decode
'''

from dataclasses import dataclass
import math

from more_itertools import last
import torch
from torch import ceil, nn
import torch.nn.functional as F

import hashlib
import json
import pathlib

from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file as _save_safetensors

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

    q = tl.load(q_ptr + row * stride_qn + q_head * stride_qh + offs_d * stride_qd,
                mask=mask_d, other=0.0)

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
        k = tl.load(k_pool_ptr + kv_offsets, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)
        score = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        score = tl.where(mask_kv, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        scale = tl.exp(m_i - m_new)
        p = tl.where(mask_kv, tl.exp(score - m_new), 0.0)

        v_offsets = phys * stride_vb + offs_s[:, None] * stride_vs \
            + kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_pool_ptr + v_offsets, mask=mask_kv[:, None] & mask_d[None, :], other=0.0)

        z_i = scale * z_i + tl.sum(p, axis=0)
        acc = scale * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(out_ptr + row * stride_on + q_head * stride_oh + offs_d * stride_od,
             acc / z_i, mask=mask_d)


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

    def __init__(self, block_size, num_kv_blocks, num_kv_heads, head_dim, device,
                 enable_prefix_caching=True, num_layers=1):
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        # 每层一份 KV：同一个 token 在每层的 K/V 不同，不能写进同一片缓存
        self.k_cache = torch.zeros(num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim, device=device)
        self.v_cache = torch.zeros(num_layers, num_kv_blocks, block_size, num_kv_heads, head_dim, device=device)
        # 每层的 [块, 块内偏移] 看成一排 token 槽位，与底层存储共享，不是副本
        self.k_flat = self.k_cache.view(num_layers, -1, num_kv_heads, head_dim)
        self.v_flat = self.v_cache.view(num_layers, -1, num_kv_heads, head_dim)
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

    def block_view(self, cache: CacheConfig, logical_block, count, layer_idx=0):
        # 直接给出池里这个物理块在某一层上的有效切片；仍是池存储的视图，不复制
        block_idx = cache.block_table[logical_block]
        return self.k_cache[layer_idx, block_idx][:count], self.v_cache[layer_idx, block_idx][:count]

    def gather(self, cache: CacheConfig, layer_idx=0):
        # 按请求逻辑位置 0..length-1 一次选出某一层的 K 和 V，按逻辑顺序返回
        # 保留作参考/调试；attention 路径不再调用它

        if cache.length == 0:
            empty = (0, self.num_kv_heads, self.head_dim)
            return self.k_cache.new_empty(empty), self.v_cache.new_empty(empty)

        slots = self._slots_of_range(cache.block_table, 0, cache.length)
        return self.k_flat[layer_idx].index_select(0, slots), self.v_flat[layer_idx].index_select(0, slots)


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


MODEL_CONFIG_NAME = "config.json"
MODEL_WEIGHTS_NAME = "model.safetensors"
FORMAT_VERSION = 1
MODEL_TYPE = "tiny_rope_decoder"
MODEL_DTYPE = "float32"

# JSON 里必须出现的结构字段；运行选项（device、后端、并发数……）不属于模型结构
_STRUCT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
                  "num_layers", "intermediate_size", "rms_norm_eps", "rope_theta")
_INT_FIELDS = ("vocab_size", "d_model", "max_seq_len", "num_q_heads", "num_kv_heads",
               "num_layers", "intermediate_size")
_FLOAT_FIELDS = ("rms_norm_eps", "rope_theta")


def save_model(model, model_dir):
    # 把模型配置和全部参数写进目录；不改动原模型（权重、device、已捕获的图都还能用）
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {"format_version": FORMAT_VERSION, "model_type": MODEL_TYPE, "dtype": MODEL_DTYPE}
    config.update(model.model_config())          # 取实际模型配置，不写死数值
    (model_dir / MODEL_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")

    # safetensors 要求稠密连续张量；保存到 CPU float32，与模型当前所在设备无关
    weights = {name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
               for name, tensor in model.state_dict().items()}
    _save_safetensors(weights, model_dir / MODEL_WEIGHTS_NAME)
    return model_dir


def load_model_config(model_dir):
    # 读取并检查配置；任何一项不对就报错，不返回半份配置
    config_path = pathlib.Path(model_dir) / MODEL_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少配置文件 {config_path}")

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as ex:
        raise ValueError(f"{config_path} 不是合法 JSON: {ex}") from ex
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} 的顶层必须是对象")

    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"不支持的 format_version={config.get('format_version')!r}，"
                         f"本实现只支持 {FORMAT_VERSION}")
    if config.get("model_type") != MODEL_TYPE:
        raise ValueError(f"不支持的 model_type={config.get('model_type')!r}，本实现只支持 {MODEL_TYPE!r}")
    if config.get("dtype") != MODEL_DTYPE:
        raise ValueError(f"不支持的 dtype={config.get('dtype')!r}，本实现只支持 {MODEL_DTYPE!r}")

    missing = [name for name in _STRUCT_FIELDS if name not in config]
    if missing:
        raise ValueError(f"配置缺少必需字段: {missing}")
    for name in _INT_FIELDS:
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError(f"配置字段 {name} 必须是整数，收到 {config[name]!r}")
    for name in _FLOAT_FIELDS:
        if isinstance(config[name], bool) or not isinstance(config[name], (int, float)):
            raise ValueError(f"配置字段 {name} 必须是数值，收到 {config[name]!r}")
    return config


def build_model_from_config(config, device, attention_backend, max_num_batched_tokens, use_cuda_graph):
    # 按配置构造模型；维度合法性由 TinyCausalLM 的校验负责（缺字段、非法维度都会明确报错）
    return TinyCausalLM(
        vocab_size=config["vocab_size"], d_model=config["d_model"], max_seq_len=config["max_seq_len"],
        num_q_heads=config["num_q_heads"], num_kv_heads=config["num_kv_heads"],
        num_layers=config["num_layers"], intermediate_size=config["intermediate_size"],
        rms_norm_eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
        device=device, attention_backend=attention_backend,
        max_num_query_tokens=max_num_batched_tokens, use_cuda_graph=use_cuda_graph)


def load_model_weights(model_dir, model):
    # 读取并检查权重，再加载进已经建在目标设备上的模型
    weights_path = pathlib.Path(model_dir) / MODEL_WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"缺少权重文件 {weights_path}")

    weights = _load_safetensors(weights_path)
    # load_state_dict 会静默把 dtype 转成目标参数的类型，所以这里自己确认是 FP32
    for name, tensor in weights.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"权重 {name} 的 dtype 是 {tensor.dtype}，本实现只支持 float32")

    # strict=True：缺参数、多参数、shape 不符都会抛，不会留下混着随机参数的模型
    model.load_state_dict(weights, strict=True)
    return model


def _resolve_device(device):
    return torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_runtime(device, attention_backend, use_cuda_graph):
    # 后端与设备组合的校验；随机初始化和从目录加载两条路都走这里
    if attention_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 attention_backend: {attention_backend!r}，可选 'torch' 或 'triton'")
    if attention_backend == "triton" and device.type != "cuda":
        raise ValueError(f"attention_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
    if use_cuda_graph and (device.type != "cuda" or attention_backend != "triton"):
        raise ValueError(f"use_cuda_graph=True 只支持 CUDA + Triton，当前 device={device.type}、"
                         f"attention_backend={attention_backend!r}")


def _rotate_half(x):
    # 前后半段配对：(0, D/2), (1, D/2+1), ...
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    # 固定频率的 RoPE：初始化时算好 cos/sin 表，按逻辑位置查表

    def __init__(self, head_dim, max_seq_len, theta):
        super().__init__()
        inv_freq = torch.pow(float(theta),
                             -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        # 查表数据，不是可学习参数；persistent=False 让它不进 state_dict
        self.register_buffer("cos_table", freqs.cos(), persistent=False)
        self.register_buffer("sin_table", freqs.sin(), persistent=False)

    def forward(self, x, positions):
        # x: [N, heads, head_dim]；positions: [N] 逻辑位置（不是打包行号）
        cos = self.cos_table[positions]
        sin = self.sin_table[positions]
        cos = torch.cat((cos, cos), dim=-1)[:, None, :]   # [N,1,head_dim]，按 head 广播
        sin = torch.cat((sin, sin), dim=-1)[:, None, :]
        return x * cos + _rotate_half(x) * sin


class RMSNorm(nn.Module):
    # RMSNorm(x) = x / sqrt(mean(x^2, 最后一维) + eps) * weight，逐 token 做，不减均值

    def __init__(self, d_model, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class DecoderLayer(nn.Module):
    # h = x + Attention(RMSNorm_1(x))
    # y = h + MLP(RMSNorm_2(h))
    # Attention 含 QKV 投影、GQA、分页 KV、head 拼接和 o_proj，不含 lm_head

    def __init__(self, d_model, num_q_heads, num_kv_heads, intermediate_size, eps):
        super().__init__()
        head_dim = d_model // num_q_heads
        self.norm1 = RMSNorm(d_model, eps)
        self.norm2 = RMSNorm(d_model, eps)

        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU：down_proj(silu(gate_proj(z)) * up_proj(z))
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, hidden):
        # MLP 部分与 attention 路径共用：入参是残差前的 h
        z = self.norm2(hidden)
        return hidden + self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))


class TinyCausalLM(nn.Module):
    
    def __init__(self, vocab_size=5, d_model=8, max_seq_len=32, device=None, attention_backend="torch",
                 attention_metadata=None, max_num_query_tokens=None, use_cuda_graph=False,
                 num_q_heads=1, num_kv_heads=1, num_layers=2, intermediate_size=64,
                 rms_norm_eps=1e-6, rope_theta=10000.0):
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

        # 结构校验：随机初始化和从目录加载都走这里
        if vocab_size <= 0:
            raise ValueError(f"vocab_size 必须为正，收到 {vocab_size}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len 必须为正，收到 {max_seq_len}")

        # head 配置：默认单头，旧行为不变
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError(f"头数必须为正，收到 num_q_heads={num_q_heads}、num_kv_heads={num_kv_heads}")
        if d_model % num_q_heads != 0:
            raise ValueError(f"d_model={d_model} 不能被 num_q_heads={num_q_heads} 整除")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(f"num_q_heads={num_q_heads} 不能被 num_kv_heads={num_kv_heads} 整除")

        if num_layers <= 0:
            raise ValueError(f"层数必须为正，收到 num_layers={num_layers}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size 必须为正，收到 {intermediate_size}")
        if rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps 必须为正，收到 {rms_norm_eps}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta 必须为正，收到 {rope_theta}")
        head_dim = d_model // num_q_heads
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE 要求 head_dim 为正偶数，当前 head_dim={head_dim}"
                             f"（d_model={d_model}, num_q_heads={num_q_heads}）")

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_q_heads // num_kv_heads  # 连续分组：kv_head = q_head // group_size
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        self.token_embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        # 位置信息不再用 embedding 加进 hidden，而是每层旋转 Q/K
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta).to(self.device)

        # 每层有自己独立的 Q/K/V/O、MLP 与两个 norm
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_q_heads, num_kv_heads, intermediate_size, rms_norm_eps)
            for _ in range(num_layers)
        ]).to(self.device)

        self.norm = RMSNorm(d_model, rms_norm_eps).to(self.device)   # 最终 norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device)
    
    def model_config(self):
        # 保存用：这个模型真实的结构，不是构造参数的默认值
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }

    def _prepare_inputs(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool):
        # 图外：算地址、推进 Python 状态、把本轮输入填进固定缓冲
        # input_ids: (N,) 只有真实 token；past_kv 与 num_scheduled_tokens 同序

        num_tokens = sum(num_scheduled_tokens)

        # 先只做校验，全部通过之后才建 Tensor、推进状态
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过固定输入缓冲容量 "
                             f"{self.max_num_query_tokens}（由 max_num_batched_tokens 决定）")
        # 位置上下界用 Python 整数判，不必先建 GPU Tensor 再取回来
        for _cache, count in zip(past_kv, num_scheduled_tokens):
            if count == 0:
                continue
            start = _cache.length
            if start < 0:
                raise ValueError(f"请求的缓存长度 {start} 为负，位置必须从 0 开始")
            if start + count > self.max_seq_len:
                raise ValueError(f"请求的位置区间 [{start}, {start + count}) 超出 max_seq_len="
                                 f"{self.max_seq_len} 的支持上限；请调大 max_seq_len")
        if self.attention_metadata is not None:
            # 容量检查也放在改状态之前，失败时不留副作用
            self.attention_metadata.validate(past_kv, num_scheduled_tokens)

        # 校验通过，才开始算地址；位置是逻辑位置，不是打包行号
        position_ids = torch.cat([
            torch.arange(_cache.length, _cache.length + count, device=self.device)
            for _cache, count in zip(past_kv, num_scheduled_tokens)
        ])
        slot_mapping = kv_cache_pool.build_slot_mapping(past_kv, num_scheduled_tokens)

        # Python 状态在这里且只在这里前进一次；warmup/capture/replay 都不再改它
        # 注意循环变量不能叫 num_tokens：for 循环的变量会泄漏出去，把上面的总数覆盖掉
        for _cache, count in zip(past_kv, num_scheduled_tokens):
            _cache.length += count

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

    def _input_embeds(self, num_tokens):
        # 整个模型只有一次 token embedding，位置信息由每层的 RoPE 提供
        return self.token_embedding(self.input_buffer[:num_tokens])

    def _layer_qkv(self, layer, hidden, num_tokens, positions):
        # 某一层的 QKV 投影 -> 拆成 head -> 按逻辑位置旋转 Q/K，V 不旋转
        q = layer.q_proj(hidden).view(num_tokens, self.num_q_heads, self.head_dim)
        k = layer.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = layer.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        return self.rotary(q, positions), self.rotary(k, positions), v

    def _write_kv(self, num_tokens, k, v, kv_cache_pool, layer_idx):
        # 纯 GPU 写：只按 slot 原位写本层，不碰 Python 状态
        slot_mapping = self.slot_buffer[:num_tokens]
        kv_cache_pool.k_flat[layer_idx].index_copy_(0, slot_mapping, k)
        kv_cache_pool.v_flat[layer_idx].index_copy_(0, slot_mapping, v)

    def _attention_out(self, layer, layer_idx, hidden, num_tokens, positions, kv_cache_pool):
        # 一层里的 Attention 段：QKV+RoPE -> 写本层 KV -> 分页 attention -> o_proj
        q, k, v = self._layer_qkv(layer, layer.norm1(hidden), num_tokens, positions)
        self._write_kv(num_tokens, k, v, kv_cache_pool, layer_idx)

        metadata = self.attention_metadata
        # 传本层的四维视图：kernel 由 grid=(N, num_q_heads) 和 seq_len 驱动，只读有效区域
        out = paged_attention(
            q,
            kv_cache_pool.k_cache[layer_idx],
            kv_cache_pool.v_cache[layer_idx],
            metadata.gpu_block_tables,
            metadata.gpu_seq_lens,
            metadata.gpu_token_to_req,
            metadata.gpu_query_pos,
            self.group_size,
        )
        # 各 head 按顺序拼回 d_model，再映射回去；这一层到此为止，不碰 lm_head
        return layer.o_proj(out.reshape(num_tokens, self.d_model))

    def gpu_forward(self, num_tokens, kv_cache_pool):
        # 图内：只做 GPU 运算，读写固定地址的缓冲区
        hidden = self._input_embeds(num_tokens)
        positions = self.position_buffer[:num_tokens]   # 各层共用同一组逻辑位置

        for layer_idx, layer in enumerate(self.layers):
            h = hidden + self._attention_out(layer, layer_idx, hidden, num_tokens, positions, kv_cache_pool)
            hidden = layer(h)   # h + MLP(RMSNorm_2(h))

        return self.lm_head(self.norm(hidden))

    def _torch_forward(self, num_tokens, num_scheduled_tokens, past_kv, kv_cache_pool):
        # 同设备参考路径：与 gpu_forward 共用准备和 KV 写入，只换 attention 算法
        hidden = self._input_embeds(num_tokens)
        positions = self.position_buffer[:num_tokens]

        for layer_idx, layer in enumerate(self.layers):
            q, k, v = self._layer_qkv(layer, layer.norm1(hidden), num_tokens, positions)
            self._write_kv(num_tokens, k, v, kv_cache_pool, layer_idx)

            heads_out = torch.empty_like(q)
            offset = 0
            for _cache, count in zip(past_kv, num_scheduled_tokens):
                query_positions = torch.arange(_cache.length - count, _cache.length, device=self.device)
                heads_out[offset:offset + count] = self.block_attention(
                    q[offset:offset + count], _cache, kv_cache_pool, query_positions, layer_idx)
                offset += count

            h = hidden + layer.o_proj(heads_out.reshape(num_tokens, self.d_model))
            hidden = layer(h)

        return self.lm_head(self.norm(hidden))

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

    def block_attention(self, q, cache: CacheConfig, kv_cache_pool: KVCachePool, query_positions, layer_idx=0):
        # 直接按逻辑块读 KV，用 online softmax 把各块结果合并成全局 attention
        # q: (Q, num_q_heads, head_dim)，Q 是本请求本轮的 query 数；返回同形状

        device = q.device
        block_size = kv_cache_pool.block_size
        num_blocks = -(-cache.length // block_size)  # ceil
        num_queries = q.shape[0]
        out = torch.empty_like(q)

        for q_head in range(self.num_q_heads):
            kv_head = q_head // self.group_size          # 连续分组
            q_head_vec = q[:, q_head, :]                 # (Q, head_dim)

            # 每个 query 的累计状态：已见最大分数、未归一化权重和、加权 value 和
            m = torch.full((num_queries, 1), float('-inf'), device=device)
            z = torch.zeros((num_queries, 1), device=device)
            u = torch.zeros((num_queries, self.head_dim), device=device)

            for logical_block in range(num_blocks):
                block_start = logical_block * block_size
                count = min(block_size, cache.length - block_start)  # 尾块只读有效部分
                k_block, v_block = kv_cache_pool.block_view(cache, logical_block, count, layer_idx)
                k_head = k_block[:, kv_head, :]          # (count, head_dim)
                v_head = v_block[:, kv_head, :]

                key_positions = torch.arange(block_start, block_start + count, device=device)
                score = torch.matmul(q_head_vec, k_head.transpose(-1, -2)) / (self.head_dim ** 0.5)
                score = score.masked_fill(key_positions.unsqueeze(0) > query_positions.unsqueeze(-1), float('-inf'))

                # 全被 mask 的行：block_max 是 -inf，m_new 保持 m，scale=1、p 全 0，该块贡献零
                block_max = score.max(dim=-1, keepdim=True).values
                m_new = torch.maximum(m, block_max)
                scale = torch.exp(m - m_new)   # 旧结果换到新基准
                p = torch.exp(score - m_new)   # (Q, count)，未归一化

                z = scale * z + p.sum(dim=-1, keepdim=True)
                u = scale * u + torch.matmul(p, v_head)
                m = m_new

            out[:, q_head, :] = u / z

        return out


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
    
    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True, device=None, attention_backend="torch", use_cuda_graph=False, num_q_heads=1, num_kv_heads=1,
                 num_layers=2, intermediate_size=64, rms_norm_eps=1e-6, rope_theta=10000.0, model=None):
        # model 给定时用它，不再按上面的维度参数随机初始化（加载路径走这里）
        # 后端与 Graph 开关也以模型上的为准，避免两边不一致
        if model is None:
            device = _resolve_device(device)
            _check_runtime(device, attention_backend, use_cuda_graph)
            model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len,
                                 device=device, attention_backend=attention_backend,
                                 max_num_query_tokens=max_num_batched_tokens,
                                 use_cuda_graph=use_cuda_graph,
                                 num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                                 num_layers=num_layers, intermediate_size=intermediate_size,
                                 rms_norm_eps=rms_norm_eps, rope_theta=rope_theta)

        self._init_runtime(model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                           on_finished, enable_prefix_caching)

    def _init_runtime(self, model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                      on_finished, enable_prefix_caching):
        # 模型已经就位（随机初始化或从目录加载），这里只装运行时：元数据、KV 池、调度器
        device = model.device
        _check_runtime(device, model.attention_backend, model.use_cuda_graph)

        self.model = model
        self.model.eval()
        self.sampler = Sampler()
        self.device = device
        self.attention_backend = model.attention_backend
        self.enable_prefix_caching = enable_prefix_caching

        # 容量按引擎配置一次分配，与首次出现的 batch 大小无关
        if model.attention_backend == "triton":
            model.attention_metadata = AttentionMetadata(
                max_num_seqs=max_num_seqs,
                max_num_query_tokens=max_num_batched_tokens,
                max_blocks_per_request=math.ceil(model.max_seq_len / block_size),
                device=device,
            )

        self.kv_cache_pool = KVCachePool(block_size, num_kv_blocks, model.num_kv_heads,
                                         model.head_dim, device, self.enable_prefix_caching,
                                         num_layers=model.num_layers)
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool)

    @classmethod
    def from_model_dir(cls, model_dir, device=None, attention_backend="torch", use_cuda_graph=False,
                       max_num_seqs=1, max_num_batched_tokens=4, block_size=4, num_kv_blocks=8,
                       on_finished=None, enable_prefix_caching=True):
        # 先从目录构建好模型并加载权重，再装运行时；失败时不会交出半个 Engine
        config = load_model_config(model_dir)
        device = _resolve_device(device)
        _check_runtime(device, attention_backend, use_cuda_graph)

        model = build_model_from_config(config, device, attention_backend,
                                        max_num_batched_tokens, use_cuda_graph)
        load_model_weights(model_dir, model)

        return cls(model=model, max_num_seqs=max_num_seqs,
                   max_num_batched_tokens=max_num_batched_tokens, block_size=block_size,
                   num_kv_blocks=num_kv_blocks, on_finished=on_finished,
                   enable_prefix_caching=enable_prefix_caching)
        
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

    # 演示：随机初始化一个 Engine -> 保存到目录 -> 用另一个随机种子只凭目录重建
    MODEL_DIR = pathlib.Path(__file__).parent / "my_tiny_model"
    CONFIG = dict(max_num_seqs=2, block_size=4, num_kv_blocks=8, vocab_size=100,
                  d_model=32, max_seq_len=32, num_q_heads=4, num_kv_heads=2,
                  num_layers=2, intermediate_size=64)
    REQUESTS = [{"request_id": "A", "prompt_ids": [0, 1, 2, 3, 4], "max_new_tokens": 4},
                {"request_id": "B", "prompt_ids": [3], "max_new_tokens": 2},
                {"request_id": "C", "prompt_ids": [0, 1], "max_new_tokens": 2}]

    def run_engine(engine):
        out = []
        engine.scheduler.on_finished = lambda r: out.append((r["request_id"], tuple(r["output_ids"])))
        for request in REQUESTS:
            engine.add_request(dict(request))
        round_id = 0
        while engine.has_unfinished_requests():
            engine.step()
            round_id += 1
        return out, round_id

    torch.manual_seed(0)
    engine = Engine(**CONFIG, on_finished=on_finished)
    print(f"随机初始化的模型: {engine.model.num_layers} 层, d_model={engine.model.d_model}, "
          f"head_dim={engine.model.head_dim}")
    print(f"保存到 {save_model(engine.model, MODEL_DIR)}")
    print(f"  目录内容: {sorted(p.name for p in MODEL_DIR.iterdir())}")

    # 换一个随机种子，确保不是「靠种子重新随机出同一套权重」
    torch.manual_seed(12345)
    # 只传运行选项：模型结构来自目录里的 config.json，不接受调用方覆盖
    loaded = Engine.from_model_dir(MODEL_DIR, max_num_seqs=CONFIG["max_num_seqs"],
                                   block_size=CONFIG["block_size"], num_kv_blocks=CONFIG["num_kv_blocks"])
    print("从目录重建的 Engine 已就绪")

    before, rounds_before = run_engine(engine)
    after, rounds_after = run_engine(loaded)
    print(f"  原模型  : {rounds_before} 轮 {before}")
    print(f"  加载的  : {rounds_after} 轮 {after}")
    print(f"  输出一致 = {before == after}")
