"""模型本体：RoPE、RMSNorm、一层 decoder、整个 TinyCausalLM。

只负责「给定一批 token 和缓存，算出 logits」，不知道请求和调度的存在。
"""

import gc

import torch
import torch.nn.functional as F
from torch import nn

from .attention import paged_attention, tiled_paged_attention
from .cache import CacheConfig, KVCachePool
from .norm import rms_norm as _fused_rms_norm
from .rope import rope

# 没声明停止规则时的默认值：旧随机模型/旧目录沿用写死的 4
DEFAULT_EOS_TOKEN_IDS = (4,)


def normalize_eos_ids(value, vocab_size, where="模型配置"):
    """停止 token 允许是一个整数，或一个非空整数列表；统一成升序 tuple 并检查范围。

    停止规则是生成控制信息，不参与 attention 计算。
    """
    ids = value if isinstance(value, (list, tuple)) else [value]
    if not ids:
        raise ValueError(f"{where} 的 eos_token_id 是空列表，停止规则不能为空")

    normalized = []
    for token_id in ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{where} 的 eos_token_id 必须是整数，收到 {token_id!r}")
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"{where} 的 eos_token_id={token_id} 超出词表范围 [0, {vocab_size})")
        normalized.append(token_id)
    return tuple(sorted(set(normalized)))


def _rotate_half(x):
    # 前后半段配对：(0, D/2), (1, D/2+1), ...
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    # 固定频率的 RoPE：初始化时算好 cos/sin 表，按逻辑位置查表
    # 角度表**恒为 FP32**，不跟着模型精度走：这个模块只搬设备，不做 dtype 转换
    #
    # 两条后端，数学完全一样，只是执行方式不同：
    #   "torch"  —— 查表 + 拼接 + 配对旋转，一串小算子；保留作 CPU 路径与参考
    #   "triton" —— 一个融合 kernel，中间量不落显存

    def __init__(self, head_dim, max_seq_len, theta, backend="torch"):
        super().__init__()
        if backend not in ("torch", "triton"):
            raise ValueError(f"未知的 rope_backend: {backend!r}，可选 'torch' 或 'triton'")
        self.backend = backend
        inv_freq = torch.pow(float(theta),
                             -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        # 查表数据，不是可学习参数；persistent=False 让它不进 state_dict
        self.register_buffer("cos_table", freqs.cos(), persistent=False)
        self.register_buffer("sin_table", freqs.sin(), persistent=False)

    def forward(self, x, positions):
        # x: [N, heads, head_dim]；positions: [N] 逻辑位置（不是打包行号）
        if self.backend == "triton":
            return rope(x, positions, self.cos_table, self.sin_table)
        # .float() 是显式的：角度和旋转都在 FP32 下算（表本身已是 FP32 时是空操作）
        cos = self.cos_table[positions].float()
        sin = self.sin_table[positions].float()
        cos = torch.cat((cos, cos), dim=-1)[:, None, :]   # [N,1,head_dim]，按 head 广播
        sin = torch.cat((sin, sin), dim=-1)[:, None, :]
        rotated = x.float() * cos + _rotate_half(x).float() * sin
        # 最后**必须**回到输入精度：FP32 的 cos/sin 会把乘法结果提升成 FP32，
        # 不转回去的话，整个网络会从这一行开始一路变成 FP32
        return rotated.to(x.dtype)


class RMSNorm(nn.Module):
    # RMSNorm(x) = x / sqrt(mean(x^2, 最后一维) + eps) * weight，沿最后一维做，不减均值
    # size 是最后一维的长度：主 norm 传 d_model，Q/K norm 传 head_dim
    #
    # 两条后端，数学完全一样，只是执行方式不同：
    #   "torch"  —— 一串小算子，保留作参考与 CPU 路径
    #   "triton" —— 一个融合 kernel，中间量不落显存

    def __init__(self, size, eps, backend="torch"):
        super().__init__()
        self.eps = eps
        self.backend = backend
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        if self.backend == "triton":
            return _fused_rms_norm(x, self.weight, self.eps)
        # 平方、均值、rsqrt 和权重缩放都在 FP32 下算，最后一次舍入回输入精度。
        # FP32 输入时 .float() 是空操作，旧结果逐位不变。
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return ((x_fp32 * torch.rsqrt(variance + self.eps)) * self.weight.float()).to(x.dtype)


class DecoderLayer(nn.Module):
    # h = x + Attention(RMSNorm_1(x))
    # y = h + MLP(RMSNorm_2(h))
    # Attention 含 QKV 投影、GQA、分页 KV、head 拼接和 o_proj，不含 lm_head；
    # 它需要缓存和元数据，放在模型里做，这一层只负责 MLP 那半段。

    def __init__(self, d_model, num_q_heads, num_kv_heads, head_dim, intermediate_size, eps,
                 use_qk_norm=False, norm_backend="torch"):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps, norm_backend)
        self.norm2 = RMSNorm(d_model, eps, norm_backend)

        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        # o_proj 的输入是「拼接后的 attention 输出」，宽度是 Q heads × head_dim，不是 d_model
        self.o_proj = nn.Linear(num_q_heads * head_dim, d_model, bias=False)

        if use_qk_norm:
            # 每个 head 单独做，权重形状是 [head_dim]（不是 [num_heads, head_dim]）
            self.q_norm = RMSNorm(head_dim, eps, norm_backend)
            self.k_norm = RMSNorm(head_dim, eps, norm_backend)

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
                 rms_norm_eps=1e-6, rope_theta=10000.0, head_dim=None, use_qk_norm=False,
                 eos_token_ids=None, dtype=torch.float32, norm_backend="torch",
                 rope_backend="torch"):
        super().__init__()

        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_backend = attention_backend
        self.attention_metadata = attention_metadata
        self.max_num_query_tokens = max_num_query_tokens
        self.use_cuda_graph = use_cuda_graph
        # 图的 key 是 (输入行数 N, 输出行数 M, 是否按行号挑选)。
        # 同样 N 可能有不同的 M：本轮谁需要采样是会变的，两者的计算形状都变了。
        self.graphs = {}          # (N, M, 挑选) -> CUDAGraph
        self.graph_outputs = {}   # 同 key -> 该图的 logits 输出（存储会被后续 replay 复用）

        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"不支持的 dtype: {dtype}，本实现只支持 torch.float32 或 torch.bfloat16")
        self.dtype = dtype
        if norm_backend not in ("torch", "triton"):
            raise ValueError(f"未知的 norm_backend: {norm_backend!r}，可选 'torch' 或 'triton'")
        if rope_backend not in ("torch", "triton"):
            raise ValueError(f"未知的 rope_backend: {rope_backend!r}，可选 'torch' 或 'triton'")
        self.rope_backend = rope_backend
        self.norm_backend = norm_backend

        if max_num_query_tokens is not None:
            # 固定容量输入缓冲：地址不变，每轮只改内容。
            # 这三个是整数索引，任何时候都不跟着模型精度走
            self.input_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)
            self.position_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)
            self.slot_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)
            # 需要采样的打包行号。同样是固定地址，每轮只改内容——Graph replay 时
            # 必须能换一批行号，不能一直用首次捕获的那组
            self.sample_buffer = torch.zeros(max_num_query_tokens, dtype=torch.long, device=self.device)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 结构校验：随机初始化和从目录加载都走这里
        if vocab_size <= 0:
            raise ValueError(f"vocab_size 必须为正，收到 {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model 必须为正，收到 {d_model}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len 必须为正，收到 {max_seq_len}")

        # head 配置：默认单头，旧行为不变
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError(f"头数必须为正，收到 num_q_heads={num_q_heads}、num_kv_heads={num_kv_heads}")
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
        # head_dim 显式给出时用它，否则按 d_model // num_q_heads 推导（这条才要求整除）
        if head_dim is None:
            if d_model % num_q_heads != 0:
                raise ValueError(f"d_model={d_model} 不能被 num_q_heads={num_q_heads} 整除；"
                                 f"要么让它整除，要么显式传 head_dim")
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
        self.use_qk_norm = use_qk_norm
        # 拼接 attention 各 head 之后的宽度；o_proj 的输入维度就是它
        self.attn_width = num_q_heads * head_dim

        # 停止规则：不参与 attention 计算，只是跟着模型一起走的生成控制信息。
        # 放在模型上是为了让 model_config()/save_model() 能把它一起存下来。
        self.eos_token_ids = normalize_eos_ids(
            DEFAULT_EOS_TOKEN_IDS if eos_token_ids is None else eos_token_ids, vocab_size)

        # 参数和主要激活用运行精度：BF16 模式下这里的每个权重都是 BF16
        self.token_embedding = nn.Embedding(vocab_size, d_model).to(device=self.device, dtype=dtype)
        # 位置信息不再用 embedding 加进 hidden，而是每层旋转 Q/K。
        # 注意只搬设备、不转 dtype：角度表保持 FP32
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta,
                                      rope_backend).to(self.device)

        # 每层有自己独立的 Q/K/V/O、MLP 与两个 norm
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_q_heads, num_kv_heads, head_dim, intermediate_size,
                         rms_norm_eps, use_qk_norm, norm_backend)
            for _ in range(num_layers)
        ]).to(device=self.device, dtype=dtype)

        self.norm = RMSNorm(d_model, rms_norm_eps, norm_backend).to(device=self.device, dtype=dtype)   # 最终 norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device=self.device, dtype=dtype)

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
            "head_dim": self.head_dim,
            "use_qk_norm": self.use_qk_norm,
        }

    def _prepare_inputs(self, input_ids, num_scheduled_tokens, past_kv, kv_cache_pool, sample_rows=None):
        # 图外：算地址、推进 Python 状态、把本轮输入填进固定缓冲
        # input_ids: (N,) 只有真实 token；past_kv 与 num_scheduled_tokens 同序
        # sample_rows: 需要采样的是哪几行（打包行号）。None 表示「全部行都算 logits」，
        #              保留给数值对照；正常 Engine 路径一律传真正的行号列表。

        num_tokens = sum(num_scheduled_tokens)

        # 先只做校验，全部通过之后才建 Tensor、推进状态
        if num_tokens > self.max_num_query_tokens:
            raise ValueError(f"本轮 query 数 {num_tokens} 超过固定输入缓冲容量 "
                             f"{self.max_num_query_tokens}（由 max_num_batched_tokens 决定）")
        if sample_rows is not None:
            if len(sample_rows) > num_tokens:
                raise ValueError(f"采样行数 {len(sample_rows)} 超过本轮 query 数 {num_tokens}")
            for row in sample_rows:
                if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < num_tokens:
                    raise ValueError(f"采样行号 {row!r} 不在 [0, {num_tokens}) 范围内")
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

        # 输出侧的行数在这里定下来：M=0 也是合法的一轮（谁都不需要采样）
        # 走哪条 attention 路径。只有 BF16 且本轮**每条请求都追加多个 token** 才分块：
        # 纯 decode 每请求只有 1 行，分块没有可复用的 query；混合 prefill/decode
        # 整批回退，不强求它提速。这个决定必须进图缓存键——同样的 N 可能是
        # 「8 条请求各 512 行」也可能是「8 条请求各 1 行」，两者的图和 grid 都不同。
        counts = [c for c in num_scheduled_tokens if c > 0]
        self.use_tiled = (self.dtype == torch.bfloat16 and len(counts) > 0
                          and all(count > 1 for count in counts))

        self.sample_rows_given = sample_rows is not None
        self.num_samples = num_tokens if sample_rows is None else len(sample_rows)
        if self.sample_rows_given and self.num_samples:
            self.sample_buffer[:self.num_samples].copy_(torch.tensor(sample_rows, dtype=torch.long))

        self.input_buffer[:num_tokens].copy_(input_ids)
        self.position_buffer[:num_tokens].copy_(position_ids)
        self.slot_buffer[:num_tokens].copy_(slot_mapping)

        if self.attention_metadata is not None:
            # seq_lens 用写完 KV 之后的长度；只有走分块路径才构建 tile 表，
            # 省掉纯 decode 每步的一遍额外遍历
            self.attention_metadata.fill(past_kv, num_scheduled_tokens,
                                         build_tiles=self.use_tiled)
            self.attention_metadata.upload()

        return num_tokens

    def _forward_append(self, input_ids: torch.Tensor, num_scheduled_tokens: list[int],
                        past_kv: list[CacheConfig], kv_cache_pool: KVCachePool, sample_rows=None):
        # 返回 logits。sample_rows 给出需要采样的是哪几行（打包行号），返回 (M, vocab_size)；
        # sample_rows=None 时全部行都算，返回 (N, vocab_size)——那是数值对照用的调试路径。
        #
        # attention / MLP / KV 仍然处理全部 N 个 token，省掉的只是最后那段
        # 最终 norm + lm_head：中间 chunk 的 token 后面还要靠它们的 KV。

        num_tokens = self._prepare_inputs(input_ids, num_scheduled_tokens, past_kv, kv_cache_pool,
                                          sample_rows)

        if self.attention_backend != "triton":
            return self._torch_forward(num_tokens, num_scheduled_tokens, past_kv, kv_cache_pool)

        if not self.use_cuda_graph:
            return self.gpu_forward(num_tokens, kv_cache_pool)

        # 键里带上走的是哪条 attention 路径：同一个 N 在不同请求划分下可能走不同的
        # kernel、grid 也不同，共用一个图会重放错的计算
        graph_key = (num_tokens, self.num_samples, self.sample_rows_given, self.use_tiled)
        graph = self.graphs.get(graph_key)
        if graph is None:
            graph = self._capture_graph(graph_key, kv_cache_pool)
        graph.replay()
        return self.graph_outputs[graph_key]

    def _input_embeds(self, num_tokens):
        # 整个模型只有一次 token embedding，位置信息由每层的 RoPE 提供
        return self.token_embedding(self.input_buffer[:num_tokens])

    def _layer_qkv(self, layer, hidden, num_tokens, positions):
        # 某一层的 QKV 投影 -> 拆成 head -> 按逻辑位置旋转 Q/K，V 不旋转
        q = layer.q_proj(hidden).view(num_tokens, self.num_q_heads, self.head_dim)
        k = layer.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = layer.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        if self.use_qk_norm:
            # 沿最后一维逐 head 归一化；V 不做
            q = layer.q_norm(q)
            k = layer.k_norm(k)
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
        if self.use_tiled:
            # 分块路径：一个 program 处理同一请求的一小组 query，K/V 在这组 query 间复用。
            # 归属与范围全部读 GPU 元数据，图内不做任何 Python 侧切分。
            out = tiled_paged_attention(
                q,
                kv_cache_pool.k_cache[layer_idx],
                kv_cache_pool.v_cache[layer_idx],
                metadata.gpu_block_tables,
                metadata.gpu_seq_lens,
                metadata.gpu_query_pos,
                metadata.gpu_tile_req,
                metadata.gpu_tile_q_start,
                metadata.gpu_tile_q_count,
                metadata.gpu_num_tiles,
                metadata.tile_capacity,
                self.group_size,
            )
        else:
            # 逐行路径：纯 decode、FP32、混合批次都走这里。
            # kernel 由 grid=(N, num_q_heads) 和 seq_len 驱动，只读有效区域
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
        return layer.o_proj(out.reshape(num_tokens, self.attn_width))

    def _final_logits(self, hidden):
        # 最后一层的输出投影。只对需要的行做最终 norm 与 lm_head：
        # 挑行在前、norm/lm_head 在后，省掉的是这两个算子的行数（N -> M），
        # 不是「先算完再切片」——后者白算的那些行照样要花算力和显存。
        if not self.sample_rows_given:
            return self.lm_head(self.norm(hidden))
        if self.num_samples == 0:
            # 没人需要采样：norm 和 lm_head 都不跑，返回空的 [0, vocab_size]
            return hidden.new_empty((0, self.vocab_size))
        picked = hidden.index_select(0, self.sample_buffer[:self.num_samples])
        return self.lm_head(self.norm(picked))

    def gpu_forward(self, num_tokens, kv_cache_pool):
        # 图内：只做 GPU 运算，读写固定地址的缓冲区
        hidden = self._input_embeds(num_tokens)
        positions = self.position_buffer[:num_tokens]   # 各层共用同一组逻辑位置

        for layer_idx, layer in enumerate(self.layers):
            h = hidden + self._attention_out(layer, layer_idx, hidden, num_tokens, positions, kv_cache_pool)
            hidden = layer(h)   # h + MLP(RMSNorm_2(h))

        return self._final_logits(hidden)

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

            h = hidden + layer.o_proj(heads_out.reshape(num_tokens, self.attn_width))
            hidden = layer(h)

        return self._final_logits(hidden)

    def _capture_graph(self, graph_key, kv_cache_pool):
        # 捕获期间不能有 Python 终结器跑：旧 CUDAGraph 被回收时会调用 cuGraphExecDestroy，
        # 那是捕获期禁止的 API，会直接把这次捕获作废（表现为 allocator 报
        # cudaStreamCaptureStatusInvalidated）。先把待回收的对象清掉再进捕获。
        gc.collect()

        # 预热用同一批真实输入反复算：只往本轮该写的 slot 写同样的 KV，Python 状态不动
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.gpu_forward(graph_key[0], kv_cache_pool)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.gpu_forward(graph_key[0], kv_cache_pool)

        self.graphs[graph_key] = graph
        self.graph_outputs[graph_key] = output
        return graph

    def block_attention(self, q, cache: CacheConfig, kv_cache_pool: KVCachePool, query_positions, layer_idx=0):
        # 直接按逻辑块读 KV，用 online softmax 把各块结果合并成全局 attention
        # q: (Q, num_q_heads, head_dim)，Q 是本请求本轮的 query 数；返回同形状

        device = q.device
        block_size = kv_cache_pool.block_size
        num_blocks = -(-cache.length // block_size)  # ceil
        num_queries = q.shape[0]
        out = torch.empty_like(q)

        # 精度边界：Q/K/V 从缓存里读出来是运行精度（可能是 BF16），
        # 点积、max、exp、分母和加权和全部在 FP32 下累计，最后再回到运行精度。
        # 必须**先转 FP32 再乘**：先按 BF16 乘、得到舍入后的结果再转 FP32，不是同一件事。
        fp32 = torch.float32
        q_fp32 = q.float()

        for q_head in range(self.num_q_heads):
            kv_head = q_head // self.group_size          # 连续分组
            q_head_vec = q_fp32[:, q_head, :]            # (Q, head_dim)，FP32

            # 每个 query 的累计状态：已见最大分数、未归一化权重和、加权 value 和
            m = torch.full((num_queries, 1), float('-inf'), device=device, dtype=fp32)
            z = torch.zeros((num_queries, 1), device=device, dtype=fp32)
            u = torch.zeros((num_queries, self.head_dim), device=device, dtype=fp32)

            for logical_block in range(num_blocks):
                block_start = logical_block * block_size
                count = min(block_size, cache.length - block_start)  # 尾块只读有效部分
                k_block, v_block = kv_cache_pool.block_view(cache, logical_block, count, layer_idx)
                k_head = k_block[:, kv_head, :].float()  # (count, head_dim)
                v_head = v_block[:, kv_head, :].float()

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

            out[:, q_head, :] = (u / z).to(out.dtype)   # 恢复运行精度

        return out


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
