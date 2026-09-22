"""Engine：模型 + 运行时（元数据、KV 池、调度器）+ 目录加载入口。

加载按「外部目录 -> 内部配置与权重 -> 模型 -> 运行时」四步走，
格式差异全在 formats/ 里消化掉，这里只认内部那套字段名。
"""

import math

import torch

from .attention import AttentionMetadata
from .cache import KVCachePool
from .formats import (CONFIG_NAME as MODEL_CONFIG_NAME,
                      WEIGHTS_NAME as MODEL_WEIGHTS_NAME,
                      native, read_raw_config, read_raw_weights)
from .model import TinyCausalLM
from .sampling import TorchSampler, apply_penalties
from .scheduler import Scheduler

# 自定义格式的公开常量，保持与之前一致
FORMAT_VERSION = native.FORMAT_VERSION
COMPATIBLE_FORMAT_VERSIONS = native.COMPATIBLE_FORMAT_VERSIONS
MODEL_TYPE = native.MODEL_TYPE
MODEL_DTYPE = native.MODEL_DTYPE


def _resolve_device(device):
    return torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_runtime(device, attention_backend, use_cuda_graph, dtype=torch.float32,
                   norm_backend="torch", rope_backend="torch"):
    # 后端、设备与精度的组合校验；随机初始化和从目录加载两条路都走这里
    if attention_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 attention_backend: {attention_backend!r}，可选 'torch' 或 'triton'")
    if attention_backend == "triton" and device.type != "cuda":
        raise ValueError(f"attention_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
    if use_cuda_graph and (device.type != "cuda" or attention_backend != "triton"):
        raise ValueError(f"use_cuda_graph=True 只支持 CUDA + Triton，当前 device={device.type}、"
                         f"attention_backend={attention_backend!r}")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"不支持的 dtype: {dtype}，本实现只支持 torch.float32 或 torch.bfloat16")
    if dtype == torch.bfloat16 and device.type != "cuda":
        raise ValueError(f"dtype=torch.bfloat16 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 float32")
    # norm 后端与 attention 后端是两件独立的事，可以自由组合做 A/B
    if norm_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 norm_backend: {norm_backend!r}，可选 'torch' 或 'triton'")
    if norm_backend == "triton" and device.type != "cuda":
        raise ValueError(f"norm_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
    # rope 后端同样与 attention / norm 独立，可以自由组合做 A/B
    if rope_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 rope_backend: {rope_backend!r}，可选 'torch' 或 'triton'")
    if rope_backend == "triton" and device.type != "cuda":
        raise ValueError(f"rope_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")


def build_model_from_config(config, device, attention_backend, max_num_batched_tokens, use_cuda_graph,
                            dtype=torch.float32, norm_backend="torch", rope_backend="torch"):
    # 按内部配置构造模型；维度合法性由 TinyCausalLM 的校验负责（缺字段、非法维度都会明确报错）
    return TinyCausalLM(
        vocab_size=config["vocab_size"], d_model=config["d_model"], max_seq_len=config["max_seq_len"],
        num_q_heads=config["num_q_heads"], num_kv_heads=config["num_kv_heads"],
        num_layers=config["num_layers"], intermediate_size=config["intermediate_size"],
        rms_norm_eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
        head_dim=config["head_dim"], use_qk_norm=config["use_qk_norm"],
        eos_token_ids=config["eos_token_ids"], dtype=dtype, norm_backend=norm_backend,
        rope_backend=rope_backend,
        device=device, attention_backend=attention_backend,
        max_num_query_tokens=max_num_batched_tokens, use_cuda_graph=use_cuda_graph)


def load_model_config(model_dir):
    # 读目录里的外部配置，翻译成内部字段；支持哪些来源由 formats/ 按 model_type 分派
    adapter, raw, generation = read_raw_config(model_dir)
    return adapter.to_internal_config(raw, generation)


def _load_weights_into(adapter, model_dir, raw, model):
    # 适配器把外部参数名翻成内部参数名，再严格装入已经建在目标设备上的模型。
    # 装入前显式转成模型的运行精度：FP32 文件进 BF16 模型就在这里舍入一次，
    # BF16 文件进 FP32 模型是精确扩宽（不恢复文件里本来就没有的信息）。
    # strict=True：缺参数、多参数、shape 不符都会抛，不会留下混着随机参数的模型
    weights = read_raw_weights(model_dir, adapter.WEIGHT_DTYPES)
    mapped = adapter.to_internal_weights(weights, raw)
    model.load_state_dict({name: t.to(model.dtype) for name, t in mapped.items()}, strict=True)
    return model


def load_model_weights(model_dir, model):
    adapter, raw, _ = read_raw_config(model_dir)
    return _load_weights_into(adapter, model_dir, raw, model)


class Engine:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True, device=None, attention_backend="torch", use_cuda_graph=False, num_q_heads=1, num_kv_heads=1,
                 num_layers=2, intermediate_size=64, rms_norm_eps=1e-6, rope_theta=10000.0,
                 head_dim=None, use_qk_norm=False, eos_token_ids=None, dtype=torch.float32,
                 norm_backend="torch", rope_backend="torch", model=None, *, on_token=None):
        # on_token 是 keyword-only：它放在**所有旧参数之后**，旧的位置参数一个都没挪位。
        # 插在中间会让 Engine(..., None, False) 里的 False 从 enable_prefix_caching
        # 变成 on_token —— Python 按位置配对，不会知道调用者的原意。
        # model 给定时用它，不再按上面的维度参数随机初始化（加载路径走这里）
        # 后端与 Graph 开关也以模型上的为准，避免两边不一致
        if model is None:
            device = _resolve_device(device)
            _check_runtime(device, attention_backend, use_cuda_graph, dtype, norm_backend,
                           rope_backend)
            model = TinyCausalLM(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len,
                                 device=device, attention_backend=attention_backend,
                                 max_num_query_tokens=max_num_batched_tokens,
                                 use_cuda_graph=use_cuda_graph,
                                 num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                                 num_layers=num_layers, intermediate_size=intermediate_size,
                                 rms_norm_eps=rms_norm_eps, rope_theta=rope_theta,
                                 head_dim=head_dim, use_qk_norm=use_qk_norm,
                                 eos_token_ids=eos_token_ids, dtype=dtype, norm_backend=norm_backend,
                                 rope_backend=rope_backend)

        self._init_runtime(model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                           on_finished, enable_prefix_caching, on_token)

    def _init_runtime(self, model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                      on_finished, enable_prefix_caching, on_token=None):
        # 模型已经就位（随机初始化或从目录加载），这里只装运行时：元数据、KV 池、调度器
        device = model.device
        _check_runtime(device, model.attention_backend, model.use_cuda_graph, model.dtype,
                       model.norm_backend, model.rope_backend)

        self.model = model
        self.model.eval()
        # 采样只保留 Torch 一条路径；Triton 采样 kernel 与 beam 已移出主线
        self.sampler = TorchSampler()
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
                                         num_layers=model.num_layers, dtype=model.dtype)
        # 增量输出回调。不传就是 None —— 那时 _sample 里连事件字典都不建
        self.on_token = on_token
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool, eos_token_ids=model.eos_token_ids,
                          vocab_size=model.vocab_size)

    @classmethod
    def from_model_dir(cls, model_dir, device=None, attention_backend="torch", use_cuda_graph=False,
                       max_num_seqs=1, max_num_batched_tokens=4, block_size=4, num_kv_blocks=8,
                       on_finished=None, enable_prefix_caching=True, dtype=torch.float32,
                       norm_backend="torch", rope_backend="torch", *, on_token=None):
        # 只给目录和运行选项，模型结构全部来自目录；失败时不会交出半个 Engine。
        # 外部配置只读一次：适配器选出来之后，配置和权重都交给它翻译。
        adapter, raw, generation = read_raw_config(model_dir)
        config = adapter.to_internal_config(raw, generation)

        device = _resolve_device(device)
        _check_runtime(device, attention_backend, use_cuda_graph, dtype, norm_backend, rope_backend)
        model = build_model_from_config(config, device, attention_backend,
                                        max_num_batched_tokens, use_cuda_graph, dtype, norm_backend,
                                        rope_backend)
        _load_weights_into(adapter, model_dir, raw, model)

        return cls(model=model, max_num_seqs=max_num_seqs,
                   max_num_batched_tokens=max_num_batched_tokens, block_size=block_size,
                   num_kv_blocks=num_kv_blocks, on_finished=on_finished, on_token=on_token,
                   enable_prefix_caching=enable_prefix_caching)

    def add_request(self, request):
        self.scheduler.add_request(request)

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()

    @staticmethod
    def _sample_plan(scheduled_items):
        # 把「哪些请求就绪、各取哪一行」整理一次，同时产出两样东西：
        #   rows  —— 传给模型的打包行号（模型只认行号，不需要认识请求对象）
        #   picked —— 每个行号对应的 scheduled_item，采样结果按同一顺序写回去
        # 行号是「片段末行」：offset 累加完该请求的 token 数后减一。
        rows = []
        picked = []
        offset = 0
        for item in scheduled_items:
            offset += item["num_scheduled_tokens"]
            if item["can_sample"]:
                rows.append(offset - 1)
                picked.append(item)
        return rows, picked

    def _sample(self, logits, picked):
        # logits 已经是「需要采样的那几行」，行序与 picked 一致，不再按原始行号二次索引
        if not picked:
            return None
        if logits.shape[0] != len(picked):
            raise RuntimeError(f"模型返回 {logits.shape[0]} 行 logits，但本轮有 {len(picked)} 个请求需要采样")

        seqs = [item["request"] for item in picked]
        # 每个请求用**自己的**参数和历史。copy=True 是必须的：Graph 的 logits
        # 输出缓冲会被下次 replay 复用，而且 FP32 张量的 .float() 不产生副本。
        rows = [apply_penalties(logits[i].to(torch.float32, copy=True),
                                seqs[i].sampling_params, seqs[i].sampling_state)
                for i in range(len(seqs))]
        tokens = self.sampler.select_batch(
            rows, [s_.sampling_params for s_ in seqs], [s_.sampling_state for s_ in seqs])

        # token id 整批回传，不逐请求 .item()
        output_ids = torch.stack(tokens).tolist()
        notify = self.on_token
        for item, output_id in zip(picked, output_ids):
            seq = item["request"]
            # 通知点就在这儿：token 已经是 Python int、马上要提交进请求状态，
            # 但请求还没被判停、没被回收。顺序沿用 picked —— 也就是本轮采样的顺序，
            # 不等于 running 列表的顺序。
            index = len(seq.output_ids)      # 提交前的长度就是这次的序号（只数生成 token）
            seq.output_ids.append(output_id)
            # 只有真正提交的输出 token 才进惩罚计数；M=0、中间 prefill 块都不经过这里
            seq.sampling_state.note_output_token(output_id)
            if notify is not None:
                # 每次新建一个独立字典，只放 CPU 上的 ID/整数：调用方存起来或改它
                # 都不会碰到引擎状态。不传 SequenceConfig，也不传内部列表。
                notify({"request_id": seq.request_id, "token_id": output_id,
                        "output_index": index})
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

                # 采样计划在 forward 之前就定好：模型只拿行号，Engine 只拿请求
                sample_rows, picked = self._sample_plan(scheduled_items)
                logits = self.model._forward_append(input_ids, num_scheduled_tokens, past_kv,
                                                    self.kv_cache_pool, sample_rows=sample_rows)
                self._sample(logits, picked)

            self.scheduler.post_step()

        return self.scheduler.step_done
