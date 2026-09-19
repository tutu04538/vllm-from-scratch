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
from .sampler import Sampler
from .scheduler import Scheduler

# 自定义格式的公开常量，保持与之前一致
FORMAT_VERSION = native.FORMAT_VERSION
COMPATIBLE_FORMAT_VERSIONS = native.COMPATIBLE_FORMAT_VERSIONS
MODEL_TYPE = native.MODEL_TYPE
MODEL_DTYPE = native.MODEL_DTYPE


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


def build_model_from_config(config, device, attention_backend, max_num_batched_tokens, use_cuda_graph):
    # 按内部配置构造模型；维度合法性由 TinyCausalLM 的校验负责（缺字段、非法维度都会明确报错）
    return TinyCausalLM(
        vocab_size=config["vocab_size"], d_model=config["d_model"], max_seq_len=config["max_seq_len"],
        num_q_heads=config["num_q_heads"], num_kv_heads=config["num_kv_heads"],
        num_layers=config["num_layers"], intermediate_size=config["intermediate_size"],
        rms_norm_eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
        head_dim=config["head_dim"], use_qk_norm=config["use_qk_norm"],
        device=device, attention_backend=attention_backend,
        max_num_query_tokens=max_num_batched_tokens, use_cuda_graph=use_cuda_graph)


def load_model_config(model_dir):
    # 读目录里的外部配置，翻译成内部字段；支持哪些来源由 formats/ 按 model_type 分派
    adapter, raw = read_raw_config(model_dir)
    return adapter.to_internal_config(raw)


def _load_weights_into(adapter, model_dir, model):
    # 适配器把外部参数名翻成内部参数名，再严格装入已经建在目标设备上的模型
    # strict=True：缺参数、多参数、shape 不符都会抛，不会留下混着随机参数的模型
    weights = adapter.to_internal_weights(read_raw_weights(model_dir))
    model.load_state_dict(weights, strict=True)
    return model


def load_model_weights(model_dir, model):
    adapter, _ = read_raw_config(model_dir)
    return _load_weights_into(adapter, model_dir, model)


class Engine:

    def __init__(self, max_num_seqs=1, max_num_batched_tokens=4,block_size=4, num_kv_blocks=8, vocab_size=5, d_model=8, max_seq_len=32, on_finished=None, enable_prefix_caching=True, device=None, attention_backend="torch", use_cuda_graph=False, num_q_heads=1, num_kv_heads=1,
                 num_layers=2, intermediate_size=64, rms_norm_eps=1e-6, rope_theta=10000.0,
                 head_dim=None, use_qk_norm=False, model=None):
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
                                 rms_norm_eps=rms_norm_eps, rope_theta=rope_theta,
                                 head_dim=head_dim, use_qk_norm=use_qk_norm)

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
        # 只给目录和运行选项，模型结构全部来自目录；失败时不会交出半个 Engine。
        # 外部配置只读一次：适配器选出来之后，配置和权重都交给它翻译。
        adapter, raw = read_raw_config(model_dir)
        config = adapter.to_internal_config(raw)

        device = _resolve_device(device)
        _check_runtime(device, attention_backend, use_cuda_graph)
        model = build_model_from_config(config, device, attention_backend,
                                        max_num_batched_tokens, use_cuda_graph)
        _load_weights_into(adapter, model_dir, model)

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
