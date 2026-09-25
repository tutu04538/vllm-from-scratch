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
from .speculative import verify_drafts

# 自定义格式的公开常量，保持与之前一致
FORMAT_VERSION = native.FORMAT_VERSION
COMPATIBLE_FORMAT_VERSIONS = native.COMPATIBLE_FORMAT_VERSIONS
MODEL_TYPE = native.MODEL_TYPE
MODEL_DTYPE = native.MODEL_DTYPE


PREEMPTION_MODES = (None, "recompute")
SCHEDULING_POLICIES = ("fcfs", "priority")
SPECULATIVE_MODES = (None, "ngram")


def _check_preemption_mode(preemption_mode):
    # 只认两种模式；写法错误在构造阶段就报出来，不要等到运行时才变成怪行为。
    #
    # 第四十六关起不再限制「recompute 必须配 enable_prefix_caching=False」：
    # 恢复时可以先复用仍然在缓存里的完整块，只重算剩下的历史，两者不再冲突。
    # 所以这里只校验模式本身，组合交给各层按自己的开关工作。
    if preemption_mode not in PREEMPTION_MODES:
        raise ValueError(f"未知的 preemption_mode: {preemption_mode!r}，"
                         f"可选 None（承诺式）或 'recompute'")


def _check_scheduling_policy(scheduling_policy, preemption_mode):
    if scheduling_policy not in SCHEDULING_POLICIES:
        raise ValueError(f"未知的 scheduling_policy: {scheduling_policy!r}，"
                         f"可选 {list(SCHEDULING_POLICIES)}")
    if scheduling_policy == "priority" and preemption_mode != "recompute":
        # 承诺式容量策略不支持强制让位：准入时按最坏情况锁了未来的块，
        # 高优先级请求顶掉别人会把那份承诺作废。明确报错，不静默降级成 fcfs。
        raise ValueError("scheduling_policy='priority' 只支持 preemption_mode='recompute'；"
                         f"当前 preemption_mode={preemption_mode!r}")


def _check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                       max_num_seqs, scheduling_policy, preemption_mode,
                       enable_prefix_caching, attention_backend, use_cuda_graph):
    """投机解码的开关与组合校验。

    第五十二关把 n-gram 模式**明确限制**在最小闭环需要的那组配置上：单请求、
    FCFS、不抢占、无前缀缓存、Torch attention、无 CUDA Graph。不支持的组合
    直接报错，不悄悄退化成普通解码——那会让人以为自己在测投机。
    """
    if speculative_mode not in SPECULATIVE_MODES:
        raise ValueError(f"未知的 speculative_mode: {speculative_mode!r}，"
                         f"可选 None（关）或 'ngram'")
    for name, value in (("num_speculative_tokens", num_speculative_tokens),
                        ("prompt_lookup_n", prompt_lookup_n)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是 >=1 的整数，收到 {value!r}")
    if speculative_mode is None:
        return
    unsupported = [
        (max_num_seqs != 1, f"max_num_seqs 必须是 1（本关只做单请求），当前 {max_num_seqs}"),
        (scheduling_policy != "fcfs", f"scheduling_policy 只能是 'fcfs'，当前 {scheduling_policy!r}"),
        (preemption_mode is not None, f"preemption_mode 必须是 None（本关不做抢占），当前 {preemption_mode!r}"),
        (enable_prefix_caching, "enable_prefix_caching 必须关闭"),
        (attention_backend != "torch", f"attention_backend 只能是 'torch'，当前 {attention_backend!r}"),
        (use_cuda_graph, "use_cuda_graph 必须关闭"),
    ]
    problems = [message for bad, message in unsupported if bad]
    if problems:
        raise ValueError("speculative_mode='ngram' 不支持的组合：" + "；".join(problems))


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
                 norm_backend="torch", rope_backend="torch", model=None, *, on_token=None,
                 preemption_mode=None, scheduling_policy="fcfs",
                 speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2):
        # on_token 是 keyword-only：它放在**所有旧参数之后**，旧的位置参数一个都没挪位。
        # 插在中间会让 Engine(..., None, False) 里的 False 从 enable_prefix_caching
        # 变成 on_token —— Python 按位置配对，不会知道调用者的原意。
        # model 给定时用它，不再按上面的维度参数随机初始化（加载路径走这里）
        # 后端与 Graph 开关也以模型上的为准，避免两边不一致
        _check_preemption_mode(preemption_mode)
        _check_scheduling_policy(scheduling_policy, preemption_mode)
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
                           on_finished, enable_prefix_caching, on_token, preemption_mode,
                           scheduling_policy, speculative_mode, num_speculative_tokens,
                           prompt_lookup_n)

    def _init_runtime(self, model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                      on_finished, enable_prefix_caching, on_token=None, preemption_mode=None,
                      scheduling_policy="fcfs", speculative_mode=None,
                      num_speculative_tokens=2, prompt_lookup_n=2):
        # 模型已经就位（随机初始化或从目录加载），这里只装运行时：元数据、KV 池、调度器
        device = model.device
        _check_runtime(device, model.attention_backend, model.use_cuda_graph, model.dtype,
                       model.norm_backend, model.rope_backend)
        # 组合校验放在这里：模型已经就位，attention 后端与 Graph 开关都以它为准
        _check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                           max_num_seqs, scheduling_policy, preemption_mode,
                           enable_prefix_caching, model.attention_backend, model.use_cuda_graph)

        self.model = model
        self.model.eval()
        # 采样只保留 Torch 一条路径；Triton 采样 kernel 与 beam 已移出主线
        self.sampler = TorchSampler()
        self.device = device
        self.attention_backend = model.attention_backend
        self.enable_prefix_caching = enable_prefix_caching
        self.preemption_mode = preemption_mode
        self.scheduling_policy = scheduling_policy
        self.speculative_mode = speculative_mode

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
                                         num_layers=model.num_layers, dtype=model.dtype,
                                         # recompute 模式允许超卖：准入不锁未来块，
                                         # 不够时由 Scheduler 选犠牲者腾地方
                                         over_subscribe=(preemption_mode == "recompute"))
        # 增量输出回调。不传就是 None —— 那时 _sample 里连事件字典都不建
        self.on_token = on_token
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool, eos_token_ids=model.eos_token_ids,
                          preemption_mode=preemption_mode,
                          scheduling_policy=scheduling_policy,
                          speculative_mode=speculative_mode,
                          num_speculative_tokens=num_speculative_tokens,
                          prompt_lookup_n=prompt_lookup_n,
                          max_seq_len=model.max_seq_len,
                          vocab_size=model.vocab_size)

    @classmethod
    def from_model_dir(cls, model_dir, device=None, attention_backend="torch", use_cuda_graph=False,
                       max_num_seqs=1, max_num_batched_tokens=4, block_size=4, num_kv_blocks=8,
                       on_finished=None, enable_prefix_caching=True, dtype=torch.float32,
                       norm_backend="torch", rope_backend="torch", *, on_token=None,
                       preemption_mode=None, scheduling_policy="fcfs",
                       speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2):
        # 只给目录和运行选项，模型结构全部来自目录；失败时不会交出半个 Engine。
        # 外部配置只读一次：适配器选出来之后，配置和权重都交给它翻译。
        _check_preemption_mode(preemption_mode)
        _check_scheduling_policy(scheduling_policy, preemption_mode)
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
                   enable_prefix_caching=enable_prefix_caching,
                   preemption_mode=preemption_mode, scheduling_policy=scheduling_policy,
                   speculative_mode=speculative_mode,
                   num_speculative_tokens=num_speculative_tokens,
                   prompt_lookup_n=prompt_lookup_n)

    def add_request(self, request):
        self.scheduler.add_request(request)

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()

    @staticmethod
    def _sample_plan(scheduled_items):
        # 把「哪些请求就绪、各取哪几行」整理一次，同时产出两样东西：
        #   rows   —— 传给模型的打包行号（模型只认行号，不需要认识请求对象）
        #   picked —— 每个请求对应的 scheduled_item，采样结果按同一顺序写回去
        # 普通项只取「片段末行」；投机项要整段输入的行——前 K 行验证草稿，
        # 最后一行给 bonus。本关投机限制单请求，所以不存在与其他请求交错的行映射。
        rows = []
        picked = []
        offset = 0
        for item in scheduled_items:
            offset += item["num_scheduled_tokens"]
            if not item["can_sample"]:
                continue
            if item["draft_ids"]:
                item["num_sample_rows"] = item["num_scheduled_tokens"]
                rows.extend(range(offset - item["num_scheduled_tokens"], offset))
            else:
                item["num_sample_rows"] = 1
                rows.append(offset - 1)
            picked.append(item)
        return rows, picked

    def _sample(self, logits, picked):
        # logits 已经是「需要采样的那几行」，行序与 picked 一致，不再按原始行号二次索引
        if not picked:
            return None
        expected = sum(item["num_sample_rows"] for item in picked)
        if logits.shape[0] != expected:
            raise RuntimeError(f"模型返回 {logits.shape[0]} 行 logits，但本轮需要 {expected} 行")

        notify = self.on_token
        # 每个请求的 logits 在自己的偏移处：普通项 1 行，投机项 K+1 行
        offsets, cursor = [], 0
        for item in picked:
            offsets.append(cursor)
            cursor += item["num_sample_rows"]

        plain = [i for i, item in enumerate(picked) if not item["draft_ids"]]
        if plain:
            seqs = [picked[i]["request"] for i in plain]
            # 每个请求用**自己的**参数和历史。copy=True 是必须的：Graph 的 logits
            # 输出缓冲会被下次 replay 复用，而且 FP32 张量的 .float() 不产生副本。
            rows = [apply_penalties(logits[offsets[i]].to(torch.float32, copy=True),
                                    seqs[k].sampling_params, seqs[k].sampling_state)
                    for k, i in enumerate(plain)]
            tokens = self.sampler.select_batch(
                rows, [s_.sampling_params for s_ in seqs], [s_.sampling_state for s_ in seqs])
            # token id 整批回传，不逐请求 .item()
            output_ids = torch.stack(tokens).tolist()
            for k, i in enumerate(plain):
                self._commit_tokens(picked[i]["request"], [output_ids[k]], notify)

        for i, item in enumerate(picked):
            if item["draft_ids"]:
                self._commit_drafts(item, logits[offsets[i]:offsets[i] + item["num_sample_rows"]],
                                    notify)
        return None

    def _commit_tokens(self, seq, token_ids, notify):
        """把 token 提交进请求状态：已提交历史、惩罚计数、回调**同进同退**。

        这里是唯一的提交点，普通路径与投机路径共用。只有真正被接受的 token 才
        会走到这儿——草稿在验证通过之前一个都不进来。

        回调顺序沿用 `picked` —— 也就是本轮采样的顺序，不等于 `running` 列表的顺序。
        """
        for token_id in token_ids:
            # 通知点就在这儿：token 已经是 Python int、马上要提交进请求状态，
            # 但请求还没被判停、没被回收。
            index = len(seq.output_ids)      # 提交前的长度就是这次的序号（只数生成 token）
            seq.append_output_ids(token_id)   # 唯一写入点：同步更新已提交历史与输出
            # 只有真正提交的输出 token 才进惩罚计数；M=0、中间 prefill 块都不经过这里
            seq.sampling_state.note_output_token(token_id)
            if notify is not None:
                # 每次新建一个独立字典，只放 CPU 上的 ID/整数：调用方存起来或改它
                # 都不会碰到引擎状态。不传 SequenceConfig，也不传内部列表。
                notify({"request_id": seq.request_id, "token_id": token_id,
                        "output_index": index})

    def _commit_drafts(self, item, rows, notify):
        """投机项：用目标模型一次 forward 的 K+1 行验证草稿，回滚 KV，再逐枚提交。

        顺序不能换：**回滚必须在 `scheduler.post_step()` 之前**。被拒绝草稿的 KV
        已经随本轮输入写进了物理块，先把 `cache.length` 退回去，post_step 里的
        发布与判停读到的才是真实进度。
        """
        seq = item["request"]
        # 本关把投机限制在贪心且无惩罚项（构造时与 add_request 时都校验过），
        # 所以验证就是逐行 argmax，不必再走 apply_penalties。批量算一次，
        # 并列时取下标最小的那个——与 TorchSampler 的贪心路径一致。
        greedy_ids = torch.argmax(rows.to(torch.float32), dim=-1).tolist()
        remaining_outputs = seq.max_new_tokens - len(seq.output_ids)
        result = verify_drafts(item["draft_ids"], greedy_ids, self.model.eos_token_ids,
                               remaining_outputs)

        # 1) 先回滚：本轮输入 [x, d0..] 里只有 x 和「被接受且还要当下一轮输入」的草稿要留
        self.kv_cache_pool.truncate(seq, item["start_cache_length"] + result.kept_inputs)
        # 2) 再提交：逐枚走和普通路径同一个入口，事件序号自然连续
        self._commit_tokens(seq, result.committed_ids, notify)

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
