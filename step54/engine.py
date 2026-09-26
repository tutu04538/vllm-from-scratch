"""Engine：模型 + 运行时（元数据、KV 池、调度器）+ 目录加载入口。

加载按「外部目录 -> 内部配置与权重 -> 模型 -> 运行时」四步走，
格式差异全在 formats/ 里消化掉，这里只认内部那套字段名。
"""

import math
from types import SimpleNamespace

import torch

from .attention import AttentionMetadata
from .cache import KVCachePool
from .formats import (CONFIG_NAME as MODEL_CONFIG_NAME,
                      WEIGHTS_NAME as MODEL_WEIGHTS_NAME,
                      native, read_raw_config, read_raw_weights)
from .model import TinyCausalLM
from .sampling import TorchSampler, apply_penalties, row_distribution
from .scheduler import Scheduler
from .speculative import verify_drafts, verify_drafts_random

# 自定义格式的公开常量，保持与之前一致
FORMAT_VERSION = native.FORMAT_VERSION
COMPATIBLE_FORMAT_VERSIONS = native.COMPATIBLE_FORMAT_VERSIONS
MODEL_TYPE = native.MODEL_TYPE
MODEL_DTYPE = native.MODEL_DTYPE


SCHEDULING_POLICIES = ("fcfs", "priority")
SPECULATIVE_MODES = (None, "ngram")


def _check_scheduling_policy(scheduling_policy):
    if scheduling_policy not in SCHEDULING_POLICIES:
        raise ValueError(f"未知的 scheduling_policy: {scheduling_policy!r}，"
                         f"可选 {list(SCHEDULING_POLICIES)}")


def _check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                       scheduling_policy, enable_prefix_caching,
                       attention_backend, use_cuda_graph):
    """投机解码的开关与组合校验。

    n-gram 模式**明确限制**在一组配置上：FCFS、无前缀缓存、Torch attention、
    无 CUDA Graph。不支持的组合直接报错，不悄悄退化成普通解码——那会让人以为
    自己在测投机。

    第五十四关放开了 `max_num_seqs=1`：一个 batch 里可以同时有多个投机请求、
    普通 decode 和中间 prefill。随之失去的是第五十二关那条「不会发生抢占」的
    推论——running 里不止一条请求，`_make_room()` 就有犠牲者可选了。这不影响
    正确性（`_preempt()` 发生在 forward **之前**，被抢占的项整个作废、走不到
    采样与回调），但「投机下不做抢占」不再是无条件成立的，见
    docs/step54_batched_speculative.md。
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
        (scheduling_policy != "fcfs", f"scheduling_policy 只能是 'fcfs'，当前 {scheduling_policy!r}"),
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
                 scheduling_policy="fcfs",
                 speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2):
        # on_token 是 keyword-only：它放在**所有旧参数之后**，旧的位置参数一个都没挪位。
        # 插在中间会让 Engine(..., None, False) 里的 False 从 enable_prefix_caching
        # 变成 on_token —— Python 按位置配对，不会知道调用者的原意。
        # model 给定时用它，不再按上面的维度参数随机初始化（加载路径走这里）
        # 后端与 Graph 开关也以模型上的为准，避免两边不一致
        _check_scheduling_policy(scheduling_policy)
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
                           on_finished, enable_prefix_caching, on_token,
                           scheduling_policy, speculative_mode, num_speculative_tokens,
                           prompt_lookup_n)

    def _init_runtime(self, model, max_num_seqs, max_num_batched_tokens, block_size, num_kv_blocks,
                      on_finished, enable_prefix_caching, on_token=None,
                      scheduling_policy="fcfs", speculative_mode=None,
                      num_speculative_tokens=2, prompt_lookup_n=2):
        # 模型已经就位（随机初始化或从目录加载），这里只装运行时：元数据、KV 池、调度器
        device = model.device
        _check_runtime(device, model.attention_backend, model.use_cuda_graph, model.dtype,
                       model.norm_backend, model.rope_backend)
        # 组合校验放在这里：模型已经就位，attention 后端与 Graph 开关都以它为准
        _check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                           scheduling_policy, enable_prefix_caching,
                           model.attention_backend, model.use_cuda_graph)

        self.model = model
        self.model.eval()
        # 采样只保留 Torch 一条路径；Triton 采样 kernel 与 beam 已移出主线
        self.sampler = TorchSampler()
        self.device = device
        self.attention_backend = model.attention_backend
        self.enable_prefix_caching = enable_prefix_caching
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
                                         num_layers=model.num_layers, dtype=model.dtype)
        # 增量输出回调。不传就是 None —— 那时 _sample 里连事件字典都不建
        self.on_token = on_token
        self.scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, block_size=block_size, enable_prefix_caching=enable_prefix_caching, on_finished=on_finished, kv_cache_pool=self.kv_cache_pool, eos_token_ids=model.eos_token_ids,
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
                       scheduling_policy="fcfs",
                       speculative_mode=None, num_speculative_tokens=2, prompt_lookup_n=2):
        # 只给目录和运行选项，模型结构全部来自目录；失败时不会交出半个 Engine。
        # 外部配置只读一次：适配器选出来之后，配置和权重都交给它翻译。
        _check_scheduling_policy(scheduling_policy)
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
                   scheduling_policy=scheduling_policy,
                   speculative_mode=speculative_mode,
                   num_speculative_tokens=num_speculative_tokens,
                   prompt_lookup_n=prompt_lookup_n)

    def add_request(self, request):
        self.scheduler.add_request(request)

    def has_unfinished_requests(self):
        return self.scheduler.has_unfinished_requests()

    @staticmethod
    def _sample_plan(scheduled_items):
        # 把「哪些请求就绪、各取哪几行」整理一次，产出两样东西：
        #   rows   —— 传给模型的**原始输入行号**（模型只认行号，不需要认识请求对象）
        #   picked —— 要采样的那些 item，采样结果按同一顺序写回去
        #
        # 这里有两个坐标系，不能混用：
        #   * 原始输入行号：本轮全部输入 token 拼成一维后的下标（中间 prefill 也占位置）
        #   * 筛选后偏移  ：模型返回的 logits 只包含选中的行，行内下标从 0 开始
        # 所以每个 picked 项都记下自己的 `num_sample_rows` 与 `sample_offset`，
        # 后者在下面的 `_sample()` 里直接用来切 Python 列表。
        #
        # 普通项只取「片段末行」；投机项取整段输入的行——前 K 行验证草稿，最后一行
        # 给 bonus。中间 prefill（`can_sample` 为假）不取行，但它的 token 照样占
        # 原始输入位置，所以 `offset` 对**每个** item 都要累加。
        rows = []
        picked = []
        offset = 0
        for item in scheduled_items:
            offset += item["num_scheduled_tokens"]
            if not item["can_sample"]:
                continue
            item["sample_offset"] = len(rows)          # 在筛选后 logits 里的起点
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

        # 1) 没有草稿的项：走采样后端（随机采样、惩罚项、按行独立的历史都在那条路上）。
        #    整批一次 select_batch + 一次 .tolist()，先把 token 算好，**提交留到下面按
        #    picked 顺序做**——在这里就提交的话，事件顺序会变成「先普通后投机」。
        plain = [item for item in picked if not item["draft_ids"]]
        plain_tokens = {}
        if plain:
            for item, token in zip(plain, self._sample_rows(logits, plain)):
                plain_tokens[id(item)] = token

        # 2) 贪心且无惩罚的投机项：保留第五十三关的整批 argmax 快路径（一次回传）。
        #    带惩罚项的贪心**不能**走这条：每一行看到的生成历史不同，argmax 必须在
        #    逐行施加惩罚之后再取。
        any_fast = any(item["draft_ids"] and self._is_greedy_without_penalty(item["request"])
                       for item in picked)
        greedy = torch.argmax(logits, dim=-1).tolist() if any_fast else None

        # 3) 严格按 picked 顺序提交：同一请求内按 token 顺序，跨请求按本轮采样顺序
        for item in picked:
            seq = item["request"]
            start, nrows = item["sample_offset"], item["num_sample_rows"]
            if not item["draft_ids"]:
                self._commit_tokens(seq, [plain_tokens[id(item)]], notify)
            elif greedy is not None and self._is_greedy_without_penalty(seq):
                self._commit_drafts(item, greedy[start:start + nrows], notify)
            else:
                self._commit_drafts_random(item, logits[start:start + nrows], notify)
        return None

    @staticmethod
    def _is_greedy_without_penalty(seq):
        """能不能走「整批 argmax」快路径：贪心，而且没有任何惩罚项。

        有惩罚项时每行看到的生成历史都不同（需求 §3），必须先逐行施加惩罚再取
        argmax——那时 one-hot 才是**该行**的目标分布。
        """
        params = seq.sampling_params
        return params.is_greedy and not params.has_penalty

    def _sample_rows(self, logits, items):
        """按每项自己的采样参数与惩罚项抽一枚 token，返回与 items 同序的 Python 列表。

        整批一次 `select_batch`、一次 `.tolist()`，不逐请求 `.item()`。
        **这里必须把 logits 转成 FP32**，和投机那条路不同：`apply_penalties()` 算
        presence / frequency 惩罚时惩罚量是 FP32，回写要 `index_put` 进 logits 那一行，
        dtype 不匹配会直接抛（BF16 模型 + 这两种惩罚，不转 FP32 就跑不起来）。

        **不复制**（不带 `copy=True`）：Graph 的输出缓冲确实会被下次 replay 覆盖，
        但 logits 只在本轮 `step()` 内被消费，`apply_penalties()` 也只读输入。
        **如果以后把采样挪进异步调度、要跨步持有 logits，这里就得改回来。**
        """
        seqs = [item["request"] for item in items]
        rows = [apply_penalties(logits[item["sample_offset"]].to(torch.float32),
                                seq.sampling_params, seq.sampling_state)
                for item, seq in zip(items, seqs)]
        tokens = self.sampler.select_batch(
            rows, [s_.sampling_params for s_ in seqs], [s_.sampling_state for s_ in seqs])
        # token id 整批回传，不逐请求 .item()
        return torch.stack(tokens).tolist()

    def _commit_drafts_random(self, item, rows, notify):
        """投机项（随机采样，或带惩罚项的贪心）：逐行构造目标分布做拒绝采样。

        **每一行的惩罚历史不同**（需求 §3）：行 j 预测的是「真实历史 + 前 j 枚草稿」
        之后那个 token。所以这里用的是一份**临时计数**——从真实计数复制一份，接受
        一枚草稿就往里加一枚；真实 `sampling_state` 与 `all_token_ids` 在
        `_commit_tokens()` 之前一个字都不动。

        行的分布按「全都接受」构造：拒绝点之后的行根本不会被读到（`verify_drafts_random`
        首次拒绝就停），而拒绝点之前的行历史恰好就是「前 j 枚都被接受」。
        """
        seq = item["request"]
        params, state = seq.sampling_params, seq.sampling_state
        draft_ids = item["draft_ids"]

        # 只复制计数，不复制整段 token 历史；与真实状态不共享底层字典
        temp = SimpleNamespace(prompt_token_ids=state.prompt_token_ids,
                               generated_counts=dict(state.generated_counts))
        row_probs = []
        for index in range(len(draft_ids) + 1):
            row_probs.append(row_distribution(rows[index].to(torch.float32), params, temp))
            if index < len(draft_ids):
                token = draft_ids[index]
                temp.generated_counts[token] = temp.generated_counts.get(token, 0) + 1

        if params.is_greedy:
            # 贪心请求没有 generator（第五十二关起只给随机采样建），这里也确实不需要：
            # 目标分布是 one-hot，`p[d]` 非 0 即 1，接受与否由 `verify_drafts_random()`
            # 直接判定、一次 uniform 都不抽；纠正与 bonus 就是该行分布的 argmax。
            def draw_uniform():
                raise AssertionError("贪心路径不该抽接受随机数")
            draw_token = lambda probs: int(torch.argmax(probs))
        else:
            def draw_uniform():
                return float(torch.rand((), generator=state.generator,
                                        device=state.generator.device))
            draw_token = lambda probs: torch.multinomial(probs, num_samples=1,
                                                         generator=state.generator)

        remaining_outputs = seq.max_new_tokens - len(seq.output_ids)
        result = verify_drafts_random(draft_ids, row_probs, self.model.eos_token_ids,
                                      remaining_outputs, draw_uniform, draw_token)

        # 先回滚（被拒草稿的 KV 已经随本轮输入写进物理块），再走唯一提交入口
        self.kv_cache_pool.truncate(seq, item["start_cache_length"] + result.kept_inputs)
        self._commit_tokens(seq, result.committed_ids, notify)

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

    def _commit_drafts(self, item, greedy_ids, notify):
        """投机项：用目标模型一次 forward 的 K+1 行验证草稿，回滚 KV，再逐枚提交。

        顺序不能换：**回滚必须在 `scheduler.post_step()` 之前**。被拒绝草稿的 KV
        已经随本轮输入写进了物理块，先把 `cache.length` 退回去，post_step 里的
        发布与判停读到的才是真实进度。

        `greedy_ids` 是**调用方整批算好、切好**的 K+1 枚目标模型贪心结果（见
        `_sample()`）。这里不再自己 argmax：批量下每个请求各做一次 argmax 会多付
        一次设备同步，而且会把「一次 forward 一次回传」拆散。
        """
        seq = item["request"]
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
