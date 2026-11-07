"""线性层对应 vLLM `model_executor/layers/linear.py`；普通行列层支持 TP。

名字保留 vLLM 的 `QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear` /
`VocabParallelEmbedding` / `ParallelLMHead`，是为了**源码映射**：读 vLLM 的 qwen3.py 时能一对
一找到这些类。普通行列层、QKV 和 Merged 支持 TP；词表及多卡模型仍未开放。

它们存在的**真正理由**是权重加载：检查点里 q/k/v 是三个独立权重、gate/up 是两个，而模型里是
打包的一个参数（少一次 GEMM 的 launch、少一次中间读写）。打包带来的问题就是"怎么把分片的
权重写进打包参数的正确区间"——那件事由每个参数自己的 `weight_loader` 回答：

    q_proj.weight        → qkv_proj.weight 的 [0 : q_size]
    k_proj.weight        → qkv_proj.weight 的 [q_size : q_size + kv_size]
    v_proj.weight        → qkv_proj.weight 的 [q_size + kv_size : ]
    gate_proj.weight     → gate_up_proj.weight 的 [0 : intermediate]
    up_proj.weight       → gate_up_proj.weight 的 [intermediate : ]

`weight_loader` 挂在 **Parameter 对象**上（`param.weight_loader = ...`），加载器按参数名取出
目标参数、再调它自己的加载器——这样"怎么装"这件事跟着参数走，而不是散在加载器里的一堆 if。

签名统一为 `weight_loader(param, loaded_weight, shard_id=None)`：只有打包层会用到 `shard_id`；
没打包的层收到它就直接报错（`_reject_shard_id`），因为那意味着加载器的打包映射指错了目标。
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def _reject_shard_id(name: str, shard_id) -> None:
    """没打包的参数收到了 shard_id：说明加载器的打包映射指错了目标。

    静默忽略会让权重**整块覆盖**（`_copy_into` 之外的路径），结果是一个"看起来加载成功"
    的错误模型——所以这里明确报错。
    """
    if shard_id is not None:
        raise ValueError(f"{name} 不是打包参数，却收到了 shard_id={shard_id!r}："
                         f"检查 packed 映射（WeightsMapper.orig_to_new_stacked）是否指错了目标")


def _copy_into(param: Parameter, loaded_weight: torch.Tensor, offset: int, size: int) -> None:
    """把 `loaded_weight` 写进 `param` 的第 `offset..offset+size` 行（输出维）。

    形状必须严格对上：少了会安静地留下未初始化的行，多了会切掉数据。两种都直接报错。
    """
    if loaded_weight.shape[0] != size:
        raise ValueError(f"权重输出维不匹配：期望 {size} 行，收到 {loaded_weight.shape[0]} 行"
                         f"（权重形状 {tuple(loaded_weight.shape)}）")
    if loaded_weight.shape[1:] != param.shape[1:]:
        raise ValueError(f"权重其余维不匹配：目标 {tuple(param.shape[1:])}，"
                         f"收到 {tuple(loaded_weight.shape[1:])}")
    with torch.no_grad():
        param[offset:offset + size].copy_(loaded_weight)


class QKVParallelLinear(nn.Module):
    """Q/K/V 独立分片后按 [Q_local,K_local,V_local] 打包。"""

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
                 total_num_kv_heads: int | None = None, bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.group, self.tp_rank, self.tp_size = _linear_tp_group(False)
        for name, value in (("hidden_size", hidden_size), ("head_size", head_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        kv_heads = total_num_heads if total_num_kv_heads is None else total_num_kv_heads
        self.num_heads = _parallel_dimension(total_num_heads, self.tp_size, "total_num_heads")
        if type(kv_heads) is not int or kv_heads <= 0:
            raise ValueError("total_num_kv_heads 必须是正整数")
        if self.tp_size >= kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = _parallel_dimension(self.tp_size, kv_heads, "TP")
        else:
            self.num_kv_heads = _parallel_dimension(kv_heads, self.tp_size, "total_num_kv_heads")
            self.num_kv_head_replicas = 1
        self.hidden_size, self.head_size, self.v_head_size = hidden_size, head_size, head_size
        self.total_num_heads, self.total_num_kv_heads = total_num_heads, kv_heads
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        self.output_sizes = [self.q_size * self.tp_size, self.kv_size * self.tp_size,
                             self.kv_size * self.tp_size]
        self.prefix = prefix
        local_size = self.q_size + 2 * self.kv_size
        self.weight = Parameter(torch.empty(local_size, hidden_size))
        self.weight.output_dim = 0
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(local_size))
            self.bias.output_dim = 0
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self._shard_load_counts: dict[str, int] = {}
        self._bias_shard_load_counts: dict[str, int] = {}

    def reset_load_state(self) -> None:
        self._shard_load_counts.clear()
        self._bias_shard_load_counts.clear()

    def validate_loaded_shards(self, required: set[str]) -> None:
        required = set(required)
        invalid = required - {"q", "k", "v"}
        if invalid:
            raise ValueError(f"未知的必需 QKV shard: {sorted(invalid)}")
        detail = []
        counts = [("weight", self._shard_load_counts)]
        if self.bias is not None:
            counts.append(("bias", self._bias_shard_load_counts))
        for name, loaded in counts:
            missing = sorted(shard for shard in required if loaded.get(shard, 0) == 0)
            duplicated = sorted(shard for shard, count in loaded.items() if count > 1)
            if missing:
                detail.append(f"{name} 缺少 {missing}")
            if duplicated:
                detail.append(f"{name} 重复覆盖 {duplicated}")
        if detail:
            raise ValueError(f"QKVParallelLinear（输出 {self.output_sizes}）的 shard 加载不完整：" + "，".join(detail))

    def _mark_shard_loaded(self, param: Parameter, shard_id: str | None) -> None:
        if param is self.weight:
            counts = self._shard_load_counts
        elif self.bias is not None and param is self.bias:
            counts = self._bias_shard_load_counts
        else:
            return
        shards = ("q", "k", "v") if shard_id is None else (shard_id,)
        for shard in shards:
            counts[shard] = counts.get(shard, 0) + 1

    def _shard_range(self, shard_id: str) -> tuple[int, int]:
        offsets = {"q": 0, "k": self.q_size, "v": self.q_size + self.kv_size}
        sizes = {"q": self.q_size, "k": self.kv_size, "v": self.kv_size}
        if shard_id not in offsets:
            raise ValueError(f"未知的 shard_id {shard_id!r}（qkv 只接受 'q' / 'k' / 'v'）")
        return offsets[shard_id], sizes[shard_id]

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id: str | None = None) -> None:
        if type(param) is not Parameter:
            raise NotImplementedError("QKV 暂不支持特殊或量化 Parameter")
        source_sizes = {"q": self.total_num_heads * self.head_size,
                        "k": self.total_num_kv_heads * self.head_size,
                        "v": self.total_num_kv_heads * self.v_head_size}
        if shard_id is None:
            if loaded_weight.ndim != param.ndim or loaded_weight.shape[0] != sum(source_sizes.values()) or loaded_weight.shape[1:] != param.shape[1:]:
                raise ValueError(f"打包权重形状不匹配：收到 {tuple(loaded_weight.shape)}")
            offset = 0
            for name in ("q", "k", "v"):
                width = source_sizes[name]
                self.weight_loader(param, loaded_weight.narrow(0, offset, width), name)
                offset += width
            return
        target_offset, size = self._shard_range(shard_id)
        if loaded_weight.ndim != param.ndim or loaded_weight.shape[0] != source_sizes[shard_id] or loaded_weight.shape[1:] != param.shape[1:]:
            raise ValueError(f"{shard_id} 权重形状不匹配：收到 {tuple(loaded_weight.shape)}")
        source_rank = self.tp_rank if shard_id == "q" else self.tp_rank // self.num_kv_head_replicas
        _copy_into(param, loaded_weight.narrow(0, source_rank * size, size), target_offset, size)
        self._mark_shard_loaded(param, shard_id)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias), None


class MergedColumnParallelLinear(nn.Module):
    """每个逻辑输出分别分片，打包顺序保持 [shard0_local, shard1_local, ...]。"""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.group, self.tp_rank, self.tp_size = _linear_tp_group(False)
        if type(input_size) is not int or input_size <= 0 or not output_sizes:
            raise ValueError("input_size 必须为正且 output_sizes 不能为空")
        self.input_size, self.output_sizes, self.prefix = input_size, list(output_sizes), prefix
        self.local_output_sizes = [_parallel_dimension(v, self.tp_size, "output_size") for v in self.output_sizes]
        self.weight = Parameter(torch.empty(sum(self.local_output_sizes), input_size))
        self.weight.output_dim = 0
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(sum(self.local_output_sizes)))
            self.bias.output_dim = 0
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
        self._shard_load_counts: dict[int, int] = {}
        self._bias_shard_load_counts: dict[int, int] = {}

    def reset_load_state(self) -> None:
        self._shard_load_counts.clear()
        self._bias_shard_load_counts.clear()

    def validate_loaded_shards(self, required: set[int]) -> None:
        required = set(required)
        if not required <= set(range(len(self.output_sizes))):
            raise ValueError("未知的 Merged shard")
        detail = []
        counts = [("weight", self._shard_load_counts)]
        if self.bias is not None:
            counts.append(("bias", self._bias_shard_load_counts))
        for name, loaded in counts:
            missing = sorted(i for i in required if loaded.get(i, 0) == 0)
            duplicate = sorted(i for i, count in loaded.items() if count > 1)
            if missing:
                detail.append(f"{name} 缺少 {missing}")
            if duplicate:
                detail.append(f"{name} 重复覆盖 {duplicate}")
        if detail:
            raise ValueError("Merged shard 加载不完整：" + "；".join(detail))

    def _shard_range(self, shard_id: int) -> tuple[int, int]:
        if type(shard_id) is not int or not 0 <= shard_id < len(self.output_sizes):
            raise ValueError(f"shard_id {shard_id!r} 越界")
        return sum(self.local_output_sizes[:shard_id]), self.local_output_sizes[shard_id]

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id: int | None = None) -> None:
        if type(param) is not Parameter:
            raise NotImplementedError("Merged 暂不支持特殊或量化 Parameter")
        if shard_id is None:
            if loaded_weight.ndim != param.ndim or loaded_weight.shape[0] != sum(self.output_sizes) or loaded_weight.shape[1:] != param.shape[1:]:
                raise ValueError(f"打包权重形状不匹配：收到 {tuple(loaded_weight.shape)}")
            offset = 0
            for index, width in enumerate(self.output_sizes):
                self.weight_loader(param, loaded_weight.narrow(0, offset, width), index)
                offset += width
            return
        target_offset, size = self._shard_range(shard_id)
        if loaded_weight.ndim != param.ndim or loaded_weight.shape[0] != self.output_sizes[shard_id] or loaded_weight.shape[1:] != param.shape[1:]:
            raise ValueError(f"shard {shard_id} 权重形状不匹配：收到 {tuple(loaded_weight.shape)}")
        _copy_into(param, loaded_weight.narrow(0, self.tp_rank * size, size), target_offset, size)
        if param is self.weight:
            counts = self._shard_load_counts
        elif self.bias is not None and param is self.bias:
            counts = self._bias_shard_load_counts
        else:
            return
        counts[shard_id] = counts.get(shard_id, 0) + 1

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias), None


def _linear_tp_group(disable_tp: bool):
    """兼容无分布式初始化的单卡；多进程默认组存在时不可静默当成 TP=1。"""
    if disable_tp:
        return None, 0, 1
    from ..distributed.parallel_state import get_tp_group_if_initialized
    group = get_tp_group_if_initialized()
    if group is None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("分布式环境已初始化但 TP 组尚未初始化")
        return None, 0, 1
    return group, group.rank_in_group, group.world_size


def _parallel_dimension(size: int, tp_size: int, name: str) -> int:
    if type(size) is not int or size <= 0 or size % tp_size:
        raise ValueError(f"{name}={size} 必须为正且能被 TP={tp_size} 整除")
    return size // tp_size


def _load_parallel_parameter(param: Parameter, loaded_weight: torch.Tensor,
                             dim: int | None, rank: int, shard_id) -> None:
    _reject_shard_id("并行线性层", shard_id)
    if type(param) is not Parameter:
        raise NotImplementedError("并行线性层暂不支持特殊或量化 Parameter")
    if dim is not None and not getattr(param, "is_sharded_weight", False):
        expected = list(param.shape)
        expected[dim] *= param.tp_size
        if tuple(loaded_weight.shape) != tuple(expected):
            raise ValueError(f"完整权重形状不匹配：期望 {tuple(expected)}，收到 {tuple(loaded_weight.shape)}")
        size = param.shape[dim]
        loaded_weight = loaded_weight.narrow(dim, rank * size, size)
    if tuple(loaded_weight.shape) != tuple(param.shape):
        raise ValueError(f"权重形状不匹配：目标 {tuple(param.shape)}，收到 {tuple(loaded_weight.shape)}")
    with torch.no_grad():
        param.copy_(loaded_weight)


class ColumnParallelLinear(nn.Module):
    """输出维分片；PyTorch 权重布局为 [out/TP, in]。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 return_bias: bool = True, prefix: str = "", *,
                 gather_output: bool = False, skip_bias_add: bool = False,
                 params_dtype: torch.dtype | None = None, quant_config=None,
                 disable_tp: bool = False) -> None:
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("并行线性层暂不支持量化参数")
        self.group, self.tp_rank, self.tp_size = _linear_tp_group(disable_tp)
        self.input_size = input_size
        self.output_size = output_size
        if type(input_size) is not int or input_size <= 0:
            raise ValueError("input_size 必须为正整数")
        self.input_size_per_partition = input_size
        self.output_size_per_partition = _parallel_dimension(output_size, self.tp_size, "output_size")
        self.output_partition_sizes = [self.output_size_per_partition]
        self.gather_output, self.skip_bias_add = gather_output, skip_bias_add
        self.return_bias, self.prefix = return_bias, prefix
        self.weight = Parameter(torch.empty(self.output_size_per_partition, input_size, dtype=params_dtype))
        self.weight.output_dim = 0
        self.weight.tp_size = self.tp_size
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            self.bias.output_dim = 0
            self.bias.tp_size = self.tp_size
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, shard_id=None) -> None:
        _load_parallel_parameter(param, loaded_weight, getattr(param, "output_dim", None), self.tp_rank, shard_id)

    def forward(self, x: torch.Tensor):
        bias = None if self.skip_bias_add else self.bias
        output = F.linear(x, self.weight, bias)
        if self.gather_output and self.tp_size > 1:
            output = self.group.all_gather(output, dim=-1)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


class RowParallelLinear(nn.Module):
    """输入维分片；PyTorch 权重布局为 [out, in/TP]。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 prefix: str = "", *, input_is_parallel: bool = True,
                 skip_bias_add: bool = False, params_dtype: torch.dtype | None = None,
                 reduce_results: bool = True, quant_config=None,
                 return_bias: bool = True, disable_tp: bool = False) -> None:
        super().__init__()
        if quant_config is not None:
            raise NotImplementedError("并行线性层暂不支持量化参数")
        self.group, self.tp_rank, self.tp_size = _linear_tp_group(disable_tp)
        self.input_size, self.output_size = input_size, output_size
        self.input_size_per_partition = _parallel_dimension(input_size, self.tp_size, "input_size")
        if type(output_size) is not int or output_size <= 0:
            raise ValueError("output_size 必须为正整数")
        if not reduce_results and bias and not skip_bias_add:
            raise ValueError("不归约时不能在线性层中加入 bias")
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.input_is_parallel, self.skip_bias_add = input_is_parallel, skip_bias_add
        self.reduce_results, self.return_bias, self.prefix = reduce_results, return_bias, prefix
        self.weight = Parameter(torch.empty(output_size, self.input_size_per_partition, dtype=params_dtype))
        self.weight.input_dim = 1
        self.weight.tp_size = self.tp_size
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(output_size, dtype=params_dtype))
            self.bias.tp_size = self.tp_size
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, shard_id=None) -> None:
        _load_parallel_parameter(param, loaded_weight, getattr(param, "input_dim", None), self.tp_rank, shard_id)

    def forward(self, x: torch.Tensor):
        if not self.input_is_parallel:
            if x.shape[-1] != self.input_size:
                raise ValueError(f"输入维度应为 {self.input_size}，收到 {x.shape[-1]}")
            x = x.narrow(-1, self.tp_rank * self.input_size_per_partition,
                         self.input_size_per_partition).contiguous()
        bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
        output = F.linear(x, self.weight, bias)
        if self.reduce_results and self.tp_size > 1:
            output = self.group.all_reduce(output)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


class VocabParallelEmbedding(nn.Module):
    """词表嵌入（TP=1：整张词表都在本卡）。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = "") -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id=None) -> None:
        _reject_shard_id(type(self).__name__, shard_id)
        if tuple(loaded_weight.shape) != tuple(param.shape):
            raise ValueError(f"嵌入权重形状不匹配：目标 {tuple(param.shape)}，"
                             f"收到 {tuple(loaded_weight.shape)}")
        with torch.no_grad():
            param.copy_(loaded_weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class ParallelLMHead(nn.Module):
    """输出头（TP=1）。`tie_word_embeddings` 时与嵌入**共享同一个 Parameter 对象**。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader
        self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: "VocabParallelEmbedding") -> "ParallelLMHead":
        """与词嵌入共享权重（vLLM 同名方法：`layer.weight = embed_tokens.weight`）。

        共享**对象**而不是复制数值：省一份显存（Qwen3-1.7B 的词表是 151936×2048 ≈ 1.2 GB），
        而且保证两边永远一致。副作用是检查点里的 `lm_head.weight` 必须**跳过不加载**
        （它是同一个参数的第二份拷贝，加载它只会做一次多余的覆盖）——见 `qwen3.py` 的
        `skip_prefixes`，以及 `AutoWeightsLoader._check_coverage` 为什么需要豁免名单。
        """
        self.weight = embed_tokens.weight
        return self

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id=None) -> None:
        _reject_shard_id(type(self).__name__, shard_id)
        if tuple(loaded_weight.shape) != tuple(param.shape):
            raise ValueError(f"lm_head 权重形状不匹配：目标 {tuple(param.shape)}，"
                             f"收到 {tuple(loaded_weight.shape)}")
        with torch.no_grad():
            param.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, None)
