"""并行线性层（**只实现 TP=1 子集**，没有通信）——对应 vLLM `model_executor/layers/linear.py`。

名字保留 vLLM 的 `QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear` /
`VocabParallelEmbedding` / `ParallelLMHead`，是为了**源码映射**：读 vLLM 的 qwen3.py 时能一对
一找到这些类。但这里没有 TP/EP/量化，也没有通信——文档与配置都必须写明这一点，不能让人以为
换 tp_size 就能跑。

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
    """把 q/k/v 三个投影打成一次 GEMM（TP=1）。

    GQA 下 q 与 k/v 的头数不同，**不能假定三等分**：输出维按
    `[num_heads*head_dim, num_kv_heads*head_dim, num_kv_heads*head_dim]` 切。
    """

    def __init__(self, hidden_size: int, head_size: int, total_num_heads: int,
                 total_num_kv_heads: int | None = None, bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_heads if total_num_kv_heads is None else total_num_kv_heads
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        # 三段输出维（q / k / v）。shard 的偏移与大小都由它推出来。
        self.output_sizes = [self.q_size, self.kv_size, self.kv_size]

        self.weight = Parameter(torch.empty(sum(self.output_sizes), hidden_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(sum(self.output_sizes)))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def _shard_range(self, shard_id: str) -> tuple[int, int]:
        offsets = {"q": 0, "k": self.q_size, "v": self.q_size + self.kv_size}
        sizes = {"q": self.q_size, "k": self.kv_size, "v": self.kv_size}
        if shard_id not in offsets:
            raise ValueError(f"未知的 shard_id {shard_id!r}（qkv 只接受 'q' / 'k' / 'v'）")
        return offsets[shard_id], sizes[shard_id]

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id: str | None = None) -> None:
        if shard_id is None:
            # 检查点本身已经是打包好的（或者只有一个分片）：整块写入
            if loaded_weight.shape[0] != param.shape[0]:
                raise ValueError(f"打包权重行数不匹配：目标 {param.shape[0]}，"
                                 f"收到 {loaded_weight.shape[0]}")
            with torch.no_grad():
                param.copy_(loaded_weight)
            return
        offset, size = self._shard_range(shard_id)
        _copy_into(param, loaded_weight, offset, size)

    def forward(self, x: torch.Tensor):
        # 返回值对齐 vLLM 的 (output, output_bias) 约定
        return F.linear(x, self.weight, self.bias), None


class MergedColumnParallelLinear(nn.Module):
    """把 gate/up 打成一次 GEMM（TP=1）。`shard_id` 是分片下标（0=gate，1=up）。"""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)
        self.weight = Parameter(torch.empty(sum(self.output_sizes), input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(sum(self.output_sizes)))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def _shard_range(self, shard_id: int) -> tuple[int, int]:
        if not 0 <= shard_id < len(self.output_sizes):
            raise ValueError(f"shard_id {shard_id} 越界（这个合并层有 "
                             f"{len(self.output_sizes)} 个分片）")
        offset = sum(self.output_sizes[:shard_id])
        return offset, self.output_sizes[shard_id]

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id: int | None = None) -> None:
        if shard_id is None:
            if loaded_weight.shape[0] != param.shape[0]:
                raise ValueError(f"打包权重行数不匹配：目标 {param.shape[0]}，"
                                 f"收到 {loaded_weight.shape[0]}")
            with torch.no_grad():
                param.copy_(loaded_weight)
            return
        offset, size = self._shard_range(shard_id)
        _copy_into(param, loaded_weight, offset, size)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias), None


class RowParallelLinear(nn.Module):
    """按输入维分片的线性层（TP=1 时就是普通 Linear）。没有通信，没有 all-reduce。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False,
                 prefix: str = "") -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor,
                      shard_id=None) -> None:
        _reject_shard_id(type(self).__name__, shard_id)
        if tuple(loaded_weight.shape) != tuple(param.shape):
            raise ValueError(f"权重形状不匹配：目标 {tuple(param.shape)}，"
                             f"收到 {tuple(loaded_weight.shape)}")
        with torch.no_grad():
            param.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias), None


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
