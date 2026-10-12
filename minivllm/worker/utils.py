"""Worker 侧的公共小工具（对应 vLLM `v1/worker/utils.py` 的一个子集）。

本仓库路径映射：本文件 ↔ 上游 `vllm/v1/worker/utils.py`。上游那个文件很大（编译期
`set_forward_context` 的辅助、DP/PP 协调、显存 profiling、各种 platform 分支），本仓库
只搬**真的有调用方**的三件：

    is_uniform_query_len           "每请求 query 长度相同"的形状判据
    get_uniform_decode_token_count 把一批请求分类成"统一 decode"（返回每请求行数）或 None
    AttentionGroup                 一组 KV cache group 的属性包（后端 + 层名 + 元数据构造器）
                                   —— 74 关的"这个后端能不能在草稿步之间原地更新元数据"
                                   就挂在它身上（`supports_draft_decode_metadata_update`）

**为什么"统一 decode"要单独有一个判据**：CUDA Graph 的键是批次形状，而"形状固定"在投机
解码里只有一种情形——批里**每条请求都恰好 `1 + K` 行**。少判一条（例如只看"最长请求是
1+K 行"）就会把"一条 4 行、一条 1 行"的批也当成统一批，那种批的行数不是请求数的整数倍，
图里的形状根本对不上（不报错，只是重放到错的形状上）。

`AttentionGroup` 在上游按 **KV cache group** 分组（滑动窗口 + 全注意力、Mamba + 注意力这类
模型会有多组）。本仓库只有一个后端、一种分页 KV，所以恒为**一组**；`kv_cache_spec` /
`kv_cache_group_id` 这两个上游字段在本仓库没有对应物（没有 KVCacheSpec 对象、没有多组），
**不留空壳字段**——留了却没有生产者，读的人会以为它是有效数据。
"""

from dataclasses import dataclass, field

from ..attention.metadata import AttentionMetadataBuilder


def is_uniform_query_len(num_reqs: int, num_tokens: int, max_query_len: int) -> bool:
    """批里每条请求是否都是 `max_query_len` 行（上游同名函数，逐字对齐）。

    纯**形状**判据：不看这批请求是在 prefill 还是在 decode。提示词分块（chunked prefill）
    也可能恰好长成 decode 的样子，所以分类要用 `get_uniform_decode_token_count()`。
    """
    return num_reqs > 0 and num_tokens == max_query_len * num_reqs


def get_uniform_decode_token_count(
    num_reqs: int, num_tokens: int, max_query_len: int, has_prefill: bool
) -> int | None:
    """统一 decode 批的"每请求行数"，不是统一批就返回 None（上游同名函数，逐字对齐）。

    两个条件缺一不可：批里**没有请求在 prefill**（有 prefill 的批不是 decode），且形状满足
    `num_tokens == max_query_len * num_reqs`。返回值就是 `max_query_len`
    （投机解码下 = `1 + K`），它同时是图键里的 `uniform_token_count`。
    """
    if not has_prefill and is_uniform_query_len(num_reqs, num_tokens, max_query_len):
        return max_query_len
    return None


@dataclass
class AttentionGroup:
    """一组注意力层的公共属性（上游 `v1/worker/utils.py::AttentionGroup` 的子集）。

    上游在这上面挂 `backend` / `layer_names` / `kv_cache_spec` / `kv_cache_group_id` /
    `metadata_builders`。本仓库只有**一个后端、一组层**，但结构与上游同形，原因是 74 关的
    一条硬要求：`AutoRegressiveSpeculator._configure_fused_multi_step_decode()` 要按
    **后端的 capability** 决定走不走融合多步 decode，而 capability 挂在**元数据构造器**上，
    由这一层转发（上游同名属性/方法）。
    """

    backend: type
    layer_names: list[str]
    metadata_builders: list[AttentionMetadataBuilder] = field(default_factory=list)

    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder:
        assert len(self.metadata_builders) > ubatch_id, (
            f"这组注意力只建了 {len(self.metadata_builders)} 个元数据构造器，"
            f"要第 {ubatch_id} 个（上游同款断言）")
        return self.metadata_builders[ubatch_id]

    @property
    def supports_draft_decode_metadata_update(self) -> bool:
        """这个后端能不能**在草稿步之间原地更新**元数据（上游同名属性）。

        True = 一步建好的元数据可以被 K−1 个草稿步复用，中间只调
        `update_draft_decode_metadata()` 刷新步相关的量 → 整个多步循环能录进一张图。
        False = 必须**逐步重建**元数据（CPU 侧 Python 循环、每步一张图），这条是回退路径，
        不是错误。
        """
        return self.get_metadata_builder().supports_draft_decode_metadata_update

    def update_draft_decode_metadata(self, attn_metadata) -> None:
        """刷新草稿 decode 步的步相关元数据（上游同名方法）。

        上游注释里那句最要紧、本仓库照抄：**图重放不会执行这个 Python 方法**，所以实现必须
        只做"捕获安全的操作"，且把要更新的状态放在**常驻缓冲**里（每轮原位改写）。本仓库的
        Torch 后端满足这条（元数据里的步相关字段就是常驻输入缓冲的切片），见
        `attention/backends/torch_sdpa.py`。
        """
        metadata = attn_metadata[self.layer_names[0]]
        self.get_metadata_builder().update_draft_decode_metadata(metadata)
