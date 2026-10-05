"""Medusa 提议者（对应 vLLM `v1/spec_decode/medusa.py::MedusaProposer`，需求 066 §2）。

**它为什么这么短**：Medusa 的草稿不来自自回归解码，而是"**同一份 target hidden 过 N 个小 head**"。
所以这个提议者没有输入工作区、没有 KV 缓存、没有自回归循环、没有随机流——那些是
`SpecDecodeBaseProposer` 解决"用一个小模型逐步解码"时才需要的东西。上游也没有让它继承基类。

    blocks   = model(target_hidden_states)        # [B, H] → N 个 [B, H]
    logits   = model.compute_logits(blocks)      # N 个 [B, V]
    draft    = torch.stack([l.argmax(-1) for l in logits], dim=1)   # [B, N]

**链式，不是树**：N 个 head 各自 argmax，拼成 N 枚草稿交给同一套 verifier（59 关）验证。
本关不做论文里的 tree attention，也不是"每个 head 随机采样"——argmax 的提议分布是点质量，
所以 `DraftTokenIds.draft_probs=None` 是**正确**的 q（与 ngram 的确定性提议同一条分支）。

**本仓库与上游的差异**（逐条见 docs/step66_alignment.md §3）：

- 上游的 `dummy_run()` 给显存 profiling（69 关的 CUDA Graph 基建）用；本仓库没有 profiling
  路径，所以 `load_model()` 末尾调它一次做形状/设备自检与预热——这是 60 关
  `NgramProposerGPU._dummy_run()` 的同一个用法（也让这个方法有真实调用方）。
- 上游用 `assert` 校验 `num_speculative_tokens`；本仓库换成明确报错（失败要在**提议**这一步
  就停摆，不能靠 `python -O` 关掉的断言）。
- 上游把"挑哪一行 hidden"写在 Runner 里（`gpu_model_runner.py:5206-5225`）；本仓库把它收进
  `select_target_hidden_states()`：那段算式的**唯一**消费者就是本提议者，放在一起才好对照
  （差异与混合 prefill 批次下的行号修正见该方法的说明）。
"""

import torch

from ..attention import set_forward_context
from .draft_model import _dtype


class MedusaProposer:
    """Medusa 多头提议者（上游同名类；不继承 `SpecDecodeBaseProposer`，上游也一样）。"""

    def __init__(self, vllm_config, device) -> None:
        self.vllm_config = vllm_config
        if vllm_config.speculative_config is None:
            raise ValueError("MedusaProposer 需要 speculative_config（配置期就该有）")
        self.spec_config = vllm_config.speculative_config
        self.device = device
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # 上游的 `load_model()` 直接用 `spec_config.draft_model_config`；本仓库的配置拿不到
        # target（见 `SpeculativeConfig.derive_medusa_draft_config` 的说明），所以在这里补上
        # "与 target 对齐词表"那一步——位置与 64 关的 `derive_extract_hidden_states_config()`
        # 相同（提议者构造时做，此时 target 配置就在手边）。
        self.draft_model_config = self.spec_config.derive_medusa_draft_config(
            vllm_config.model_config)
        # head 的输入宽度 = draft 配置的 hidden_size（与 target 的 hidden 一致，
        # `derive_medusa_draft_config()` 在配置期已经校验过）
        self.hidden_size = int(self.draft_model_config.hf_config["hidden_size"])
        self.dtype = vllm_config.model_config.dtype
        self.num_speculative_tokens = self.spec_config.num_speculative_tokens
        self.model = None

    # -------- 加载 --------

    def load_model(self) -> None:
        """装 Medusa head（上游 `load_model()`）。

        上游在 `set_model_tag("medusa_head")` 上下文里加载（给 torch.compile 的图分个标签，
        本仓库没有编译路径，不需要标签）；加载完断言"MoE + EPLB 不支持"。
        """
        from ..model_loader import get_model

        self.model = get_model(self.draft_model_config, self.device)
        self._reject_unsupported_eplb()
        # 上游这里没有这一步（它的 dummy_run 由显存 profiling 触发）；本仓库用它做启动期
        # 形状/设备自检与预热，理由见模块说明。
        self.dummy_run(num_tokens=self.max_num_tokens)
        return self.model

    def _reject_unsupported_eplb(self) -> None:
        """上游 `medusa.py:66-69` 的 `assert not (is_mixture_of_experts(model) and eplb)`。

        EPLB（专家负载均衡）会重排 MoE 的专家，而 Medusa 的 head 是在**固定的 hidden 分布**上
        训练的，两者一起用没有意义，上游直接拒绝。本仓库既没有 MoE 模型也没有 EPLB（没有并行
        配置），所以这条在当前代码里**不可能成立**；写成显式检查是为了把边界留在代码里：
        将来谁把 `parallel_config.enable_eplb` 加进来，这里会立刻报错而不是静默跑。
        """
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if bool(getattr(parallel_config, "enable_eplb", False)) and \
                bool(getattr(self.model, "is_mixture_of_experts", False)):
            raise ValueError(
                "EPLB for Medusa is not supported（上游原文）：MoE 的专家重排会改变 head 看到的 "
                "hidden 分布，Medusa 的 head 不能在这种组合下工作")

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        """按**最大工作区宽度**跑一遍全零 hidden（上游同名方法逐行照抄）。"""
        hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                    dtype=_dtype(self.dtype), device=self.device)
        with set_forward_context({}, num_tokens=num_tokens):
            self.model(hidden_states)

    # -------- 提议 --------

    def propose(self, num_speculative_tokens: int, target_hidden_states: torch.Tensor,
                sampling_metadata=None, slot_mappings=None) -> torch.Tensor:
        """`[B, H]` → `[B, num_heads]`（上游同名方法；两个后置参数上游收下但**不用**）。

        上游签名里的 `sampling_metadata` / `slot_mappings` 在本方法里没有任何读取：Medusa 是
        argmax 提议，不看采样参数、也不写 KV。本仓库保留参数（调用形态与上游逐字一致），
        但 Runner 只传真正用得上的那两个。
        """
        if num_speculative_tokens != self.num_speculative_tokens:
            raise RuntimeError(
                f"Medusa 提议者按 K={self.num_speculative_tokens} 建的 head，"
                f"收到 num_speculative_tokens={num_speculative_tokens}："
                f"head 数与调度侧的 K 不一致时，草稿列数与验证行数会对不上")
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)
        # 每个 head 一个 argmax，拼成 [B, num_heads]（顺序就是 head 顺序，列 0 = head 0）
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        return draft_tokens

    @staticmethod
    def select_target_hidden_states(target_hidden_states: torch.Tensor,
                                    rows_per_request: list[int],
                                    num_sampled_tokens: list[int]) -> torch.Tensor:
        """挑出每条请求"**最后一个已经算过的 token**"的 hidden 行（上游 Runner 的那段算式）。

        为什么是这一行：Medusa 的 head 条件在"当前序列最后一个 token 的 hidden"上。一轮投机
        验证里，请求的 query 是 `[b][d1]…[dK]`（K+1 行，见 `spec_decode/metadata.py`），
        采样后新序列的最后一个 token 是 bonus（**本轮没有算过它的 hidden**），所以可用的是
        "产出 bonus 的那一行"，也就是 `len(tokens) - 1` 那一行：

            接受 a 枚时 → 第 a 行（d_a 的 hidden，它的 logits 采出了 bonus）
            首枚就被拒（a=0）→ 第 0 行（b 的 hidden）

        上游（`v1/worker/gpu_model_runner.py:5206-5225`）两条分支：

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                hidden_states = sample_hidden_states          # 整批每请求只有 1 行（没有草稿）
            else:
                for num_draft, tokens in zip(num_draft_tokens, sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1

        这里合成一条：`rows_per_request[i]` 是请求 i 本轮**真正算了几行**（调度快照的
        `num_scheduled_tokens`）。没有草稿时它恒为 1 → `indices == [0, 1, …, B-1]`，与上游
        第一条分支**逐值相同**；有草稿时它 = K_i + 1 = 上游的 `num_draft + 1`，与第二条相同。

        **修正**（差异见 docs/step66_alignment.md §3）：一批里混进"中间 prefill 块"
        （本轮排了 n>1 行、没有草稿）时，上游的 `offset += num_draft + 1` 只前进 1，
        它后面所有请求的行号整体错位；本仓库按调度快照前进。
        """
        if len(rows_per_request) != len(num_sampled_tokens):
            raise ValueError("rows_per_request 与 num_sampled_tokens 必须一一对应")
        indices: list[int] = []
        offset = 0
        for rows, num_tokens in zip(rows_per_request, num_sampled_tokens):
            if num_tokens <= 0:
                # 中间 prefill 块（本轮没有采到 token）：上游会算出 `offset - 1` 这种行号
                # （第一个请求时就是 -1，即最后一行），那是**错的**。本仓库在 Runner 里
                # 直接跳过这类请求（不提草稿），所以走到这里就是调用方漏了过滤。
                raise ValueError("select_target_hidden_states 只接受「采到了 token」的请求："
                                 "中间 prefill 块没有可用作条件的 hidden 行")
            if num_tokens > rows:
                raise ValueError(
                    f"一行最多采出 {rows} 个 token，收到 {num_tokens}："
                    f"采样数与本轮 query 行数口径不一致")
            indices.append(offset + num_tokens - 1)
            offset += rows
        if offset != target_hidden_states.shape[0]:
            raise RuntimeError(
                f"本轮各请求的 query 行数之和 {offset} 与 target 的 hidden 行数 "
                f"{target_hidden_states.shape[0]} 不一致：调度快照与执行侧的行口径不同")
        index_tensor = torch.tensor(indices, dtype=torch.int64,
                                    device=target_hidden_states.device)
        return target_hidden_states.index_select(0, index_tensor)
