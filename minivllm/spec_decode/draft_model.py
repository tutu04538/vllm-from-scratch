"""用一个**小模型**当提议者（对应 vLLM `v1/spec_decode/llm_base_proposer.py` +
`draft_model.py`）。

拆成两个类，位置与 vLLM 一致：

    SpecDecodeBaseProposer   提议的**步骤**：自己的输入缓冲、自己的 KV、自己的 attention 元数据
    DraftModelProposer       与"用哪个模型"有关的：造 draft 的 VllmConfig、加载、规格校验

### 它为什么不是"第二套引擎"

它没有调度器、没有请求状态、没有输出提交：只做一件事——**给定本轮 target 侧的事实，提 K 枚草稿**。

    第一遍 forward：把 [本轮起点, 采样后有效历史末尾) 喂给 draft 模型（含 prefix 命中之后的续算），
                    写进自己的 KV；扩容行的 hidden 顺便得到**第一枚**草稿
    自回归 K-1 步：每步一行（上一枚草稿当输入），**复用同一个工作区**，得到其余草稿

第一遍为什么需要：draft 要提出**像样**的草稿，它的 KV 必须覆盖与 target 相同的那段历史；
57 的设计是"每轮与 target 跑同一段位置"（中间 prefill 块也同步），所以发布出去的完整块
在 draft 那几层也都写过（199 §9 的不变量）。

### 58：第一遍的输入怎么组织（padded + mask，与上游同形）

普通自回归 draft 的第一遍，每条请求的**物理行**是：

    [有效行 (n - num_rejected)] + [1 行扩容行（新采出的 token）] + [num_rejected 行被拒行]
    总物理行 = Σ(target 本轮物理行数 + 1)          ← 与 target 的 input_budget 对得上

- **有效行** = target 本轮 query 里真正成为历史的那部分（起点 `start` 到采样后有效历史末尾）；
- **扩容行** = target 刚采出的那个 token（普通 draft 比 target 多要的就是这 1 行）；
  中间 prefill 块没有新 token，用 backup（本段历史最后一个 token）占位，草稿不提；
- **被拒行** = 留在工作区里但被屏蔽：token=padding、position=0、`slot=PADDING_SLOT_ID(-1)`，
  于是它既不写 KV、也不进新提议的上下文。展开规则与上游
  `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=False`）逐值一致，见 `utils.py`。

**prefix 命中**：起点直接用调度快照里的 `num_computed_tokens`（含命中起点），命中段不再重算——
它的前提正是上面的 199 §9 不变量（缓存交回来的块在 draft 每一层都有效）。58 之前用
`_draft_computed` 当第二套权威，新请求=0 于是命中 120 个 token 也会从 0 重算。

**固定工作区**：`input_ids/positions/slot_mapping/masks/query_start_loc/seq_lens/block_table`
在初始化时按 `max_num_batched_tokens` / `max_num_seqs` 开好，每轮只覆盖前 `[:num_tokens]` /
`[:num_reqs]`，`data_ptr()` 稳定；自回归步骤复用同一块工作区，不再重复计入输入预算。

### KV：共用一个 group，但每层是自己的 tensor

draft 与 target 共用**逻辑块表**与分配生命周期（同一张块表、同一套 slot 编号），
但每个 Attention 层绑自己的物理 tensor。所以同一块号在两边代表同一段位置，**不是同一份 K/V**。
加载时校验规格（词表、dtype、KV head 数、head_size、block_size），不兼容就明确报错——
不假装"所有小模型都能配对"。

### 本关的差异（写清楚，不假装已实现）

- `num_lookahead_tokens` = K：草稿里"下一轮才验证"的那 K 枚写在 target 本轮 query **之外**，
  Scheduler 分配块时多留 K 个槽位；上下文快满时预留会被 `max_model_len` 截掉，所以自回归
  每写一枚前还要过逻辑上界 + `BlockTable.covers` 两道检查（205 §3），过不了就少提几枚。
- 仍然**没有** vLLM 的 EAGLE / MTP / 并行提议（PARD/DFlash）分支：`max_num_new_slots_for_drafting`
  只有普通 draft 的 1 与 ngram 的 0 两条路径；`is_masked_token_mask` 缓冲留着但对齐的是
  "并行提议的多 query 槽位"，本关恒为 False。
- 中间 prefill 块**不提草稿**（只同步 KV）：vLLM 跑完 drafter 再让 Scheduler 丢掉草稿，
  本关在采样阶段就不采（少跑 K 次试探性前向，57E 起就记在差异账本里）。
- draft 不共享 target 的 random stream：自己按 `seed` 建 generator（可复现），
  与 target 的采样流相互独立（vLLM 也把 draft 的随机数分开算）。
- 不做 CUDA Graph / 编译（那是 69 关）：本关只保证输入工作区稳定复用，不要求中间算子零分配。
"""

import torch

from ..outputs import DraftTokenIds
from ..sample import Sampler
from ..sample.metadata import SAMPLING_EPS
from ..sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from ..attention import Attention, AttentionMetadataBuilder, set_forward_context
from ..attention.backends.torch_sdpa import TorchAttentionBackend
from .utils import (DraftInputRows, FirstPassPlan, TargetRows, compute_new_slot_mapping,
                    expand_draft_inputs, extend_all_queries_by_N)


def _dtype(name) -> torch.dtype:
    """配置里的 dtype 字符串 → torch dtype（本关只用到这两种）。"""
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}.get(str(name), torch.float32)


class SpecDecodeBaseProposer:
    """提议步骤的骨架：**没有调度、没有请求状态**，只有"本轮哪些行进、草稿怎么出"。

    63/65 关：EAGLE 系（含 MTP）的提议者还吃 target 的 hidden states，所以这里留了两个开关——
    `pass_hidden_states_to_model`（第一遍要传特征，上游是构造参数）与
    `model_returns_tuple()`（模型返回 `(hidden_for_logits, hidden_for_next_step)` 两个张量还是
    一个，上游同名方法；**按家族不同**：EAGLE3 与 DeepSeek/Kimi 的 MTP 是两个，
    Qwen3-Next 的 MTP 是一个）。
    """

    # 普通 draft 只吃 token；EAGLE 子类改成 True（上游 `EagleProposer.__init__` 传的就是它）
    pass_hidden_states_to_model = False
    # 69 关：这个提议者收不收"padded 批"的两个逐请求索引
    # （`token_indices_to_sample` / `num_rejected_tokens_gpu`，上游 `prepare_inputs_padded()`
    # 的产物）。所有 LLM 系提议者（draft_model / EAGLE / MTP）都收；ngram / suffix /
    # medusa / extract / 用户插件不吃 token 与特征，没有这两个概念。
    supports_padded_first_pass = True

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        self.spec_config = spec_config
        self.method = spec_config.method      # 上游 `SpecDecodeBaseProposer` 同样存这份
        self.vllm_config = vllm_config
        self.device = device
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.block_size = vllm_config.cache_config.block_size
        # **逻辑**上界。块表容量是按块向上取整的（10 个位置可能给 12 个槽位），
        # 所以"物理槽位够"不等于"模型允许写这个位置"——两个边界要分别检查（205 §3）。
        self.max_model_len = vllm_config.model_config.max_model_len
        # 普通自回归 draft：第一遍比 target query 多要 1 行输入（= 新采出的那个 token）
        self.num_new_slots_per_request = spec_config.max_num_new_slots_for_drafting
        # 72 关（并行提议）：与上游 `llm_base_proposer.py:112-119` 逐行同义，**两个量不要混**：
        #   extra_slots_per_request       这一块里"要采样"的行数 = 1 个锚点 + (K−1) 个 mask
        #   net_num_new_slots_per_request 比 target 已经给的那一行**多占**几行
        # 串行时 extra=1（只采锚点）；net：draft=1（尾部扩容行是新行）、EAGLE=0（左移复用那一行）。
        self.parallel_drafting = bool(spec_config.parallel_drafting)
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens)
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0)
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0
        self.parallel_drafting_token_id = 0
        self.parallel_drafting_hidden_state_tensor = None
        # 67 关（TLI）：异构词表的映射表。基类默认 None = "draft 与 target 同词表"，
        # 只有 `DraftModelProposer` 在 `use_heterogeneous_vocab` 时建它（上游同款位置）。
        self.vocab_mapping = None
        self.use_heterogeneous_vocab = bool(getattr(spec_config, "use_heterogeneous_vocab", False))
        self.model = None                     # 子类加载
        self.kv_caches: dict[str, torch.Tensor] = {}
        # 草稿模型与 target 用同一个注意力后端（能力档位一致）；草稿侧的图属 74 关
        self.metadata_builder = TorchAttentionBackend.get_builder_cls()(self.block_size)
        self.sampler = Sampler()
        # **观测字段，不参与任何决策**（58 §5）：上一次第一遍把 draft 的 KV 覆盖到哪。
        # 58 之前的版本用它当"从哪开始补算"的第二套权威（新请求=0 于是命中前缀也整段重算）；
        # 现在起点由本轮调度快照的 `TargetRows.start` 决定，这里只留给测试/排查对账。
        self._draft_computed: dict[str, int] = {}
        # 自回归步的两个起点（每轮第一遍之后重设，见 `propose()`）：
        #   _ar_anchor_position  采样行自己的 position（第 k 枚草稿的位置 = 它 + k）
        #   _ar_base_seq_len     第一遍 seq_lens 减掉被拒行数（第 k 枚的上下文 = 它 + k）
        # 两个都**不能**从 `history_end` 之类的量反推：EAGLE 与 draft 的采样行不是同一行。
        self._ar_anchor_position: dict[str, int] = {}
        self._ar_base_seq_len: dict[str, int] = {}
        self._draft_generators: dict[str, torch.Generator] = {}
        # 63 关：AR 步要用的 draft 自己的 hidden（每请求一行，`_forward` 的第二返回值）
        self._ar_hidden: dict[str, torch.Tensor] = {}
        self.num_drafts_proposed = 0
        # ---------------- 固定输入工作区（58 §7） ----------------
        # 一次性开好、每轮只覆盖前 N 行/N 个请求：**不每轮新建张量**（`data_ptr()` 稳定，
        # 为后面的编译/CUDA Graph 留地基）。CPU 上 staging 与 device 侧是同一份；
        # CUDA 上先写 staging、再只上传有效前缀。
        scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = scheduler_config.max_num_seqs
        # 容量口径与 Scheduler 的 input_budget 起点**同一个配置**（58 §5），不另加一个
        # "调大一点逃避预算"的旋钮。
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_blocks_per_req = max(1, -(-self.max_model_len // self.block_size))

        def _buffer(size, dtype):
            cpu = torch.zeros(size, dtype=dtype)
            if device == "cpu":
                return cpu, cpu
            return cpu, torch.zeros(size, dtype=dtype, device=device)

        self.input_ids_cpu, self.input_ids = _buffer((self.max_num_tokens,), torch.int64)
        self.positions_cpu, self.positions = _buffer((self.max_num_tokens,), torch.int64)
        self.slot_mapping_cpu, self.slot_mapping = _buffer((self.max_num_tokens,), torch.int64)
        self.is_rejected_token_mask_cpu, self.is_rejected_token_mask = _buffer(
            (self.max_num_tokens,), torch.bool)
        # 并行提议（DFlash/PARD）才用到的 mask；普通 draft 恒 False，缓冲先留着对齐布局
        self.is_masked_token_mask_cpu, self.is_masked_token_mask = _buffer(
            (self.max_num_tokens,), torch.bool)
        self.query_start_loc_cpu, self.query_start_loc = _buffer((self.max_num_reqs + 1,),
                                                                 torch.int64)
        self.seq_lens_cpu, self.seq_lens = _buffer((self.max_num_reqs,), torch.int64)
        self.block_table_cpu, self.block_table = _buffer(
            (self.max_num_reqs, self.max_blocks_per_req), torch.int64)
        # 63 关：EAGLE 的第一遍要带 target 的 hidden states（工作区定长、原位覆盖）
        hidden_size = int((vllm_config.model_config.hf_config or {}).get("hidden_size", 0))
        self.hidden_size = hidden_size
        # EAGLE3 收的是**多个辅助层拼接后**的特征（宽度 = hidden × 辅助层数），EAGLE-1 是 1 层
        draft_hf = (spec_config.draft_model_config.hf_config or {}) \
            if spec_config.draft_model_config is not None else {}
        aux_ids = draft_hf.get("eagle_aux_hidden_state_layer_ids") or \
            (draft_hf.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
        num_aux = len(aux_ids) if aux_ids else int(draft_hf.get("num_aux_layers", 1) or 1)
        self.num_aux_layers = max(int(num_aux), 1)
        dtype = _dtype(vllm_config.model_config.dtype)
        # 模型收到的是**投影后**的特征（宽度 = draft 的 hidden）：多辅助层的拼接与 fc 投影
        # 由提议者在写缓冲之前用 `combine_hidden_states()` 完成（上游同在 proposer 里做）
        self.hidden_states_cpu, self.hidden_states = _buffer(
            (self.max_num_tokens, hidden_size), dtype)
        # 72 关：并行提议的 mask token / mask hidden（上游 `_init_parallel_drafting_params()`，
        # 调用点同在基类 `__init__` 的末尾）。缺字段当场报错——静默用一个错误的 mask token
        # 不会报错，只会让 K 枚草稿全部基于错的输入。
        if self.parallel_drafting:
            self._init_parallel_drafting_params()

    # -------- 交给子类 --------

    def load_model(self) -> None:
        raise NotImplementedError

    def model_returns_tuple(self) -> bool:
        """模型是否返回 `(for_logits, for_next_step)` 两个 hidden（上游 `llm_base_proposer.py:1015`）。

        上游的规则：`method == "mtp"` 时只有 `DeepSeekMTPModel`/`KimiK3MTPModel` 返回两个；
        其余 EAGLE 系都返回两个，`draft_model`/`dflash` 返回一个。本仓库实现了
        EAGLE3（两个）与 Qwen3 家族的 MTP（一个，与上游 `Qwen3NextMTP` 一致）。
        """
        return False

    # -------- 69 关：draft 侧的图模式 --------

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        """draft 侧的图键（对应上游 `llm_base_proposer.py:419-434`）。

        上游规则：`mixed_mode()` 是 PIECEWISE/FULL 时给 draft 建 **PIECEWISE** 键，否则 NONE
        ——也就是说 **EAGLE 系只在分段图下走图**（它的自回归循环每步形状都不同，只有"注意力
        拆到图外"的分段图能容纳）。

        本仓库没有 PIECEWISE（`CompilationConfig` 构造期就拒绝），所以这条规则退化成**恒 NONE**：
        draft 侧不做图。这里把原因与"上游会怎么做"都记在对象上，而不是沉默地什么都不做——
        `self.cudagraph_mode` 是"draft 这一侧到底有没有图"的唯一答案，测试盯它。
        """
        from ..config import CUDAGraphMode

        self.requested_cudagraph_mode = cudagraph_mode
        needs_graph = cudagraph_mode.mixed_mode() in (CUDAGraphMode.PIECEWISE,
                                                      CUDAGraphMode.FULL)
        self.cudagraph_mode = CUDAGraphMode.NONE
        if needs_graph:
            # 不 raise：target 侧照样有图可用（FULL_DECODE_ONLY 下 mixed_mode() 本来就是 NONE，
            # 走到这里说明用户显式要了含 mixed 图的模式）。draft 侧回退 eager 这件事是**配置的
            # 结果**，而且被 `self.cudagraph_mode` 与文档三态矩阵记下来了，不是静默降级。
            self.cudagraph_unsupported_reason = (
                "draft 侧的图要求 PIECEWISE（上游 eagle 只支持分段图），本仓库未实现该模式")
        else:
            self.cudagraph_unsupported_reason = None

    # -------- 提议 --------

    def propose(self, rows: list[TargetRows], all_token_ids: dict[str, list[int]], input_batch,
                reset_req_ids: set[str] | None = None,
                target_hidden_states: dict[str, torch.Tensor] | None = None,
                target_token_ids: torch.Tensor | None = None,
                target_positions: torch.Tensor | None = None,
                token_indices_to_sample: torch.Tensor | None = None,
                num_rejected_tokens_gpu: torch.Tensor | None = None,
                num_speculative_tokens: int | None = None) -> DraftTokenIds:
        """对每个被调度的请求都跑一遍：**同步 KV**，并给其中 ready 的那些提草稿。

        `rows` 是本轮 target 侧的事实（`TargetRows`：起点 `start`、本轮行数 `target_rows`、
        被拒数 `num_rejected`、采样后有效历史末尾 `history_end`）。**起点直接来自调度快照**
        ——prefix 命中过的请求 `start` 就是命中末尾，draft 不再从位置 0 重算（58 §4/§6）。

        中间 prefill 块也要同步（`ready=False`）：同一个逻辑块在 target/draft 的每一层都有
        各自的 tensor，draft 没写过的那一层会让这个块不能算"完整可复用"，而 target 的完整块
        这一轮就会发布出去（199 §9）。**草稿只给 ready 的请求提**：中间 prefill 块没有可验证的
        next token，提了 Scheduler 也会丢（vLLM 的 `update_draft_token_ids` 同款规则）。

        **71 关：`num_speculative_tokens` 是本轮要提几枚**（`None` = 用配置的 K，直接调用本方法的
        单测不受影响）。上游在 `propose()` 开头就是 `self.num_speculative_tokens =
        num_speculative_tokens`（`llm_base_proposer.py:533`），本仓库照抄这个**逐轮改写**：
        自回归循环的上界、`_apply_num_rejected_to_seq_lens` 的 K>1 分支都读它。
        容量（工作区/KV lookahead/块表）仍然按配置的最大 K 开，**不随 K 重建**。
        **K=0 不是"什么都不做"**：第一遍照样跑（draft 的 KV 必须继续与 target 同源同步），
        只是不提任何草稿就返回——上游同款（"the prefill forward pass above already ran to
        keep the drafter KV cache in sync, so just return an empty tensor"）。跳过第一遍会让
        之后恢复 K>0 的那一轮缺历史（不报错，只是草稿质量变差）。

        **生命周期**（205 §4.4，与 Controller 的分工）：没被调度的请求根本不进 `rows`，
        状态保留；抢占恢复（`reset_req_ids`）只重置进度、随机流继续；结束/abort 由
        `remove_requests()` 显式删除。
        """
        if num_speculative_tokens is None:
            num_speculative_tokens = self.spec_config.num_speculative_tokens
        if not 0 <= num_speculative_tokens <= self.spec_config.num_speculative_tokens:
            raise ValueError(
                f"本轮 K={num_speculative_tokens} 超出 [0, "
                f"{self.spec_config.num_speculative_tokens}]：工作区/KV 预留按最大 K 开，"
                f"逐轮 K 只能是它的前缀（调度侧的表也被这个上界裁剪）")
        # 上游同款：本轮的 K 写在实例上，自回归循环与 seq_lens 修正都读它
        self.num_speculative_tokens = num_speculative_tokens
        self._reset_requests(reset_req_ids or set())
        req_ids = [target.req_id for target in rows]
        # 69 关：把 Runner 按上游口径算出来的两个索引收下（长度 = 本轮批的请求数）。
        # **必须逐值对齐**：它代表"从哪一行取 hidden 采样 / 有几行是被拒的 padding"，
        # 与 CPU 侧自己算的那份不一致时，说明两套坐标系已经分叉——那种错不会报错，
        # 只会让草稿基于另一个 token 的条件产生（草稿质量悄悄变差，甚至验证出错误的候选）。
        self.last_padded_inputs = None
        if token_indices_to_sample is not None:
            if token_indices_to_sample.shape[0] != len(rows):
                raise ValueError(
                    f"token_indices_to_sample 长度 {token_indices_to_sample.shape[0]} 与本轮 "
                    f"请求数 {len(rows)} 不一致：padded 索引必须按批行序逐请求给")
            self.last_padded_inputs = {
                "token_indices_to_sample": token_indices_to_sample.tolist(),
                "num_rejected_tokens_gpu": (None if num_rejected_tokens_gpu is None
                                           else num_rejected_tokens_gpu.tolist()),
            }
        drafts: dict[str, list[int]] = {req_id: [] for req_id in req_ids}
        probs: dict[str, list[torch.Tensor]] = {req_id: [] for req_id in req_ids}
        if not rows:
            return DraftTokenIds(req_ids=[], draft_token_ids=[], draft_probs=None)

        # ---- 第一遍：把 [start, history_end) 这段写进 draft 的 KV（含 prefix 命中的跳过）----
        self._fill_block_table_rows([target.row for target in rows], input_batch.block_table)
        plan = self.set_inputs_first_pass(rows, all_token_ids, target_hidden_states,
                                          target_token_ids, target_positions)
        hidden = self._forward(plan.num_tokens, plan.num_reqs)
        hidden = self._split_hidden(hidden, plan, rows)
        for target in rows:
            self._draft_computed[target.req_id] = target.history_end      # 观测用
        self._check_padded_sample_rows(plan, rows)
        self._apply_num_rejected_to_seq_lens(plan, rows)
        # 自回归步的起点 = 第一遍**采样行自己的位置**（上游同源：`positions =
        # self.positions[token_indices_to_sample]`）。K=0 时没有自回归步，但这两个字典照样
        # 填好——它们只是"这一轮第一遍的事实"，不参与任何判断。
        self._ar_anchor_position = dict(zip(plan.sample_req_ids, plan.sample_positions))
        # 71 关：K=0 时**第一遍已经跑完**（KV 与 target 同步过了），但一枚草稿都不采——
        # 采样会消耗随机流、也会白算一次 lm_head；上游在同样的位置直接返回 `[B, 0]`。
        if self.num_speculative_tokens == 0:
            return DraftTokenIds(
                req_ids=list(req_ids),
                draft_token_ids=[[] for _ in req_ids],
                draft_probs=None)
        if self.parallel_drafting:
            # ---- 72 关（PARD / P-EAGLE）：**一次** forward 出 K 枚 ----
            # 上游同一处（`llm_base_proposer.py:627-632`）：`K == 1 or parallel_drafting` 共用
            # 一条早退分支——在"锚点 + K−1 个 mask"这 K 行上各采一枚，然后 `view(-1, K)`。
            # 采样行按"请求序 × 行内序"给出（`_parallel_first_pass()` 保证），所以每条请求拿到的
            # 就是 [第1枚, 第2枚, …, 第K枚]；q 也**逐位置**保存（不能广播同一行）。
            if plan.sample_rows:
                self._sample_draft_tokens(hidden,
                                          list(zip(plan.sample_req_ids, plan.sample_rows)),
                                          input_batch, drafts, probs)
            probs_rows = [row for req_id in req_ids for row in probs[req_id]]
            self.num_drafts_proposed += sum(len(drafts[req_id]) for req_id in req_ids)
            draft_probs = None if (not probs_rows or any(row is None for row in probs_rows)) \
                else torch.stack(probs_rows)
            return DraftTokenIds(
                req_ids=list(req_ids),
                draft_token_ids=[drafts[req_id] for req_id in req_ids],
                draft_probs=draft_probs)
        if plan.sample_rows:
            self._sample_draft_tokens(hidden,
                                      list(zip(plan.sample_req_ids, plan.sample_rows)),
                                      input_batch, drafts, probs)

        # ---- 自回归补足 K 枚：上一枚当输入，位置接在它后面（**复用同一个工作区**）----
        #
        # 起点 = **第一遍采样行自己的 position**（`plan.sample_positions`），之后每步 +1：
        # 上游就是 `positions = self.positions[token_indices_to_sample]` 再交给
        # `_update_positions_dependent_metadata()`（每步 `position + 1`）。
        # 为什么不能在外面反推：两种布局的采样行不同——EAGLE 是 target 行块的**最后一行**
        # （位置原样、内容被换成新采样的 token），draft 是尾部**扩容行**；用 `history_end`
        # 之类的量反推只在 `rejected == 1` 时巧合相等，其它情况会把自回归行写到错的位置
        # （实测 `rejected=3` 时落回第一遍刚写过的行，覆盖掉刚写的 KV）。
        #
        # 第 k 枚草稿写在采样行位置 + k，那是 target 本轮 query 之外的位置：
        # 调度侧为此预留了 `num_lookahead_tokens` 个 KV 槽位。但预留可能被 `max_model_len`
        # 截掉、上下文也可能刚好走到尽头，所以每写一枚都要过**两个**边界：模型自己的位置范围
        # （逻辑）与块表覆盖（物理）。过不了就少提几枚——草稿只是候选，不写就不会越界。
        while True:
            pending: list[tuple[TargetRows, int]] = []
            for target in rows:
                req_id = target.req_id
                if not 0 < len(drafts[req_id]) < self.num_speculative_tokens:
                    continue
                anchor = self._ar_anchor_position.get(req_id)
                if anchor is None:
                    continue
                position = anchor + len(drafts[req_id])
                if not 0 <= position < self.max_model_len:
                    continue
                if not input_batch.block_table.covers(target.row, position):
                    continue
                pending.append((target, position))
            if not pending:
                break
            self._set_autoregressive_inputs(pending, drafts, input_batch)
            hidden = self._forward(len(pending), len(pending))
            if self.model_returns_tuple():
                logits_hidden, next_hidden = hidden
                for index, (target, _) in enumerate(pending):
                    self._ar_hidden[target.req_id] = next_hidden[index]
                hidden = logits_hidden
            elif self.pass_hidden_states_to_model:
                # 单返回值的 EAGLE 系（本仓库的 MTP-Qwen3）：**同一个张量既是下一步要回灌的
                # hidden，也是过 lm_head 的那份**（上游 base proposer 里
                # `last_hidden_states = hidden_states = ret_hidden_states`）。
                # 不记下来的话，AR 步会一直用第一遍的 hidden——草稿仍然"能跑"，但第 2 枚起
                # 就不再条件于第 1 枚了（最典型的一类静默错，tests/step65 用错位反证盯着）。
                for index, (target, _) in enumerate(pending):
                    self._ar_hidden[target.req_id] = hidden[index]
            self._sample_draft_tokens(
                hidden, [(target.req_id, index) for index, (target, _) in enumerate(pending)],
                input_batch, drafts, probs)

        probs_rows = [row for req_id in req_ids for row in probs[req_id]]
        self.num_drafts_proposed += sum(len(drafts[req_id]) for req_id in req_ids)
        # TLI 的草稿是点质量（`None` 占位）：只要有一个 None，整批就不带 q（59 关的
        # NO_DRAFT_PROBS 分支——点质量提议本来就不需要 q），不能把 draft 空间的概率混进来。
        draft_probs = None if (not probs_rows or any(row is None for row in probs_rows)) \
            else torch.stack(probs_rows)
        return DraftTokenIds(
            req_ids=list(req_ids),
            draft_token_ids=[drafts[req_id] for req_id in req_ids],
            draft_probs=draft_probs)

    def _split_hidden(self, hidden, plan: FirstPassPlan, rows: list[TargetRows]):
        """拆开 `(for_logits, for_next_step)`（tuple 模型）／记住 AR 要用的那份（单返回值模型）。

        第一遍每请求的采样行在 `plan.sample_rows`（= 扩容行的全局行号），AR 步用的 hidden 就是
        这一行对应请求的那份。
        """
        if not self.model_returns_tuple():
            if self.pass_hidden_states_to_model:
                for req_id, row in zip(plan.sample_req_ids, plan.sample_rows):
                    self._ar_hidden[req_id] = hidden[row]
            return hidden
        logits_hidden, next_hidden = hidden
        for req_id, row in zip(plan.sample_req_ids, plan.sample_rows):
            self._ar_hidden[req_id] = next_hidden[row]
        return logits_hidden

    # -------- 第一遍输入（对应上游 set_inputs_first_pass） --------

    def set_inputs_first_pass(self, rows: list[TargetRows],
                              all_token_ids: dict[str, list[int]],
                              target_hidden_states=None,
                              target_token_ids: torch.Tensor | None = None,
                              target_positions: torch.Tensor | None = None) -> FirstPassPlan:
        """把第一遍的输入写进工作区，返回物理行数、采样行与各请求的 AR 起点。

        每条请求的物理行 = [有效行 (n - num_rejected)] + [1 行扩容行] + [被拒行]，展开规则与
        上游 `copy_and_expand_eagle_inputs_kernel`（`shift_input_ids=False`）一致，见
        `spec_decode/utils.py`；槽位用上游同款 `compute_new_slot_mapping()` 算，query/seq
        长度用 `extend_all_queries_by_N()` 扩。

        **拒绝尾部留在工作区里但被屏蔽**：token 取 padding、position 取 0、slot 取哨兵，
        于是它既不写 KV、也不进新提议的上下文（58 §6）。
        """
        if self.parallel_drafting:
            # 72 关（PARD）：不左移 —— 有效行照抄、锚点与 K−1 个 mask 接在后面（一次 forward 出 K 枚）
            return self._parallel_first_pass(rows, target_hidden_states, target_token_ids,
                                             target_positions)
        input_rows = []
        for target in rows:
            tokens = all_token_ids[target.req_id]
            valid_token_ids = [int(tokens[position])
                               for position in range(target.start,
                                                     target.start + target.num_valid)]
            next_token_id = target.next_token_id
            if self.vocab_mapping is not None:
                # 67 关（TLI）：喂进 draft 的 token 必须是**草稿空间**的 id（上游
                # `set_inputs_first_pass()` 里同一处：`map_target_to_draft_ids(target_token_ids / next_token_ids)`）。
                # 不在交集里的历史 token 按 unk→eos→报错 处理（映射表里已经填好兜底 id）。
                valid_token_ids, next_token_id = self._to_draft_space(
                    valid_token_ids, next_token_id)
            input_rows.append(DraftInputRows(
                valid_token_ids=valid_token_ids,
                start=target.start,
                next_token_id=next_token_id,
                num_rejected=target.num_rejected))
        input_ids, positions, is_rejected, sample_indices = expand_draft_inputs(input_rows)
        num_tokens = len(input_ids)
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"draft 第一遍要 {num_tokens} 行，超过输入工作区 {self.max_num_tokens} 行："
                f"Scheduler 的 input_budget 没兜住，属于控制面/执行面口径不一致（不是模型问题）")
        query_lens = [target.target_rows for target in rows]
        # target 的 query_start_loc 与 seq_lens，交给 extend_all_queries_by_N 各 +1 行/+1 长度
        query_start_loc = [0]
        for length in query_lens:
            query_start_loc.append(query_start_loc[-1] + length)
        seq_lens = [target.start + target.target_rows for target in rows]
        query_start_loc, seq_lens = extend_all_queries_by_N(
            query_start_loc, seq_lens, self.num_new_slots_per_request)

        self.input_ids_cpu[:num_tokens] = torch.tensor(input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = torch.tensor(positions, dtype=torch.int64)
        self.is_rejected_token_mask_cpu[:num_tokens] = torch.tensor(is_rejected, dtype=torch.bool)
        self.query_start_loc_cpu[:len(query_start_loc)] = torch.tensor(query_start_loc,
                                                                      dtype=torch.int64)
        self.seq_lens_cpu[:len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int64)
        # slot mapping 在 **CPU** 上算：块表镜像是 CPU 结构，索引也必须是 CPU 张量
        # （否则会撞上 "Expected all tensors to be on the same device"，204 §5）。
        self.slot_mapping_cpu[:num_tokens] = compute_new_slot_mapping(
            self.block_table_cpu[:len(rows)], query_lens, self.positions_cpu[:num_tokens],
            self.is_rejected_token_mask_cpu[:num_tokens], self.block_size,
            self.num_new_slots_per_request, self.max_model_len)
        self._check_valid_positions(rows)
        ready = [index for index, target in enumerate(rows) if target.ready]
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=len(rows),
            sample_rows=[sample_indices[index] for index in ready],
            sample_req_ids=[rows[index].req_id for index in ready],
            # 采样行（= 扩容行）自己的 position：自回归步从这里 +1 开始（上游同款）
            sample_positions=[positions[sample_indices[index]] for index in ready],
            seq_lens=list(seq_lens),
            history_end={target.req_id: target.history_end for target in rows})

    def _check_padded_sample_rows(self, plan: FirstPassPlan, rows: list[TargetRows]) -> None:
        """把 device 侧的 `token_indices_to_sample` 与 CPU 侧的事实按同一公式对一遍。

        **两套坐标系的区别要说清楚**（这是本关最容易搞错的一处）：

            device（上游 `eagle_prepare_inputs_padded_kernel`）：索引 **target 的输入行**
                —— "最后一枚有效 token 所在的行"。每条请求的 target 块是 `[b, d1..dK]`（K+1 行），
                所以行号 = `块起点 + target_rows - 1 - num_rejected`。
            本仓库的 draft 工作区：索引 **第一遍拼出来的行**。两种提议者的布局还不一样：
                `DraftModelProposer`：`[有效行][扩容行][被拒行]` → 采样行 = 块起点 + num_valid
                `EagleProposer`      ：沿用 target 的行块、把扩容 token 打在**最后一行** →
                                       采样行 = 块起点 + target_rows - 1

        所以这里校验的是**协议事实与 TargetRows 事实算出的 target 行号必须一致**（两个独立来源：
        左边来自 `cu_num_draft_tokens` + 有效采样数，右边来自调度快照 + 记账结果）。
        工作区行号与 target 行号的换算由各自的 `set_inputs_first_pass()` 决定，测试里按布局分别钉住
        （`tests/step69/test_drafter_padding.py`）——**不在这里假设某一种布局**，否则这个校验就会
        变成"用布局 A 的公式去检查布局 B"，报一堆假警。
        """
        recorded = self.last_padded_inputs
        if recorded is None:
            return
        actual = recorded["token_indices_to_sample"]
        cursor = 0
        for index, target in enumerate(rows):
            if target.target_rows <= 0:
                raise RuntimeError(f"{target.req_id!r} 本轮的 target 行数为 0，无法定位采样行")
            want = cursor + target.target_rows - 1 - target.num_rejected
            cursor += target.target_rows
            got = actual[index]
            if got != want:
                raise RuntimeError(
                    f"{target.req_id!r} 的采样行两套口径不一致：device 侧（上游 "
                    f"prepare_inputs_padded 算法）说第 {got} 行，按本轮 TargetRows"
                    f"（起点 {target.start}、{target.target_rows} 行、被拒 {target.num_rejected} 行）"
                    f"算出来应是第 {want} 行。这会让第一枚草稿条件在错误的 token 上"
                    f"（不报错、只是草稿变差）")

    def _apply_num_rejected_to_seq_lens(self, plan: FirstPassPlan,
                                        rows: list[TargetRows]) -> None:
        """把被拒的 padding 行从 draft 的上下文长度里减掉（上游同名动作）。

        上游（`llm_base_proposer.py:654-660`）：

            if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
                common_attn_metadata.seq_lens -= num_rejected_tokens_gpu

        为什么必须减：第一遍的 `seq_lens` 是**乐观值**（把 K 枚草稿全算上），而被拒的那几行
        既没写 KV、也不该成为上下文。不减的话，自回归步的 query 位置会落在"从未写过的位置"上
        （读到上一轮的残留 KV），草稿的条件就是错的——不报错，只是草稿变差。

        起点是**第一遍自己的 `seq_lens`**（`plan.seq_lens`），不是外面重算的公式：
        上游减的就是 `common_attn_metadata.seq_lens`（第一遍用的那份）。EAGLE 的第一遍
        `seq_lens` 不含额外槽位（net=0），draft 的含 1 个扩容行——用同一个公式反推会在其中
        一种布局上多算一行。

        为什么在**第一遍之后**才减：第一遍的 query 行是本轮的连续排布，
        `seq_len - query_len` 必须等于本轮的起点；减了它，第一遍的位置假设就塌了。
        自回归步每请求只有 1 行，减完正好是"真历史"的长度，之后每步 +1。

        本仓库的 AR 步位置是由采样行位置 +1 递推的，所以这里的产物是**上下文长度**：
        设备侧的 `seq_lens` 也要一起改（AR 步的内核按"上一步 + 1"更新它），否则 device 与
        CPU 两套账会在自回归步里差出被拒行数。
        """
        plan.num_rejected = [target.num_rejected for target in rows]
        recorded = self.last_padded_inputs
        if recorded is not None and recorded["num_rejected_tokens_gpu"] is not None:
            device_rejected = recorded["num_rejected_tokens_gpu"]
            if device_rejected != plan.num_rejected:
                raise RuntimeError(
                    f"被拒行数两套口径不一致：device 侧（上游 prepare_inputs_padded 算法）"
                    f"{device_rejected}，CPU 侧（58 关的 TargetRows）{plan.num_rejected}："
                    f"draft 的上下文长度会按错误的值修剪")
        if self.num_speculative_tokens <= 1:
            # 上游同款条件：K=1 时没有自回归步，第一遍的乐观长度就是最终长度
            return
        if len(plan.seq_lens) != len(rows):
            raise RuntimeError(
                f"第一遍的 seq_lens 有 {len(plan.seq_lens)} 项，本轮请求数 {len(rows)}："
                f"`set_inputs_first_pass()` 必须逐请求填全（否则上下文长度会张冠李戴）")
        for index, target in enumerate(rows):
            base = plan.seq_lens[index] - target.num_rejected
            self._ar_base_seq_len[target.req_id] = base
            self.seq_lens_cpu[index] = base
        if self.device != "cpu":
            self.seq_lens[:len(rows)].copy_(self.seq_lens_cpu[:len(rows)])

    # -------- 72 关：并行提议（PARD / P-EAGLE） --------

    def _init_parallel_drafting_params(self) -> None:
        """解析并行提议要的两个东西（上游 `_init_parallel_drafting_params()`，L350-379）。

        1. **mask token**：并行槽位填哪个 token id。上游按固定顺序找**模型自带**的字段，
           一个都没有就报错（不许猜）：

               dflash_config.mask_token_id → mask_token_id → dspark_noise_token_id
               → pard_token → ptd_token_id

           为什么必须来自 checkpoint：这个 token 是模型**训练时**约定的占位符（PARD 的
           `pard_token`、P-EAGLE 的 `mask_token_id`），换一个 token 不会报错，只会让 K 枚草稿
           全部基于模型没见过的输入。
        2. **mask hidden**（只有吃 target hidden 的 EAGLE 系要）：并行槽位的 hidden 要换成
           模型自带的常量向量 `mask_hidden`（`load_model()` 之后从模型缓冲里取，见
           `_maybe_fill_parallel_drafting_hidden_state()`）。
        """
        draft_hf = (self.spec_config.draft_model_config.hf_config or {}) \
            if self.spec_config.draft_model_config is not None else {}
        dflash_config = draft_hf.get("dflash_config") or {}
        candidates = (
            ("dflash_config.mask_token_id", dflash_config.get("mask_token_id")),
            ("mask_token_id", draft_hf.get("mask_token_id")),
            ("dspark_noise_token_id", draft_hf.get("dspark_noise_token_id")),
            ("pard_token", draft_hf.get("pard_token")),
            ("ptd_token_id", draft_hf.get("ptd_token_id")),
        )
        for name, value in candidates:
            if value is not None:
                self.parallel_drafting_token_id = int(value)
                break
        else:
            raise ValueError(
                "开了 parallel_drafting 但 draft 配置里找不到 mask token：上游要求 "
                "`dflash_config.mask_token_id` / `mask_token_id` / `dspark_noise_token_id` / "
                "`pard_token` / `ptd_token_id` 至少有一个（并行槽位要填模型训练时约定的占位符，"
                "猜一个不会报错、只会让 K 枚草稿全部基于错的输入）")
        if self.pass_hidden_states_to_model:
            # 上游：`torch.empty(self.hidden_size, dtype, device)`；值在模型装好之后从
            # `self.model.mask_hidden` 取（见 `_maybe_fill_parallel_drafting_hidden_state()`）
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=_dtype(self.vllm_config.model_config.dtype),
                device=self.device if self.device != "cpu" else "cpu")

    def _maybe_fill_parallel_drafting_hidden_state(self) -> None:
        """把 `mask_hidden` 从 draft 模型搬进 mask 缓冲（上游 `load_model()` 末尾那段，L1416-1424）。

        上游的 EAGLE3 draft 在 `parallel_drafting=True` 时会注册一个**非持久** buffer
        `mask_hidden`（形状 `(1, fc_input_size)`），权重文件里必须带这一项，否则加载期直接报错
        （"mask_hidden not found in weights but model is configured for parallel drafting"）——
        也就是说**串行训练的 EAGLE3 权重不能拿来开并行**，上游用加载期错误挡住了这件事。
        本仓库照抄这条边界：模型没带 `mask_hidden` 就在这里报错，不静默用一个零向量。
        """
        if not (self.parallel_drafting and self.pass_hidden_states_to_model):
            return
        model = self.model
        mask_hidden = getattr(model, "mask_hidden", None)
        if mask_hidden is None:
            raise ValueError(
                "parallel_drafting=True 的 EAGLE 系 draft 必须在权重里带 `mask_hidden`"
                "（上游模型在并行模式下注册这个 buffer、加载器找不到就直接报错）。"
                "本机这份 draft 权重是**串行训练**的：不能用它做并行提议，"
                "请换成按并行草稿训练的 checkpoint（需求 072 §4 最后一条）")
        flat = mask_hidden.reshape(-1).to(dtype=self.parallel_drafting_hidden_state_tensor.dtype,
                                          device=self.parallel_drafting_hidden_state_tensor.device)
        if self.method == "eagle3" and hasattr(model, "combine_hidden_states"):
            # EAGLE3：mask_hidden 存的是**多个辅助层**的拼接，先过 fc 投影（上游同款）
            flat = model.combine_hidden_states(flat.reshape(1, -1)).reshape(-1)
        if flat.numel() != self.parallel_drafting_hidden_state_tensor.numel():
            raise ValueError(
                f"mask_hidden 投影后是 {flat.numel()} 维，缓冲是 "
                f"{self.parallel_drafting_hidden_state_tensor.numel()} 维：draft 的 hidden_size "
                f"配置与权重不一致")
        self.parallel_drafting_hidden_state_tensor.copy_(flat)

    def _parallel_first_pass(self, rows: list[TargetRows], target_hidden_states,
                             target_token_ids, target_positions) -> FirstPassPlan:
        """并行提议的第一遍：**一次**排出 `[有效行][锚点][K−1 mask][被拒行]`（72 关）。

        与串行的区别只在布局（上游用同一个 kernel，靠 `shift_input_ids` 分叉）：

            P-EAGLE（shift=True，吃 hidden）：跳过 target 块第 0 个 token，锚点复用最后一行
            PARD   （shift=False，不吃 hidden）：整块原样复制，锚点与 mask 接在后面

        采样行 = 锚点 + 所有 mask 行（共 K 行），**按"请求序 × 行内序"排列** —— 这正是
        `propose()` 里一次 forward 之后能直接 `view(-1, K)` 的原因。

        与串行路径共享的不变量一条不少：输入与 positions 来自**本轮真正喂进 target 的那份缓冲**
        （63 关）、被拒尾部打 `is_rejected` 掩码（槽位是哨兵、不算上下文）、
        槽位用 `compute_new_slot_mapping(num_new_tokens=net)` 按新 positions 重算。
        """
        from .utils import ParallelFirstPass, expand_parallel_draft_inputs, \
            compute_new_slot_mapping, extend_all_queries_by_N

        if target_token_ids is None or target_positions is None:
            raise ValueError(
                "并行提议的第一遍需要本轮 target 的原始输入（target_token_ids / target_positions）："
                "上游 kernel 就是在这两份缓冲上展开的，本仓库不自己重建行")
        num_tokens_in = sum(int(target.target_rows) for target in rows)
        tokens = [int(t) for t in target_token_ids[:num_tokens_in]]
        positions_in = [int(p) for p in target_positions[:num_tokens_in]]
        if len(tokens) != num_tokens_in or len(positions_in) != num_tokens_in:
            raise RuntimeError(
                f"本轮 target 的输入行数 {len(tokens)} 与调度快照的行数之和 {num_tokens_in} "
                f"不一致：控制面/执行面口径不一致（不是模型问题）")

        expanded: ParallelFirstPass = expand_parallel_draft_inputs(
            tokens, positions_in, rows,
            extra_slots_per_request=self.extra_slots_per_request,
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            shift_input_ids=self.pass_hidden_states_to_model)
        num_tokens = expanded.num_tokens
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"并行第一遍要 {num_tokens} 行，超过输入工作区 {self.max_num_tokens} 行："
                f"Scheduler 的 input_budget 没按 net={self.net_num_new_slots_per_request} "
                f"兜住（控制面/执行面口径不一致）")

        self.input_ids_cpu[:num_tokens] = torch.tensor(expanded.input_ids, dtype=torch.int64)
        self.positions_cpu[:num_tokens] = torch.tensor(expanded.positions, dtype=torch.int64)
        self.is_rejected_token_mask_cpu[:num_tokens] = torch.tensor(expanded.is_rejected,
                                                                   dtype=torch.bool)
        self.is_masked_token_mask_cpu[:num_tokens] = torch.tensor(expanded.is_masked,
                                                                  dtype=torch.bool)
        query_start_loc = [0]
        for target in rows:
            query_start_loc.append(query_start_loc[-1] + int(target.target_rows))
        seq_lens = [int(target.start) + int(target.target_rows) for target in rows]
        query_start_loc, seq_lens = extend_all_queries_by_N(
            query_start_loc, seq_lens, self.net_num_new_slots_per_request)
        self.query_start_loc_cpu[:len(query_start_loc)] = torch.tensor(
            query_start_loc, dtype=torch.int64)
        self.seq_lens_cpu[:len(seq_lens)] = torch.tensor(seq_lens, dtype=torch.int64)
        self.slot_mapping_cpu[:num_tokens] = compute_new_slot_mapping(
            self.block_table_cpu[:len(rows)], [int(target.target_rows) for target in rows],
            self.positions_cpu[:num_tokens], self.is_rejected_token_mask_cpu[:num_tokens],
            self.block_size, self.net_num_new_slots_per_request, self.max_model_len)
        self._check_valid_positions(rows)

        if self.pass_hidden_states_to_model:
            self._write_parallel_hidden(rows, expanded, target_hidden_states)

        # 采样行：并行槽位按 extra 分组写出来，这里按"请求序 × 行内序"取出来
        sample_rows: list[int] = []
        sample_req_ids: list[str] = []
        for request_index, target in enumerate(rows):
            base = request_index * self.extra_slots_per_request
            for local in range(self.extra_slots_per_request):
                if not target.ready:
                    continue          # 中间 prefill 块不采样（上游会让 Scheduler 丢掉它的草稿）
                sample_rows.append(expanded.token_indices_to_sample[base + local])
                sample_req_ids.append(target.req_id)
        return FirstPassPlan(
            num_tokens=num_tokens, num_reqs=len(rows),
            sample_rows=sample_rows, sample_req_ids=sample_req_ids,
            sample_positions=[expanded.positions[row] for row in sample_rows],
            seq_lens=list(seq_lens),
            history_end={target.req_id: target.history_end for target in rows})

    def _write_parallel_hidden(self, rows: list[TargetRows], expanded, target_hidden_states) -> None:
        """把特征写进并行第一遍的缓冲（上游 `shift_input_ids=True` 分支的后半段）。

        两件事，顺序不能反：

            1. hidden **不跟着 token 移**：源行 `q_start + i`（target 的第 i 行）的特征写到
               目标行 `out_start + i`（`out_hidden_state_mapping` 给的映射）；
            2. **mask 行换成模型自带的常量向量** `mask_hidden`（并行槽位没有真实特征可用）。

        不做第 2 步不会报错，只会让 K 枚草稿全部条件在一个"看起来是特征、其实是上一行残留"的
        输入上——正是 63 关记过的那类静默错。
        """
        if target_hidden_states is None:
            raise ValueError("并行提议（吃 hidden 的 EAGLE 系）需要本轮 target 的特征")
        if self.parallel_drafting_hidden_state_tensor is None:
            raise RuntimeError("mask hidden 缓冲还没建：`_init_parallel_drafting_params()` 没跑过")
        hidden_rows: list[torch.Tensor] = []
        num_tokens = expanded.num_tokens
        for target in rows:
            rows_hidden = target_hidden_states[target.req_id]
            if self.method == "eagle3":
                rows_hidden = self.model.model.combine_hidden_states(rows_hidden)
            hidden_rows.extend(rows_hidden[i] for i in range(int(target.target_rows)))
        # **顺序不能反**（上游就是这两句的先后）：先把真实特征按映射搬过去，再用 mask 向量
        # 覆盖 mask 行。反过来的话，被拒行多的时候（`num_valid + 1 .. target_rows` 落在 mask 区
        # 里）映射会把 mask 行的常量向量又盖回真实特征——不报错，只是那些槽位的条件全错。
        # 先整批搬到 **staging 的设备/dtype**（切片赋值能跨设备拷贝，但**花式索引赋值不行**：
        # `cpu[mask_rows] = cuda_tensor` 会报 "Expected all tensors to be on the same device"）
        hidden_matrix = torch.stack(hidden_rows).to(
            device=self.hidden_states_cpu.device, dtype=self.hidden_states_cpu.dtype)
        for source_row, destination_row in expanded.hidden_state_mapping.items():
            self.hidden_states_cpu[destination_row] = hidden_matrix[source_row]
        mask_rows = [index for index, flag in enumerate(expanded.is_masked) if flag]
        if mask_rows:
            mask_row = self.parallel_drafting_hidden_state_tensor.to(
                device=self.hidden_states_cpu.device, dtype=self.hidden_states_cpu.dtype)
            self.hidden_states_cpu[mask_rows] = mask_row

    def _check_valid_positions(self, rows: list[TargetRows]) -> None:
        """内部错误检查（对应 205 §3 保留的那条断言）：有效行/扩容行必须在模型位置上界内。

        被拒行不在检查范围（它们是 padding，position=0，本来就不参与上下文）。
        """
        for target in rows:
            last = target.start + target.num_valid          # 扩容行的位置
            if not 0 <= last < self.max_model_len:
                raise RuntimeError(
                    f"{target.req_id!r} 的 draft 第一遍要写位置 {last}，超出 "
                    f"max_model_len={self.max_model_len}：调度快照与 draft 输入口径不一致")

    def _set_autoregressive_inputs(self, pending: list[tuple[TargetRows, int]],
                                   drafts: dict, input_batch) -> None:
        """后续自回归步骤：**复用同一个工作区的前 B 行**（每活跃请求一行，58 §7）。"""
        num_reqs = len(pending)
        tokens = [drafts[target.req_id][-1] for target, _ in pending]
        if self.vocab_mapping is not None:
            # 67 关（TLI）：`drafts` 里存的是**交回调度器的 target id**，而模型吃的是草稿空间的 id
            # （上游自回归循环里同一处：`input_ids = self.vocab_mapping.map_target_to_draft_ids(input_ids)`）。
            tokens = self._to_draft_space_tokens(tokens)
        positions = [position for _, position in pending]
        batch_rows = [target.row for target, _ in pending]
        self.input_ids_cpu[:num_reqs] = torch.tensor(tokens, dtype=torch.int64)
        self.positions_cpu[:num_reqs] = torch.tensor(positions, dtype=torch.int64)
        if self.pass_hidden_states_to_model:
            # 第 k 枚草稿的输入特征 = 第 k-1 枚那一步 draft 自己吐出的 hidden（上游同款）
            rows_hidden = [self._ar_hidden[target.req_id] for target, _ in pending]
            self.hidden_states_cpu[:num_reqs] = torch.stack(rows_hidden).to(
                self.hidden_states_cpu.dtype)
        # 槽位与 target 同一份公式、同一张块表（物理容量已由 covers() 确认过）
        self.slot_mapping_cpu[:num_reqs] = input_batch.block_table.compute_slot_mapping(
            input_batch.num_reqs, torch.tensor(positions, dtype=torch.int64),
            torch.tensor(batch_rows, dtype=torch.int64))
        self.is_rejected_token_mask_cpu[:num_reqs] = False
        self.query_start_loc_cpu[:num_reqs + 1] = torch.arange(num_reqs + 1, dtype=torch.int64)
        # 第 k 枚草稿的上下文 = 修正后的第一遍长度 + k（本步之前已经写进去 k-1 行，
        # 加上本步这一行）。上游同序：`seq_lens` 先减被拒数，再由
        # `eagle_step_update_slot_mapping_and_metadata()` 每步 +1。
        self.seq_lens_cpu[:num_reqs] = torch.tensor(
            [self._ar_base_seq_len[target.req_id] + len(drafts[target.req_id])
             for target, _ in pending], dtype=torch.int64)
        self._fill_block_table_rows(batch_rows, input_batch.block_table)
        # 69 关：上面几步就是上游 `_update_positions_dependent_metadata()` 里的那三件事
        # （位置 +1、查块表得槽位、上下文 +1），上游把它们融进一个 kernel。这里在 device 上用
        # 同一套算法**复算一遍并逐值校验**（见下面的方法），通过之后才让 `_upload()` 把
        # （相等的）staging 值推上去给模型。
        self._check_ar_metadata_on_device(pending, positions, num_reqs)

    def _check_ar_metadata_on_device(self, pending, positions: list[int],
                                     num_reqs: int) -> None:
        """AR 步的 `(positions, slot_mapping, seq_lens)` 用上游同款算法在 device 上更新。

        输入是**上一枚草稿的位置**（`self.positions` 里此刻存的就是它），函数把它们各 +1 写进
        同一块位置缓冲、按位置查块表得到槽位、并把 `seq_lens` 原地 +1 —— 与上游
        `eagle_step_update_slot_mapping_and_metadata()` 的调用形态逐条一致
        （上游 `_update_positions_dependent_metadata()` 就是这么调的）。

        CPU 上 staging 与工作区是同一份张量，所以这条分支只在 CUDA 上有意义；
        CPU 上直接返回（值已经由 `_set_autoregressive_inputs` 的 CPU 公式写好了）。
        """
        from .utils import eagle_step_update_slot_mapping_and_metadata

        if self.device == "cpu":
            return                      # CPU 上 staging 与工作区本就是同一份张量
        # 上游传的是"上一步选中的那些行的位置"（`positions = self.positions[token_indices_to_sample]`），
        # 也就是本步位置 - 1。**不能直接从 `self.positions` 读**：那块缓冲此刻装的是第一遍
        # 的整块位置（[0,1,2,...]），不是"上一步选中的行"。读错的表现是位置从 1 开始递增
        # （而不是从历史末尾开始）——不报错，但草稿全写在错的位置上。
        previous_positions = torch.tensor([position - 1 for position in positions],
                                          dtype=torch.int64, device=self.device)
        # 复算到 scratch：不动 live 缓冲（live 缓冲随后由 `_upload()` 写，值是这里校验过的）
        out_positions = torch.empty(num_reqs, dtype=torch.int64, device=self.device)
        out_slots = torch.empty(num_reqs, dtype=torch.int64, device=self.device)
        # `self.seq_lens` 此刻还是**上一步**的值（本步的值在 staging 里），正是内核的输入语义
        out_seq_lens = self.seq_lens[:num_reqs].clone()
        eagle_step_update_slot_mapping_and_metadata(
            previous_positions, self.block_table[:num_reqs], out_seq_lens,
            self.block_size, self.max_model_len, out_positions, out_slots,
            input_batch_size=num_reqs)
        # 与 CPU staging 比对：位置/槽位/长度三样都必须一样（不一样就是两套口径分叉）
        for name, device_values, cpu_values in (
                ("positions", out_positions, self.positions_cpu[:num_reqs]),
                ("slot_mapping", out_slots, self.slot_mapping_cpu[:num_reqs]),
                ("seq_lens", out_seq_lens, self.seq_lens_cpu[:num_reqs])):
            got = device_values.tolist()
            want = cpu_values.tolist()
            if got != want:
                raise RuntimeError(
                    f"draft 自回归步的 {name} 两套算法不一致：device 侧（上游 "
                    f"eagle_step_update_slot_mapping_and_metadata 算法）{got}，"
                    f"CPU 侧（58 关的工作区拼行）{want}：草稿会写在错误的位置上")

    def _fill_block_table_rows(self, batch_rows: list[int], block_table) -> None:
        """把这几条请求的块表行拷进**工作区块表**（只覆盖前 len(batch_rows) 行）。

        尾部残留的块号不会被读到：metadata 只带 `[:num_reqs]` 的有效切片（58 §7）。
        """
        for index, row in enumerate(batch_rows):
            count = block_table.num_blocks(row)
            if count > self.max_blocks_per_req:
                raise RuntimeError(
                    f"第 {row} 行有 {count} 个块，超过工作区 {self.max_blocks_per_req} 列："
                    f"工作区是按 max_model_len/block_size 开的，说明块表与本配置不匹配")
            self.block_table_cpu[index, :count] = torch.tensor(
                block_table.cpu[row, :count].tolist(), dtype=torch.int64)
            self.block_table_cpu[index, count:].zero_()

    def _upload(self, num_tokens: int, num_reqs: int) -> None:
        """只上传有效前缀（CPU 上 staging 与 device 侧是同一份，直接返回）。

        上传的内容**必须已经与 device 侧算法校验过**（见 `_check_ar_metadata_on_device`）：
        AR 步的 positions/slot_mapping/seq_lens 由上游同款算法在 device 上复算并逐值比对，
        再看这里把（相等的）值推上去——"谁算的"与"谁被模型读到"因此不会分叉。
        """
        if self.device == "cpu":
            return
        self.input_ids[:num_tokens].copy_(self.input_ids_cpu[:num_tokens])
        self.positions[:num_tokens].copy_(self.positions_cpu[:num_tokens])
        self.slot_mapping[:num_tokens].copy_(self.slot_mapping_cpu[:num_tokens])
        self.is_rejected_token_mask[:num_tokens].copy_(
            self.is_rejected_token_mask_cpu[:num_tokens])
        self.query_start_loc[:num_reqs + 1].copy_(self.query_start_loc_cpu[:num_reqs + 1])
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs])
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs])
        if self.pass_hidden_states_to_model:
            # **特征也必须上传**（63/65 关；这是 65 关的错位反证用例抓出来的一个真 bug）。
            # 漏了这一步，CUDA 上 `_forward()` 传进模型的 `self.hidden_states` 就是 device 侧
            # 那份**从没被写过**的缓冲（全零），而 `hidden_states_cpu` 上 staging 的特征永远
            # 到不了模型：草稿照样出、不报错，只是完全没吃到 target 的 hidden（EAGLE3/MTP 一起
            # 退化成"只看 token"）。CPU 上 staging 与 device 是同一份张量，所以只在 CPU 跑
            # 测试永远发现不了——必须在 CUDA 上比"模型实际收到的特征"。
            self.hidden_states[:num_tokens].copy_(self.hidden_states_cpu[:num_tokens])

    def _forward(self, num_tokens: int, num_reqs: int, hidden_states=None):
        """把工作区的前 `num_tokens` / `num_reqs` 行交给 draft 模型，返回 hidden states。

        **只传有效切片**：缓冲尾部的残留 token / 块号不能被 attention 读到——用长度表达有效，
        不是"内容恰好是 0"（58 §7）。
        """
        self._upload(num_tokens, num_reqs)
        metadata = self.metadata_builder.build(
            query_start_loc=self.query_start_loc[:num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            block_table=self.block_table[:num_reqs],
            slot_mapping=self.slot_mapping[:num_tokens],
            num_reqs=num_reqs)
        attn_metadata = {name: metadata for name in self.kv_caches}
        with set_forward_context(attn_metadata, num_tokens=num_tokens):
            if self.pass_hidden_states_to_model:
                features = self.hidden_states if hidden_states is None else hidden_states
                out = self.model(self.input_ids[:num_tokens], self.positions[:num_tokens],
                                 features[:num_tokens])
                if self.model_returns_tuple():
                    # 上游：`last_hidden_states, hidden_states = ret_hidden_states`
                    # —— lm_head 用前者，下一步 draft 用后者（EAGLE3 的 prenorm）
                    for_logits, for_next = out
                    return for_logits, for_next
                return out
            return self.model(self.input_ids[:num_tokens], self.positions[:num_tokens])

    def _reset_requests(self, reset_req_ids: set[str]) -> None:
        """丢掉不再成立的 draft 侧**进度**：请求刚被抢占恢复（块表整表换过）。

        恢复之后旧物理编号上的 KV 已经不属于它了，**必须**从头补，不能接着用。
        随机流**不重置**：请求还活着，只是换了块，重新 seed 会改变它的采样序列
        （205 §4.4：抢占恢复 ≠ 新请求）。
        """
        for req_id in reset_req_ids:
            self._draft_computed.pop(req_id, None)

    def remove_requests(self, req_ids) -> None:
        """请求**结束/abort**：进度与随机流一并删除（草稿概率 q 存在 Runner 上、
        每轮整体重算，不需要按请求清）。

        只有控制端明确说"这条结束了"才调它。**不能**用"不在本轮的 req_ids 里"代替：
        预算不够没排上、被抢占等待恢复的请求都不在 batch 里，但它们的状态必须留着
        （205 §4.1/§4.3 就是这两种情况混在一起造成的）。
        0-token 的结束清理轮也要调——否则最后一条请求结束后，复用的 ID 会继承旧进度。
        """
        for req_id in req_ids:
            self._draft_computed.pop(req_id, None)
            self._draft_generators.pop(req_id, None)

    # -------- 67 关：异构词表的 id 搬运 --------

    def _to_draft_space(self, valid_token_ids: list[int],
                        next_token_id: int) -> tuple[list[int], int]:
        """把一条请求的历史行 + 扩容行 token 从 **target 空间** 搬到 **草稿空间**（TLI）。

        只做"逐位置换 id"，**不重新分词**：位置数、行数、`start/history_end` 全都不变（需求 067 §5），
        所以 KV 槽位/块表/attention 元数据一律照旧——这也是 token 级 TLI 与"字符串桥接"的根本区别。
        不在交集里的 token 由映射表填 `draft_unk_token_id`（上游同款）。
        """
        tokens = torch.tensor([*valid_token_ids, int(next_token_id)], dtype=torch.int64)
        mapped = self.vocab_mapping.map_target_to_draft_ids(tokens)
        mapped = [int(value) for value in mapped.tolist()]
        return mapped[:-1], mapped[-1]

    def _to_draft_space_tokens(self, tokens: list[int]) -> list[int]:
        """自回归步用：上一枚草稿（target id）→ 草稿空间 id（上游同一处的 map 调用）。"""
        tensor = torch.tensor(tokens, dtype=torch.int64)
        return [int(value)
                for value in self.vocab_mapping.map_target_to_draft_ids(tensor).tolist()]

    # -------- 采样草稿 --------

    def _sample_draft_tokens(self, hidden: torch.Tensor, row_refs: list[tuple[str, int]],
                             input_batch, drafts: dict, probs: dict) -> None:
        """对给定行各采一枚草稿，同时记下它来自的分布（q）。

        `q` 必须是**实际提议时用的分布**（199 §7），所以这里与 `Sampler.sample` 走同一条
        约束链（温度 → top-k/top-p → 指数竞赛），把 softmax 之后的整行留下来。
        贪心行的"分布"是草稿位置上的点质量（one-hot），与 `draft_probs=None` 的点质量提议同义。
        """
        # **先过 LM head**：hidden 是隐藏态，采样要的是词表上的 logits（与 Runner 同一条路：
        # `compute_logits` 只对需要的行做词表 GEMM）
        logits = self.model.compute_logits(
            hidden[torch.tensor([row for _, row in row_refs],
                                dtype=torch.int64, device=hidden.device)]).to(torch.float32)
        if self.vocab_mapping is not None:
            # 67 关（TLI）：只留交集里的列（交集外的草稿 token 在 target 词表里没有对应物），
            # 采完再把 id 映回 **target 空间**——交出去的草稿必须是 target id（上游 `_greedy_sample()` 同款）。
            # 这条路上草稿是 argmax（点质量 q），所以 `draft_probs` 记 `None`；`_row_probs` 的
            # 概率是 **draft 空间** 的，宽度与 target 词表不符，绝不能交给验证器（需求 067 §3.5 的边界）。
            logits = self.vocab_mapping.constrain_draft_logits(logits)
            sampled = self.vocab_mapping.map_draft_to_target_ids(logits.argmax(dim=-1))
            for index, (req_id, _) in enumerate(row_refs):
                drafts[req_id].append(int(sampled[index]))
                probs[req_id].append(None)
            return
        for index, (req_id, _) in enumerate(row_refs):
            parameter = input_batch.sampling_params[input_batch.req_id_to_index[req_id]]
            row_logits = logits[index]
            probs_row = self._row_probs(row_logits, parameter)
            token = int(row_logits.argmax()) if parameter.is_greedy else int(
                random_sample(probs_row.unsqueeze(0),
                              {0: self._generator(req_id, parameter)})[0])
            drafts[req_id].append(token)
            probs[req_id].append(probs_row)

    def _row_probs(self, row_logits: torch.Tensor, parameter) -> torch.Tensor:
        """一行的提议分布：贪心 → one-hot（点质量）；否则与普通采样同一条约束链。"""
        if parameter.is_greedy:
            one_hot = torch.zeros_like(row_logits)
            one_hot[int(row_logits.argmax())] = 1.0
            return one_hot
        logits = row_logits / max(parameter.temperature, SAMPLING_EPS)
        top_k = None if parameter.top_k in (-1, 0) else torch.tensor(
            [parameter.top_k], device=logits.device)
        top_p = None if parameter.top_p >= 1.0 else torch.tensor(
            [parameter.top_p], device=logits.device)
        logits = apply_top_k_top_p(logits.unsqueeze(0), top_k, top_p)
        return logits.softmax(dim=-1, dtype=torch.float32)[0]

    def _generator(self, req_id: str, parameter):
        """draft 自己的随机流：按请求的 seed 建一次，之后**每轮接着用**（不是每轮重置）。"""
        generator = self._draft_generators.get(req_id)
        if generator is None and parameter.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(parameter.seed + 1)   # 与 target 的流分开，避免逐位重合
            self._draft_generators[req_id] = generator
        return generator

    def _attention_layer_names(self) -> list[str]:
        return sorted(self.kv_caches)



class DraftModelProposer(SpecDecodeBaseProposer):
    """用**另一个模型目录**当提议者（对应 vLLM `v1/spec_decode/draft_model.py`）。"""

    def __init__(self, spec_config, vllm_config, device: str) -> None:
        super().__init__(spec_config, vllm_config, device)
        if spec_config.draft_model_config is None:
            raise ValueError("draft_model 投机必须在 SpeculativeConfig 里给 draft_model_config")
        self.draft_model_config = spec_config.draft_model_config
        # 67 关（TLI）：异构词表时在这里建映射表——上游 `DraftModelProposer.__init__` 也在这个位置
        # （构造完就持有，模型加载时用它校验/约束）。**不让 target 模型改自己的词表**：映射只存在于
        # 提议者这边，target 全程用自己原来的 id 空间（需求 067 §2）。
        if self.use_heterogeneous_vocab:
            from .vocab_mapping import VocabMapping, load_tokenizer

            self.vocab_mapping = VocabMapping(
                target_tokenizer=load_tokenizer(
                    vllm_config.model_config.tokenizer_path),
                draft_tokenizer=load_tokenizer(self.draft_model_config.tokenizer_path),
                target_vocab_size=int((vllm_config.model_config.hf_config or {})["vocab_size"]),
                draft_vocab_size=int((self.draft_model_config.hf_config or {})["vocab_size"]),
                device=device)

    # -------- 加载与校验 --------

    def load_model(self) -> None:
        from ..model_loader import get_model

        self._validate_configs()
        self.model = get_model(self.draft_model_config, self.device)
        self._allocate_kv_caches()
        # 72 关：并行提议要从权重里的 `mask_hidden` 取出 mask 槽位的常量特征（普通 draft 无此步）
        self._maybe_fill_parallel_drafting_hidden_state()
        return self.model

    def _validate_configs(self) -> None:
        """词表与 KV 规格必须兼容（199 §9）：不满足就**明确报错**，不静默降级。"""
        target_config = self.vllm_config.model_config.hf_config or {}
        draft_config = self.draft_model_config.hf_config or {}
        target_vocab = target_config.get("vocab_size")
        draft_vocab = draft_config.get("vocab_size")
        if target_vocab != draft_vocab and not self.use_heterogeneous_vocab:
            # 63 关：EAGLE3 允许 draft 词表更小 + 带 `d2t` 偏移映射（同 tokenizer、缩小词表；
            # 与 67 关"两套 tokenizer 的 TLI 交集"不是一回事）。映射本身由 draft 的
            # `compute_logits()` 完成（scatter 回 target 宽度，上游 llama_eagle3.py:339-356 同款），
            # 所以这里只要"配置里有 draft_vocab_size 且加载到了 d2t"就放行；缺了就明确报错。
            if not (self.method == "eagle3" and draft_config.get("draft_vocab_size")):
                raise ValueError(
                    f"draft 与 target 的词表不一致（{draft_vocab} vs {target_vocab}）："
                    f"草稿的 token 在 target 的词表里是另一个意思，验证没有意义"
                    f"（要跨两套 tokenizer 用 method='draft_model' + use_heterogeneous_vocab，"
                    f"那会按 token 级交集建映射表）")
        if str(self.draft_model_config.dtype) != str(self.vllm_config.model_config.dtype):
            raise ValueError("draft 与 target 的 dtype 必须一致（KV 缓存要放进同一个 group）")
        for key in ("num_key_value_heads", "head_dim", "max_position_embeddings"):
            if draft_config.get(key) != target_config.get(key):
                raise ValueError(
                    f"draft 与 target 的 {key} 不一致"
                    f"（{draft_config.get(key)} vs {target_config.get(key)}）："
                    f"本关只支持 KV 规格相同的 draft/target 共用一个 KV group")

    def _allocate_kv_caches(self) -> None:
        """给 draft 的每个 Attention 层分配**自己的**物理缓存。

        与 target 同形、同块数：块号在两边代表同一段位置。但**不是同一份 K/V**——
        target 写它自己的 tensor，draft 写自己的。
        """
        cache_config = self.vllm_config.cache_config
        dtype = next(self.model.parameters()).dtype
        self.kv_caches = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, Attention):
                continue
            cache = torch.zeros(2, cache_config.num_gpu_blocks, self.block_size,
                                module.num_kv_heads, module.head_size, dtype=dtype,
                                device=self.device)
            module.kv_cache = cache
            self.kv_caches[name] = cache

