"""`StructuredOutputManager`：**引擎级**的结构化输出管理器（对应 vLLM `v1/structured_output/__init__.py`）。

它回答三个问题（068 §2）：

1. **谁来编译 grammar**：`grammar_init(request)` —— 每条请求一份 FSM（同一个 schema 的不同请求
   状态必须独立，所以不共享 matcher）。
2. **每轮每行允许哪些 token**：`grammar_bitmask(...)` —— 为**每个待定位置**生成一份掩码。
   常规生成是"1 行"；投机是"K 个候选位 + 1 个 bonus 位 = K+1 行"，因为候选位各自面对
   **不同的假设历史**（068 §1）。
3. **试走之后要收尾**：候选位的掩码靠"假设前面都被接受"地试走得到，算完必须
   `rollback(state_advancements)` 把 FSM 退回原位；**只有 Scheduler 在实际采样结果回来后**
   才真正 `accept_tokens`（068 §2：不要让 proposer 永久推进 grammar）。

本仓库是上游的同步子集，三处差异（docs/step68_alignment.md 全部记账）：

| 上游 | 本仓库 | 为什么 |
|---|---|---|
| `__init__` 里就加载 tokenizer | 第一次 `grammar_init` 才加载 | tiny 模型目录没有 tokenizer 文件；不用结构化输出就不该读它 |
| 异步编译（线程池 + Future） | 同步编译，失败即 `add_request` 报错 | 上游在 external_launcher 下也走同步分支；异步属"尚未接入" |
| 超过 128 条请求时并行填掩码 | 一律串行 | 同一批结果逐位相同，只是 CPU 并行度；本项目的批很小 |

`-1` 的处理与上游**逐行**一致（`_fill_bitmasks` 在判 `-1` 之前调用）：

    `-1` 那一位     仍然按"当前试走状态"填一份掩码（它是 padding 位，不会被采样）
    `-1` **之后**   不再填掩码（全允许）、也不再推进 FSM
    整段结束        统一 `rollback(state_advancements)`，把试走的账还清

（`-1` 是异步调度的占位符语义，见 70 关；本仓库的调度侧目前会先把无效草稿裁掉，
所以生产路径上很少出现，但语义要对齐。）
"""

from __future__ import annotations

import torch

from .backend_types import StructuredOutputGrammar
from .backend_xgrammar import XgrammarBackend, validate_xgrammar_grammar
from .request import StructuredOutputRequest


class StructuredOutputManager:
    def __init__(self, vllm_config) -> None:
        self.vllm_config = vllm_config
        self.backend: XgrammarBackend | None = None
        self.tokenizer = None
        # 掩码缓冲：`[max_num_seqs * (1 + K), ceil(vocab/32)]` 的 int32。**懒分配**——
        # 分配时才知道 vocab_size 与后端布局（上游同款：第一次要掩码时分配）。
        self._grammar_bitmask: torch.Tensor | None = None
        # "这一行不受约束"的填充值：全 1 的掩码（bit=0 表示禁止，所以全 1 = 全允许）
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

    # -------- 编译 --------

    def grammar_init(self, request) -> None:
        """给一条请求编译 grammar（上游同名方法）。

        **失败就在这里抛**：本仓库在"提交请求"这一同步路径上编译，所以一条语法写错的请求
        会在 `add_request` 处直接报错，而不是先在引擎里排一会儿队再失败。上游把它包成
        Future 让 Scheduler 只失败这一条请求（异步编译的代价与收益），我们不做异步。
        """
        struct_request = request.structured_output_request
        if struct_request is None:
            return
        if self.backend is None:
            backend_name = getattr(
                getattr(self.vllm_config, "structured_outputs_config", None), "backend",
                "auto")
            if backend_name not in ("auto", "xgrammar"):
                raise NotImplementedError(
                    f"structured_outputs.backend={backend_name!r} 本项目没接：本仓库只接 "
                    f"xgrammar（068 §3.4「选择本机已有 backend，不重造 JSON parser」）。"
                    f"改配置为 'xgrammar'/'auto'，或者按三态矩阵接受这个缺口——"
                    f"不能把参数收下却按别的后端跑（068 §3.6 禁止静默忽略）")
            from ..tokenizer_utils import cached_tokenizer_from_config
            self.tokenizer = cached_tokenizer_from_config(self.vllm_config.model_config)
            self.backend = XgrammarBackend(
                vllm_config=self.vllm_config, tokenizer=self.tokenizer,
                vocab_size=self.vllm_config.model_config.get_vocab_size())
        struct_request.grammar = self._create_grammar(request)

    def _create_grammar(self, request) -> StructuredOutputGrammar:
        struct_request = request.structured_output_request
        assert struct_request is not None
        request_type, grammar_spec = struct_request.structured_output_key
        assert self.backend is not None
        stop_token_ids = (request.sampling_params.all_stop_token_ids
                          if request.sampling_params is not None else None)
        return self.backend.compile_grammar(request_type, grammar_spec,
                                            stop_token_ids=stop_token_ids)

    # -------- 每轮的掩码 --------

    def grammar_bitmask(self, requests: dict, structured_output_request_ids: list[str],
                        scheduled_spec_decode_tokens: dict):
        """为这一轮的每个待定位置生成掩码（上游同名方法，串行分支 + 同步子集）。

        返回 `[num_masks, ceil(vocab/32)]` 的 **numpy int32**（上游也用 numpy 跨进程传：
        序列化比 torch 张量便宜）。
        """
        if not structured_output_request_ids:
            return None

        spec_config = getattr(self.vllm_config, "speculative_config", None)
        max_num_spec_tokens = (spec_config.num_speculative_tokens
                               if spec_config is not None else 0)

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            # 每个待定位置一行：K 个候选位 + 1 个 bonus 位（非投机时 K=0，就是 1 行）
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens))

        cumulative_index = 0
        for req_id in structured_output_request_ids:
            request = requests[req_id]
            grammar = request.structured_output_request.grammar
            assert grammar is not None, f"{req_id!r} 的 grammar 还没编译好（grammar_init 漏了）"
            apply_bitmask = self.should_fill_bitmask(request)

            state_advancements = 0
            req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
            for token in req_tokens:
                self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                advance_grammar = apply_bitmask
                if token == -1:
                    # padding 的候选位：**不推进** FSM；它之后的候选位也不再填掩码
                    # （上游同款：填充之后的位置都没有可信的假设历史）。注意掩码是在这之前
                    # 填的 —— 上游的 `_fill_bitmasks` 调用也在 `-1` 判断之前，所以这一行
                    # 拿到的仍是当前试走状态的掩码。
                    apply_bitmask = False
                    advance_grammar = False
                if advance_grammar and not grammar.is_terminated():
                    accepted = grammar.accept_tokens(req_id, [token])
                    if accepted:
                        state_advancements += 1
                    else:
                        # 走到这里说明"草稿没被语法预筛过"。上游直接断言失败：
                        # 与其带着错误的假设继续填掩码（后面每一行都错），不如停在这里。
                        raise AssertionError(
                            f"{req_id!r} 的草稿 {token} 没通过语法：草稿必须在 "
                            f"update_draft_token_ids() 里先用 validate_tokens 裁过"
                            f"（Scheduler 的职责），不能把非法草稿交给验证器。"
                            f"scheduled_spec_decode_tokens={scheduled_spec_decode_tokens!r}")
                cumulative_index += 1

            # bonus 位：只有"这一轮确实在受约束"时才填掩码（上游的 bonus_apply）
            bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
            self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
            cumulative_index += 1

            # 试走的账要还清：FSM 退回原位，真正推进留给 Scheduler 的 accept_tokens
            if state_advancements > 0:
                grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        return bitmask_tensor.numpy()

    def _fill_bitmasks(self, batch) -> None:
        """给若干行填掩码（上游同名方法；`apply_bitmask=False` 的行填"全允许"）。"""
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # 不受约束 / 已经结束的行：填成全 1（= 全允许），否则上一轮的旧掩码会残留
                self._grammar_bitmask[index].fill_(self._full_mask)

    # -------- 语义（本仓库没有思考模式，这两条都退化成常量）--------

    def should_fill_bitmask(self, request) -> bool:
        """这条请求这一轮要不要受约束。

        上游在这里判"思考是否结束"（reasoner 存在且没结束时先不约束）。本仓库没有
        reasoning parser，所以恒为 True —— 与上游在没有 reasoner 时**同一条路径**。
        """
        return True

    def should_advance(self, request, new_token_ids: list[int] | None = None) -> bool:
        """Scheduler 要不要在提交 token 之后推进 grammar。

        上游的 `new_token_ids`/`reasoning_ended` 都服务于"思考结束后才开始约束"。
        本仓库恒为"有结构化输出就推进"，即上游无 reasoner 时的返回值。
        """
        return bool(request.use_structured_output)

    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()


# 请求期校验的入口（上游在 `SamplingParams._validate_structured_output` 里调）
def validate_structured_output(sampling_params, structured_outputs_config, tokenizer=None):
    """校验 + 规范化一条请求的结构化输出参数（上游同位置逻辑的本仓库子集）。

    做三件事：
      1. 引擎级配置里的 backend 必须是我们接得住的那个（`auto` 会解析成 `xgrammar`）；
      2. 请求里指定的 backend（上游 `structured_outputs._backend`）不许和引擎级冲突；
      3. 让选中的后端校验一遍规格（`validate_xgrammar_grammar` 还会把 `choice` 改写成 EBNF）。

    第 3 步**必须有**：不校验就编译，一份写错的 regex 会在第一次采样时才炸；更糟的是
    `choice` 不会被改写成 grammar，`compile_grammar` 会拿到没有分支的 CHOICE 类型。
    """
    if sampling_params is None or sampling_params.structured_outputs is None:
        return

    backend = getattr(structured_outputs_config, "backend", "auto") or "auto"
    request_backend = sampling_params.structured_outputs._backend
    if request_backend:
        if backend != request_backend and not (backend == "auto"
                                               and sampling_params.structured_outputs
                                               ._backend_was_auto):
            raise ValueError(
                f"结构化输出后端只能由引擎决定：请求里写的是 {request_backend!r}，"
                f"引擎初始化的是 {backend!r}（上游同样不支持请求级选择后端）")
    else:
        sampling_params.structured_outputs._backend = backend

    if backend == "auto":
        validate_xgrammar_grammar(sampling_params)
        sampling_params.structured_outputs._backend = "xgrammar"
        sampling_params.structured_outputs._backend_was_auto = True
        return
    if backend == "xgrammar":
        validate_xgrammar_grammar(sampling_params)
        return
    raise NotImplementedError(
        f"structured_outputs.backend={backend!r} 本项目没接：三态矩阵里属「本项目尚未接入」"
        f"（本机装了 outline/guidance/lmfe 的部分依赖，但本关只对齐 xgrammar 这条路径，"
        f"068 §3.4）。请把 backend 设为 'xgrammar' 或 'auto'")
