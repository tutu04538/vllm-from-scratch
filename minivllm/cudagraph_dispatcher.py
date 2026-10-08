"""CUDA Graph 运行时分派器（对应 vLLM `v1/cudagraph_dispatcher.py`，逐方法对齐）。

它解决的问题：**这一轮该不该走图？走哪张图？**

CUDA Graph 的键是"补齐后的批次形状"（`BatchDescriptor`）。而真实批次每轮都不一样
（请求数、总行数、每条请求的行数都在变），所以中间需要一个"翻译"：

    真实形状  ──补齐到最近的档位──▶  图键（BatchDescriptor）  ──▶  模式（FULL / PIECEWISE / NONE）

三个方法分工（与上游同名）：

    initialize_cudagraph_keys()   启动时把"**可能**被用到的键"全列出来（= 要捕获哪些图）
    dispatch()                    运行时把真实形状翻译成键，键不存在就回退 NONE（eager）
    get_capture_descs()           捕获阶段读第一份名单（按 num_tokens 从大到小，先占显存）

**为什么键里必须有 num_reqs**：FULL 图把整段前向录成一张图，图里的 kernel 网格、workspace
大小、注意力元数据的形状都按请求数定死；请求数变了就是另一张图。PIECEWISE 图则允许任意请求数
（注意力被拆在图外，图里的部分只看 token 数），所以那里的键把 `num_reqs` 置空——上游用
`replace(batch_desc, num_reqs=None, uniform=False)` 表达这件事，本仓库照抄。

**`uniform` 是投机解码与图的接缝**：只有"每请求恰好 `1 + K` 行"的批，行数才是请求数的整数倍，
形状才固定。上游因此给它单独起名 uniform decode，并把 `uniform_decode_query_len = 1 + K`
作为参数传给分派器；补齐时按这个宽度取整，`num_reqs = num_tokens_padded // (1 + K)`。
"""

from itertools import product

from .config import CUDAGraphMode
from .forward_context import BatchDescriptor


class CudagraphDispatcher:
    """维护"哪些图存在"的**唯一权威**，并按真实形状分派。

    与上游的差异（都只涉及本仓库没有的能力，不是"简化了算法"）：

    1. **没有 LoRA**：上游的 `_get_lora_cases()` 会按 `lora_config` 决定要不要为"带 LoRA"
       单独捕获一套图；本仓库没有 LoRA，所以那一支恒为 `[0]`（= 只有"不带 LoRA"一种情况）。
       `BatchDescriptor.has_lora / num_active_loras` 字段照抄保留，值恒 False/0。
    2. 没有 `breakable_cudagraph` / sequence-parallel / DP 协调：那三样都要求多卡或编译路径。
    3. **PIECEWISE 的切分点是手工的**（模型结构里的注意力边界），不是上游那种"编译期按
       `splitting_ops` 拆 fx 图"；对分派器来说两者无差别（键与模式完全一样）。
    """

    def __init__(self, vllm_config) -> None:
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        # 统一 decode 批的每请求行数：普通 decode 是 1，投机是 1 + K
        self.uniform_decode_query_len = 1 + vllm_config.num_speculative_tokens

        # 两类图的键表。**这是"图是否存在"的唯一真相**：运行时只查这两张表，
        # 不自己"按最近档位猜"（自造规则会让重放落到没捕获过的形状上）。
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        self.keys_initialized = False
        # 未调用 initialize_cudagraph_keys() 之前一律 NONE（宁可 eager，不可猜测）
        self.cudagraph_mode = CUDAGraphMode.NONE
        # 记录每类图捕获时的档位（`get_capture_descs()` 用得上，也是给测试看的账本）
        self.captured_lora_counts: list[int] = []

    # -------- 键的构造 --------

    def _compute_bs_to_padded_graph_size(self) -> None:
        """预计算"批大小 → 补齐后的档位"（上游同名方法，含边界规则）。

        规则（逐行对齐上游）：档位表升序 `[c1, c2, ..., cn]`，则

            bs == c_i          → c_i        （正好命中档位，不补）
            c_{i-1} < bs < c_i → c_i        （补到**下一个**档位）
            bs  > max          → 不在这里处理（`dispatch()` 直接判成 NONE）

        注意 `[0] + capture_sizes` 那个错位：`(0, c1]` 全部映射到 `c1`，`(c1, c2]` 映射到 `c2`。
        """
        max_size = self.compilation_config.max_cudagraph_capture_size
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        if max_size is None or capture_sizes is None:
            raise RuntimeError(
                "启用 CUDA Graph 时必须有 max_cudagraph_capture_size 与 "
                "cudagraph_capture_sizes（由 VllmConfig._set_cudagraph_sizes() 解析）")
        self._bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
            for bs in range(start, end):
                # 正好等于档位起点 → 不补；否则补到本段终点
                self._bs_to_padded_graph_size[bs] = start if bs == start else end

        # 上游的一条保护：compile_sizes 里的形状不能被补齐改变（否则"编译的形状"与
        # "重放时喂进去的形状"不是同一个，编译白做）。本仓库没有编译，这段只在用户
        # 同时配了两者时才有意义——配了编译会被 CompilationConfig 直接拒掉，所以这里只留校验。
        if self.compilation_config.compile_sizes and self.cudagraph_mode != CUDAGraphMode.NONE:
            for size in self.compilation_config.compile_sizes:
                size = int(size)
                if size <= max_size and self._bs_to_padded_graph_size[size] != size:
                    raise ValueError(
                        f"compile_sizes 里的 {size} 会被 padding 改成 "
                        f"{self._bs_to_padded_graph_size[size]}：编译的形状与重放的形状必须一致")

    def _get_lora_cases(self) -> list[int]:
        """要捕获哪几种 `num_active_loras`（上游同名方法）。

        本仓库没有 LoRA 实现（`lora_config` 这个配置根本不存在），所以只有一种情况：
        `[0]` = 不带 LoRA。上游的 `lora_config is None` 分支返回的也是它。
        """
        return [0]

    def _create_padded_batch_descriptor(self, num_tokens: int, uniform_decode: bool,
                                        has_lora: bool,
                                        num_active_loras: int = 0) -> BatchDescriptor:
        """真实形状 → 图键（上游同名方法）。

        补齐之后 `num_reqs` 也要**跟着补齐算**，不能沿用真实请求数：
        FULL 图里的注意力元数据是按请求数定形状的。

            统一 decode（且模式含 FULL）→ num_reqs = padded_tokens // (1 + K)，且必须整除
            其它                        → num_reqs = min(padded_tokens, max_num_seqs)
        """
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            if num_tokens_padded % uniform_decode_query_len != 0:
                raise ValueError(
                    f"补齐后的档位 {num_tokens_padded} 不是 uniform_decode_query_len="
                    f"{uniform_decode_query_len} 的整数倍：统一 decode 图要求每请求恰好 "
                    f"1+K 行，档位表必须由 (1+K) 的倍数构成")
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(num_tokens=num_tokens_padded, num_reqs=num_reqs,
                               uniform=uniform_decode, has_lora=has_lora,
                               num_active_loras=num_active_loras)

    def add_cudagraph_key(self, runtime_mode: CUDAGraphMode,
                          batch_descriptor: BatchDescriptor) -> None:
        if runtime_mode not in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
            raise ValueError(f"图键只能挂在 PIECEWISE / FULL 下，收到 {runtime_mode}")
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    # -------- 初始化 --------

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode,
                                  uniform_decode_query_len: int = 1) -> None:
        """按模式列出所有**可能**被用到的键（上游同名方法）。

        调用时机有讲究：**必须在注意力后端确定之后**——因为"这个后端支不支持图"决定最终模式。
        本仓库只有一个 Torch 后端，它也支持图（统一 decode 那条路径，见
        `attention/backends/torch_sdpa.py`），所以模式就是配置里解析出来的那个。

        键是"可能用到"，不是"一定用到"：多列一些不会错（`dispatch()` 找不到就回退 eager），
        少列了才会出现"形状没图可走"。所以这里按**档位表 × LoRA 情况**全量生成。
        """
        self.uniform_decode_query_len = uniform_decode_query_len
        self.cudagraph_mode = cudagraph_mode

        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        lora_cases = self._get_lora_cases()
        self.captured_lora_counts = [count for count in lora_cases if count]

        # (a) 混合 prefill/decode 批的键。只在"混合批也要图"的模式下才有意义
        #     （FULL_DECODE_ONLY 的混合模式是 NONE → 混合批一律 eager）。
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            if self.compilation_config.cudagraph_capture_sizes is None:
                raise RuntimeError("mixed 模式启用时必须有档位表")
            for bs, num_active_loras in product(
                    self.compilation_config.cudagraph_capture_sizes, lora_cases):
                batch_desc = self._create_padded_batch_descriptor(
                    bs, False, num_active_loras > 0, num_active_loras)
                # PIECEWISE 能处理任意请求数（注意力在图外）；FULL 必须精确匹配请求数
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    from dataclasses import replace

                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # (b) 统一 decode 批的键：只有"decode 走 FULL 且两种批分开走"时才单独建一套。
        #     上限是 `(1+K) * max_num_seqs`（再多也不可能有这么长的统一 decode 批）。
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and cudagraph_mode.separate_routine()):
            max_num_tokens = uniform_decode_query_len * \
                self.vllm_config.scheduler_config.max_num_seqs
            if self.compilation_config.cudagraph_capture_sizes is None:
                raise RuntimeError("decode FULL 模式启用时必须有档位表")
            decode_sizes = [size for size in self.compilation_config.cudagraph_capture_sizes
                            if uniform_decode_query_len <= size <= max_num_tokens]
            for bs, num_active_loras in product(decode_sizes, lora_cases):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(
                        bs, True, num_active_loras > 0, num_active_loras))

        self.keys_initialized = True

    # -------- 运行时 --------

    def dispatch(self, num_tokens: int, uniform_decode: bool = False,
                 has_lora: bool = False, num_active_loras: int = 0,
                 valid_modes=None, invalid_modes=None,
                 ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """真实形状 → `(运行模式, 图键)`（上游同名方法）。

        返回的键**可能不是**调用方传进来的形状：补齐过的 `num_tokens` 与算出来的 `num_reqs`
        都在里面，调用方必须拿它去准备输入（"我要跑多少行"由分派结果说了算）。

        回退规则（上游的五条早退）：
          1. 键还没初始化                        → NONE
          2. 模式是 NONE                          → NONE
          3. 没有档位上限                         → NONE
          4. `num_tokens` 超过最大档位             → NONE
          5. 允许的模式里只剩 NONE                → NONE
        然后：先查 FULL 键（要求完全相等的键）、再查**放宽过的** PIECEWISE 键
        （`num_reqs=None, uniform=False`），都不中就回退 NONE（= 这一轮跑 eager）。
        """
        allowed_modes = (valid_modes if valid_modes is not None
                         else CUDAGraphMode.valid_runtime_modes())
        if invalid_modes:
            allowed_modes = set(allowed_modes) - set(invalid_modes)
        if not allowed_modes:
            raise ValueError(f"没有任何允许的图模式：valid_modes={valid_modes}, "
                             f"invalid_modes={invalid_modes}")
        max_size = self.compilation_config.max_cudagraph_capture_size

        if (not self.keys_initialized
                or self.cudagraph_mode == CUDAGraphMode.NONE
                or max_size is None
                or num_tokens > max_size
                or allowed_modes <= {CUDAGraphMode.NONE}):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        if has_lora:
            # 没有 LoRA 实现：真出现就是调用方搞错了，不能"当作没有 LoRA"继续跑（那会算出
            # 不带适配器的结果）
            raise ValueError("本仓库没有 LoRA 实现，has_lora 只能是 False")

        # `uniform` 只有在"两种批分开走"的模式下才有意义：FULL_DECODE_ONLY / FULL_AND_PIECEWISE
        # 是分开的；纯 FULL 是同一套图处理所有批，键里不能带 uniform（否则同一形状会有两个键）。
        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, num_active_loras)

        if CUDAGraphMode.FULL in allowed_modes:
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            from dataclasses import replace

            relaxed = replace(batch_desc, num_reqs=None, uniform=False)
            if relaxed in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, relaxed

        if CUDAGraphMode.NONE not in allowed_modes:
            raise ValueError(
                f"没有匹配的图，且 NONE 不在允许的模式里：allowed_modes={allowed_modes}")
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """捕获阶段要用的名单：`[(模式, [键...]), ...]`，PIECEWISE 在前、FULL 在后。

        每个模式内部**按 num_tokens 从大到小**排（上游注释：先捕获大图，小图就能复用大图
        占下的显存池；反过来的话每张小图都要单独开池）。
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []
        result = []
        for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
            descs = list(self.cudagraph_keys[mode])
            if descs:
                descs.sort(key=lambda d: (d.num_tokens, d.num_active_loras), reverse=True)
                result.append((mode, descs))
        return result
