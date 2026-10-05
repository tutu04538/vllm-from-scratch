"""异构词表的 token 级对齐（对应 vLLM `v1/spec_decode/vocab_mapping.py::VocabMapping`，需求 067 §2）。

**它解决什么**：target 的 token 17 与 draft 的 token 17 未必是同一段文字。两套 tokenizer 的 id 空间
不能直接互换，只比 `vocab_size` 相等也挡不住（不同 tokenizer 完全可能撞上同一个词表大小）。所以初始化
时按 **token 字符串** 建一张交集表，运行期把 id 在两套空间之间搬运：

    draft_to_target_ids      [Vdraft]   草稿 id → target id，缺失 = -1
    target_to_draft_ids      [Vtarget]  target id → 草稿 id，缺失 = -1
    intersection_mask_draft  [Vdraft]   bool，给草稿 logits 做掩码用
    target_unk_token_id / draft_unk_token_id   不在交集里的 token 各自的兜底 id

**四条路径**（上游 `llm_base_proposer.py` 的三个调用点 + 采样点）：

    ① 第一遍：target 的历史行与"新采出的那个 token" → `map_target_to_draft_ids` → 才喂进 draft
    ② 自回归步：上一枚草稿（**target 空间的 id**）→ `map_target_to_draft_ids` → 才喂回 draft
    ③ 采样：草稿 logits 先用 `constrain_draft_logits` 把非交集列置 -inf（**交集外的 token 永远选不到**）
    ④ 交回：采出的草稿 id → `map_draft_to_target_ids` → 调度器/target 验证全程用 **target 空间的 id**

**边界（需求 §3.5，照抄上游）**：只支持 `method="draft_model"`，且只支持 **greedy 草稿**
（概率草稿要把 q 从 draft 空间搬到 target 空间，上游还没做——代码里留着 TODO；配置期直接拒绝）。

**本仓库与上游的差异**（记在 docs/step67_alignment.md §3）：上游的 map 方法默认索引在 GPU 上的表，
本仓库的"第一遍输入"是在 **CPU** 上组织的（`input_ids` 是 Python 列表 / CPU 缓冲），所以 map 方法会把
表挪到输入所在的设备（`.to(ids.device)`，已经同设备时是空操作）。语义与上游逐位一致。

**不做什么**（需求 §5）：不实现"重新分词"式的通用字符串桥接——本类对齐的是 **token 级交集**，
交集外的词按 unk→eos→报错 处理，不假装支持任意 tokenizer 组合。
"""

import warnings

import torch

# 交集小于这个数就告警（上游 `VocabMapping.__init__` 的阈值，逐字相同）
SMALL_INTERSECTION_THRESHOLD = 100


def _detect_space_prefix(tokenizer) -> tuple[str, ...]:
    """探测"词首空格"用什么字符标记（上游同名函数，逐行照抄）。

    BPE 系用 `Ġ`（U+0120）、SentencePiece 系用 `▁`（U+2581）。运行时探测而不是写死，是为了
    正确支持混合家族的两套 tokenizer（例如 BPE draft + SentencePiece target）。
    """
    try:
        space_ids = tokenizer.encode(" a", add_special_tokens=False)
        if space_ids:
            tok_str = tokenizer.convert_ids_to_tokens(space_ids[0])
            if (isinstance(tok_str, str) and len(tok_str) > 1 and tok_str.endswith("a")
                    and tok_str[0] not in (" ", "\t")):
                return (tok_str[:-1],)
    except Exception:                     # noqa: BLE001 —— 探测失败就退回"两种都试"
        pass
    return ("\u0120", "\u2581")


def _normalize_token(token: str, space_prefixes: tuple[str, ...]) -> str:
    """把词首空格标记还原成真正的空格，让两套 tokenizer 的同一个词得到同一个字符串。"""
    for prefix in space_prefixes:
        if token.startswith(prefix):
            return " " + token[len(prefix):]
    return token


def _get_unk_token_id(tokenizer, role: str) -> int:
    """不在交集里的 token 用哪个 id 兜底（上游同名函数）：`unk → eos → 报错`。

    只能用 `is not None` 判空：**0 是合法的 unk id**（很多 tokenizer 就是 0），写成 `unk or eos`
    会把它悄悄换成 eos（需求 §3.4 专门点了这一条）。
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        return unk
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        warnings.warn(
            f"VocabMapping: {role} 没有 unk_token_id，交集外的 token 退回 eos_token_id={eos}",
            stacklevel=2)
        return eos
    raise ValueError(
        f"VocabMapping: {role} 既没有 unk_token_id 也没有 eos_token_id，"
        f"交集外的 token 无处可去（上游同样直接报错）")


def load_tokenizer(path: str):
    """按目录加载 tokenizer（对应上游 `get_tokenizer`）。

    TLI 需要**两套** tokenizer 才能建交集表，所以这是本仓库第一次在提议者里读 tokenizer 文件；
    读不到就明确报错（不静默退回"同词表"——那会让草稿和 target 的 id 语义悄悄错位）。
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:              # noqa: BLE001 —— 包成"缺 tokenizer"这一件事
        raise RuntimeError(
            f"use_heterogeneous_vocab 需要 {path!r} 下有可加载的 tokenizer 文件"
            f"（tokenizer.json / tokenizer_config.json 等）：{type(exc).__name__}: {exc}") from exc


class VocabMapping:
    """两套 tokenizer 的 token 级交集表（上游同名类；字段与语义逐个对应）。"""

    def __init__(self, target_tokenizer, draft_tokenizer, target_vocab_size: int,
                 draft_vocab_size: int, device) -> None:
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device
        self.target_unk_token_id = _get_unk_token_id(target_tokenizer, "target tokenizer")
        self.draft_unk_token_id = _get_unk_token_id(draft_tokenizer, "draft tokenizer")

        target_prefixes = _detect_space_prefix(target_tokenizer)
        draft_prefixes = _detect_space_prefix(draft_tokenizer)

        # 规范化后建表：**同一个规范化字符串只留第一次出现的那一个 id**（上游同款规则，
        # 词表顺序由 tokenizer 自己决定）。两边的空格标记不同（Ġ / ▁）也能对上同一个词。
        target_normalized: dict[str, int] = {}
        for token, tid in target_tokenizer.get_vocab().items():
            norm = _normalize_token(token, target_prefixes)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized: dict[str, int] = {}
        for token, tid in draft_tokenizer.get_vocab().items():
            norm = _normalize_token(token, draft_prefixes)
            if norm not in draft_normalized:
                draft_normalized[norm] = tid

        common_tokens = set(target_normalized.keys()) & set(draft_normalized.keys())

        draft_to_target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
        target_to_draft = torch.full((target_vocab_size,), -1, dtype=torch.long)
        intersection_mask_draft = torch.zeros(draft_vocab_size, dtype=torch.bool)

        for norm_token in common_tokens:
            t_id = target_normalized[norm_token]
            d_id = draft_normalized[norm_token]
            # 词表里可能含有超出**模型** vocab_size 的 id（tokenizer 多出来的 special token）：
            # 越界的直接不要（上游同款判断）
            if t_id < target_vocab_size and d_id < draft_vocab_size:
                draft_to_target[d_id] = t_id
                target_to_draft[t_id] = d_id
                intersection_mask_draft[d_id] = True

        self.draft_to_target_ids = draft_to_target.to(device)
        self.target_to_draft_ids = target_to_draft.to(device)
        self.intersection_mask_draft = intersection_mask_draft.to(device)
        self.intersection_size = int(intersection_mask_draft.sum().item())

        # 上游用 logger.info 打这三个数；本仓库不引 logger，改成**字段 + 文档**（需求 §5 要求记录
        # 交集大小）。字段留着，脚本/测试直接读。
        self.stats = {
            "target_vocab_size": target_vocab_size,
            "draft_vocab_size": draft_vocab_size,
            "intersection_size": self.intersection_size,
            "draft_coverage": (self.intersection_size / draft_vocab_size
                               if draft_vocab_size else 0.0),
            "target_coverage": (self.intersection_size / target_vocab_size
                                if target_vocab_size else 0.0),
        }
        if self.intersection_size < SMALL_INTERSECTION_THRESHOLD:
            warnings.warn(
                f"VocabMapping: 两套 tokenizer 的交集只有 {self.intersection_size} 个 token，"
                f"草稿能说的话极少，接受率会很低（上游同样只告警、不报错）", stacklevel=2)

    # -------- 两个方向的映射 --------

    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        """target 空间 → 草稿空间；不在交集里的位置填 `draft_unk_token_id`（**不是 -1**）。"""
        table = self.target_to_draft_ids.to(target_ids.device)
        draft_ids = table[target_ids]                    # 新张量：不改写入参（需求 §4）
        missing = draft_ids == -1
        if missing.any():
            draft_ids[missing] = self.draft_unk_token_id
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """草稿空间 → target 空间；不在交集里的位置填 `target_unk_token_id`。"""
        table = self.draft_to_target_ids.to(draft_ids.device)
        target_ids = table[draft_ids]
        missing = target_ids == -1
        if missing.any():
            target_ids[missing] = self.target_unk_token_id
        return target_ids.to(draft_ids.dtype)

    # -------- 采样前约束 --------

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """把草稿词表里**不在交集**的列置 `-inf`（`masked_fill` 返回新张量，不改写入参）。

        为什么必须做：交集外的草稿 token 在 target 词表里没有对应物，说出来也无法验证——掩掉之后
        `argmax`/采样永远选不到它，于是"草稿交回的 id 一定能在 target 空间找到对应"这条前提成立。
        """
        mask = self.intersection_mask_draft.to(logits.device)
        return logits.masked_fill(~mask, float("-inf"))
