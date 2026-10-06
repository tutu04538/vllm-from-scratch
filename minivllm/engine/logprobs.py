"""把 EngineCore 交回的 logprobs 逐位置变成用户可见的 `{token_id: Logprob}`（对应 vLLM
`v1/engine/logprobs.py::LogprobsProcessor`）。

数据在两侧的形状不同，这一层就是翻译：

    执行侧（`LogprobsLists`）     numpy，行 = 一个生成位置，列 = [选中 token, top-1, top-2, …]
    用户侧（`SampleLogprobs`）    逐位置的 dict：`{token_id: Logprob(logprob, rank, decoded_token)}`

三件事只有这里做：

1. **累计**：`cumulative_logprob += 选中 token 的 logprob`（用户拿它算 perplexity）。
   注意加的是**交付的那一份**（raw 或 processed 由引擎配置决定），不是两套都加。
2. **逐请求裁剪宽度**：整批按最大宽度算 top-k，这里按这条请求要的 `num_logprobs` 截
   （上游靠 `zip` 在更短的那一边停住，本项目同样用 zip）。
3. **解码 token 文本**：`decoded_token` 要给人看。byte-level BPE 会把一个多字节汉字拆到
   相邻 token 上，单个 token 解码会得到替换字符 `�`——所以要用**前面的已采样 token**当上下文
   重新解码（上游 `_verify_tokens` / `_correct_decoded_token`，本项目逐字照抄这套修正）。
   没有 tokenizer 时 decoded_token 一律 None（不是空串：空串会被误读成"这个 token 解码为空"）。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..logprobs import SampleLogprobs, append_logprobs_for_next_position

#: "没有 tokenizer"时给每个候选的占位（上游同名常量：`itertools.repeat(None)`）
NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    tokenizer: object | None
    logprobs: SampleLogprobs
    cumulative_logprob: float
    #: 这条请求要几个（None = 不要 logprobs；-1 = 全词表）
    num_logprobs: int | None

    @classmethod
    def from_new_request(cls, tokenizer, sampling_params) -> "LogprobsProcessor | None":
        """按请求的采样参数建一个；没要 logprobs 就返回 None（不占内存、不做事）。"""
        num_logprobs = sampling_params.num_logprobs
        if num_logprobs is None:
            return None
        return cls(tokenizer=tokenizer, logprobs=[], cumulative_logprob=0.0,
                   num_logprobs=num_logprobs)

    # -------- 一轮的增量 --------

    def update_from_output(self, output) -> None:
        """吃下 `EngineCoreOutput.new_logprobs`（上游同名方法的子集：没有 prompt logprobs）。"""
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)

    def _update_sample_logprobs(self, logprobs_lists) -> None:
        """把这一轮的每个位置追加进容器（上游同名方法）。

        `zip` 在三边里最短的那边停住 —— 这就是"逐请求裁剪"：整批按最大宽度算，
        每条请求只留自己要的几个（上游靠同一处 zip 完成这件事）。
        """
        assert self.num_logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists
        for rank_np, logprobs_np, token_ids_np in zip(ranks_lst, logprobs_lst,
                                                      token_ids_lst):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()

            if self.tokenizer is None:
                decoded_tokens: list[str] | itertools.repeat = NONES
            else:
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=self.tokenizer.batch_decode(
                        [[token_id] for token_id in token_ids]),
                    tokens=token_ids,
                    context_token_ids=self._get_sampled_context_ids(self.logprobs),
                )

            # 第 0 列是**实际采到的 token**（`gather_logprobs` 的约定）
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            append_logprobs_for_next_position(
                self.logprobs, token_ids, logprobs, decoded_tokens, rank,
                self.num_logprobs)

    # -------- 解码修正（上游同款，逐字照抄）--------

    def _get_sampled_context_ids(self, logprobs_source, max_context: int = 4) -> list[int]:
        """取最近的几个**已采样** token 当解码上下文。

        位置 dict 的第 0 项就是那个位置实际采到的 token（`append_logprobs_for_next_position`
        先插它），所以 `next(iter(entry))` 拿到的就是它。4 个足够覆盖任何 UTF-8 多字节序列
        （最长 4 字节）。
        """
        if not logprobs_source:
            return []
        n = len(logprobs_source)
        start = max(0, n - max_context)
        result: list[int] = []
        for i in range(start, n):
            entry = logprobs_source[i]
            if entry is not None:
                result.append(next(iter(entry)))
        return result

    def _verify_tokens(self, decoded_tokens_list: list[str], tokens: list[int],
                       context_token_ids: list[int] | None = None) -> list[str]:
        """把"以替换字符结尾"的候选解码文本用上下文修回来（上游同名方法）。

        `tokens` 是**同一个位置**的几个候选（选中 + top-k），不是连续 token；
        上下文只来自前面已采样的 token。
        """
        if context_token_ids is None:
            context_token_ids = self._get_sampled_context_ids(self.logprobs)

        corrected: dict[int, str] = {}
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("\ufffd"):
                corrected[idx] = self._correct_decoded_token(tokens[idx],
                                                            context_token_ids)
        for idx, text in corrected.items():
            decoded_tokens_list[idx] = text
        return decoded_tokens_list

    def _correct_decoded_token(self, token_id: int, context_token_ids: list[int]) -> str:
        """用一个多字节字符的"前半截"上下文，重新解码出这个 token 补完的部分（上游同名方法）。

        例（byte-level BPE）："你好" 可能被拆成 `<0xE4><0xBD> + <0xA0><0xE5> + <0xA5><0xBD>`，
        单独解码每个 token 都是 `�`。这里把前面 1~4 个已采样 token 与它拼起来解码，
        再减掉"干净前缀"，就得到这个 token 真正贡献的那几个字节。
        """
        assert self.tokenizer is not None

        max_ctx = min(len(context_token_ids), 4)
        for num_ctx in range(1, max_ctx + 1):
            context = context_token_ids[-num_ctx:]
            full_decoded = self.tokenizer.decode(context + [token_id])
            if full_decoded.endswith("\ufffd"):
                continue

            # 上下文里也带"半个字符"的 token，它们的文本要算在这个补完的 token 头上
            clean_end = len(context)
            for j in range(len(context) - 1, -1, -1):
                if self.tokenizer.decode([context[j]]).endswith("\ufffd"):
                    clean_end = j
                else:
                    break
            clean_prefix = (self.tokenizer.decode(context[:clean_end])
                            if clean_end > 0 else "")
            if full_decoded.startswith(clean_prefix):
                return full_decoded[len(clean_prefix):]

            # tokenizer 的规范化可能让前缀对不上：退化成"最长公共前缀"
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]

        return ""
