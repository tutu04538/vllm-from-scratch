"""V2 的原生 MTP 提议者（对应 vLLM `v1/worker/gpu/spec_decode/mtp/speculator.py`）。

上游这个文件只有 58 行，做两件事：

    1. 加载：`load_draft_model()` 与 EAGLE 走同一条加载路径，外加一次
       "**这台模型支不支持 top-k 索引复用**"的探测；
    2. 四个 hook：把 `skip_topk` 开关与 top-k 索引压缩插到"prefill 之前/之后、
       草稿 decode 之前/之后"这四点上。

**为什么 hook 必须成对、而且 prefill 那个必须无条件清除**：`skip_topk=True` 的含义是
"草稿步不要自己算 top-k 索引，直接读共享缓冲里那份"。如果上一轮在中间炸掉、没走到
`on_multi_step_decode_end`，标志就会**残留成 True**，下一轮的第一步就会读到**过期的索引**
（不报错，只是草稿质量偷偷变差）。所以 `on_prefill_begin()` 里那一次清除是**无条件**的
——这正是需求 074 §3.6 那条"异常后下一轮不能残留 skip_topk 状态"。

V2 的接线边界（写在 `init_speculator` 与 `docs/step74_alignment.md`）：MTP 的权重在 **target 的
checkpoint 里**，draft 配置要用 `SpeculativeConfig.derive_mtp_draft_config(target_config)` 从
target 派生。"什么时候派生"在上游是配置期（`SpeculativeConfig.__post_init__` 拿得到 target），
本仓库的 `SpeculativeConfig` 拿不到 target（65 关记录的差异）——V1 由 Runner 派生，V2 侧
**要求显式传入派生好的 draft 配置**，否则明确拒绝（不猜）。
"""

import torch.nn as nn

from ..autoregressive.speculator import AutoRegressiveSpeculator
from ..eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    #: 这台 draft 能不能复用 step 0 算出的 top-k 索引（由 `load_draft_model()` 探测）。
    #: 类属性给默认值，实例属性在加载后覆盖（上游同款写法）。
    share_mtp_topk_indices: bool = False

    @staticmethod
    def _supports_index_sharing(hf_config, draft_model) -> bool:
        """只认 **`model.model`** 这一层接口（与冻结 `DeepSeekMTP` 同构）。

        配置声明 `index_share_for_mtp_iteration=True`，且内层真的实现了
        `set_skip_topk` / `compact_topk_indices`，才启用复用。缺任一项就当不支持，
        不去猜别的属性路径。
        """
        inner = getattr(draft_model, "model", None)
        return bool(
            (hf_config or {}).get("index_share_for_mtp_iteration", False)
            and inner is not None
            and hasattr(inner, "set_skip_topk")
            and hasattr(inner, "compact_topk_indices")
        )

    def _index_model(self):
        """`model.model`：索引缓存/开关的拥有者；缺失时明确报错。"""
        inner = getattr(self.model, "model", None)
        if inner is None:
            raise AttributeError(
                "MTPSpeculator 期望 `model.model` 提供 set_skip_topk/"
                "compact_topk_indices（与冻结 DeepSeekMTP 同构）")
        return inner

    def load_draft_model(self, target_model: nn.Module) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config if spec_config is not None else None
        )
        self.share_mtp_topk_indices = self._supports_index_sharing(
            draft_hf_config, draft_model)
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        """Step 0 自己算 top-k。**无条件**执行，所以中途死掉的一步不会留下复用状态。"""
        if self.share_mtp_topk_indices:
            self._index_model().set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        """把 step 0 写下的 top-k 索引压到"每条请求的最后一枚 token"上，供 step 1+ 复用。"""
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self._index_model().compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        """切到复用模式：step 1+ 跳过索引算子，直接读共享缓冲里那份。"""
        if self.share_mtp_topk_indices:
            self._index_model().set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self._index_model().set_skip_topk(False)
