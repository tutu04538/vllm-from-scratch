"""Medusa 多头草稿（对应 vLLM `model_executor/models/medusa.py`）。

**Medusa 是什么**（论文 <https://arxiv.org/abs/2401.10774>，参考实现 FasterDecoding/Medusa）：
在 target 旁边挂 N 个**小 head**，每个 head 都是"若干层残差 MLP + 自己的 lm_head"：

    blocks[h] :  x → x + SiLU(Linear(x))  × num_hidden_layers        （残差块）
    logits[h] = lm_heads[h](blocks[h](hidden))                       （每个 head 一个词表头）

关键是**所有 head 读同一份 target hidden**（就是 target 本轮算出的最后一个 token 的 hidden）：

    hidden ──┬─→ blocks[0] → lm_head[0] → argmax → 草稿 0
             ├─→ blocks[1] → lm_head[1] → argmax → 草稿 1
             └─→ blocks[2] → lm_head[2] → argmax → 草稿 2      ← N 个 head 并行，不是串行

于是 **N 个 head 提 N 枚草稿**，第 2 枚不再条件于第 1 枚（这就是"多头并行"与 EAGLE/MTP
"自回归迭代"的本质区别：并行省时间，但候选之间互相"听不见"）。

**本仓库与上游的差异**（逐条见 docs/step66_alignment.md §3）：

- 上游 `Medusa.__init__` 从 `vllm_config.speculative_config.draft_model_config.hf_config` 里读配置；
  本仓库的模型只吃一个 config dict（与 Qwen3/EAGLE3/MTP 一致），Runner 已经把 draft 配置放好。
- `original_lm_head` 那一条（所有 head 共享**同一个** lm_head，权重名 `lm_head.weight`）照抄；
  上游把 `self.lm_heads` 写成普通 list（同名参数只出现一次），本仓库同样处理。
- `load_weights()` 上游对"检查点里有、模型里没有"的名字**静默丢弃**；本仓库把两类**故意不加载**
  的名字显式记账（旧 checkpoint 的 fc bias、超出 K 的 head），其余未知名字当场报错——
  这正是"这个家族的块没实现"的唯一信号（AGENTS §8 的约定，差异记在 §3）。
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from ..layers import LogitsProcessor, ParallelLMHead
from ..model_loader.weight_utils import default_weight_loader
from .qwen3 import maybe_prefix


class ResidualBlock(nn.Module):
    """一个 head 的残差 MLP 块（对应上游同名类）：`x = x + SiLU(Linear(x))` 叠 `num_layers` 次。

    `medusa_fc_bias`：FasterDecoding 训练脚本里的 `nn.Linear` **默认带 bias**，所以旧 checkpoint
    里一般有 `{h}.{l}.linear.bias`；vLLM 默认按**不带 bias** 建（这个开关默认 False），
    要用 bias 就得在 draft 配置里写 `medusa_fc_bias: true`。上游是
    `getattr(config, "medusa_fc_bias", False)`，本仓库的配置是 dict，语义完全相同。
    """

    def __init__(self, config: dict, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=bool(config.get("medusa_fc_bias", False)))
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


class Medusa(nn.Module):
    """Medusa head 本体（对应上游 `Medusa`；注册名 `MedusaModel`）。

    上游 docstring 里那两条"与参考实现不同"照抄在下面（它们决定了本关的边界）：

    1. 只从 top-1 token 生成候选（每个 head 一个 argmax）——**不做论文里的树**；
    2. 可选 `token_map`：把草稿词表截断成"最常用的 k 个 token"（`truncated_vocab_size <
       vocab_size` 且检查点里有 `token_map` 时才启用），`compute_logits()` 会把截断 logits
       散回原词表、其余位置填 `-inf`。它只减少草稿头的采样开销，不影响接受率多少。
    """

    def __init__(self, config: dict, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            ResidualBlock(config=config, hidden_size=config["hidden_size"],
                          num_layers=config["num_hidden_layers"])
            for _ in range(config["num_heads"])
        ])
        self.orig_vocab_size = config["vocab_size"]
        self.truncated_vocab_size = config["truncated_vocab_size"]

        if config.get("original_lm_head", False):
            # 所有 head **共享同一个** lm_head：权重名只有 `lm_head.weight` 一份
            # （上游同样写成普通 list，于是 named_parameters 里只出现一次）
            self.lm_head = ParallelLMHead(self.truncated_vocab_size, config["hidden_size"],
                                          prefix=maybe_prefix(prefix, "lm_head"))
            self.lm_heads = [self.lm_head for _ in range(config["num_heads"])]
        else:
            self.lm_heads = nn.ModuleList([
                ParallelLMHead(config["vocab_size"], config["hidden_size"],
                               prefix=maybe_prefix(prefix, f"lm_heads.{i}"))
                for i in range(config["num_heads"])
            ])

        self.logits_processor = LogitsProcessor(
            config["vocab_size"], self.truncated_vocab_size, config.get("logit_scale", 1.0))

        self.token_map: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        """**所有 head 并行**读同一份 hidden（每个 head 一个块输出，list 长度 = num_heads）。"""
        return [block(hidden_states) for block in self.blocks]

    def compute_logits(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        """逐 head 过自己的 lm_head，返回 logits 列表（长度 = num_heads）。"""
        logits_lst: list[torch.Tensor] = []
        for hidden, lm_head in zip(hidden_states, self.lm_heads):
            _logits = self.logits_processor(lm_head, hidden)
            if self.token_map is None:
                logits_lst.append(_logits)
            else:
                # 截断词表：把 k 宽的 logits 散回原词表，其余位置 -inf（采样永远选不到）
                logits_lst.append(
                    -torch.inf * torch.ones(
                        size=(*_logits.shape[:-1], self.orig_vocab_size),
                        device=_logits.device, dtype=_logits.dtype))
                logits_lst[-1][..., self.token_map] = _logits
        return logits_lst

    @staticmethod
    def remap_old_checkpoint_key(name: str) -> str:
        """旧 FasterDecoding checkpoint 的名字 → 本模型的名字（上游 `_remap_old_checkpoint_key`）。

        真实旧文件的键（本机实测 `FasterDecoding/medusa-vicuna-7b-v1.3/medusa_lm_head.pt`
        的 state_dict，见 docs/step66_alignment.md §2）：

            `0.0.linear.weight` / `0.0.linear.bias`   → `blocks.0.layers.0.weight` / `.bias`
            `0.1.weight`                              → `lm_heads.0.weight`

        即"head 号 . 块内序号 . 参数名"：带 `linear` 的是残差块，不带的是那个 head 的 lm_head。
        有些转换脚本还会留下 `medusa_heads.` 前缀（上游在 `load_weights()` 里先剥掉再调本函数）。
        """
        parts = name.split(".")
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            head, layer = parts[0], parts[1]
            rest = parts[2:]
            if "linear" in rest:
                param = ".".join(rest[rest.index("linear") + 1:])
                return f"blocks.{head}.layers.{layer}.{param}"
            return f"lm_heads.{head}.{'.'.join(rest)}"
        return name

    @staticmethod
    def _is_unused_bias(name: str) -> bool:
        """检查点里有、但本配置**故意不建**的 bias 槽位。

        `blocks.{h}.layers.{l}.bias`：配置没开 `medusa_fc_bias` 时残差块不带 bias
        （旧 FasterDecoding 的 `nn.Linear` 默认带，所以文件里通常有这一项）；
        `lm_heads.{h}.bias` / `lm_head.bias`：本模型的 lm_head 不带 bias（上游同款）。
        """
        if not name.endswith(".bias"):
            return False
        parts = name.split(".")
        if len(parts) == 5 and parts[0] == "blocks" and parts[1].isdigit() \
                and parts[2] == "layers" and parts[3].isdigit():
            return True
        return len(parts) == 3 and parts[0] in ("lm_heads", "lm_head")

    def _is_deliberately_unused(self, name: str) -> bool:
        """检查点里有、上游也**静默丢弃**的那几类名字（本仓库记账后丢，见 `load_weights`）。

        1. `blocks.{h≥K}.*` / `lm_heads.{h≥K}.*`：K 改写了 `num_heads`，多出来的 head 没人要；
        2. `lm_heads.{h≥1}.weight` **且配置开了 `original_lm_head`**：所有 head 共享同一个
           lm_head，检查点里逐 head 的那几份只用第 0 份（第 0 份在上面被改名成 `lm_head.weight`）；
        3. 本配置不建的 bias（见 `_is_unused_bias`）。
        """
        if self._is_unused_bias(name):
            return True
        parts = name.split(".")
        if len(parts) < 3 or parts[0] not in ("blocks", "lm_heads") or not parts[1].isdigit():
            return False
        if int(parts[1]) >= len(self.blocks):
            return True
        return bool(self.config.get("original_lm_head", False)) and parts[0] == "lm_heads"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """挑出本模型的参数并加载，返回**已填写的参数名集合**（上游同款返回）。

        名字要先归一（旧格式 → `blocks.*` / `lm_heads.*`）。检查点里有、本模型**故意不要**的
        名字（`_is_deliberately_unused()`）记账后丢掉；其余认不出的名字**当场报错**
        （上游对两者一律静默丢弃，差异见 docs/step66_alignment.md §3）。
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        weights_map: dict[str, torch.Tensor] = {}
        dropped: list[str] = []

        for name, loaded_weight in weights:
            # 上游同款：某些转换脚本留下的 `medusa_heads.` 前缀先剥掉
            original_name = name
            name = name.replace("medusa_heads.", "")
            name = self.remap_old_checkpoint_key(name)

            if name == "token_map":
                # 只有"截断词表"配置才收这份映射（上游同款判据）
                if self.truncated_vocab_size < self.orig_vocab_size:
                    self.token_map = nn.Parameter(loaded_weight, requires_grad=False)
                else:
                    dropped.append(name)
            elif name in params_dict:
                weights_map[name] = loaded_weight
            elif (self.config.get("original_lm_head", False)
                  and name == "lm_heads.0.weight"):
                # 共享 lm_head：检查点里是逐 head 的名字，模型里只有一份
                weights_map["lm_head.weight"] = loaded_weight
            elif self._is_deliberately_unused(name):
                dropped.append(name)
            else:
                raise ValueError(
                    f"Medusa 检查点里有认不出的参数 {original_name!r}（归一后 {name!r}）："
                    f"本模型有的参数是 {sorted(params_dict)}；如果是**这个家族的块没实现**，"
                    f"应当在这里报错，静默跳过会让真 checkpoint 跑出一个悄悄错的模型")

        for name, loaded_weight in weights_map.items():
            if ("lm_head" in name and self.token_map is not None
                    and loaded_weight.shape[0] > self.token_map.shape[0]):
                # 截断词表：检查点里是整份词表，先按 token_map 选出那 k 行
                loaded_weight = loaded_weight[self.token_map]
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if self.token_map is not None:
            # 上游同款：token_map 是加载期才挂上来的 Parameter，挪到与 lm_head 同一个设备
            self.token_map = nn.Parameter(self.token_map.to(
                device=self.lm_heads[0].weight.device), requires_grad=False)
            loaded_params.add("token_map")

        if self.truncated_vocab_size < self.orig_vocab_size and self.token_map is None:
            raise ValueError(
                f"draft 配置声明了 truncated_vocab_size={self.truncated_vocab_size} < "
                f"vocab_size={self.orig_vocab_size}，但检查点里没有 token_map："
                f"截断词表的 logits 散不回原词表（上游这里是 assert），不能继续")
        self._check_coverage(params_dict, loaded_params)
        self.dropped_weights = tuple(dropped)
        return loaded_params

    def _check_coverage(self, params_dict: dict, loaded_params: set[str]) -> None:
        """**每个参数都必须被填上**（与 `AutoWeightsLoader._check_coverage` 同一条约定）。

        旧 checkpoint 的 head 数比 K 多时，多出来的 head 根本不在 `params_dict` 里（模型只建了
        K 个块），所以覆盖检查只盯"模型自己的参数"；反过来 K 比检查点里的 head 多时，
        缺的那些权重会让这里当场报错——**不会**留下 `torch.empty` 的随机参数。
        """
        missing = sorted(name for name in params_dict if name not in loaded_params)
        if missing:
            raise ValueError(
                f"Medusa 有 {len(missing)} 个参数没有被加载：{missing}\n"
                f"（最常见的原因是 K（num_speculative_tokens）比检查点里的 head 多："
                f"本模型按 K 建 blocks/lm_heads，缺的那些没法凭空造出来——"
                f"上游会把 num_heads 直接改写成 K，然后静默留下未初始化的参数）")
