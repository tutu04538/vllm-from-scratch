"""按名字把权重流灌进模型（对应 vLLM `model_executor/models/utils.py` 的
`WeightsMapper` 与 `AutoWeightsLoader`）。

它回答一个问题：**加载器怎么知道 `...self_attn.q_proj.weight` 该进哪个参数的哪一段？**

两步，缺一不可：

    1. 名字路由（WeightsMapper）   ...self_attn.q_proj.weight → ...self_attn.qkv_proj.weight
                                  并且**顺带记下**它是打包参数的第几段（shard_id="q"）
    2. 段内写入（Parameter.weight_loader）  交给目标参数自己的加载器：q_proj 写 [0:q_size]

第 2 步"怎么装"跟着**参数**走（`param.weight_loader`，见 `layers/linear.py`），加载器不写
一堆 if。"第几段"这个信息 vLLM 是**挂在张量对象的属性上**（`tensor.shard_id = "q"，
`loader` 再 `getattr(weight, "shard_id", None)` 取回来——`vllm/model_executor/layers/
linear.py` 的 `weight_loader_v2` 就是这么做的，本关照抄这条通道，因为它让"改名"和"分段"
在一次遍历里同时完成，不必构造中间对象。

`AutoWeightsLoader` 是**递归委派**的：

    _load_module("", Qwen3ForCausalLM, weights)
      └─ 按第一段名字分组："model" / "lm_head"
           ├─ "lm_head" 在 skip_prefixes 里 → 整组丢（tied embedding，见 qwen3.py）
           └─ "model" 是子模块，且它自己有 load_weights() → **把子迭代器交给它**
                └─ Qwen3Model.load_weights 里再来一个 AutoWeightsLoader（带它自己的
                   hf_to_vllm_mapper），于是打包映射只写在"离参数最近"的那一层

**这是简化版**：vLLM 的 WeightsMapper 还有正则/前缀/后缀/重命名五类映射与 `__or__` 合并，
本关只需要 `orig_to_new_stacked`（打包映射）一种，其余不实现（差异账本里记着）。但遍历子模块、
委派子 `load_weights`、缺失/未知/重复检查、覆盖检查这些**职责**是真的实现了，不是一个改名的
`load_state_dict`。
"""

import itertools
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.parameter import Parameter

from .weight_utils import default_weight_loader

# 打包参数里"第几段"：QKV 用 'q'/'k'/'v'，gate/up 用 0/1（与 vLLM 的 ShardId 同形）
ShardId = str | int


@dataclass(frozen=True)
class WeightsMapper:
    """检查点里的名字 → 模型里的名字，外加"这是打包参数的第几段"。

    `orig_to_new_stacked` 的每一项是 `子串: (替换成的子串, shard_id)`，与 vLLM 同构：

        ".q_proj": (".qkv_proj", "q")      → ...self_attn.qkv_proj.weight, shard_id="q"
        ".gate_proj": (".gate_up_proj", 0)

    用**子串替换而不是整名匹配**，是为了让 `model.layers.3.self_attn.q_proj.weight` 这种
    带层级前缀的名字一次就能改对（层号不同、后缀相同）。
    """

    orig_to_new_stacked: Mapping[str, tuple[str, ShardId]] = field(default_factory=dict)
    # 63 关（EAGLE3）需要"整段前缀改名"：检查点里 draft 的那一层叫 `midlayer.*`，
    # 模型里叫 `layers.0.*`（上游用同一个字段）。先做 substring 再做 stacked。
    orig_to_new_substr: Mapping[str, str] = field(default_factory=dict)

    def _map_name_with_shard(self, key: str) -> tuple[str, ShardId | None] | None:
        """返回 `(新名字, shard_id)`；`None` 表示这个名字要丢掉（本关没有这种规则，留给后续）。"""
        shard_id: ShardId | None = None
        for substring, new_substring in self.orig_to_new_substr.items():
            if substring in key:
                key = key.replace(substring, new_substring)
        for substring, (new_key, new_shard_id) in self.orig_to_new_stacked.items():
            if substring in key:
                key = key.replace(substring, new_key, 1)
                shard_id = new_shard_id
        return key, shard_id

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for name, data in weights:
            result = self._map_name_with_shard(name)
            if result is None:
                continue
            out_name, shard_id = result
            if shard_id is not None:
                data.shard_id = shard_id          # 挂在张量上，见模块说明
            yield out_name, data


class AutoWeightsLoader:
    """递归遍历模块树，把权重流里的每一项交给对应的参数。"""

    def __init__(self, module: nn.Module, *, skip_prefixes: list[str] | None = None,
                 skip_substrs: list[str] | None = None) -> None:
        self.module = module
        # 明确"这些名字我们**故意**不加载"（tied embedding 的 lm_head）。它同时决定覆盖检查
        # 的豁免名单——否则"故意不加载"会被当成"漏加载"。
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []

    # -------- 名字工具 --------

    @staticmethod
    def _groupby_prefix(weights: Iterable[tuple[str, torch.Tensor]]):
        """按**第一段**名字分组，组内把名字剥掉第一段再交下去。

        `itertools.groupby` 只合并相邻的同一前缀，所以要求权重流按模块分组连续产出——
        `weight_utils.iter_weights()` 的预检保证了这一点（`_check_grouping`）。
        """
        parts_and_data = ((name.split(".", 1), data) for name, data in weights)
        for prefix, group in itertools.groupby(parts_and_data, key=lambda item: item[0][0]):
            yield prefix, (("" if len(parts) == 1 else parts[1], data)
                           for parts, data in group)

    @staticmethod
    def _get_qualname(prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix
        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return (any(qualname.startswith(prefix) for prefix in self.skip_prefixes)
                or any(substring in qualname for substring in self.skip_substrs))

    # -------- 递归加载 --------

    def _load_module(self, base_prefix: str, module: nn.Module,
                     weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[str]:
        """把一组（名字已剥掉本层前缀的）权重分派给 `module` 的子模块或参数。

        三种去向，顺序就是优先级：

        1. 子模块**自己声明了 `load_weights`** → 整组交给它（本关的 Qwen3Model 就是这样，
           打包映射写在那一层）；
        2. 名字对应本层的一个参数 → 调它的 `weight_loader`；
        3. 都不是 → 报错，并把本层可用的参数名列出来（未知名字不能静默丢掉）。
        """
        # 递归进来时先看"这个模块会不会自己加载"。`module != self.module` 是为了避免无限递归：
        # 本函数通常就是从模块自己的 load_weights 里调起来的。
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    raise ValueError(
                        f"{type(module).__name__}.load_weights() 没有返回已加载参数名集合，"
                        f"无法做覆盖检查（本关要求它返回 set[str]）")
                for name in loaded_params:
                    yield self._get_qualname(base_prefix, name)
                return

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    continue
                yield from self._load_module(prefix, child_modules[child_prefix], child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    continue
                yield from self._load_param(prefix, child_params[child_prefix], child_weights)
            elif self._can_skip(prefix + ".") or self._can_skip(prefix):
                continue
            else:
                available = sorted(name for name, _ in module.named_parameters(recurse=True))
                raise ValueError(
                    f"{type(self.module).__name__} 里没有名为 {prefix!r} 的子模块或参数；"
                    f"{type(module).__name__} 下可用的参数：{available}")

    def _load_param(self, base_prefix: str, param: Parameter,
                    weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[str]:
        """一个参数对应权重流里的**一项**（组里多余的项说明名字指到了叶子下面，直接报错）。"""
        for weight_name, weight_data in weights:
            qualname = self._get_qualname(base_prefix, weight_name)
            if self._can_skip(qualname):
                continue
            if weight_name != "":
                raise ValueError(
                    f"权重 {qualname!r} 想加载到单个参数 {base_prefix!r} 里面："
                    f"参数没有子结构，名字要么正好等于它，要么就是拼错了")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            shard_id = getattr(weight_data, "shard_id", None)
            weight_loader(param, weight_data, shard_id)
            yield qualname

    # -------- 入口 --------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]],
                     *, mapper: WeightsMapper | None = None) -> set[str]:
        """加载并返回**已填写的参数名集合**（相对本模块，形如 `layers.0.self_attn...`）。"""
        if mapper is not None:
            weights = mapper.apply(weights)
        # 先按 skip 过滤，再分组：被跳过的名字不该参与 groupby（否则空组会掉进"未知名字"分支）
        weights = ((name, data) for name, data in weights if not self._can_skip(name))
        loaded = set(self._load_module("", self.module, weights))
        self._check_coverage(loaded)
        return loaded

    def _check_coverage(self, loaded: set[str]) -> None:
        """**每个参数都必须被填上**——这是本关对 `strict=False` 的替代。

        vLLM 用 `named_parameters()` 与加载记录对账（漏了 q_norm 这类权重是最典型的静默错误：
        模型能跑、输出只是"有点不对"）。这里做完账后直接报错，把问题挡在加载阶段。

        两个豁免：`skip_prefixes` 里声明过的名字（故意不加载，如 tied 的 lm_head），以及
        重复参数——`named_parameters()` 本身对共享的 Parameter 只报第一个名字，所以 tied
        embedding 只会以 `model.embed_tokens.weight` 出现一次。

        只对账**参数**不对账 buffer：本关唯一的 buffer 是 RoPE 的 `cos_sin_cache`，它是算出来的、
        检查点里没有；vLLM 的 `_add_loadable_non_param_tensors` 处理的是 batchnorm 统计量这类
        检查点里确实有的 buffer，本关不需要。
        """
        expected = {name for name, _ in self.module.named_parameters()}
        missing = sorted(name for name in expected - loaded if not self._can_skip(name))
        if missing:
            raise ValueError(
                f"{type(self.module).__name__} 有 {len(missing)} 个参数没有被加载：{missing}\n"
                f"（加载器不允许 strict=False 式的静默跳过：漏掉的参数会保持随机初始化，"
                f"模型照样能跑，只是结果不对）")
