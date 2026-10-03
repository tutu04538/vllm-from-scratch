"""从本地模型目录读出 `(名字, 张量)`（对应 vLLM `model_executor/model_loader/weight_utils.py` 的一小部分）。

本关只做一件事：**把一个模型目录变成一条权重流**。

    get_safetensors_files(model_dir)   单文件 / index 分片 两种布局
    iter_weights(model_dir)            逐 shard 打开、按 key 惰性取张量、yield

**为什么是"流"而不是"字典"**：真实模型几十 GB，先读进一个大 dict 会把内存打满（本关要加载的
本机 Qwen3-1.7B 是 3.8 GB）。safetensors 支持按 key 惰性读取（`safe_open` + `get_tensor`），
所以按 shard 打开、读完一个再开下一个，峰值内存只有"一个 shard + 正在写的那几个参数"。
这也是 vLLM 的做法（`DefaultModelLoader.get_all_weights()` 返回迭代器）。

**校验**（198 §8 要求"严格检查"，不能靠 `strict=False` 掩盖漏加载）。先跑一遍**只看名字**的
预检（不读张量，开销只是目录元数据），全过了才开始 yield 张量——这样"检查点坏了"不会变成
"加载到一半才发现"：

1. index 点名 的 shard 文件必须都存在；
2. 一个张量必须真的来自 index 说它所在的那个 shard（错放的检查点要被抓出来）；
3. index 里的名字必须**全部**出现（缺失 → 报错）；
4. 同一个名字出现两次 → 报错；
5. **名字必须按模块分组**（每个前缀组连续出现）——这一条服务于下游 `AutoWeightsLoader`
   的 `itertools.groupby`：`groupby` 只合并**相邻**的相同前缀，交错的名字会让同一层的权重
   被分成好几批、后一批覆盖前一批。见 `_check_grouping()`。

注意 3 只覆盖"检查点里声明的名字"。"模型里的参数有没有全部被填上"是另一件事，由
`AutoWeightsLoader` 的覆盖检查回答（`auto_weights_loader.py`）。
"""

import json
import os
from collections.abc import Iterator

import torch
from safetensors import safe_open

# HF 的分片索引文件名。它不是模型文件列表，而是"权重名 → 哪个 shard"，所以既用来找文件，
# 也用来做上面的 2/3 两条校验。
INDEX_FILE = "model.safetensors.index.json"

# 本关支持的权重文件后缀。`.bin`（torch.save 的旧格式）不在这里：它需要反序列化整个文件，
# 没有 safetensors 的惰性读取，本关不实现——但要**明确报错**，不能静默跳过变成"没加载"。
_SUPPORTED_SUFFIX = ".safetensors"


def _read_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _shard_names(path: str) -> list[str]:
    """只读一个 shard 的**元数据**，拿到里面的权重名（不读张量）。"""
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def get_safetensors_files(model_dir: str) -> tuple[list[str], dict[str, str] | None]:
    """找出模型目录里的权重文件。返回 `(文件列表, 权重名→文件名)`；单文件时第二个是 None。

    两种布局都支持（HF 的两种常见形态）：

        model.safetensors                                    单文件
        model.safetensors.index.json + model-00001-of-00002.safetensors ...   分片
    """
    index_path = os.path.join(model_dir, INDEX_FILE)
    if os.path.isfile(index_path):
        weight_map = _read_json(index_path).get("weight_map")
        if not weight_map:
            raise ValueError(f"{index_path} 里没有 weight_map：分片索引必须有这张表")
        # 文件列表从 weight_map 推出来（而不是 glob 目录）：index 说用哪些 shard 就用哪些，
        # 目录里多出来的文件（上一版的残留、别的精度的副本）不该被读进来。
        files = sorted({os.path.join(model_dir, name) for name in weight_map.values()})
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"{INDEX_FILE} 点名了这些 shard，但目录里没有：{missing}")
        return files, weight_map

    single = os.path.join(model_dir, "model.safetensors")
    if os.path.isfile(single):
        return [single], None

    # 都没有：说清楚目录里到底有什么，别让调用方猜
    entries = sorted(os.listdir(model_dir)) if os.path.isdir(model_dir) else []
    unsupported = [name for name in entries
                   if name.endswith((".bin", ".pt", ".pth", ".gguf"))]
    if unsupported:
        raise NotImplementedError(
            f"{model_dir} 里只有本关不支持的权重格式 {unsupported}；"
            f"只支持 {_SUPPORTED_SUFFIX}（见 weight_utils.py 的模块说明）")
    raise FileNotFoundError(
        f"{model_dir} 里既没有 model.safetensors 也没有 {INDEX_FILE}；目录内容：{entries}")


def _check_grouping(names: list[str]) -> None:
    """检查"每个前缀组连续出现"（`AutoWeightsLoader` 的 groupby 依赖它）。

    做法：按顺序走，维护"上一个名字的各层前缀路径"。与新名字的最长公共前缀之下的那些路径
    就算**关闭**了；新名字里任何一条已关闭的路径再出现，就是交错。

    为什么值得单独写一个检查：交错不会报错，只会让同一层的权重被加载两遍（后一遍覆盖前一遍），
    或者让"打包参数"的先到 shard 被后到的覆盖——是最难查的一类错。
    """
    closed: set[tuple[str, ...]] = set()
    previous: tuple[str, ...] = ()
    for name in names:
        parts = tuple(name.split("."))
        common = 0
        while (common < len(previous) and common < len(parts)
               and previous[common] == parts[common]):
            common += 1
        for depth in range(common + 1, len(previous) + 1):
            closed.add(previous[:depth])          # 上一个名字里更深的前缀，从此关闭
        for depth in range(common + 1, len(parts) + 1):
            if parts[:depth] in closed:
                raise ValueError(
                    f"权重流不是按模块分组的：{'.'.join(parts[:depth])!r} 这一组已经结束，"
                    f"现在又出现了 {name!r}。加载器用 itertools.groupby 按前缀分组，"
                    f"只认相邻的同一组；请让权重按名字分组后连续产出")
        previous = parts


def iter_weights(model_dir: str) -> Iterator[tuple[str, torch.Tensor]]:
    """流式产出 `(权重名, 张量)`。名字是**检查点里的原名**（如 `model.layers.0.self_attn.q_proj.weight`），
    改名与打包路由不是这一层的事（那是 `AutoWeightsLoader` + 模型的 `load_weights`）。
    """
    files, weight_map = get_safetensors_files(model_dir)

    # ---- 预检：只看名字，全过了再开始读张量 ----
    all_names: list[str] = []
    seen: set[str] = set()
    for path in files:
        shard = os.path.basename(path)
        for name in _shard_names(path):
            if name in seen:
                raise ValueError(
                    f"权重 {name!r} 出现了两次（现在在 {shard}）："
                    f"检查点里同一个参数只能有一个来源")
            expected_shard = shard if weight_map is None else weight_map.get(name)
            if expected_shard != shard:
                raise ValueError(
                    f"权重 {name!r} 出现在 {shard}，但索引说它在 {expected_shard}："
                    f"检查点与索引对不上")
            seen.add(name)
            all_names.append(name)
    if weight_map is not None:
        missing = sorted(set(weight_map) - seen)
        if missing:
            raise ValueError(
                f"索引里有这些权重名，但 shard 里一个都没读到：{missing[:8]}"
                f"{' …' if len(missing) > 8 else ''}（共 {len(missing)} 个）")
    _check_grouping(all_names)

    # ---- 正式读：按 shard 打开，逐个 key 惰性取张量 ----
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def default_weight_loader(param, loaded_weight: torch.Tensor, shard_id=None) -> None:
    """参数没挂 `weight_loader` 时的兜底：整块覆盖，形状必须严格相同。

    对应 vLLM 的同名函数。本关所有参数都挂了自己的加载器（见 `layers/linear.py`），这个兜底
    仍然保留：**"没挂加载器"应该是"整块覆盖"这种平庸行为，而不是静默跳过**。
    """
    if shard_id is not None:
        raise ValueError(f"参数 {tuple(param.shape)} 没有挂 weight_loader，"
                         f"却收到了 shard_id={shard_id!r}：打包映射指错了目标")
    if tuple(param.shape) != tuple(loaded_weight.shape):
        raise ValueError(f"权重形状不匹配：目标 {tuple(param.shape)}，"
                         f"收到 {tuple(loaded_weight.shape)}")
    with torch.no_grad():
        param.copy_(loaded_weight)
