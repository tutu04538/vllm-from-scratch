"""tokenizer 的加载入口（对应 vLLM `vllm/tokenizers/__init__.py` 的 `cached_tokenizer_from_config`）。

**模块名不叫 `tokenizers`**：本仓库的入口脚本 `python minivllm/demo.py` 会把 `minivllm/`
放进 `sys.path[0]`，那样 `from tokenizers import ...`（transformers 内部要用的第三方包）
会解析到本文件，直接 ImportError（实测）。上游没这个问题是因为它的 `vllm/tokenizers/`
在 `vllm` 包里面、入口脚本的目录永远是仓库根。

本仓库历史上只有两处读 tokenizer 文件：67 关的异构词表（TLI，要建两套 tokenizer 的 token 级
交集）和 68 关的结构化输出（xgrammar 要按 tokenizer 建词表信息）。两处合到这一个入口，
**同一条路径只加载一次**（按目录缓存）：同一个进程里 TLI 的 target tokenizer 与结构化输出
用的是同一份对象，不会出现"两份词表信息不一致"这种事。

上游 README 里对 tokenizer 的三种模式（hf / mistral / deepseek 等）本项目都没有，只有 HF
这一条；`local_files_only=True` 是硬要求——本项目在离线机器上跑，静默去联网下载会让测试
挂在网络上（67 关的 `load_tokenizer` 就是这么做的，这里继承那条约定）。
"""

from __future__ import annotations

_CACHE: dict[str, object] = {}


def load_tokenizer(path: str):
    """按目录加载 HF tokenizer（读不到就明确报错，不静默换一个）。"""
    cached = _CACHE.get(path)
    if cached is not None:
        return cached

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:              # noqa: BLE001 —— 包成"缺 tokenizer"这一件事
        raise RuntimeError(
            f"{path!r} 下没有可加载的 tokenizer 文件（tokenizer.json / "
            f"tokenizer_config.json 等）：{type(exc).__name__}: {exc}") from exc
    _CACHE[path] = tokenizer
    return tokenizer


def cached_tokenizer_from_config(model_config):
    """`ModelConfig` → tokenizer（对应上游同名函数）。

    用 `tokenizer_path`（没单独指定 tokenizer 时就是模型目录，与上游默认一致）。
    """
    return load_tokenizer(model_config.tokenizer_path)
