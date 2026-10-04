"""自定义 Proposer 的接入（需求 62）：把一个"用户自己写的类"变成 Runner 的提议者。

对应上游 `vllm/v1/spec_decode/custom_class_proposer.py:12-73::create_custom_proposer`，
本文件逐条对齐它，包括**错误分类**（哪一步失败就说清是哪一步）：

    config 里没给 model                          → 配置期 ValueError（config.py 里）
    model 不是 module.Class 形式（没有点号）      → ValueError
    import 失败（模块不存在 / 依赖缺失）          → ImportError（带原始异常链）
    模块里没有这个类                              → AttributeError
    类构造失败（签名不接受 VllmConfig 等）        → RuntimeError（带原始异常链）
    实例没有 propose / propose 不可调用           → AttributeError

它解决什么痛点：**换一个候选来源不应该动 Engine**。上游和本仓库都只把 `VllmConfig` 交给这个类，
不把它塞进 Runner、也不给它 Scheduler 的 `Request` 或 `KVCacheManager`；它的唯一契约是
`propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None)` 返回
`list[list[int]]`。所以候选算法可以被替换，而 Scheduler / KV / verifier 一行都不用改。

**不做套壳 Adapter**：上游注释写明"returned directly so the caller can use it without any
wrapper"，验收也要求"合法类实例就是 Runner 持有的对象"。原因是各提议者的签名本来就不一样
（ngram 多一个 `num_speculative_tokens`、ngram_gpu 收显存张量、draft_model 收 `TargetRows`），
硬造一个统一基类就得改掉各自的签名——那属于"自己重新设计接口"，上游没这么做。
"""

import importlib


def create_custom_proposer(vllm_config):
    """导入并实例化用户提供的提议者类（上游同名函数，逐行对齐）。

    类路径来自 `speculative_config.model`（例如 `"my_module.MyCustomProposer"`）。
    返回的对象必须有可调用的 `propose` 方法；**不做包装、不加适配层**。
    """
    assert vllm_config.speculative_config is not None
    spec_config = vllm_config.speculative_config

    backend = spec_config.model
    assert backend is not None

    # 1) 必须是 module.Class 形式（上游在工厂里才检查点号，config 只保证非空）
    if "." not in backend:
        raise ValueError(
            f"Invalid custom proposer module path '{backend}'. "
            "It must be a full module path (e.g., 'module.MyProposerClass')."
        )

    module_path, class_name = backend.rsplit(".", 1)

    # 2) 模块必须能导入（模块不存在、模块内部 import 失败都归到这一类）
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' for custom proposer '{backend}': {e}"
        ) from e

    # 3) 类必须存在（注意用 getattr(..., None) 而不是 hasattr+getattr 两次查找）
    user_class = getattr(module, class_name, None)
    if user_class is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}' "
            f"(speculative_config.model='{backend}')"
        )

    # 4) 用 VllmConfig 构造；构造器抛任何异常都转成"构造失败"并保留原因（__cause__）
    try:
        instance = user_class(vllm_config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate custom proposer class '{backend}': {e}. "
            "The class constructor must accept VllmConfig as argument."
        ) from e

    # 5) `propose` 必须存在**且可调用**（`propose = 123` 这种要在启动期就挡住）
    if not hasattr(instance, "propose"):
        raise AttributeError(
            f"Custom proposer class '{backend}' must have a 'propose' method."
        )
    if not callable(instance.propose):
        raise AttributeError(
            f"Custom proposer class '{backend}' has a 'propose' attribute "
            "but it is not callable."
        )

    # 上游这里打 logger.info（"Loaded custom proposer class '%s' with
    # num_speculative_tokens=%d"）；本仓库没有 logger，改由调用方/文档记录，
    # 不为了对齐而引入日志设施（差异记在 docs/step62_alignment.md §3）。

    return instance
