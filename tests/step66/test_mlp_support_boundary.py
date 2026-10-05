"""step66：`mlp_speculator` 的**支持缺口**（需求 066 §3）。

本关对 MLP speculator 的交付是"**正确识别缺口**"，不是"实现它"：

1. 上游配置层认得它：`transformers_utils/configs/mlp_speculator.py::MLPSpeculatorConfig`
   能解析 `model_type: mlp_speculator` 的 config.json，`SpeculativeConfig` 里也有
   `method = "mlp_speculator"` 的自动推断分支；
2. 但**模型层没有它**：`model_executor/models/registry.py:689` 那行是注释掉的
   （`# Temporarily disabled.` / `# TODO(woosuk): Re-enable this once the MLP Speculator is
   supported in V1.`），所以 `ModelRegistry._try_inspect_model_cls("MLPSpeculatorPreTrainedModel")`
   返回 `None`，`ModelConfig(...)` 当场 `ValidationError: ... are not supported for now.`；
3. `model_executor/models/mlp_speculator.py` 里 `MLPSpeculator` **没有 `forward`**
   （只剩 `__init__` / `load_weights`，是 V0 时代的遗留件）；
4. V1 的 `GPUModelRunner` 里**一次都没提** `mlp_speculator`——提议者分派会落到
   `ValueError: Unknown speculative decoding method`。

本仓库的做法：生产路径**保持不支持**，但把失败点变成**明确报错**（`NotImplementedError` +
"版本缺口"字样），并且**不写一套自创的 MLP 实现去骗过枚举**（那是自创算法冒充对齐）。
下面每条都是"预期失败"的 `pytest.raises` 断言，**不是** skip 之后在文档里写"已支持"。
"""

import ast
import json
from pathlib import Path

import pytest

from minivllm import ModelConfig, SpeculativeConfig
from minivllm.config import MLP_SPECULATOR_MODEL_TYPE

ROOT = Path(__file__).resolve().parents[2]
import vllm  # noqa: E402  （差分用：只允许测试 import 上游）

VLLM_DIR = Path(vllm.__file__).parent

# 一份"像真的" MLP speculator 配置（字段名与上游 `MLPSpeculatorConfig` 一致；
# 官方 checkpoint 是 `ibm-fms/*-mlp-speculator` 那一族）
MLP_CONFIG = {
    "architectures": ["MLPSpeculatorPreTrainedModel"],
    "model_type": "mlp_speculator",
    "hidden_size": 32,
    "vocab_size": 11,
    "n_predict": 3,
    "num_layers": 2,
    "top_k_tokens_per_head": [3, 2, 1],
}


def write_mlp_dir(tmp_path) -> str:
    directory = tmp_path / "mlp_speculator"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(MLP_CONFIG, indent=2) + "\n")
    return str(directory)


# ---------------------------------------------------------------- 本仓库：明确拒绝


def test_method_is_rejected_as_version_gap():
    """显式给 `method="mlp_speculator"` → 版本缺口报错（不是静默回归 draft_model）。"""
    with pytest.raises(NotImplementedError) as excinfo:
        SpeculativeConfig(method="mlp_speculator", num_speculative_tokens=3,
                          draft_model_config=ModelConfig(
                              model="/nonexistent", dtype="float32", max_model_len=64,
                              hf_config=MLP_CONFIG))
    message = str(excinfo.value)
    assert "版本缺口" in message and "mlp_speculator" in message
    # 报错信息要写清"以后怎么才能做"，而不是一句"不支持"
    assert "上游提交" in message and "不得悄悄换参考版本" in message


def test_method_is_inferred_from_config_then_rejected(tmp_path):
    """draft 配置自己声明 `model_type=mlp_speculator` → 配置推断认得它，随后明确拒绝。"""
    directory = write_mlp_dir(tmp_path)
    # 注意：报错本身就证明"认出来了"——如果把 model_type 当没看见、默认成 draft_model，
    # 这个构造会**成功**（本仓库的 draft_model 只要有 draft 配置就能建）。这条判定的上游出处是
    # `vllm/config/speculative.py` 的 `elif ... == "mlp_speculator": self.method = ...`
    with pytest.raises(NotImplementedError, match="版本缺口"):
        SpeculativeConfig(num_speculative_tokens=3,
                          draft_model_config=ModelConfig(
                              model=directory, dtype="float32", max_model_len=64,
                              hf_config=MLP_CONFIG))


def test_our_registry_has_no_mlp_model_class():
    """本仓库注册表里同样没有这个结构（与上游的 `_try_inspect_model_cls → None` 对应）。"""
    from minivllm.models.registry import get_model_class

    with pytest.raises(ValueError, match="没有注册的模型结构"):
        get_model_class("MLPSpeculatorPreTrainedModel")


def test_runner_dispatch_has_no_mlp_branch(tmp_path):
    """就算绕过配置校验，Runner 的分派也没有这条路（落到"未知的投机方法"）。"""
    from minivllm.worker.gpu_model_runner import GPUModelRunner

    directory = write_mlp_dir(tmp_path)
    spec = SpeculativeConfig(method="medusa", num_speculative_tokens=2,
                             draft_model_config=ModelConfig(
                                 model=directory, dtype="float32", max_model_len=64,
                                 hf_config={**MLP_CONFIG, "model_type": "medusa",
                                            "architectures": ["MedusaModel"]}))
    object.__setattr__(spec, "method", MLP_SPECULATOR_MODEL_TYPE)   # 绕过配置期校验
    from minivllm import CacheConfig, DeviceConfig, SchedulerConfig, VllmConfig
    from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir

    target_dir = tiny_qwen3_dir("tiny_gqa")
    config = VllmConfig(
        model_config=ModelConfig(model=target_dir, dtype="float32", max_model_len=64,
                                 hf_config=tiny_qwen3_config("tiny_gqa")),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=8),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=16),
        device_config=DeviceConfig(device="cpu"), speculative_config=spec)
    runner = GPUModelRunner(config, "cpu")
    with pytest.raises(ValueError, match="未知的投机方法"):
        runner._build_proposer()


def test_production_package_keeps_no_mlp_implementation():
    """"不加一套自创实现骗过枚举"：生产包里除了那条版本缺口报错，没有别的 MLP 代码。"""
    mentions = set()
    for path in (ROOT / "minivllm").rglob("*.py"):
        if "mlp_speculator" in path.read_text(encoding="utf-8"):
            mentions.add(path.relative_to(ROOT).as_posix())
    assert mentions == {"minivllm/config.py"}, mentions


# ---------------------------------------------------------------- 上游：缺口证据


def test_upstream_config_class_parses_the_type(tmp_path):
    """证据 1：上游**配置层**认得 `mlp_speculator`（所以缺口不在配置解析）。"""
    from vllm.transformers_utils.configs.mlp_speculator import MLPSpeculatorConfig

    directory = write_mlp_dir(tmp_path)
    config = MLPSpeculatorConfig.from_pretrained(directory)
    assert config.model_type == "mlp_speculator"
    assert config.n_predict == MLP_CONFIG["n_predict"]
    assert config.hidden_size == MLP_CONFIG["hidden_size"]
    # MLP speculator 里与 K 对应的字段是 `num_lookahead_tokens`（= n_predict），
    # 与 Medusa 的 `num_lookahead_tokens → num_heads` 是同一个字段名的两种用法
    assert config.num_lookahead_tokens == MLP_CONFIG["n_predict"]
    # 上游的自动推断分支确实存在（不然"配置认得"这句话没有出处）
    speculative_src = (VLLM_DIR / "config" / "speculative.py").read_text(encoding="utf-8")
    assert 'model_type == "mlp_speculator"' in speculative_src


def test_upstream_registry_line_is_commented_out():
    """证据 2：注册表里那行是注释掉的 → `_try_inspect_model_cls` 返回 None。"""
    from vllm.model_executor.models.registry import ModelRegistry

    assert ModelRegistry._try_inspect_model_cls("MLPSpeculatorPreTrainedModel") is None
    registry_src = (VLLM_DIR / "model_executor" / "models" / "registry.py").read_text(
        encoding="utf-8")
    lines = [line for line in registry_src.splitlines() if "MLPSpeculatorPreTrainedModel" in line]
    assert lines, "上游 registry 里连那行注释都没有了：本关的版本缺口说明需要重新核对"
    assert all(line.lstrip().startswith("#") for line in lines), lines
    assert "Temporarily disabled" in registry_src


def test_upstream_model_config_rejects_the_architecture(tmp_path):
    """证据 2 的后果：拿这份配置建 `ModelConfig` 当场 ValidationError（死在第一步）。"""
    import pydantic
    from vllm.config import ModelConfig as UpstreamModelConfig

    directory = write_mlp_dir(tmp_path)
    with pytest.raises(pydantic.ValidationError, match="not supported for now"):
        UpstreamModelConfig(model=directory, dtype="float32", max_model_len=64)


def test_upstream_model_class_has_no_forward():
    """证据 3：`MLPSpeculator` 只有 `__init__` / `load_weights`，没有 `forward`。"""
    source = (VLLM_DIR / "model_executor" / "models" / "mlp_speculator.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "MLPSpeculator" in classes and "MLPSpeculatorLayerNorm" in classes
    methods = {node.name for node in classes["MLPSpeculator"].body
               if isinstance(node, ast.FunctionDef)}
    assert "forward" not in methods, methods
    assert {"__init__", "load_weights"} <= methods, methods


def test_upstream_runner_has_no_mlp_dispatch_branch():
    """证据 4：V1 的 Runner 里一次都没提 `mlp_speculator`（分派会落到"未知方法"）。"""
    runner_src = (VLLM_DIR / "v1" / "worker" / "gpu_model_runner.py").read_text(
        encoding="utf-8")
    assert "mlp_speculator" not in runner_src
    assert "Unknown speculative decoding method" in runner_src
