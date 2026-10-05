"""**只给测试用**的 tiny Qwen3 模型目录生成器（不再往仓库里放 safetensors）。

原来的 `fixtures/step30_qwen3/{tiny_gqa,tiny_mqa}` 是提交进仓库的随机初始化权重（约 200 KB），
由外部工具用 transformers `save_pretrained` 导出。现在改为**现场生成**：

- 配置与形状与原来**完全一致**（对照脚本依赖 `vocab_size=11/13`、GQA 4/2 与 MQA 4/1、
  `head_dim`、`rms_norm_eps` 这些规格，不能变）；
- 权重按固定 `seed` 生成 → 同一 seed 每次一致，但**与原来那两份权重无关**；
  所以任何"记住某个 token id / 某条 logits"的断言都不能依赖它，只能做自洽对照
  （与 HF / 真实 vLLM 读同一份目录比数值）；
- 生成到临时目录，进程内缓存：多个用例共用一份，不重复写盘。

    from minivllm.testing.tiny_models import tiny_qwen3_dir
    model_dir = tiny_qwen3_dir("tiny_gqa")

要按**旧仓库布局**重建（给按 `fixtures/step30_qwen3/...` 路径写的老脚本/验收探针用）：

    python -m minivllm.testing.tiny_models --out fixtures/step30_qwen3

生成器不属于生产路径：`minivllm` 的引擎代码不 import 它，只依赖 torch + safetensors。
"""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

# 与旧 fixtures 的 config.json 逐字段一致（含 transformers 写下的默认字段）
_TINY_GQA = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dtype": "float32",
    "eos_token_id": 4,
    "head_dim": 16,
    "hidden_act": "silu",
    "hidden_size": 32,
    "initializer_range": 0.02,
    "intermediate_size": 48,
    "layer_types": ["full_attention"] * 2,
    "max_position_embeddings": 64,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "pad_token_id": None,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "5.14.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 11,
}

_TINY_MQA = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": None,
    "dtype": "float32",
    "eos_token_id": 4,
    "head_dim": 6,
    "hidden_act": "silu",
    "hidden_size": 30,
    "initializer_range": 0.02,
    "intermediate_size": 41,
    "layer_types": ["full_attention"] * 3,
    "max_position_embeddings": 64,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 4,
    "num_hidden_layers": 3,
    "num_key_value_heads": 1,
    "pad_token_id": None,
    "rms_norm_eps": 0.003,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "5.14.1",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 13,
}

_CONFIGS = {"tiny_gqa": _TINY_GQA, "tiny_mqa": _TINY_MQA}
_CACHE: dict[tuple[str, int], str] = {}


def tiny_qwen3_config(name: str) -> dict:
    """取模型配置（深拷贝：调用方改它不会污染模板）。"""
    if name not in _CONFIGS:
        raise ValueError(f"没有名为 {name!r} 的 tiny 模型；可选：{sorted(_CONFIGS)}")
    return json.loads(json.dumps(_CONFIGS[name]))


def write_tiny_qwen3(name: str, out_dir, seed: int = 0) -> Path:
    """把 tiny 模型写到 `out_dir`（config.json + generation_config.json + model.safetensors）。

    权重是**随机但确定性**的：同一 `seed` 每次字节一致，与旧 fixtures 无关。
    目录里**没有 tokenizer**——本项目的引擎只读 config 与 safetensors。
    """
    config = tiny_qwen3_config(name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": config["bos_token_id"], "eos_token_id": config["eos_token_id"],
         "pad_token_id": config["pad_token_id"]}, indent=2) + "\n")
    save_file(_weights(config, seed), str(out / "model.safetensors"))
    return out


def tiny_qwen3_dir(name: str, seed: int = 0) -> str:
    """生成到临时目录并返回路径（**进程内缓存**：同名同 seed 只写一次）。

    返回 `str` 而不是 `Path`：调用方多数在拼 `f"{dir}/config.json"` 这类字符串。
    """
    key = (name, seed)
    cached = _CACHE.get(key)
    if cached is None:
        cached = tempfile.mkdtemp(prefix=f"minivllm_{name}_")
        write_tiny_qwen3(name, cached, seed)
        _CACHE[key] = cached
    return cached


def tiny_eagle3_dir(target_name: str = "tiny_gqa", *, num_aux_layers: int = 2,
                    aux_layers=(0, 1), seed: int = 0) -> str:
    """生成一份**与 tiny target 规格匹配**的 EAGLE3 draft（随机但确定；只给测试）。

    与 `tiny_qwen3_dir` 的区别（这些就是 63 关要适配的东西）：
      - 只有 **1 层**（`midlayer.*`），forward 语义是 EAGLE3 那套（见 `models/qwen3_eagle3.py`）；
      - `fc.weight` 把 `num_aux_layers` 个 target 辅助层特征投到 hidden（输入宽度 = hidden × 层数）；
      - 检查点里**没有 embed_tokens**（与 target 共享）、`lm_head` 是 draft 自己那份（本 tiny 用同词表）；
      - `eagle_aux_hidden_state_layer_ids` 显式给出（tiny target 只有 2 层，上游默认值
        `(2, n//2, n-3)` 对 2 层不成立，所以这里必须显式写）。
    """
    target_config = tiny_qwen3_config(target_name)
    hidden = target_config["hidden_size"]
    head_dim = target_config["head_dim"]
    num_heads = target_config["num_attention_heads"]
    num_kv_heads = target_config["num_key_value_heads"]
    vocab = target_config["vocab_size"]
    config = {
        "architectures": ["Eagle3Qwen3ForCausalLM"],
        "model_type": "qwen3",                      # Qwen3 风格 → 带 q/k norm
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": target_config["intermediate_size"],
        "vocab_size": vocab,                        # 同词表：异构词表的采样空间语义属 67 关
        "max_position_embeddings": target_config["max_position_embeddings"],
        "rms_norm_eps": target_config["rms_norm_eps"],
        "rope_theta": target_config.get("rope_theta", 10000.0),
        "target_hidden_size": hidden,
        "num_aux_layers": num_aux_layers,
        "eagle_aux_hidden_state_layer_ids": list(aux_layers),
        "tie_word_embeddings": False,               # draft 的 lm_head 是自己的（同词表）
        "torch_dtype": target_config.get("torch_dtype", "float32"),
    }
    generator = torch.Generator().manual_seed(seed)
    scale = 0.02
    def rand(*shape):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.float32)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    weights = {
        "fc.weight": rand(hidden, hidden * num_aux_layers),
        "norm.weight": torch.ones(hidden, dtype=torch.float32),
        "lm_head.weight": rand(vocab, hidden),
        "midlayer.hidden_norm.weight": torch.ones(hidden, dtype=torch.float32),
        "midlayer.input_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
        "midlayer.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.float32),
        # 第一层的 qkv 输入是 2*hidden（embeds + 特征拼接）
        "midlayer.self_attn.q_proj.weight": rand(q_size, 2 * hidden),
        "midlayer.self_attn.k_proj.weight": rand(kv_size, 2 * hidden),
        "midlayer.self_attn.v_proj.weight": rand(kv_size, 2 * hidden),
        "midlayer.self_attn.o_proj.weight": rand(hidden, q_size),
        "midlayer.mlp.gate_proj.weight": rand(config["intermediate_size"], hidden),
        "midlayer.mlp.up_proj.weight": rand(config["intermediate_size"], hidden),
        "midlayer.mlp.down_proj.weight": rand(hidden, config["intermediate_size"]),
        "midlayer.self_attn.q_norm.weight": torch.ones(head_dim, dtype=torch.float32),
        "midlayer.self_attn.k_norm.weight": torch.ones(head_dim, dtype=torch.float32),
    }
    out = Path(tempfile.mkdtemp(prefix=f"minivllm_eagle3_{target_name}_"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(out / "model.safetensors"))
    return str(out)


def tiny_mtp_dir(target_name: str = "tiny_gqa", *, naming: str = "mtp",
                 num_mtp_layers: int = 1, seed: int = 0) -> str:
    """生成一份"**含 target 层 + spec 层**"的 MTP checkpoint（随机但确定；只给测试）。

    这就是 65 关要对付的真实形态：MTP 权重与 target 在**同一个文件**里。两种命名约定各生成一份
    （需求 §3.1 要求"loader 按源码识别并改写 spec 层权重名"，两派都要能认）：

        naming="mtp"       `mtp.fc.weight` / `mtp.layers.0.*` / `mtp.norm.weight` …
                           （上游 `qwen3_next_mtp.py` 那一派；本仓库的 MTP 用这套）
        naming="absolute"  `model.layers.{N+i}.fc.weight` / `…self_attn.q_proj.weight` …
                           （上游 DeepSeek 那一派：spec 层排在 target 最后一层后面，
                            用**绝对层号**，加载时改成相对层号）

    配置里带 `num_nextn_predict_layers`（MTP 层数）——没有它就无法知道 target 后面附了几层，
    上游也是靠它做别名归一与 K 的整除校验。
    """
    if naming not in ("mtp", "absolute"):
        raise ValueError(f"naming 只能是 'mtp' / 'absolute'，收到 {naming!r}")
    target_config = tiny_qwen3_config(target_name)
    hidden = target_config["hidden_size"]
    inter = target_config["intermediate_size"]
    heads = target_config["num_attention_heads"]
    kv_heads = target_config["num_key_value_heads"]
    head_dim = target_config["head_dim"]
    vocab = target_config["vocab_size"]
    base = target_config["num_hidden_layers"]

    # 配置里**保持 target 的样子**（architectures 仍指向 target 的类）：真实 MTP checkpoint 就是
    # 这样——"draft 是 MTP 模型"这件事由 `SpeculativeConfig.derive_mtp_draft_config()` 在派生
    # 配置时改写（上游 `hf_config_override` 也是改写 draft 配置，不动 target 的）。
    config = dict(target_config)
    config["num_nextn_predict_layers"] = num_mtp_layers

    generator = torch.Generator().manual_seed(seed + 1000)

    def normal(*shape):
        return torch.empty(*shape, dtype=torch.float32).normal_(
            0.0, target_config["initializer_range"], generator=generator)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32)

    # 先放 target 自己的权重（MTP 与它同一个文件）
    weights = _weights(target_config, seed)
    # target 那份 embed/lm_head 就是 MTP 共享的那两份（上游 `shared_weight_names`）
    weights[".mtp_shared_marker"] = torch.zeros(0)     # 占位，稍后删掉
    for index in range(num_mtp_layers):
        if naming == "mtp":
            # 上游 qwen3_next_mtp.py 那一派：predictor 的前缀是 `mtp`，它内部的层是
            # `mtp.layers.{i}.*`，胶水（fc/norm/pre_fc_norm_*/embed_tokens）在 `mtp.` 下
            glue = "mtp."
            block = f"mtp.layers.{index}."
        else:
            # 上游 DeepSeek 那一派：spec 层用**绝对层号**，block 直接挂在 `model.layers.{base+i}.` 下，
            # 胶水也挂在那下面（加载时按"是不是胶水"决定提到顶层还是改层号）
            glue = block = f"model.layers.{base + index}."
        # 胶水：embedding 侧的归一化、hidden 侧的归一化、拼接投影、最终归一化
        weights[glue + "pre_fc_norm_embedding.weight"] = ones(hidden)
        weights[glue + "pre_fc_norm_hidden.weight"] = ones(hidden)
        weights[glue + "fc.weight"] = normal(hidden, hidden * 2)
        weights[glue + "norm.weight"] = ones(hidden)
        # 一层解码器（与 target 同规格）
        weights[block + "input_layernorm.weight"] = ones(hidden)
        weights[block + "post_attention_layernorm.weight"] = ones(hidden)
        weights[block + "self_attn.q_proj.weight"] = normal(heads * head_dim, hidden)
        weights[block + "self_attn.k_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[block + "self_attn.v_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[block + "self_attn.o_proj.weight"] = normal(hidden, heads * head_dim)
        weights[block + "self_attn.q_norm.weight"] = ones(head_dim)
        weights[block + "self_attn.k_norm.weight"] = ones(head_dim)
        weights[block + "mlp.gate_proj.weight"] = normal(inter, hidden)
        weights[block + "mlp.up_proj.weight"] = normal(inter, hidden)
        weights[block + "mlp.down_proj.weight"] = normal(hidden, inter)
    del weights[".mtp_shared_marker"]

    out = Path(tempfile.mkdtemp(prefix=f"minivllm_mtp_{target_name}_"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(out / "model.safetensors"))
    return str(out)


def tiny_medusa_dir(target_name: str = "tiny_gqa", *, num_heads: int = 3,
                    num_layers: int = 1, naming: str = "old", fc_bias: bool = False,
                    original_lm_head: bool = False, truncated_vocab: int | None = None,
                    seed: int = 0) -> str:
    """生成一份**与 tiny target 规格匹配**的 Medusa head（随机但确定；只给测试）。

    66 关要对付的真实形态（本机实测 `FasterDecoding/medusa-vicuna-7b-v1.3/medusa_lm_head.pt`
    的 state_dict 键 + 它的 `config.json`）：

        config.json   `{"medusa_num_heads": 2, "medusa_num_layers": 1, "base_model_name_or_path": ...}`
                      —— 连 model_type / vocab_size / hidden_size / architectures 都没有
        head 权重     `0.0.linear.weight` / `0.0.linear.bias`（残差块）
                      `0.1.weight`（那个 head 的 lm_head，序号 = num_layers）
                      `1.0.linear.weight` / `1.1.weight` / …（每个 head 一组）

    三种命名各生成一份（`naming`）：

        "old"           `{h}.{l}.linear.weight` / `{h}.{num_layers}.weight`（真实旧文件）
        "medusa_heads"  同上但带 `medusa_heads.` 前缀（上游 `load_weights` 会先剥掉它）
        "vllm"          `blocks.{h}.layers.{l}.weight` / `lm_heads.{h}.weight`（已经是本模型的名字）

    默认的 config 里**带 hidden_size**（tiny target 的 32；真实旧文件缺这一项时
    `MedusaConfig` 的默认值 4096 只对 7B 有效）但**不带 vocab_size**——于是
    `derive_medusa_draft_config()` 的"与 target 对齐词表"这一步真的有东西可做。

    `fc_bias=True` 让残差块带 bias（旧文件里通常有）；`original_lm_head=True` +
    `truncated_vocab=k` 生成"共享一个 lm_head + token_map 截断词表"的那一套
    （上游就只有这条路走得通：lm_head 的参数宽度等于 truncated_vocab_size）。
    """
    if naming not in ("old", "medusa_heads", "vllm"):
        raise ValueError(f"naming 只能是 'old' / 'medusa_heads' / 'vllm'，收到 {naming!r}")
    target_config = tiny_qwen3_config(target_name)
    hidden = target_config["hidden_size"]
    vocab = target_config["vocab_size"]

    config = {
        # 旧 FasterDecoding 的字段名（`MedusaConfig` 会把它们改写成 num_heads/num_hidden_layers）
        "medusa_num_heads": num_heads,
        "medusa_num_layers": num_layers,
        "base_model_name_or_path": f"tiny://{target_name}",
        "transformers_version": "4.31.0",
        "hidden_size": hidden,
    }
    if fc_bias:
        config["medusa_fc_bias"] = True
    if original_lm_head:
        config["original_lm_head"] = True
        config["vocab_size"] = vocab          # 与 target 一致 → 不触发词表对齐
        config["truncated_vocab_size"] = int(truncated_vocab or vocab)

    generator = torch.Generator().manual_seed(seed + 2000)

    def normal(*shape):
        return torch.empty(*shape, dtype=torch.float32).normal_(
            0.0, target_config["initializer_range"], generator=generator)

    weights: dict[str, torch.Tensor] = {}
    for head in range(num_heads):
        for layer in range(num_layers):
            weights[f"{head}.{layer}.linear.weight"] = normal(hidden, hidden)
            if fc_bias:
                weights[f"{head}.{layer}.linear.bias"] = normal(hidden)
        # 这个 head 的 lm_head：真实检查点里存的是**整份词表**（截断词表靠加载时按
        # `token_map` 选行，见 `Medusa.load_weights`）
        weights[f"{head}.{num_layers}.weight"] = normal(vocab, hidden)
        if fc_bias:
            # 旧文件里 lm_head 不带 bias；这里**故意**造一份出来，用来钉住"本模型不建 bias
            # → 显式记账后丢弃"这条规则（真实转换脚本也偶尔留下这种项）
            weights[f"{head}.{num_layers}.bias"] = normal(vocab)
    if original_lm_head and truncated_vocab:
        # 截断词表的映射表（上游：只有 config 里 truncated_vocab_size < vocab_size 且检查点有
        # `token_map` 时才启用）
        weights["token_map"] = torch.arange(int(truncated_vocab), dtype=torch.int64)

    if naming == "medusa_heads":
        weights = {f"medusa_heads.{name}": value for name, value in weights.items()}
    elif naming == "vllm":
        renamed: dict[str, torch.Tensor] = {}
        for name, value in weights.items():
            if name == "token_map":
                renamed[name] = value
                continue
            # 旧名字的形状是 `{head}.{层号}.{linear.}?{参数}`：带 `linear.` 的是残差块，
            # 不带的是那个 head 的 lm_head（序号 = num_layers）
            head, _, rest = name.partition(".")
            layer, _, param = rest.partition(".")
            if param.startswith("linear."):
                renamed[f"blocks.{head}.layers.{layer}.{param[len('linear.'):]}"] = value
            else:
                renamed[f"lm_heads.{head}.{param}"] = value
        weights = renamed

    out = Path(tempfile.mkdtemp(prefix=f"minivllm_medusa_{target_name}_"))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(out / "model.safetensors"))
    return str(out)


# 67 关的 TLI 集成测试用：一对"同 KV 规格、不同词表、不同 tokenizer"的 tiny 模型。
# 两边的**同一个词**故意放在不同的 id 上，空格标记也故意用两族的写法（SentencePiece 的 ▁ 与
# BPE 的 Ġ），这样"不能直接换 id"和"要按 token 字符串建交集"两件事都被真正跑到。
HETERO_TARGET_TOKENS = {
    "<unk>": 0, "<eos>": 1, "\u2581hello": 2, "\u2581world": 3, "\u2581the": 4,
    "\u2581kv": 5, "\u2581cache": 6, "\u2581is": 7, "\u2581a": 8, "\u2581b": 9,
    "\u2581onlyt": 10,                      # target 独有 → 进 draft 空间时变 draft unk
}
HETERO_DRAFT_TOKENS = {
    "<unk>": 0, "<eos>": 1, "\u0120the": 2, "\u0120kv": 3, "\u0120cache": 4,
    "\u0120is": 5, "\u0120hello": 6, "\u0120world": 7, "\u0120a": 8, "\u0120b": 9,
    "\u0120onlyd": 10, "\u0120x": 11, "\u0120y": 12,   # draft 独有 → logits 被掩掉
}


def write_wordlevel_tokenizer(out_dir, vocab: dict[str, int], *, space_marker: str,
                              unk_token: str = "<unk>", eos_token: str = "<eos>",
                              model_max_length: int = 64) -> Path:
    """写一份**真正能被 transformers 加载**的极小 tokenizer（WordLevel + 空格预处理）。

    `space_marker` 取 `"\u2581"`（SentencePiece 系）或 `"\u0120"`（BPE 系）：两种都让
    " hello" 变成一个以该字符开头的 token（`▁hello` / `Ġhello`），于是 `VocabMapping` 的
    `_detect_space_prefix()` 走的是它两条真实分支，而不是被喂一个假的探测结果。
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(vocab=dict(vocab), unk_token=unk_token))
    if space_marker == "\u2581":
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="\u2581", prepend_scheme="always", split=True)
        tokenizer.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always",
                                              split=True)
    elif space_marker == "\u0120":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    else:
        raise ValueError(f"space_marker 只支持 '▁' / 'Ġ'，收到 {space_marker!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": unk_token, "eos_token": eos_token, "bos_token": None,
        "pad_token": None, "model_max_length": model_max_length,
    }, indent=2) + "\n")
    return out


def tiny_hetero_pair(target_name: str = "tiny_gqa", *, seed: int = 0):
    """生成 67 关用的一对"异构词表"模型目录：`(target_dir, draft_dir, 说明)`。

    - target：tiny target 的配置与权重（vocab=11）+ SentencePiece 风格的 tokenizer；
    - draft：**同 KV 规格**（同 hidden/heads/head_dim/interleaved/max_position_embeddings）但
      **vocab=13**、BPE 风格 tokenizer 的 Qwen3 模型——两边同一个词的 id 不同，交集 10 个 token。

    为什么 KV 规格必须相同：draft 与 target 共用逻辑块表（同一个 KV group），规格不同就谈不到一起
    （`DraftModelProposer._validate_configs` 会报错）。TLI 只是让**词表**可以不同。
    """
    target_config = tiny_qwen3_config(target_name)
    draft_config = dict(target_config)
    draft_config["vocab_size"] = max(HETERO_DRAFT_TOKENS.values()) + 1        # 13
    draft_config["eos_token_id"] = HETERO_DRAFT_TOKENS["<eos>"]

    target_dir = Path(tempfile.mkdtemp(prefix=f"minivllm_hetero_target_{target_name}_"))
    draft_dir = Path(tempfile.mkdtemp(prefix=f"minivllm_hetero_draft_{target_name}_"))
    (target_dir / "config.json").write_text(json.dumps(target_config, indent=2) + "\n")
    (draft_dir / "config.json").write_text(json.dumps(draft_config, indent=2) + "\n")
    save_file(_weights(target_config, seed), str(target_dir / "model.safetensors"))
    save_file(_weights(draft_config, seed + 3000), str(draft_dir / "model.safetensors"))
    write_wordlevel_tokenizer(target_dir, HETERO_TARGET_TOKENS, space_marker="\u2581")
    write_wordlevel_tokenizer(draft_dir, HETERO_DRAFT_TOKENS, space_marker="\u0120")
    info = {
        "target_vocab_size": target_config["vocab_size"],
        "draft_vocab_size": draft_config["vocab_size"],
        "target_tokens": dict(HETERO_TARGET_TOKENS),
        "draft_tokens": dict(HETERO_DRAFT_TOKENS),
        "prompt_token_ids": [HETERO_TARGET_TOKENS["\u2581hello"],
                             HETERO_TARGET_TOKENS["\u2581kv"],
                             HETERO_TARGET_TOKENS["\u2581cache"]],
    }
    return str(target_dir), str(draft_dir), info


def _weights(config: dict, seed: int) -> dict:
    """按 HF 的初始化方式生成权重：线性/嵌入 ~ N(0, initializer_range)，RMSNorm 全 1。

    `initializer_range` 与形状全部从 config 推导——改配置就不会出现形状对不上。
    """
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab = config["vocab_size"]
    num_layers = config["num_hidden_layers"]
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape):
        return torch.empty(*shape, dtype=torch.float32).normal_(
            0.0, config["initializer_range"], generator=generator)

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32)

    weights = {
        "model.embed_tokens.weight": normal(vocab, hidden),
        "lm_head.weight": normal(vocab, hidden),
        "model.norm.weight": ones(hidden),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = ones(hidden)
        weights[prefix + "post_attention_layernorm.weight"] = ones(hidden)
        weights[prefix + "self_attn.q_proj.weight"] = normal(heads * head_dim, hidden)
        weights[prefix + "self_attn.k_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.v_proj.weight"] = normal(kv_heads * head_dim, hidden)
        weights[prefix + "self_attn.o_proj.weight"] = normal(hidden, heads * head_dim)
        weights[prefix + "self_attn.q_norm.weight"] = ones(head_dim)
        weights[prefix + "self_attn.k_norm.weight"] = ones(head_dim)
        weights[prefix + "mlp.gate_proj.weight"] = normal(inter, hidden)
        weights[prefix + "mlp.up_proj.weight"] = normal(inter, hidden)
        weights[prefix + "mlp.down_proj.weight"] = normal(hidden, inter)
    return weights


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="生成 tiny Qwen3 测试模型目录")
    parser.add_argument("--out", required=True,
                        help="输出根目录；会在其下写 tiny_gqa/ 与 tiny_mqa/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="目录非空时也覆盖（默认跳过，避免盖掉手头的东西）")
    args = parser.parse_args(argv)

    root = Path(args.out)
    for name in _CONFIGS:
        target = root / name
        if target.exists() and any(target.iterdir()) and not args.force:
            print(f"跳过（已存在，非空）：{target}")
            continue
        write_tiny_qwen3(name, target, seed=args.seed)
        print(f"已生成：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
