# Step30 的外部 Qwen3 目录

由 Transformers 本地随机初始化、save_pretrained 导出；不是预训练模型，没有语言能力，无网络下载。

- tiny_gqa：2 层，hidden=32，Q/KV heads=4/2，head_dim=16。
- tiny_mqa：3 层，hidden=30，Q/KV heads=4/1，head_dim=6。

你的加载器只读取各模型目录中的 config.json 和 model.safetensors；generation_config.json 本关可忽略。根目录 manifest.json 是验收记录，不是模型加载输入，不得用它代替配置/权重。

生成工具：vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/tools/make_step30_qwen3_fixtures.py。已有非空输出目录不会覆盖；重建可指定另一个 --output。
