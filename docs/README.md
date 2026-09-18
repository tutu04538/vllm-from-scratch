# docs —— 改动记录

每次改动一个文件：`docs/stepNN_<主题>.md`，与 `stepNN.py` 同名对应。

每篇记录固定写这几节：

0. **需求大概**：这次要做什么、边界在哪，两三句话讲清。
1. **改动内容**：改了哪些方法、为什么。
2. **设计要点**：架构层面的事实，尤其是容易被后续改动破坏的约定。
3. **验证**：跑了什么、结论是什么；没跑的就写没跑。
4. **接口变化与遗留**：公开接口的变动、已知未做的部分。

## 索引

| 文档 | 对应代码 | 主题 |
|---|---|---|
| [step18_prefix_cache.md](step18_prefix_cache.md) | `step18.py` | 跨请求前缀缓存、共享块引用、LRU 淘汰 |
| [step19_packed_forward.md](step19_packed_forward.md) | `step19.py` | 无 padding 扁平打包、prefill/decode 一次 forward |
| [step20_slot_mapping.md](step20_slot_mapping.md) | `step20.py` | slot mapping、整批 KV 写入与向量化读取 |
| [step21_block_attention.md](step21_block_attention.md) | `step21.py` | 按块读 KV、online softmax 合并 attention |
| [step22_triton_attention.md](step22_triton_attention.md) | `step22.py` | 整批请求一次 Triton attention kernel |
| [step23_metadata_buffer.md](step23_metadata_buffer.md) | `step23.py` | attention 元数据固定缓冲、一次 H2D 上传 |
| [step24_cuda_graph.md](step24_cuda_graph.md) | `step24.py` | GPU forward 接入 CUDA Graph（按 N 缓存、replay） |
| [step25_multi_head_gqa.md](step25_multi_head_gqa.md) | `step25.py` | 多头 attention、GQA/MQA、KV 池按 KV head 存 |
