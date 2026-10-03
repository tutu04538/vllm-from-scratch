"""**只给测试用**的替身（FakeRunner）与 tiny 模型生成器（tiny_models）。

生产路径（`minivllm/engine`、`minivllm/executor`、`minivllm/worker`）**不 import 这个包**：
"没有真模型就退回假执行"是被明确禁止的，替身只能由测试显式注入。
"""
