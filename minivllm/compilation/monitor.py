"""捕获期开关（对应 vLLM `compilation/monitor.py` 的子集）。

解决的问题：**"什么时候允许捕获 CUDA Graph"必须只有一个答案**。

捕获图是一次昂贵的、会占住显存的全局操作，而它只能在启动阶段的固定窗口里做
（`GPUModelRunner.capture_model()`）。如果没有这道闸门，任何一次"第一次遇到某个形状"的
前向都会顺手捕获——表现是"跑着跑着突然卡一下、显存涨一截"，而且在服务中把请求的延迟
拉出尖峰。上游的做法就是这两个函数：

    set_cudagraph_capturing_enabled(True/False)   由 Runner 在捕获窗口前后开关
    validate_cudagraph_capturing_enabled()        图包装器在**真的要捕获**之前问一句

`torch.compile` 那一半（`set_torch_compile_enabled`）本仓库没有编译路径，所以不留
——留一个永远不会被置位的开关，读代码的人会以为编译被支持。
"""

#: 允许捕获图吗？默认 False：进程刚起来时任何捕获都是意外。
_cudagraph_capturing_enabled = False


def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global _cudagraph_capturing_enabled
    _cudagraph_capturing_enabled = bool(enabled)


def validate_cudagraph_capturing_enabled() -> None:
    """图包装器在捕获前调用：不在窗口里就报错（而不是"顺手捕获一张"）。"""
    if not _cudagraph_capturing_enabled:
        raise RuntimeError(
            "现在不允许捕获 CUDA Graph（不在 capture_model() 的捕获窗口里）。"
            "出现这个错通常意味着某个形状没被 dummy_run 预热过，导致第一次真实前向就想捕获："
            "请检查 cudagraph_capture_sizes 是否覆盖了实际形状")
