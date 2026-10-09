"""数学小工具（对应 vLLM `vllm/utils/math_utils.py` 的子集：只搬 V2 通路用到的 `cdiv`）。"""


def cdiv(a: int, b: int) -> int:
    """Ceiling division（上游同款：负数也取上界）。"""
    return -(a // -b)
