"""`tests/` 的全局环境准备（只做一件事）。

**为什么需要它**：V2 通路（`worker/gpu/`）依赖 UVA，而 UVA 判定
（`minivllm/.../buffer_utils.py` 与上游 `vllm/v1/worker/gpu/buffer_utils.py`）都被
`functools.cache` 缓存过。这个缓存的取值只看**第一次调用时**环境变量在不在——
在整套测试里，`tests/step73` 的用例会在运行期调用它，而"谁先 import/先调用"取决于
pytest 的收集顺序。于是同一组用例：

    单跑 step73           → 通过（模块 import 时已经 setdefault 了环境变量）
    跑 step58..step73     → 3 个 UVA 相关用例失败（前面某个模块先把它缓存成 False 了）

实测：在提交 `1f62833`（76 关之前）上，不带 shell 环境变量跑 `tests/step58..75` 就会
稳定复现这 3 个失败；把环境变量 export 到 shell 里则 772 项全绿。这不是被测代码的差异，
而是"环境变量的设置时机"——所以在**任何测试模块 import 之前**统一设好。

`setdefault` 而不是覆盖：显式设成 0 的人（想验证"没有 UVA 时的报错"）仍然说了算。
"""

import os

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
