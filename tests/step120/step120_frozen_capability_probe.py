"""独立进程中对冻结 Marlin 的能力谓词做 CPU 静态设备模拟。"""
import hashlib
import json
from pathlib import Path

import torch
import vllm.model_executor.kernels.linear.mixed_precision.marlin as marlin
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.scalar_type import scalar_types

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/model_executor/kernels/linear/mixed_precision/marlin.py'
if hashlib.sha256(Path(marlin.__file__).read_bytes()).digest() != hashlib.sha256(FREEZE.read_bytes()).digest():
    raise RuntimeError('上游安装文件与冻结源码不同')
# 仅用 CPU 模拟平台查询值；不替换 can_implement 或格式能力函数。
current_platform.get_device_capability = lambda *args, **kwargs: DeviceCapability(8, 9)
cases = [
    ('awq', (4096, 4096), (4096, 4096), scalar_types.uint4, 128, True, False),
    ('gptq', (4096, 4096), (2048, 4096), scalar_types.uint4b8, 128, False, False),
    ('group_crosses_rank', (4096, 4096), (2000, 4096), scalar_types.uint4, 128, True, False),
    ('bad_type_zp', (4096, 4096), (4096, 4096), scalar_types.uint4b8, 128, True, False),
    ('bad_group', (4096, 4096), (4096, 4096), scalar_types.uint4, 96, True, False),
]
result = {}
for name, full, part, weight, group, zp, gidx in cases:
    c = MPLinearLayerConfig(full_weight_shape=full, partition_weight_shape=part,
                            weight_type=weight, act_type=torch.bfloat16,
                            group_size=group, zero_points=zp, has_g_idx=gidx)
    ok, reason = marlin.MarlinLinearKernel.can_implement(c)
    result[name] = {'ok': ok, 'reason': reason}
print(json.dumps(result))
