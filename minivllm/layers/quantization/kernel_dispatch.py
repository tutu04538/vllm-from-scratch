"""冻结 vLLM 的 mixed-precision 选择边界；当前仅描述 SM89 W4A16 子集。"""
from dataclasses import dataclass
import torch

CUDA_CANDIDATE_ORDER = ('CutlassW4A8LinearKernel', 'MacheteLinearKernel', 'AllSparkLinearKernel',
                        'MarlinLinearKernel', 'ConchLinearKernel', 'ExllamaLinearKernel',
                        'TritonW4A16LinearKernel', 'HummingLinearKernel')


@dataclass(frozen=True)
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]
    partition_weight_shape: tuple[int, int]
    weight_type: str
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool
    out_type: torch.dtype | None = None


def can_plan_marlin(config: MPLinearLayerConfig, *, platform: str, capability: int):
    """当前项目计划的子集；不是整个上游 Marlin can_implement。"""
    if platform != 'cuda':
        return False, 'Marlin 仅支持 CUDA'
    if type(capability) is not int or capability != 89:
        return False, '当前只冻结 SM89；其他设备的上游候选顺序尚未覆盖'
    if not isinstance(config, MPLinearLayerConfig):
        return False, '无效的 MPLinearLayerConfig'
    full, part = config.full_weight_shape, config.partition_weight_shape
    if (not isinstance(full, tuple) or not isinstance(part, tuple) or len(full) != 2 or len(part) != 2
            or any(type(v) is not int or v <= 0 for v in full + part)
            or any(p > f or f % p for f, p in zip(full, part))):
        return False, 'full/partition shape 必须为正且能对应合法分片'
    if type(config.zero_points) is not bool or type(config.has_g_idx) is not bool:
        return False, 'zero_points 和 has_g_idx 必须为 bool'
    if config.act_type not in (torch.float16, torch.bfloat16):
        return False, '本项目 W4A16 仅支持 FP16/BF16 激活'
    if config.out_type not in (None, config.act_type):
        return False, '不支持与激活不同的输出 dtype'
    if config.group_size != 128:
        return False, '当前仅支持 group_size=128'
    if config.has_g_idx:
        return False, '当前不支持 g_idx/act-order'
    expected = 'uint4' if config.zero_points else 'uint4b8'
    if config.weight_type != expected:
        return False, f'该 zero-point 模式要求 {expected}'
    if part[0] % 128:
        return False, '本地 K 必须按 group_size=128 对齐'
    if part[1] % 8:
        return False, '本地 N 必须按 8 个 4-bit 权重对齐'
    return True, None


def plan_mp_kernel(config: MPLinearLayerConfig, *, platform: str, capability: int):
    """仅返回选择计划，不能把该计划当作已经可执行的 kernel。"""
    ok, reason = can_plan_marlin(config, platform=platform, capability=capability)
    if not ok:
        raise NotImplementedError(f'无法选择当前项目的 Marlin 子集: {reason}')
    # 在 SM89 / G128 / BF16或FP16 / 无g_idx 条件下，上游前三候选均不适用：
    # CUTLASS W4A8 和 Machete 要 SM90；AllSpark 对 SM89 不支持 group 128。
    return {'backend': 'MarlinLinearKernel', 'scope': 'sm89_w4a16_group128',
            'candidate_order': CUDA_CANDIDATE_ORDER, 'execution_available': False,
            'reason': '本地 Marlin 数学实现与 kernel 布局仍待 121/122 验收'}


def require_mp_kernel(config: MPLinearLayerConfig, *, platform: str, capability: int):
    plan = plan_mp_kernel(config, platform=platform, capability=capability)
    raise NotImplementedError(plan['reason'])
