"""配置与后端的组合校验：不合法的组合在**构造阶段**就报出来。

这些函数不碰实例、不碰模型，只回答「这组配置能不能一起用」。放在引擎之外，
是因为它们与「运行时怎么装、前向怎么编排」无关，Engine 只是调用方之一。

Engine 在 `_init_runtime()` 里调它们，`check_runtime` 在两条构造路径上都会走到。
"""

import torch

SCHEDULING_POLICIES = ("fcfs", "priority")
SPECULATIVE_MODES = (None, "ngram")


def check_scheduling_policy(scheduling_policy):
    if scheduling_policy not in SCHEDULING_POLICIES:
        raise ValueError(f"未知的 scheduling_policy: {scheduling_policy!r}，"
                         f"可选 {list(SCHEDULING_POLICIES)}")


def check_speculative(speculative_mode, num_speculative_tokens, prompt_lookup_n,
                       scheduling_policy, enable_prefix_caching,
                       attention_backend, use_cuda_graph):
    """投机解码的开关与组合校验。

    只剩两条硬约束，都是**实现方式**决定的，不是「懒得验」：

    - `attention_backend="torch"`：拒绝采样现在是逐请求的 Torch 参考循环，
      没有 Triton rejection kernel（需求明确不做）；
    - `use_cuda_graph=False`：采样要在图**外**按请求逐行做设备同步，图里做不到。

    其余组合都放开并验过：`max_num_seqs` 从第五十三关起不限；`scheduling_policy`
    与 `enable_prefix_caching` 从第五十四关起不限（见
    benchmarks/check_step54_combinations.py）。放开抢占那一条尤其值得说明：抢占
    发生在 `schedule()` 里、forward **之前**，被抢占的项整个作废、走不到采样与回调，
    所以「投机 + 抢占」不需要额外机制；前缀缓存那一条靠的是「被回滚的整块从来没被
    发布过」（回滚目标 >= 本轮起点 + 1，而发布的块严格在起点之前）。
    """
    if speculative_mode not in SPECULATIVE_MODES:
        raise ValueError(f"未知的 speculative_mode: {speculative_mode!r}，"
                         f"可选 None（关）或 'ngram'")
    for name, value in (("num_speculative_tokens", num_speculative_tokens),
                        ("prompt_lookup_n", prompt_lookup_n)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是 >=1 的整数，收到 {value!r}")
    if speculative_mode is None:
        return
    unsupported = [
        (attention_backend != "torch", f"attention_backend 只能是 'torch'，当前 {attention_backend!r}"),
        (use_cuda_graph, "use_cuda_graph 必须关闭"),
    ]
    problems = [message for bad, message in unsupported if bad]
    if problems:
        raise ValueError("speculative_mode='ngram' 不支持的组合：" + "；".join(problems))


def resolve_device(device):
    return torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_runtime(device, attention_backend, use_cuda_graph, dtype=torch.float32,
                   norm_backend="torch", rope_backend="torch"):
    # 后端、设备与精度的组合校验；随机初始化和从目录加载两条路都走这里
    if attention_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 attention_backend: {attention_backend!r}，可选 'torch' 或 'triton'")
    if attention_backend == "triton" and device.type != "cuda":
        raise ValueError(f"attention_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
    if use_cuda_graph and (device.type != "cuda" or attention_backend != "triton"):
        raise ValueError(f"use_cuda_graph=True 只支持 CUDA + Triton，当前 device={device.type}、"
                         f"attention_backend={attention_backend!r}")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"不支持的 dtype: {dtype}，本实现只支持 torch.float32 或 torch.bfloat16")
    if dtype == torch.bfloat16 and device.type != "cuda":
        raise ValueError(f"dtype=torch.bfloat16 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 float32")
    # norm 后端与 attention 后端是两件独立的事，可以自由组合做 A/B
    if norm_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 norm_backend: {norm_backend!r}，可选 'torch' 或 'triton'")
    if norm_backend == "triton" and device.type != "cuda":
        raise ValueError(f"norm_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
    # rope 后端同样与 attention / norm 独立，可以自由组合做 A/B
    if rope_backend not in ("torch", "triton"):
        raise ValueError(f"未知的 rope_backend: {rope_backend!r}，可选 'torch' 或 'triton'")
    if rope_backend == "triton" and device.type != "cuda":
        raise ValueError(f"rope_backend='triton' 需要 CUDA 设备，当前是 {device.type}；CPU 上请用 'torch'")
