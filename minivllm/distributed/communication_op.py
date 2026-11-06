"""TP 通信入口，对应 vLLM distributed.communication_op。"""
from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_):
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_, dim=-1):
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(input_, dim=-1):
    return get_tp_group().reduce_scatter(input_, dim)
