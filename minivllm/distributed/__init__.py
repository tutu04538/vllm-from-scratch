"""单机 TP/PP 通信；不开放多卡引擎。"""
from .parallel_state import (GroupCoordinator, destroy_distributed_environment,
                             destroy_model_parallel, get_pp_group, get_tp_group,
                             init_distributed_environment, initialize_model_parallel)

__all__ = ['GroupCoordinator', 'init_distributed_environment', 'initialize_model_parallel',
           'get_tp_group', 'get_pp_group', 'destroy_model_parallel',
           'destroy_distributed_environment']
