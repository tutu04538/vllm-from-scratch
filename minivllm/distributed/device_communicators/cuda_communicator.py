"""CUDA communication backend selection for the implemented TP primitives."""
from .pynccl import PyNcclCommunicator


class CudaCommunicator:
    def __init__(self, cpu_group, device):
        self.pynccl_comm = PyNcclCommunicator(cpu_group, device)
        self.backend = "pynccl"
        self.custom_all_reduce_reason = "本项目未实现 custom all-reduce 后端"

    def all_reduce(self, tensor):
        return self.pynccl_comm.all_reduce(tensor)

    def all_gather(self, output, tensor):
        self.pynccl_comm.all_gather(output, tensor)

    def destroy(self):
        self.pynccl_comm.destroy()
