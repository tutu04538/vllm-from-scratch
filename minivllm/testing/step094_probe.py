"""094 仅用于测试的 Worker：按 FakeRunner 协议执行，不代表真实模型。"""
import time
import os
from pathlib import Path
from .fake_runner import FakeRunner
from .step093_probe import ProbeWorker


class RPCProbeWorker(ProbeWorker):
    def load_model(self):
        self.model_runner = FakeRunner(tokens={'A':[7,8,9], 'B':[5,6,7]})
        self.last_snapshot = None

    def get_cache_config(self):
        return self.vllm_config.cache_config

    def initialize_from_config(self, config):
        self.kv_cache_config = config
        self.model_runner.initialize_kv_cache(config)

    def compile_or_warm_up_model(self):
        pass

    def execute_model(self, scheduler_output, non_block=False):
        self.last_snapshot = scheduler_output
        return super().execute_model(scheduler_output, non_block=non_block)

    def sample_tokens(self, grammar_output=None, non_block=False):
        result = super().sample_tokens(grammar_output, non_block=non_block)
        result.source_rank = self.rank
        return result

    def snapshot(self):
        return self.last_snapshot

    def rank_identity(self):
        return self.rank

    def echo(self, value):
        return value

    def delayed(self, value, seconds):
        if self.rank == 1:
            time.sleep(seconds)
        return (self.rank, value)

    def fail_rank(self, target_rank):
        if self.rank == target_rank:
            raise RuntimeError(f'injected RPC failure on rank {self.rank}')
        return self.rank

    def exit_rank(self, target_rank):
        if self.rank == target_rank:
            os._exit(72)
        return self.rank


class GatedRPCProbeWorker(RPCProbeWorker):
    def execute_model(self, scheduler_output, non_block=False):
        if self.rank == 1:
            root = Path(os.environ['MINIVLLM_STEP094_GATE'])
            (root / 'waiting').touch()
            deadline = time.monotonic() + 60
            while not (root / 'release').exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError('094 测试 gate 等待超时')
                time.sleep(0.02)
        return super().execute_model(scheduler_output, non_block=non_block)
