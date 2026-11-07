"""095 real spawned GPU processes and deliberately failed worker paths."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests/step094'))
from step094_rpc_scenarios import config, snapshot
from minivllm.executor import MultiprocExecutor
from minivllm.testing.step094_probe import RPCProbeWorker
from minivllm.testing.step093_probe import FailInitWorker, FailLoadWorker
from minivllm.engine import LLMEngine
from minivllm.sampling_params import SamplingParams


class Step095Worker(RPCProbeWorker):
    def fail_phase(self, phase):
        if os.environ.get('MINIVLLM_STEP095_CASE') != phase:
            return
        if self.rank == 0:
            time.sleep(30)
        if self.rank == 1:
            raise RuntimeError(f'095 injected {phase} failure')

    def initialize_from_config(self, value):
        self.fail_phase('initialize')
        return super().initialize_from_config(value)

    def compile_or_warm_up_model(self):
        self.fail_phase('capture')
        return super().compile_or_warm_up_model()

    def execute_model(self, scheduler_output, non_block=False):
        self.fail_phase('execute')
        return super().execute_model(scheduler_output, non_block)

    def sample_tokens(self, grammar_output=None, non_block=False):
        self.fail_phase('sample')
        if os.environ.get('MINIVLLM_STEP095_CASE') == 'propose':
            self.propose()
        return super().sample_tokens(grammar_output, non_block)

    def propose(self):
        self.fail_phase('propose')

    def take_draft_token_ids(self):
        self.fail_phase('output')
        return super().take_draft_token_ids()

    def hang_and_exit(self):
        if self.rank == 0:
            time.sleep(30)
        if self.rank == 1:
            os._exit(72)
        return self.rank

    def ignore_terminate_and_exit_peer(self):
        if self.rank == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
            return
        time.sleep(1)
        os._exit(72)

    def collective_and_exit(self):
        if self.rank == 0:
            import torch
            import torch.distributed as dist
            value = torch.ones(1, device=f'cuda:{self.device_index}')
            dist.all_reduce(value)
            return value.item()
        time.sleep(1)
        os._exit(72)


def wait_until(predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('timed out waiting for state')
        time.sleep(0.02)


def clean(ex):
    directory = Path(ex._rendezvous.name)
    names = list(ex._response_shm_names)
    ex.shutdown()
    ex.shutdown()
    assert all(not h.process.is_alive() for h in ex.workers)
    assert not directory.exists()
    assert all(not (Path('/dev/shm') / n).exists() for n in names)


def run(case):
    if case in ('init', 'load'):
        cls = FailInitWorker if case == 'init' else FailLoadWorker
        try:
            MultiprocExecutor(config(tp=2, pp=1), _worker_cls=cls, _startup_timeout=40)
        except RuntimeError as exc:
            import multiprocessing
            assert 'rank 1' in str(exc) and 'injected' in str(exc)
            assert not multiprocessing.active_children()
            return {'error': str(exc), 'children_reaped': True}
        raise AssertionError('startup failure not propagated')

    os.environ['MINIVLLM_STEP095_CASE'] = case
    cfg = config(tp=2, pp=2 if case == 'abrupt4' else 1)
    ex = MultiprocExecutor(cfg, _worker_cls=Step095Worker, _startup_timeout=60)
    try:
        if case in ('initialize', 'capture'):
            try:
                LLMEngine(cfg, ex)
            except RuntimeError as exc:
                error = str(exc)
                assert 'rank 1' in error and f'095 injected {case} failure' in error, error
                assert ex.failure is not None and ex._closed
                return {'error': error, 'engine_constructor_cleaned': True}
            raise AssertionError('EngineCore initialization failure not propagated')
        if case == 'engine':
            engine = LLMEngine(config(tp=2, pp=1), ex)
            engine.add_request('A', [1, 2], SamplingParams(max_tokens=2, temperature=0.0))
            engine.abort_request(['A'])
            assert ex.failure is None
            ex.check_health()
            engine.add_request('B', [1, 2], SamplingParams(max_tokens=2, temperature=0.0))
            os.kill(ex.workers[1].process.pid, signal.SIGKILL)
            wait_until(lambda: ex.failure is not None)
            core = engine.engine_core.engine_core
            def no_schedule():
                raise AssertionError('schedule called after failure')
            core.scheduler.schedule = no_schedule
            try:
                engine.step()
            except RuntimeError as exc:
                assert '失败' in str(exc) and 'rank 1' in str(exc)
            else:
                raise AssertionError('failed engine scheduled again')
            engine.shutdown()
            fresh_ex = MultiprocExecutor(config(tp=2, pp=1), _worker_cls=Step095Worker, _startup_timeout=60)
            fresh = LLMEngine(config(tp=2, pp=1), fresh_ex)
            try:
                fresh.add_request('A', [1, 2], SamplingParams(max_tokens=2, temperature=0.0))
                outputs = []
                while fresh.has_unfinished_requests():
                    outputs.extend(fresh.step())
                assert outputs[-1].token_ids == [7, 8]
            finally:
                fresh.shutdown()
            return {'error': ex.failure, 'schedule_blocked': True,
                    'cancel_is_not_failure': True, 'fresh_engine_tokens': [7, 8]}
        if case in ('abrupt', 'abrupt4', 'collective', 'kill_escalation'):
            method = {'abrupt': 'hang_and_exit', 'abrupt4': 'hang_and_exit',
                      'collective': 'collective_and_exit',
                      'kill_escalation': 'ignore_terminate_and_exit_peer'}[case]
            future = ex.collective_rpc(method, non_block=True)
        else:
            methods = {'initialize': ('initialize_from_config', (ex.vllm_config.cache_config,)),
                       'capture': ('compile_or_warm_up_model', ()),
                       'execute': ('execute_model', (snapshot(),)),
                       'sample': ('sample_tokens', (None,)),
                       'propose': ('sample_tokens', (None,)),
                       'output': ('take_draft_token_ids', ())}
            method, args = methods[case]
            future = ex.collective_rpc(method, args=args, non_block=True)
        started = time.monotonic()
        try:
            future.result()
        except RuntimeError as exc:
            error = str(exc)
            assert 'rank 1' in error, error
            if case not in ('abrupt', 'abrupt4', 'collective', 'kill_escalation'):
                assert f'095 injected {case} failure' in error, error
        else:
            raise AssertionError('RPC failure not propagated')
        elapsed = time.monotonic() - started
        assert elapsed < 10, elapsed
        assert ex.failure is not None
        try:
            ex.collective_rpc('rank_identity')
        except RuntimeError:
            pass
        else:
            raise AssertionError('failed executor accepted RPC')
        return {'error': error, 'elapsed_seconds': elapsed, 'future_done': future.done()}
    finally:
        clean(ex)
        if case == 'kill_escalation':
            assert ex.workers[0].process.exitcode == -signal.SIGKILL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True, choices=['init', 'load', 'initialize', 'capture', 'execute', 'sample', 'propose', 'output', 'abrupt', 'abrupt4', 'collective', 'kill_escalation', 'engine', 'fresh'])
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    try:
        if a.case == 'fresh':
            ex = MultiprocExecutor(config(tp=2, pp=1), _worker_cls=Step095Worker, _startup_timeout=60)
            try:
                assert ex.collective_rpc('rank_identity') == [0, 1]
                ex.check_health()
                result = {'fresh_ranks': [0, 1]}
            finally:
                clean(ex)
        else:
            result = run(a.case)
        if a.case == 'kill_escalation':
            result['rank0_exitcode'] = -signal.SIGKILL  # clean() 后已断言实际退出码
        result.update(status='passed', case=a.case)
        # 真实应用中 fatal 应使父进程非零退出；测试记录预期非零。
        code = 1 if a.case not in ('engine', 'fresh') else 0
    except BaseException as exc:
        import traceback
        result = {'status': 'failed', 'case': a.case, 'error': str(exc), 'traceback': traceback.format_exc()}
        code = 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
