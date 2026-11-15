"""测试和诊断专用的 CUDA Graph 分派观察器。

显式安装后才记录真实请求的选择；生产 Runner 不保留随请求数增长的历史。
"""
from __future__ import annotations

from functools import wraps
from inspect import currentframe
from weakref import WeakKeyDictionary


def install_graph_observer():
    from minivllm.worker.gpu.cudagraph_utils import CudaGraphManager
    from minivllm.worker.gpu.model_runner import GPUModelRunner as RunnerV2
    from minivllm.worker.gpu_model_runner import GPUModelRunner as RunnerV1
    from minivllm.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    from minivllm.worker.gpu.spec_decode.multi_module_mtp.speculator import MultiModuleMTPSpeculator

    if getattr(CudaGraphManager, '_test_observer_installed', False):
        return
    histories = WeakKeyDictionary()
    v1_histories = WeakKeyDictionary()

    def history(obj):
        return histories.setdefault(obj, [])

    def save(manager, desc, num_reqs, num_tokens):
        history(manager).append({'num_reqs': num_reqs, 'num_tokens': num_tokens,
            'desc_num_reqs': desc.num_reqs, 'desc_num_tokens': desc.num_tokens,
            'mode': str(desc.cg_mode), 'replayed': desc in manager.graphs})

    original_dispatch = CudaGraphManager.dispatch
    @wraps(original_dispatch)
    def dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
        desc = original_dispatch(manager, num_reqs, num_tokens, *args, **kwargs)
        caller = currentframe().f_back
        # V2 target 会在 dispatch 后检查 prefill 并丢弃不兼容描述符；
        # 改在 prepare_inputs 观察最终选择，避免记录未经使用的候选。
        if not (caller.f_code.co_name == 'execute_model'
                and caller.f_globals.get('__name__') == RunnerV2.__module__):
            save(manager, desc, num_reqs, num_tokens)
        return desc
    CudaGraphManager.dispatch = dispatch
    CudaGraphManager.selections = property(history)

    original_prepare = RunnerV2.prepare_inputs
    @wraps(original_prepare)
    def prepare(runner, scheduler_output, batch_req_state, batch_desc=None):
        caller = currentframe().f_back
        if (caller.f_code.co_name == 'execute_model'
                and caller.f_globals.get('__name__') == RunnerV2.__module__
                and batch_desc is not None and runner.target_cudagraph_manager is not None):
            save(runner.target_cudagraph_manager, batch_desc,
                 len(batch_req_state.req_ids), batch_req_state.num_tokens)
        return original_prepare(runner, scheduler_output, batch_req_state, batch_desc)
    RunnerV2.prepare_inputs = prepare

    original_determine = RunnerV1._determine_batch_execution_and_padding
    @wraps(original_determine)
    def determine(runner, *args, **kwargs):
        mode, desc = original_determine(runner, *args, **kwargs)
        caller = currentframe().f_back
        if caller.f_code.co_name == 'execute_model':
            num_tokens = kwargs['num_tokens']
            num_reqs = kwargs['num_reqs']
            v1_histories.setdefault(runner, []).append({
                'num_tokens': num_tokens, 'num_reqs': num_reqs,
                'padded_tokens': desc.num_tokens, 'padded_reqs': desc.num_reqs,
                'uniform': desc.uniform, 'mode': str(mode), 'key': str(desc)})
        return mode, desc
    RunnerV1._determine_batch_execution_and_padding = determine
    RunnerV1.cudagraph_selections = property(lambda runner: v1_histories.setdefault(runner, []))

    def auto_selections(spec):
        result = []
        for name, manager in (('prefill', spec.prefill_cudagraph_manager),
                              ('decode', spec.decode_cudagraph_manager)):
            if manager is not None:
                result.extend(dict(entry, graph=name) for entry in history(manager))
        return result
    AutoRegressiveSpeculator.cudagraph_selections = property(auto_selections)
    MultiModuleMTPSpeculator.cudagraph_selections = property(
        lambda spec: [] if spec.cudagraph_manager is None else
        [dict(entry, graph='multi_module') for entry in history(spec.cudagraph_manager)])
    CudaGraphManager._test_observer_installed = True
