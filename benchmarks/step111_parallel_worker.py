"""单配置单引擎子进程；只在正式计时前后采集只读快照。"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from benchmarks.check_step090_models import resolve_snapshot
from spec_decode_metrics import AcceptanceAggregate, RequestTiming, summarize_run

REVISIONS = {'8b': '9629dfce7a4a10564dd48d3e5485c3976095653c',
             '14b': '3d13517724e81cb409ddf1d4650772ec52f1e18e'}


def snapshot(worker, upstream):
    import torch
    import torch.distributed as dist
    runner = worker.model_runner
    def storages(model):
        result = {}
        if model is not None:
            for tensor in model.parameters():
                storage = tensor.untyped_storage()
                result[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return result
    params = storages(runner.model)
    speculator = getattr(runner, 'speculator', None)
    draft = storages(getattr(speculator, 'model', None))
    shared = params.keys() & draft.keys()
    kv = getattr(runner, 'kv_caches', None)
    tensors = list(kv.values()) if isinstance(kv, dict) else list(kv or [])
    kv_storages = {}
    def add(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            kv_storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        elif isinstance(value, (list, tuple)):
            for child in value: add(child)
    for tensor in tensors: add(tensor)
    from benchmarks.step103_memory_hooks import snapshot as known_workspace_snapshot
    workspace = known_workspace_snapshot(worker, upstream)
    manager = getattr(runner, 'cudagraph_manager' if upstream else 'target_cudagraph_manager', None)
    graphs = getattr(manager, 'graphs', None) if manager is not None else None
    return {'rank': dist.get_rank() if dist.is_initialized() else 0,
            'device': str(worker.device),
            'device_uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
            'device_name':torch.cuda.get_device_name(worker.device), 'parameters_bytes': sum(params.values()),
            'draft_parameters_bytes': sum(draft.values()),
            'shared_parameters_bytes': sum(params[key] for key in shared),
            'kv_unique_bytes': sum(kv_storages.values()),
            'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'known_workspace_bytes':workspace['known_workspace_bytes'],
            'workspace_storage':workspace['workspace_storage'],
            'workspace_scope':'identified attention/worker buffers; excludes unknown allocator and library workspaces',
            'target_graph_count':len(graphs) if graphs is not None else 0,
            'gate_graph_replays':getattr(worker, '_step111_graph_replays', None),
            'capture_stats': getattr(runner, 'cudagraph_capture_stats', None)}


def run_once(engine, prompts, max_tokens, upstream, *, warmup=False):
    if upstream:
        from vllm import SamplingParams
    else:
        from minivllm import SamplingParams
    timings = {}
    wall_start = time.perf_counter()
    for i, ids in enumerate(prompts):
        req = f'warm{i}' if warmup else str(i)
        timings[req] = RequestTiming(req, time.perf_counter())
        engine.add_request(req, ids, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True))
    tokens, steps = {}, 0
    acceptance = AcceptanceAggregate()
    while engine.has_unfinished_requests():
        outputs = engine.step(); now = time.perf_counter(); steps += 1
        for item in outputs:
            values = list(item.outputs[0].token_ids) if upstream and item.outputs else ([] if upstream else list(item.token_ids))
            timings[item.request_id].observe(now, len(values)); tokens[item.request_id] = values
        if not upstream:
            stats = engine.last_scheduler_stats
            if stats is not None and stats.spec_decoding_stats is not None: acceptance.observe(stats.spec_decoding_stats)
        if steps > 20000: raise RuntimeError('step limit exceeded')
    wall = time.perf_counter() - wall_start
    if len(tokens) != len(prompts) or any(len(t) != max_tokens for t in tokens.values()):
        raise RuntimeError(f'incomplete output: {[len(x) for x in tokens.values()]}')
    metrics = summarize_run(timings=timings, acceptance=acceptance, wall_time_s=wall, engine_steps=steps)
    if upstream:
        metrics['acceptance'] = None
        metrics['acceptance_unavailable_reason'] = 'frozen engine counters were not collected'
    for name in ('ttft_s', 'tpot_s', 'inter_burst_latency_s'):
        metrics[name + '_detail'] = metrics[name]; metrics[name] = metrics[name]['mean']
    metrics['requests_per_s'] = len(prompts) / wall
    metrics['completion_s'] = {r: t.last_output_at - wall_start for r, t in timings.items()}
    metrics['request_timing'] = {r:{'submitted_at':t.submitted_at,
                                     'first_output_at':t.first_output_at,
                                     'last_output_at':t.last_output_at} for r,t in timings.items()}
    metrics['clock'] = 'time.perf_counter; local to this process'
    metrics['tokens'] = tokens
    return metrics


def install_gate(rpc, upstream, k):
    """Only correctness subprocesses install hooks; performance stays uninstrumented."""
    if k:
        from benchmarks.step106_reference_probe import install
        from benchmarks.step109_real_probe import install_state_trace
        rpc(install, upstream)
        rpc(install_state_trace, upstream)
    else:
        from benchmarks.step103_hooks import install
        rpc(install, upstream)


def install_graph_gate(worker, upstream):
    runner = worker.model_runner
    manager = getattr(runner, 'cudagraph_manager' if upstream else 'target_cudagraph_manager', None)
    if manager is None:
        raise RuntimeError('requested graph has no target graph manager')
    original = manager.run_fullgraph
    worker._step111_graph_replays = 0
    def replay(*args, **kwargs):
        worker._step111_graph_replays += 1
        return original(*args, **kwargs)
    manager.run_fullgraph = replay
    return True


def flush_gate(rpc, directory, k):
    directory.mkdir(parents=True, exist_ok=True)
    if k:
        from benchmarks.step109_real_probe import flush_rank
        return rpc(flush_rank, str(directory.resolve()))
    from benchmarks.step103_hooks import flush
    return rpc(flush, str(directory.resolve()), 'run0')


def trace_begin(worker):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.cuda.synchronize(worker.device)
    torch.cuda.nvtx.range_push(f'step111_steady_rank_{rank}')
    return rank


def trace_install_collectives(worker, upstream):
    """Count Python-issued collective input bytes; graph replay is separately traced."""
    import torch
    import torch.distributed as dist
    counts = {'all_reduce': [], 'all_gather': []}
    worker._step111_collectives = counts
    if not dist.is_initialized():
        return True
    if upstream:
        from vllm.distributed.parallel_state import get_tp_group
    else:
        from minivllm.distributed.parallel_state import get_tp_group
    group = get_tp_group()
    if group.world_size <= 1:
        return True
    for name in counts:
        original = getattr(group, name)
        def wrapped(tensor, *args, _name=name, _original=original, **kwargs):
            size = int(tensor.numel() * tensor.element_size())
            capturing = torch.cuda.is_current_stream_capturing()
            counts[_name].append({'bytes':size, 'capturing':capturing})
            if capturing: torch.cuda.nvtx.range_push(f'step111_capture_{_name}_{size}')
            try:
                return _original(tensor, *args, **kwargs)
            finally:
                if capturing: torch.cuda.nvtx.range_pop()
        setattr(group, name, wrapped)
    return True


def trace_collectives(worker, reset=False):
    import torch.distributed as dist
    raw = worker._step111_collectives
    result = {'rank':dist.get_rank() if dist.is_initialized() else 0,
            'coverage':'Python-issued collectives only; CUDA graph replay is not counted',
            'operations':{name:{'calls':len(events), 'input_bytes':sum(item['bytes'] for item in events),
                                'events':list(events)} for name, events in raw.items()}}
    if reset:
        for events in raw.values(): events.clear()
    return result


def trace_scheduler(engine):
    """Observe only the scheduler output in the separate diagnostic run."""
    scheduler = engine.engine_core.engine_core.scheduler
    original = scheduler.schedule
    records = []
    def schedule(*args, **kwargs):
        output = original(*args, **kwargs)
        scheduled = output.num_scheduled_tokens
        records.append({'at':time.perf_counter(), 'request_ids':list(scheduled),
                        'requests':len(scheduled), 'scheduled_tokens':sum(scheduled.values()),
                        'running':len(scheduler.running), 'waiting':len(scheduler.waiting)})
        return output
    scheduler.schedule = schedule
    return records


def measured_runs(engine, prompts, output_length, upstream, repeats, schedule_records):
    samples = []
    for _ in range(repeats):
        start = len(schedule_records) if schedule_records is not None else 0
        sample = run_once(engine, prompts, output_length, upstream)
        if schedule_records is not None:
            records = schedule_records[start:]
            first = {}
            for row in records:
                for req in row['request_ids']:
                    first.setdefault(str(req).split('-', 1)[0], row['at'])
            queue = {}
            for req, timing in sample['request_timing'].items():
                if req not in first:
                    raise RuntimeError(f'no first scheduler timestamp for {req}')
                queue[req] = first[req] - timing['submitted_at']
                if queue[req] < 0: raise RuntimeError(f'negative queue time for {req}')
            sample['queue_seconds'] = sum(queue.values()) / len(queue)
            sample['queue_seconds_by_request'] = queue
            sample['batch_schedule'] = records
        samples.append(sample)
    return samples


def install_schedule_gate(engine):
    """Record protocol snapshots without changing the Scheduler's state."""
    scheduler = engine.engine_core.engine_core.scheduler
    original = scheduler.schedule
    records = []
    def reqid(value): return str(value).split('-', 1)[0]
    def schedule(*args, **kwargs):
        output = original(*args, **kwargs)
        cached = output.scheduled_cached_reqs
        new = [{'request_id':reqid(item.req_id), 'prompt_token_ids':item.prompt_token_ids,
                'block_ids':item.block_ids, 'num_computed_tokens':item.num_computed_tokens}
               for item in output.scheduled_new_reqs]
        records.append({'new':new, 'cached':{
            'req_ids':[reqid(item) for item in cached.req_ids],
            'num_computed_tokens':list(cached.num_computed_tokens),
            'num_output_tokens':list(cached.num_output_tokens),
            'new_block_ids':list(cached.new_block_ids),
            'resumed_req_ids':sorted(reqid(item) for item in cached.resumed_req_ids)},
            'scheduled_tokens':[(reqid(req), count) for req,count in output.num_scheduled_tokens.items()],
            'spec_tokens':[(reqid(req), list(ids)) for req,ids in output.scheduled_spec_decode_tokens.items()],
            'finished_req_ids':sorted(reqid(item) for item in output.finished_req_ids)})
        return output
    scheduler.schedule = schedule
    return records


def check_identity(config, case):
    scheduler = config.scheduler_config
    cache = config.cache_config
    mode = config.compilation_config.cudagraph_mode
    mode = mode.name if hasattr(mode, 'name') else str(mode)
    compile_mode = config.compilation_config.mode
    compile_mode = compile_mode.name if hasattr(compile_mode, 'name') else str(compile_mode)
    dtype = str(config.model_config.dtype).removeprefix('torch.')
    backend = getattr(config, 'attention_backend', None)
    if backend is None:
        backend = getattr(config.attention_config, 'backend', None)
    backend = backend.name if hasattr(backend, 'name') else str(backend)
    backend = 'TRITON_ATTN' if backend.lower() in ('triton', 'triton_attn') else backend
    expected_mode = 'FULL_DECODE_ONLY' if case['mode'] == 'graph' else 'NONE'
    actual = {'tp':config.parallel_config.tensor_parallel_size,
              'pp':config.parallel_config.pipeline_parallel_size, 'dp':config.parallel_config.data_parallel_size,
              'max_num_seqs':scheduler.max_num_seqs,
              'max_num_batched_tokens':scheduler.max_num_batched_tokens,
              'async_scheduling':scheduler.async_scheduling,
              'block_size':cache.block_size,
              'enable_prefix_caching':cache.enable_prefix_caching,
              'cudagraph_mode':mode,
              'compilation_mode':compile_mode, 'dtype':dtype, 'seed':config.model_config.seed, 'attention_backend':backend,
              'max_model_len':config.model_config.max_model_len,
              'num_speculative_tokens':getattr(config.speculative_config, 'num_speculative_tokens', 0) or 0}
    expected = {'tp':case['tp'], 'pp':1, 'dp':1, 'max_num_seqs':case['batch'], 'max_num_batched_tokens':128,
                'async_scheduling':False, 'block_size':16, 'enable_prefix_caching':False,
                'cudagraph_mode':expected_mode, 'compilation_mode':'NONE', 'dtype':'bfloat16', 'seed':0, 'attention_backend':'TRITON_ATTN',
                'max_model_len':2048, 'num_speculative_tokens':case['k']}
    if actual != expected:
        raise ValueError(f'resolved configuration differs: actual={actual}, expected={expected}')
    return actual


def trace_end(worker):
    import torch
    torch.cuda.synchronize(worker.device)
    torch.cuda.nvtx.range_pop()
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True); p.add_argument('--implementation', choices=('mini', 'frozen'), required=True)
    p.add_argument('--repeats', type=int, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('gate', 'performance'), required=True)
    a = p.parse_args(); case = json.loads(a.case); model, tp, k = case['model'], case['tp'], case['k']
    target, _ = resolve_snapshot(model)
    draft = Path('/data/syh/models/converted/AngelSlim') / f'Qwen3-{model.upper()}_eagle3' / REVISIONS[model]
    hf = json.loads((target / 'config.json').read_text())
    source = json.loads((ROOT / 'benchmarks/results/step090/numerics_contract.json').read_text())['models'][model]['prompt_token_ids']
    # Version 2 is frozen before cross-TP reference registration. The initial
    # long-input diagnostic used offset 0 and remains separate from this run.
    prompts = [[source[(j + 19 * i + (1 if n == 1024 else 0)) % len(source)]
               for j in range(n)] for i, n in enumerate(case['lengths'])]
    sizes = [1, 2, 3, 4, 6, 12] if k else [1, 2, 4]
    batch = case['batch']; graph = case['mode'] == 'graph'; blocks = 800
    start = time.perf_counter()
    if a.implementation == 'mini':
        from minivllm import CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
        from minivllm.config import CompilationConfig, ParallelConfig
        from minivllm.executor.multiproc_executor import MultiprocExecutor
        from minivllm.executor.uniproc_executor import UniProcExecutor
        spec = None if k == 0 else SpeculativeConfig(method='eagle3', num_speculative_tokens=k,
                    draft_model_config=ModelConfig(str(draft), dtype='bfloat16', max_model_len=2048,
                    hf_config=json.loads((draft / 'config.json').read_text())))
        cfg = VllmConfig(model_config=ModelConfig(str(target), dtype='bfloat16', max_model_len=2048, hf_config=hf, seed=0),
            cache_config=CacheConfig(block_size=16, num_gpu_blocks=blocks, enable_prefix_caching=False),
            scheduler_config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=128, async_scheduling=False),
            device_config=DeviceConfig('cuda'), parallel_config=ParallelConfig(tensor_parallel_size=tp),
            compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if graph else 'none',
                                                cudagraph_capture_sizes=sizes if graph else None),
            attention_backend='triton', speculative_config=spec)
        identity = check_identity(cfg, case)
        executor = MultiprocExecutor(cfg, _startup_timeout=1500) if tp > 1 else UniProcExecutor(cfg)
        executor_init_s = time.perf_counter() - start
        engine = None
        def rpc(fn, *args):
            return executor.collective_rpc(fn, args=args) if tp > 1 else [fn(executor.driver_worker, *args)]
        try:
            if os.environ.get('STEP111_TRACE') == '1': rpc(trace_install_collectives, False)
            engine = LLMEngine(cfg, executor, tokenizer=None)
            build_s = time.perf_counter() - start
            output_length = 32 if a.phase == 'gate' else 128
            before = rpc(snapshot, False); warmup_start = time.perf_counter(); run_once(engine, prompts, output_length, False, warmup=True)
            warmup_s = time.perf_counter() - warmup_start
            schedule_gate = None
            if a.phase == 'gate':
                install_gate(rpc, False, k); schedule_gate = install_schedule_gate(engine)
                if graph: rpc(install_graph_gate, False)
            trace_batches = trace_scheduler(engine) if a.phase == 'performance' else None
            trace_setup_communication = None
            if os.environ.get('STEP111_TRACE') == '1':
                trace_setup_communication = rpc(trace_collectives, True); rpc(trace_begin)
            samples = measured_runs(engine, prompts, output_length, False, 1 if a.phase == 'gate' else a.repeats, trace_batches)
            trace_communication = None
            if os.environ.get('STEP111_TRACE') == '1':
                rpc(trace_end); trace_communication = rpc(trace_collectives)
            traces = flush_gate(rpc, a.output.with_suffix(''), k) if a.phase == 'gate' else None
            after = rpc(snapshot, False)
        finally:
            if engine is None: executor.shutdown()
            else: engine.shutdown()
    else:
        os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
        from benchmarks.check_step090_reference import verify_oracle
        verify_oracle()
        from vllm import LLM
        local_heads = max(1, hf['num_key_value_heads'] // tp)
        block_bytes = 2 * 16 * local_heads * hf['head_dim'] * 2 * hf['num_hidden_layers']
        llm = LLM(model=str(target), skip_tokenizer_init=True, dtype='bfloat16', tensor_parallel_size=tp,
            enforce_eager=not graph, async_scheduling=False, max_model_len=2048, max_num_seqs=batch,
            max_num_batched_tokens=128, block_size=16, gpu_memory_utilization=.9, kv_cache_memory_bytes=blocks*block_bytes,
            attention_config={'backend':'TRITON_ATTN'}, enable_prefix_caching=False, disable_log_stats=True, seed=0,
            compilation_config={'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY', 'cudagraph_capture_sizes':sizes} if graph else {'mode':0},
            speculative_config={'method':'eagle3', 'model':str(draft), 'num_speculative_tokens':k} if k else None)
        executor_init_s = None
        try:
            assert llm.llm_engine.vllm_config.scheduler_config.async_scheduling is False
            identity = check_identity(llm.llm_engine.vllm_config, case)
            engine = llm.llm_engine; build_s = time.perf_counter()-start
            def rpc(fn, *args): return llm.collective_rpc(fn, args=args)
            before = rpc(snapshot, True)
            output_length = 32 if a.phase == 'gate' else 128
            warmup_start = time.perf_counter(); run_once(engine, prompts, output_length, True, warmup=True)
            warmup_s = time.perf_counter()-warmup_start
            schedule_gate = None
            if a.phase == 'gate':
                install_gate(rpc, True, k); schedule_gate = install_schedule_gate(engine)
                if graph: rpc(install_graph_gate, True)
            trace_batches = trace_scheduler(engine) if a.phase == 'performance' else None
            trace_setup_communication = None
            if os.environ.get('STEP111_TRACE') == '1':
                rpc(trace_install_collectives, True); rpc(trace_begin)
            samples = measured_runs(engine, prompts, output_length, True, 1 if a.phase == 'gate' else a.repeats, trace_batches)
            trace_communication = None
            if os.environ.get('STEP111_TRACE') == '1':
                rpc(trace_end); trace_communication = rpc(trace_collectives)
            traces = flush_gate(rpc, a.output.with_suffix(''), k) if a.phase == 'gate' else None
            after = rpc(snapshot, True)
        finally:
            del llm; gc.collect()
    capture = [((item.get('capture_stats') or {}).get('capture_seconds')) for item in before]
    payload = {'case': case, 'implementation': a.implementation, 'target':str(target), 'draft':str(draft) if k else None,
               'target_revision':target.name, 'draft_revision':draft.name if k else None,
               'target_config_sha256':hashlib.sha256((target/'config.json').read_bytes()).hexdigest(),
               'draft_config_sha256':hashlib.sha256((draft/'config.json').read_bytes()).hexdigest() if k else None,
               'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'identity':identity,
               'prompt_ids':prompts, 'prompt_fixture':'step111-v2',
               'build_seconds':build_s, 'capture_seconds_by_rank':capture, 'warmup_seconds':warmup_s,
               'executor_init_including_load_seconds':executor_init_s,
               'rank_before':before, 'rank_after':after, 'samples':samples, 'traces':traces,
               'trace_communication':trace_communication, 'trace_setup_communication':trace_setup_communication,
               'trace_scheduler_batches':trace_batches,
               'schedule_gate':schedule_gate}
    if a.phase == 'gate' and k:
        tokens = [samples[0]['tokens'][str(i)] for i in range(batch)]
        summary = {'implementation':a.implementation, 'model':model, 'tp':tp, 'batch':batch,
                   'k':k, 'mode':case['mode'], 'prompts':prompts, 'tokens':tokens}
        (a.output.with_suffix('')/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    a.output.write_text(json.dumps(payload, indent=2)+'\n')

if __name__ == '__main__': main()
