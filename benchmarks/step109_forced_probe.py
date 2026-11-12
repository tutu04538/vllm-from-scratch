"""Replay one fixed real EAGLE3 history for step109."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
from benchmarks.check_step090_models import resolve_snapshot
from benchmarks.check_step090_reference import verify_oracle
from benchmarks.step103_workload import prompts
from benchmarks.step106_reference_probe import install, flush


def flush_rank(worker, directory):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.save(worker.model_runner._step109_records, Path(directory)/f'state_rank{rank}.pt')
    return {'rank': rank, 'events': flush(worker, str(Path(directory)/f'rank{rank}.pt'))}


def install_state_trace(worker, upstream):
    import torch
    runner = worker.model_runner
    spec = runner.speculator
    runner._step109_records = []
    runner._step109_graph_counts = {}
    def cpu(value):
        return value.detach().cpu().clone() if value is not None else None
    target = runner.model
    original_logits = target.compute_logits
    def logits(hidden):
        result = original_logits(hidden)
        runner._step109_target_logits = cpu(result)
        return result
    target.compute_logits = logits
    original_sample = runner.sample
    def sample(hidden, batch, grammar):
        rows = cpu(hidden[batch.logits_indices])
        result = original_sample(hidden, batch, grammar)
        output, sampled, rejected = result
        counts = cpu(sampled)
        sampled_ids = cpu(output.sampled_token_ids)
        runner._step109_records.append({'kind':'sample', 'req_ids':[str(req).split('-', 1)[0] for req in batch.req_ids], 'hidden':rows,
            'logits':runner._step109_target_logits, 'num_sampled':counts,
            'num_rejected':cpu(rejected), 'sampled_ids':[sampled_ids[i,:int(n)] for i,n in enumerate(counts)],
            'input_ids':cpu(batch.input_ids[:batch.num_tokens]),
            'positions':cpu(batch.positions[:batch.num_tokens]),
            'query_start_loc':cpu(batch.query_start_loc[:batch.num_reqs+1])})
        return result
    runner.sample = sample
    original_propose = spec.propose
    def propose(batch, *args, **kwargs):
        start = len(runner._step106_records)
        result = original_propose(batch, *args, **kwargs)
        source = next(e for e in runner._step106_records[start:] if e['event'] == 'propose')
        record = {'kind':'propose', 'draft_ids':cpu(result),
            'draft_logits':cpu(spec.draft_logits[batch.idx_mapping[:batch.num_reqs]])
                           if spec.draft_logits is not None else None}
        for index, value in enumerate(source['target_valid_kv_per_request']):
            record[f'target_valid_kv_{index}'] = value
        for index, value in enumerate(source['draft_valid_kv_per_request']):
            record[f'draft_valid_kv_{index}'] = value
        runner._step109_records.append(record)
        return result
    spec.propose = propose
    target_manager = runner.cudagraph_manager if upstream else runner.target_cudagraph_manager
    managers = {'target':target_manager, 'draft_prefill':spec.prefill_cudagraph_manager,
                'draft_decode':spec.decode_cudagraph_manager}
    for name, manager in managers.items():
        runner._step109_graph_counts[name] = {'captured':0 if manager is None else len(manager.graphs), 'replays':0}
        if manager is None:
            continue
        original = manager.run_fullgraph
        def run(*args, _name=name, _original=original, **kwargs):
            runner._step109_graph_counts[_name]['replays'] += 1
            return _original(*args, **kwargs)
        manager.run_fullgraph = run
    return True


def graph_stats(worker):
    import torch
    import torch.distributed as dist
    runner = worker.model_runner
    def info(manager):
        if manager is None:
            return None
        return {'captured': len(manager.graphs), 'replays': manager.num_replays,
                'token_counts': manager.captured_token_counts()}
    spec = runner.speculator
    return {'rank': dist.get_rank() if dist.is_initialized() else 0,
            'uuid': str(torch.cuda.get_device_properties(worker.device).uuid),
            'target': info(runner.target_cudagraph_manager),
            'draft_prefill': info(spec.prefill_cudagraph_manager),
            'draft_decode': info(spec.decode_cudagraph_manager),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(worker.device),
            'observed':runner._step109_graph_counts}


def observed_graph_stats(worker):
    import torch
    import torch.distributed as dist
    return {'rank': dist.get_rank() if dist.is_initialized() else 0,
            'uuid': str(torch.cuda.get_device_properties(worker.device).uuid),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(worker.device),
            'observed':worker.model_runner._step109_graph_counts}


def install_layer_diagnostics(worker):
    runner = worker.model_runner
    runner._step109_layers = {}
    for index, layer in enumerate(runner.model.model.layers):
        def hook(module, args, output, _index=index):
            if _index not in runner._step109_layers:
                values = output if isinstance(output, tuple) else (output,)
                runner._step109_layers[_index] = [v.detach().cpu().clone() if v is not None else None for v in values]
        layer.register_forward_hook(hook)
    layer = runner.model.model.layers[0]
    modules = {'input_norm':layer.input_layernorm, 'qkv':layer.self_attn.qkv_proj,
               'q_norm':layer.self_attn.q_norm, 'k_norm':layer.self_attn.k_norm,
               'attention':layer.self_attn.attn, 'o_proj':layer.self_attn.o_proj,
               'post_norm':layer.post_attention_layernorm, 'mlp':layer.mlp}
    for name, module in modules.items():
        def module_hook(module, args, output, _name=name):
            key = 'first_' + _name
            if key not in runner._step109_layers:
                values = output if isinstance(output, tuple) else (output,)
                runner._step109_layers[key] = [v.detach().cpu().clone() if isinstance(v, torch.Tensor) else None for v in values]
        import torch
        module.register_forward_hook(module_hook)
    def post_norm_input(module, args):
        if 'first_post_norm_input' not in runner._step109_layers:
            runner._step109_layers['first_post_norm_input'] = [v.detach().cpu().clone() for v in args]
    layer.post_attention_layernorm.register_forward_pre_hook(post_norm_input)
    return True


def flush_layer_diagnostics(worker, directory):
    import torch
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.save(worker.model_runner._step109_layers, Path(directory)/f'layers_rank{rank}.pt')
    return rank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=['mini', 'frozen'], required=True)
    p.add_argument('--model', choices=['8b', '14b'], required=True)
    p.add_argument('--tp', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--batch', type=int, choices=[1, 4], required=True)
    p.add_argument('--k', type=int, choices=[1, 2, 4], required=True)
    p.add_argument('--mode', choices=['eager', 'graph'], required=True)
    p.add_argument('--gpu-memory-utilization', type=float, default=.60)
    p.add_argument('--layer-diagnostics', action='store_true')
    p.add_argument('--forced-from', type=Path, required=True)
    p.add_argument('--probabilistic', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.model == '14b' and a.tp == 1:
        p.error('14B TP1 is outside this matrix')
    if not 0 < a.gpu_memory_utilization < 1:
        p.error('gpu-memory-utilization must be in (0, 1)')
    target, _ = resolve_snapshot(a.model)
    revisions = {'8b': '9629dfce7a4a10564dd48d3e5485c3976095653c',
                 '14b': '3d13517724e81cb409ddf1d4650772ec52f1e18e'}
    draft_root = Path('/data/syh/models/converted/AngelSlim') / f'Qwen3-{a.model.upper()}_eagle3'
    draft = draft_root / revisions[a.model]
    if not (draft/'config.json').is_file():
        raise RuntimeError(f'fixed converted draft is missing: {draft}')
    hf = json.loads((target/'config.json').read_text())
    dhf = json.loads((draft/'config.json').read_text())
    sizes = sorted({1, 2, 4, *((a.k + 1) * n for n in (1, 2, 4))})
    a.output.mkdir(parents=True, exist_ok=True)
    if a.batch != 1:
        p.error('fixed-history replay currently requires batch 1')
    import torch
    prior = json.loads((a.forced_from/'summary.json').read_text())
    if any(prior[key] != getattr(a, key) for key in ('model', 'tp', 'batch', 'k', 'mode')):
        raise ValueError('fixed history and configuration differ')
    forced_history = prior['tokens'][0]
    events = torch.load(a.forced_from/'rank0.pt', map_location='cpu', weights_only=True)
    forced_drafts = [event['output'].tolist() for event in events if event['event'] == 'propose']
    base_prompt = prompts(a.model, 1)[0]
    if len(base_prompt) != 128:
        raise ValueError(f'expected 128-token base prompt, got {len(base_prompt)}')
    prompt_ids = [base_prompt[i:] + base_prompt[:i] for i in range(a.batch)]
    start = time.monotonic()
    if a.implementation == 'mini':
        from minivllm import (CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams,
                              SchedulerConfig, SpeculativeConfig, VllmConfig)
        from minivllm.config import CompilationConfig, ParallelConfig
        from minivllm.executor.multiproc_executor import MultiprocExecutor
        from minivllm.executor.uniproc_executor import UniProcExecutor
        cfg = VllmConfig(model_config=ModelConfig(str(target), dtype='bfloat16', max_model_len=2048,
                           hf_config=hf, seed=0), cache_config=CacheConfig(block_size=16, num_gpu_blocks=192,
                           enable_prefix_caching=False), scheduler_config=SchedulerConfig(max_num_seqs=a.batch,
                           max_num_batched_tokens=128 * a.batch, async_scheduling=False), device_config=DeviceConfig('cuda'),
                           parallel_config=ParallelConfig(tensor_parallel_size=a.tp),
                           compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if a.mode == 'graph' else 'none',
                           cudagraph_capture_sizes=sizes if a.mode == 'graph' else None), attention_backend='triton',
                           speculative_config=SpeculativeConfig(method='eagle3', draft_sample_method='probabilistic' if a.probabilistic else 'greedy', num_speculative_tokens=a.k,
                           draft_model_config=ModelConfig(str(draft), dtype='bfloat16', max_model_len=2048, hf_config=dhf)))
        executor = MultiprocExecutor(cfg, _startup_timeout=1500) if a.tp > 1 else UniProcExecutor(cfg)
        engine = None
        def rpc(fn, *args):
            return executor.collective_rpc(fn, args=args) if a.tp > 1 else [fn(executor.driver_worker, *args)]
        try:
            engine = LLMEngine(cfg, executor, tokenizer=None)
            identity = rpc(install, False, forced_history, forced_drafts)
            rpc(install_state_trace, False)
            if a.layer_diagnostics:
                rpc(install_layer_diagnostics)
            for i, prompt in enumerate(prompt_ids):
                engine.add_request(str(i), prompt, SamplingParams(temperature=.8 if a.probabilistic else 0, max_tokens=32, ignore_eos=True, seed=123))
            outputs = {}
            while engine.has_unfinished_requests():
                for item in engine.step():
                    outputs[item.request_id] = list(item.token_ids)
            stats = rpc(graph_stats)
            counts = rpc(flush_rank, str(a.output.resolve()))
            if a.layer_diagnostics:
                rpc(flush_layer_diagnostics, str(a.output.resolve()))
            tokens = [outputs[str(i)] for i in range(a.batch)]
        finally:
            if engine is None:
                executor.shutdown()
            else:
                engine.shutdown()
    else:
        verify_oracle()
        from vllm import LLM, SamplingParams
        local_heads = max(1, hf['num_key_value_heads'] // a.tp)
        block_bytes = 2 * 16 * local_heads * hf['head_dim'] * 2 * hf['num_hidden_layers']
        llm = LLM(model=str(target), skip_tokenizer_init=True, dtype='bfloat16', tensor_parallel_size=a.tp,
                  enforce_eager=a.mode == 'eager', async_scheduling=False, max_model_len=2048, max_num_seqs=a.batch,
                  max_num_batched_tokens=128 * a.batch, gpu_memory_utilization=a.gpu_memory_utilization,
                  kv_cache_memory_bytes=192 * block_bytes, attention_config={'backend':'TRITON_ATTN'},
                  enable_prefix_caching=False, disable_log_stats=True, seed=0,
                  compilation_config={'mode':0, 'cudagraph_mode':'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': sizes} if a.mode == 'graph' else {'mode':0},
                  speculative_config={'method':'eagle3', 'model':str(draft), 'num_speculative_tokens':a.k, 'draft_sample_method':'probabilistic' if a.probabilistic else 'greedy'})
        assert llm.llm_engine.vllm_config.scheduler_config.async_scheduling is False
        identity = llm.collective_rpc(install, args=(True, forced_history, forced_drafts))
        llm.collective_rpc(install_state_trace, args=(True,))
        if a.layer_diagnostics:
            llm.collective_rpc(install_layer_diagnostics)
        generated = llm.generate([{'prompt_token_ids': ids} for ids in prompt_ids],
            SamplingParams(temperature=.8 if a.probabilistic else 0, max_tokens=32, ignore_eos=True, seed=123), use_tqdm=False)
        tokens = [list(o.outputs[0].token_ids) for o in generated]
        counts = llm.collective_rpc(flush_rank, args=(str(a.output.resolve()),))
        if a.layer_diagnostics:
            llm.collective_rpc(flush_layer_diagnostics, args=(str(a.output.resolve()),))
        stats = llm.collective_rpc(observed_graph_stats)
    assert all(len(t) == 32 for t in tokens)
    if a.mode == 'graph':
        for rank_stats in stats:
            for component in ('target', 'draft_prefill') + (('draft_decode',) if a.k > 1 else ()):
                observed = rank_stats['observed'][component]
                if not observed['captured'] or not observed['replays']:
                    raise RuntimeError(f'{component} was not captured and replayed on rank {rank_stats["rank"]}: {observed}')
    payload = {'implementation':a.implementation, 'model':a.model, 'tp':a.tp, 'batch':a.batch,
               'k':a.k, 'mode':a.mode, 'async_scheduling':False, 'target':str(target), 'draft':str(draft), 'prompts':prompt_ids,
               'tokens':tokens, 'identity':identity, 'graph_stats':stats, 'traces':counts,
               'elapsed_seconds':time.monotonic()-start}
    (a.output/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({'status':'passed', 'implementation':a.implementation, 'model':a.model,
                      'tp':a.tp, 'batch':a.batch, 'k':a.k, 'mode':a.mode}))

if __name__ == '__main__':
    main()
