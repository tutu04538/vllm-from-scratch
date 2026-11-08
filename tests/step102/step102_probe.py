"""Lifecycle trace with the real production Worker; no worker subclass."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '1'
import torch
from minivllm import CacheConfig, DeviceConfig, LLMEngine, ModelConfig, SamplingParams, SchedulerConfig, VllmConfig
from minivllm.config import CompilationConfig, ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.executor.uniproc_executor import UniProcExecutor
from minivllm.testing.tiny_models import tiny_qwen3_config, tiny_qwen3_dir
from step102_hooks import install, snapshot, fail_rank_one


def run(tp, prefix, mode, output):
    blocks = 5 if mode == 'preempt' else 12
    hf = tiny_qwen3_config('tiny_gqa')
    cfg = VllmConfig(model_config=ModelConfig(tiny_qwen3_dir('tiny_gqa'), dtype='float32', max_model_len=20 if mode == 'preempt' else 32, hf_config=hf, seed=123),
        cache_config=CacheConfig(block_size=4, num_gpu_blocks=blocks, enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=24),
        parallel_config=ParallelConfig(tensor_parallel_size=tp), device_config=DeviceConfig(device='cuda'),
        compilation_config=CompilationConfig(cudagraph_mode='none'), attention_backend='triton')
    ex = MultiprocExecutor(cfg) if tp > 1 else UniProcExecutor(cfg)
    engine = None
    def rpc(fn):
        return ex.collective_rpc(fn) if tp > 1 else [fn(ex.driver_worker)]
    try:
        engine = LLMEngine(cfg, ex, tokenizer=None)
        core = engine.engine_core.engine_core
        rpc(install)
        prompt = [1,2,3,5,6,7] if mode == 'preempt' else [1,2,3,5,6,7,8,9,10]
        params = lambda n, seed: SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, seed=seed)
        engine.add_request('a',prompt,params(8 if mode=='preempt' else 3,17))
        rounds=[]
        def step(label):
            outs=engine.step()
            ranks=rpc(snapshot)
            pool=core.kv_cache_manager.block_pool
            rounds.append({'label':label, 'ranks':ranks,
                'outputs':{o.request_id:list(o.token_ids) for o in outs},
                'scheduler':{name:{'computed':r.num_computed_tokens, 'tokens':list(r.all_token_ids),
                    'blocks':core.kv_cache_manager.get_blocks(name).get_block_ids()} for name,r in core.scheduler.requests.items()},
                'trace':core.scheduler.trace[-1] if core.scheduler.trace else None,
                'refcounts':[b.ref_cnt for b in pool.blocks], 'free':pool.get_num_free_blocks(),
                'preemptions':core.scheduler.num_preemptions})
        if mode == 'fail':
            rpc(fail_rank_one)
            before = list(core.scheduler.requests['a'].output_token_ids)
            try:
                engine.step()
                raise AssertionError('rank failure was not propagated')
            except RuntimeError as exc:
                assert 'step102 injected rank 1 sample failure' in str(exc), str(exc)
            after = list(core.scheduler.requests['a'].output_token_ids)
            assert before == after
            steps = core.scheduler.num_steps
            try:
                core.step()
                raise AssertionError('failed core accepted another step')
            except RuntimeError as exc:
                assert '失败' in str(exc)
            assert core.scheduler.num_steps == steps
            Path(output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({'tp':tp, 'mode':mode, 'before':before, 'after':after, 'steps':steps, 'failure':core.failure},output)
            return
        if mode == 'prefix':
            while engine.has_unfinished_requests(): step('a')
            engine.add_request('b',prompt,params(3,23))
            while engine.has_unfinished_requests(): step('b')
            step('cleanup')
        elif mode == 'preempt':
            step('a-start')
            engine.add_request('b',prompt,params(8,23))
            for i in range(35):
                if not engine.has_unfinished_requests(): break
                step('contend')
            step('cleanup')
        else:
            step('a-start')
            if mode == 'cancel_before':
                engine.abort_request(['a'])
                engine.add_request('c',[10,9,8,7,6],params(3,31))
                step('cancel-before')
            else:
                original_sample = ex.sample_tokens
                def cancel_before_delivery(*args, **kwargs):
                    result = original_sample(*args, **kwargs)
                    engine.abort_request(['a'])
                    return result
                ex.sample_tokens = cancel_before_delivery
                step('cancel-after-execute')
                ex.sample_tokens = original_sample
                engine.add_request('c',[10,9,8,7,6],params(3,31))
                step('reuse')
            while engine.has_unfinished_requests(): step('finish')
            step('cleanup')
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        torch.save({'tp':tp,'prefix':prefix,'mode':mode,'capacity':core.kv_cache_manager.num_gpu_blocks,'rounds':rounds},output)
        print(json.dumps({'tp':tp,'prefix':prefix,'mode':mode,'rounds':len(rounds),'preemptions':core.scheduler.num_preemptions,'hits':[r['trace']['hits'] for r in rounds if r['trace']]}))
    finally:
        if engine is not None: engine.shutdown()
        else: ex.shutdown()

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--tp',type=int,required=True);p.add_argument('--prefix',action='store_true');p.add_argument('--mode',choices=['prefix','preempt','cancel_before','cancel_after','fail'],required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.tp,a.prefix,a.mode,a.output)
