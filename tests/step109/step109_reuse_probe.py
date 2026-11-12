"""EAGLE3 graph reuse across requests, sequence lengths, and engine instances."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
from minivllm import CacheConfig,DeviceConfig,LLMEngine,ModelConfig,SamplingParams,SchedulerConfig,SpeculativeConfig,VllmConfig
from minivllm.config import CompilationConfig,ParallelConfig
from minivllm.executor.multiproc_executor import MultiprocExecutor
from minivllm.testing.tiny_models import tiny_eagle3_dir,tiny_qwen3_dir,tiny_qwen3_config
from step109_tiny_probe import step109_graph_stats


def run(mode):
    draft=Path(tiny_eagle3_dir('tiny_gqa'))
    cfg=VllmConfig(model_config=ModelConfig(tiny_qwen3_dir('tiny_gqa'),dtype='float32',max_model_len=64,hf_config=tiny_qwen3_config('tiny_gqa'),seed=0),
        cache_config=CacheConfig(block_size=16,num_gpu_blocks=32),scheduler_config=SchedulerConfig(max_num_seqs=2,max_num_batched_tokens=64),
        device_config=DeviceConfig('cuda'),parallel_config=ParallelConfig(tensor_parallel_size=2),attention_backend='triton',
        compilation_config=CompilationConfig(cudagraph_mode='full_decode_only' if mode=='graph' else 'none',
                                              cudagraph_capture_sizes=[1,2,3,6] if mode=='graph' else None),
        speculative_config=SpeculativeConfig(method='eagle3',num_speculative_tokens=2,
            draft_model_config=ModelConfig(str(draft),dtype='float32',max_model_len=64,hf_config=json.loads((draft/'config.json').read_text()))))
    ex=MultiprocExecutor(cfg);engine=LLMEngine(cfg,ex,tokenizer=None)
    params=SamplingParams(temperature=0,max_tokens=14,ignore_eos=True,seed=123)
    try:
        before=ex.collective_rpc(step109_graph_stats)
        engine.add_request('a',[1,2,3,4,5,6],params)
        observations=[];outputs={}
        for step in range(60):
            if step==3:
                engine.abort_request(['a'])
                engine.add_request('b',[2,3,4,5,6,7,8,9,10],params)
            for item in engine.step():outputs[item.request_id]=list(item.token_ids)
            stats=ex.collective_rpc(step109_graph_stats)
            observations.append(stats)
            if mode=='graph':
                for start,end in zip(before,stats):
                    for key in ('target','prefill','decode'):
                        assert start[key]['captured']==end[key]['captured'], (start,end)
            if step>=3 and not engine.has_unfinished_requests():break
        assert len(outputs.get('b',[]))==14
        if mode=='graph':
            for first, last in zip(observations[2], observations[-1]):
                for name in ('target','prefill','decode'):
                    assert first[name]['captured'] and first[name]['replays']
                    assert last[name]['replays'] > first[name]['replays'], (name,first,last)
        return {'outputs':outputs,'graph_stats':observations}
    finally:engine.shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    eager=run('eager');first=run('graph');second=run('graph')
    assert first['outputs']['b']==eager['outputs']['b']==second['outputs']['b']
    a.output.write_text(json.dumps({'status':'passed','eager':eager,'first':first,'second':second})+'\n')

if __name__=='__main__':main()
