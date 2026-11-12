"""Run immutable real-model workload on installed hash-verified frozen vLLM."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
from benchmarks.step103_workload import prompts
from benchmarks.step103_hooks import install,flush
from benchmarks.check_step090_reference import verify_oracle
from benchmarks.check_step090_models import resolve_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['8b','14b'],required=True)
    p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--batch',type=int,choices=[1,4],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--gpu-memory-utilization',type=float,default=.86);a=p.parse_args()
    if a.model=='14b' and a.tp==1:p.error('14B TP1 exceeds card memory')
    if a.repeats<2:p.error('reference needs two repeats')
    a.output.mkdir(parents=True,exist_ok=True)
    oracle=verify_oracle(); path,_=resolve_snapshot(a.model); cfg=json.loads((path/'config.json').read_text())
    kv_heads=max(1,cfg['num_key_value_heads']//a.tp)
    bytes_per_block=2*16*kv_heads*cfg['head_dim']*2*cfg['num_hidden_layers']; budget=bytes_per_block*192
    from vllm import LLM,SamplingParams
    start=time.monotonic()
    llm=LLM(model=str(path),skip_tokenizer_init=True,dtype='bfloat16',tensor_parallel_size=a.tp,
            enforce_eager=True,max_model_len=2048,max_num_seqs=a.batch,max_num_batched_tokens=128,
            kv_cache_memory_bytes=budget,gpu_memory_utilization=a.gpu_memory_utilization,
            attention_config={'backend':'TRITON_ATTN'},enable_prefix_caching=False,disable_log_stats=True,seed=0)
    initial=llm.collective_rpc(install,args=(True,))
    prompt_ids=prompts(a.model,a.batch);runs=[]
    for i in range(a.repeats):
        generated=llm.generate([{'prompt_token_ids':p} for p in prompt_ids],
             SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)
        ranks=llm.collective_rpc(flush,args=(str(a.output.resolve()),f'run{i}'))
        runs.append({'outputs':[{'request_id':o.request_id,'tokens':list(o.outputs[0].token_ids),
                                  'finish_reason':o.outputs[0].finish_reason} for o in generated],'ranks':ranks})
    result={'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch,'prompts':prompt_ids,
            'kv_budget_bytes_per_rank':budget,'kv_num_blocks_requested':192,
            'gpu_memory_utilization':a.gpu_memory_utilization,'initial_ranks':initial,'runs':runs,'oracle':oracle,
            'elapsed_seconds':time.monotonic()-start}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','model':a.model,'tp':a.tp,'batch':a.batch}))
if __name__=='__main__':main()
