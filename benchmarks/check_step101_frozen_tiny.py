"""Run frozen vLLM tiny Qwen3 reference; output hooks for all layers and steps."""
import argparse,json,sys
from pathlib import Path
from functools import partial
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.check_step090_reference import verify_oracle
from benchmarks.step101_frozen_hooks import install,flush

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--tp',type=int,choices=[1,2,4],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['TRITON_ATTN','FLEX_ATTENTION'],default='TRITON_ATTN');a=p.parse_args()
 oracle=verify_oracle()
 from vllm import LLM,SamplingParams
 a.output.mkdir(parents=True,exist_ok=True)
 llm=LLM(model=str(a.model_dir.resolve()),skip_tokenizer_init=True,dtype='float32',tensor_parallel_size=a.tp,enforce_eager=True,max_model_len=64,max_num_seqs=1,max_num_batched_tokens=64,gpu_memory_utilization=0.25,kv_cache_memory_bytes=33554432,attention_config={'backend':a.backend},enable_prefix_caching=False,disable_log_stats=True,seed=0)
 llm.apply_model(install)
 prompt=list(range(1,18));out=llm.generate([{'prompt_token_ids':prompt}],SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)[0].outputs[0]
 records=llm.apply_model(partial(flush,directory=str(a.output.resolve())))
 result={'oracle':oracle,'tp':a.tp,'output_tokens':list(out.token_ids),'records':records,'status':'passed'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','output_tokens':result['output_tokens']}))
if __name__=='__main__':main()
