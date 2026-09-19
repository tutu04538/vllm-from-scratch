"""Read-only real Qwen3 acceptance + measured Engine timings. Reference work excluded from timings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1'
import argparse,copy,gc,hashlib,json,statistics,sys,time
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import torch,transformers
from transformers import AutoTokenizer,Qwen3ForCausalLM
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录');sys.path.insert(0,str(RECORDS/'tools'))
from step31_source import source_digest,source_files
from step31_test_helpers import m
from step31_external_process import mapped
from step31.step31 import encode
MODEL=Path('/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B')

def summary(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive');med=statistics.median(xs)
    return dict(median_ms=med,iqr_over_median=(q[2]-q[0])/med,samples_ms=xs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['torch','eager','graph']);ap.add_argument('--repeats',type=int,default=5);a=ap.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sha=source_digest();file_stats={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in MODEL.iterdir() if p.is_file()}
    t=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tokenizer_load_ms=(time.perf_counter()-t)*1000
    questions=['用一句话解释什么是 KV cache。','请只回答：1+1等于几？','用一句话解释什么是 Transformer。'];prompts=[]
    for q in questions:
        official_text=tokenizer.apply_chat_template([dict(role='user',content=q)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(official_text,add_special_tokens=False)['input_ids'];assert ids==encode(tokenizer,q);prompts.append(ids)
    encoding=[]
    for _ in range(11):
        t=time.perf_counter();[encode(tokenizer,q) for q in questions];encoding.append((time.perf_counter()-t)*1000)
    opts=dict(device='cuda',attention_backend='torch' if a.mode=='torch' else 'triton',use_cuda_graph=a.mode=='graph',max_num_seqs=3,max_num_batched_tokens=16,num_kv_blocks=64,block_size=16,enable_prefix_caching=True)
    torch.cuda.synchronize();t=time.perf_counter();e=m.Engine.from_model_dir(MODEL,**opts);torch.cuda.synchronize();load_ms=(time.perf_counter()-t)*1000
    assert set(e.model.eos_token_ids)=={151643,151645} and e.scheduler.eos_token_ids=={151643,151645}
    assert all(p.dtype==torch.float32 for p in e.model.parameters()) and e.kv_cache_pool.k_cache.dtype==torch.float32
    assert not e.model.graphs and not e.scheduler.waiting and not e.scheduler.running and not e.kv_cache_pool.block_hash
    reqs=[dict(request_id=f'r{i}',prompt_ids=p,max_new_tokens=8) for i,p in enumerate(prompts)]
    def generate(requests=reqs):
        torch.cuda.synchronize();start=time.perf_counter();results=[];steps=0;times={r['request_id']:[] for r in requests};lengths={k:0 for k in times}
        for r in requests:e.add_request(copy.deepcopy(r))
        refs={s.request_id:s for s in e.scheduler.waiting+e.scheduler.running}
        while e.has_unfinished_requests():
            results.extend(copy.deepcopy(e.step()));steps+=1;assert steps<200;torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            for k,s in refs.items():
                if len(s.output_ids)>lengths[k]:times[k].append(elapsed);lengths[k]=len(s.output_ids)
        return dict(results={r['request_id']:r['output_ids'] for r in results},elapsed_ms=(time.perf_counter()-start)*1000,steps=steps,times_ms=times)
    with torch.inference_mode():first=generate()
    # Check every loaded parameter against its actual BF16 file value, not metadata alone.
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as f:
        for k,v in e.model.state_dict().items():assert torch.equal(v.cpu(),f.get_tensor(mapped(k)).float()),k
    hf=Qwen3ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda')
    errors=[];kv_errors=[];calls=[];original=e.model._forward_append
    def wrapped(ids,counts,caches,pool):
        seqs=[next(s for s in e.scheduler.running if s.cache is c) for c in caches];lengths=[c.length for c in caches]
        histories=[(s.prompt_ids+s.output_ids)[:l+n] for s,l,n in zip(seqs,lengths,counts)]
        out=original(ids,counts,caches,pool)
        for s,c,h,l,n,rows in zip(seqs,caches,histories,lengths,counts,out.split(counts)):
            want=hf(torch.tensor([h],device='cuda'),use_cache=True);expected=want.logits[0,l:]
            torch.testing.assert_close(rows,expected,atol=3e-4,rtol=3e-4);errors.append(float((rows-expected).abs().max()))
            slots=torch.arange(c.length,device='cuda');blocks=torch.tensor(c.block_table,device='cuda')[slots//pool.block_size];offsets=slots%pool.block_size
            for storage,attr in [(pool.k_cache,'keys'),(pool.v_cache,'values')]:
                got=storage[:,blocks,offsets];ref=torch.stack([getattr(layer,attr)[0].transpose(0,1) for layer in want.past_key_values.layers])
                torch.testing.assert_close(got,ref,atol=3e-4,rtol=3e-4);kv_errors.append(float((got-ref).abs().max()))
            assert c.length==l+n
            calls.append(dict(request_id=s.request_id,history=l,count=n,total=c.length))
        return out
    with torch.inference_mode(),patch.object(e.model,'_forward_append',side_effect=wrapped):verified=generate()
    expected={}
    with torch.inference_mode():
        for r in reqs:
            ids=torch.tensor([r['prompt_ids']],device='cuda');past=None;tokens=[]
            for _ in range(r['max_new_tokens']):
                out=hf(ids,past_key_values=past,use_cache=True);tok=int(out.logits[0,-1].argmax());tokens.append(tok);past=out.past_key_values
                if tok in e.model.eos_token_ids:break
                ids=torch.tensor([[tok]],device='cuda')
            expected[r['request_id']]=tokens
    assert first['results']==verified['results']==expected
    assert any(c['request_id']=='r0' and c['history']>=16 and c['count']<len(prompts[0]) for c in calls)
    assert not any(e.kv_cache_pool.block_usage)
    del hf;gc.collect();torch.cuda.empty_cache()
    long_result=None
    if a.mode=='eager':
        with torch.inference_mode():long_result=generate([dict(request_id='long',prompt_ids=prompts[0],max_new_tokens=32)])
        golden=json.loads((RECORDS/'step31_local_reference_20260919T123344.681266Z.json').read_text());assert long_result['results']['long']==golden['output_ids'] and long_result['results']['long'][-1] in e.model.eos_token_ids
    # Benchmark without prompt-cache hits. No active references; preserve graph/input/KV addresses.
    e.enable_prefix_caching=False;e.scheduler.enable_prefix_caching=False;e.kv_cache_pool.enable_prefix_caching=False;e.kv_cache_pool.block_hash.clear();e.kv_cache_pool.block_to_hash.clear()
    with torch.inference_mode():
        for _ in range(2):assert generate()['results']==expected
        samples=[]
        for _ in range(a.repeats):
            value=generate();assert value['results']==expected;samples.append(value)
    assert not any(e.kv_cache_pool.block_usage) and not e.kv_cache_pool.block_hash
    assert file_stats=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in MODEL.iterdir() if p.is_file()} and sha==source_digest()
    itl=[[b-a for ts in s['times_ms'].values() for a,b in zip(ts,ts[1:])] for s in samples]
    ttft=[statistics.mean(ts[0] for ts in s['times_ms'].values()) for s in samples]
    data=dict(status='passed',mode=a.mode,source_sha256=sha,source_files=source_files(),torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),dtype='float32',tf32=False,threads=1,model_dir=str(MODEL),options=opts,questions=questions,prompt_ids=prompts,results=expected,texts={k:tokenizer.decode(v,skip_special_tokens=True) for k,v in expected.items()},tokenizer_load_ms=tokenizer_load_ms,encoding_three_prompts=summary(encoding),model_load_ms=load_ms,first_workload=first,logits_max_abs_error=max(errors),kv_max_abs_error=max(kv_errors),verified_calls=calls,long_eos_result=long_result,steady=summary([s['elapsed_ms'] for s in samples]),ttft_request_mean=summary(ttft),itl_request_mean=summary([statistics.mean(v) for v in itl]),samples=samples,scope='One fresh process per backend. Load excludes tokenizer and pre-initialized CUDA context. First full workload includes first GPU library work/JIT/graph capture; on-disk caches retained. Reference and full parameter/KV checks are outside timing. Warm benchmark has prefix caching OFF,2 warmups,5 complete workloads by default,3 requests max8 output tokens each. TTFT is request submission to first sampled token (not text streaming), reported mean over requests per run. ITL includes scheduling and other requests/prefill, not pure kernel time. Encoding measured separately; timing not a claim of stable cross-backend speedup. Source files never edited.')
    p=ROOT/'benchmarks/results'/f"step31_real_{a.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(p);print(a.mode,'load_ms',load_ms,'steady_ms',data['steady']['median_ms'],'maxerror',max(errors))
if __name__=='__main__':main()
