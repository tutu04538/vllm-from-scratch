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
from step32_precision_reference import reference as precision_reference
from step32_source import source_digest,source_files
from step32_test_helpers import m
from step32_external_process import mapped
from step32.step32 import encode
MODEL=Path('/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B')

def summary(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive');med=statistics.median(xs)
    return dict(median_ms=med,iqr_over_median=(q[2]-q[0])/med,samples_ms=xs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['torch','eager','graph']);ap.add_argument('--dtype',choices=['float32','bfloat16'],required=True);ap.add_argument('--repeats',type=int,default=5);a=ap.parse_args();dt=getattr(torch,a.dtype)
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
    torch.cuda.synchronize();t=time.perf_counter();e=m.Engine.from_model_dir(MODEL,**opts,dtype=dt);torch.cuda.synchronize();load_ms=(time.perf_counter()-t)*1000
    assert set(e.model.eos_token_ids)=={151643,151645} and e.scheduler.eos_token_ids=={151643,151645}
    assert all(p.dtype==dt for p in e.model.parameters()) and e.kv_cache_pool.k_cache.dtype==dt
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
    memory=dict(parameter_bytes=sum(x.numel()*x.element_size() for x in e.model.parameters()),kv_bytes=sum(x.numel()*x.element_size() for x in [e.kv_cache_pool.k_cache,e.kv_cache_pool.v_cache]),loaded_allocated=torch.cuda.memory_allocated(),loaded_reserved=torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():first=generate()
    memory['first_peak_allocated']=torch.cuda.max_memory_allocated();memory['first_peak_reserved']=torch.cuda.max_memory_reserved()
    # Check every loaded parameter against its actual BF16 file value, not metadata alone.
    with safe_open(MODEL/'model.safetensors',framework='pt',device='cpu') as f:
        for k,v in e.model.state_dict().items():assert torch.equal(v.cpu(),f.get_tensor(mapped(k)).to(dt)),k
    hf=Qwen3ForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=dt,attn_implementation='eager').eval().to('cuda')
    errors=[];kv_errors=[];calls=[];precision_errors=[];precision_kv_errors=[];disagreements=[];state=e.model.state_dict();original=e.model._forward_append
    def wrapped(ids,counts,caches,pool):
        seqs=[next(s for s in e.scheduler.running if s.cache is c) for c in caches];lengths=[c.length for c in caches]
        histories=[(s.prompt_ids+s.output_ids)[:l+n] for s,l,n in zip(seqs,lengths,counts)]
        out=original(ids,counts,caches,pool)
        for s,c,h,l,n,rows in zip(seqs,caches,histories,lengths,counts,out.split(counts)):
            want=hf(torch.tensor([h],device='cuda'),use_cache=True);expected=want.logits[0,l:]
            
            if dt==torch.float32:torch.testing.assert_close(rows,expected,atol=3e-4,rtol=3e-4)
            errors.append(float((rows.float()-expected.float()).abs().max()))
            assert torch.isfinite(rows).all()
            if dt==torch.bfloat16:
                exact,exact_kv=precision_reference(state,h,e.model.num_q_heads,e.model.num_kv_heads,e.model.rms_norm_eps,e.model.rope_theta)
                precision_errors.append(float((rows.float()-exact[l:].float()).abs().max()))
                top=rows[-1].float().topk(2);ref_top=expected[-1].float().topk(2)
                disagreements.append(dict(request=s.request_id,history=len(h),candidate_token=int(top.indices[0]),hf_token=int(ref_top.indices[0]),candidate_gap=float(top.values[0]-top.values[1]),hf_gap=float(ref_top.values[0]-ref_top.values[1]),precision_token=int(exact[-1].argmax())))
            slots=torch.arange(c.length,device='cuda');blocks=torch.tensor(c.block_table,device='cuda')[slots//pool.block_size];offsets=slots%pool.block_size
            for storage,attr in [(pool.k_cache,'keys'),(pool.v_cache,'values')]:
                got=storage[:,blocks,offsets];ref=torch.stack([getattr(layer,attr)[0].transpose(0,1) for layer in want.past_key_values.layers])
                
                if dt==torch.float32:torch.testing.assert_close(got,ref,atol=3e-4,rtol=3e-4)
                kv_errors.append(float((got.float()-ref.float()).abs().max()))
                if dt==torch.bfloat16:precision_kv_errors.append(float((got.float()-exact_kv[0 if attr=='keys' else 1].float()).abs().max()))
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
    hf_expected=expected
    if dt==torch.float32:assert first['results']==verified['results']==expected
    assert first['results']==verified['results']
    expected=first['results']
    assert any(c['request_id']=='r0' and c['history']>=16 and c['count']<len(prompts[0]) for c in calls)
    assert not any(e.kv_cache_pool.block_usage)
    del hf,out,past,ids;gc.collect();torch.cuda.empty_cache()
    long_result=None
    if a.mode=='eager' and dt==torch.float32:
        with torch.inference_mode():long_result=generate([dict(request_id='long',prompt_ids=prompts[0],max_new_tokens=32)])
        golden=json.loads((RECORDS/'step31_local_reference_20260919T123344.681266Z.json').read_text());assert long_result['results']['long']==golden['output_ids'] and long_result['results']['long'][-1] in e.model.eos_token_ids
    # Benchmark without prompt-cache hits. No active references; preserve graph/input/KV addresses.
    e.enable_prefix_caching=False;e.scheduler.enable_prefix_caching=False;e.kv_cache_pool.enable_prefix_caching=False;e.kv_cache_pool.block_hash.clear();e.kv_cache_pool.block_to_hash.clear()
    with torch.inference_mode():
        for _ in range(2):assert generate()['results']==expected
        memory['steady_base_allocated']=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(a.repeats):
            value=generate();assert value['results']==expected;samples.append(value)
    memory['steady_peak_allocated']=torch.cuda.max_memory_allocated();memory['steady_peak_reserved']=torch.cuda.max_memory_reserved()
    assert not any(e.kv_cache_pool.block_usage) and not e.kv_cache_pool.block_hash
    assert file_stats=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in MODEL.iterdir() if p.is_file()} and sha==source_digest()
    itl=[[b-a for ts in s['times_ms'].values() for a,b in zip(ts,ts[1:])] for s in samples]
    ttft=[statistics.mean(ts[0] for ts in s['times_ms'].values()) for s in samples]
    data=dict(measurement_revision=2,reference_outputs_deleted=True,status='measured' if dt==torch.bfloat16 else 'passed',mode=a.mode,source_sha256=sha,source_files=source_files(),torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),dtype=a.dtype,memory=memory,hf_results=hf_expected,hf_comparisons=disagreements,precision_logits_max_abs_error=max(precision_errors,default=0),precision_kv_max_abs_error=max(precision_kv_errors,default=0),tf32=False,threads=1,model_dir=str(MODEL),options=opts,questions=questions,prompt_ids=prompts,results=expected,texts={k:tokenizer.decode(v,skip_special_tokens=True) for k,v in expected.items()},tokenizer_load_ms=tokenizer_load_ms,encoding_three_prompts=summary(encoding),model_load_ms=load_ms,first_workload=first,logits_max_abs_error=max(errors),kv_max_abs_error=max(kv_errors),verified_calls=calls,long_eos_result=long_result,steady=summary([s['elapsed_ms'] for s in samples]),ttft_request_mean=summary(ttft),itl_request_mean=summary([statistics.mean(v) for v in itl]),samples=samples,scope='One fresh process per backend. Load excludes tokenizer and pre-initialized CUDA context. First full workload includes first GPU library work/JIT/graph capture; on-disk caches retained. BF16 official comparisons are diagnostic (different rounding policy); BF16 precision-aware dense oracle metrics are recorded for review, not silently accepted with FP32 tolerance. Memory is PyTorch allocated/reserved, not nvidia-smi, reference excluded from first/steady peaks. Reference and full parameter/KV checks are outside timing. Warm benchmark has prefix caching OFF,2 warmups,5 complete workloads by default,3 requests max8 output tokens each. TTFT is request submission to first sampled token (not text streaming), reported mean over requests per run. ITL includes scheduling and other requests/prefill, not pure kernel time. Encoding measured separately; timing not a claim of stable cross-backend speedup. Source files never edited.')
    p=ROOT/'benchmarks/results'/f"step32_real_{a.dtype}_{a.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(p);print(a.mode,'load_ms',load_ms,'steady_ms',data['steady']['median_ms'],'maxerror',max(errors))
if __name__=='__main__':main()
