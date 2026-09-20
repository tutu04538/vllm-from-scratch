"""Read-only real Qwen3 acceptance + measured Engine timings. Reference work excluded from timings."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1'
import argparse,copy,gc,hashlib,json,statistics,sys,time,importlib
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import torch,transformers
from transformers import AutoTokenizer,Qwen3ForCausalLM
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];RECORDS=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录');sys.path.insert(0,str(RECORDS/'tools'))
from step34_precision_reference import reference as precision_reference
from step34_source import source_digest,source_files
from step34_test_helpers import m
from step34_external_process import mapped
from step34.step34 import encode
MODEL=Path('/home/user/proj/KuiperLLama/Qwen/Qwen3-0.6B')

def summary(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive');med=statistics.median(xs)
    return dict(median_ms=med,iqr_over_median=(q[2]-q[0])/med,samples_ms=xs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,choices=[33,34],required=True);ap.add_argument('--mode',required=True,choices=['torch','eager','graph']);ap.add_argument('--dtype',choices=['float32','bfloat16'],required=True);ap.add_argument('--norm-backend',choices=['torch','triton'],required=True);ap.add_argument('--repeats',type=int,default=5);a=ap.parse_args();dt=getattr(torch,a.dtype)
    global m,source_digest,source_files
    m=importlib.import_module(f'step{a.version}')
    def source_files():
        root=ROOT/f'step{a.version}';return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}
    def source_digest():return hashlib.sha256(''.join(f'{k}\0{v}\n' for k,v in source_files().items()).encode()).hexdigest()
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
    opts=dict(norm_backend=a.norm_backend,device='cuda',attention_backend='torch' if a.mode=='torch' else 'triton',use_cuda_graph=a.mode=='graph',max_num_seqs=3,max_num_batched_tokens=16,num_kv_blocks=64,block_size=16,enable_prefix_caching=True)
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
    norm_calls=[];head_calls=[];errors=[];kv_errors=[];calls=[];precision_errors=[];precision_kv_errors=[];disagreements=[];state=e.model.state_dict();original=e.model._forward_append
    def wrapped(ids,counts,caches,pool,sample_rows=None):
        seqs=[next(s for s in e.scheduler.running if s.cache is c) for c in caches];lengths=[c.length for c in caches]
        histories=[(s.prompt_ids+s.output_ids)[:l+n] for s,l,n in zip(seqs,lengths,counts)]
        out=original(ids,counts,caches,pool,**({'sample_rows':sample_rows} if a.version==34 else {}))
        assert sample_rows is not None if a.version==34 else sample_rows is None
        offset=0
        for s,c,h,l,n in zip(seqs,caches,histories,lengths,counts):
            chosen=list(range(n)) if sample_rows is None else [r-offset for r in sample_rows if offset<=r<offset+n]
            packed=list(range(offset,offset+n)) if sample_rows is None else [i for i,r in enumerate(sample_rows) if offset<=r<offset+n]
            rows=out[packed];offset+=n
            want=hf(torch.tensor([h],device='cuda'),use_cache=True);expected=want.logits[0,l:][chosen]
            
            if dt==torch.float32:torch.testing.assert_close(rows,expected,atol=3e-4,rtol=3e-4)
            if rows.numel():errors.append(float((rows.float()-expected.float()).abs().max()))
            assert torch.isfinite(rows).all()
            if dt==torch.bfloat16:
                exact,exact_kv=precision_reference(state,h,e.model.num_q_heads,e.model.num_kv_heads,e.model.rms_norm_eps,e.model.rope_theta)
                if rows.numel():
                    precision_errors.append(float((rows.float()-exact[l:][chosen].float()).abs().max()))
                    top=rows[-1].float().topk(2);ref_top=expected[-1].float().topk(2)
                    disagreements.append(dict(request=s.request_id,history=len(h),candidate_token=int(top.indices[0]),hf_token=int(ref_top.indices[0]),candidate_gap=float(top.values[0]-top.values[1]),hf_gap=float(ref_top.values[0]-ref_top.values[1]),precision_token=int(exact[l:][chosen][-1].argmax())))
            slots=torch.arange(c.length,device='cuda');blocks=torch.tensor(c.block_table,device='cuda')[slots//pool.block_size];offsets=slots%pool.block_size
            for storage,attr in [(pool.k_cache,'keys'),(pool.v_cache,'values')]:
                got=storage[:,blocks,offsets];ref=torch.stack([getattr(layer,attr)[0].transpose(0,1) for layer in want.past_key_values.layers])
                
                if dt==torch.float32:torch.testing.assert_close(got,ref,atol=3e-4,rtol=3e-4)
                kv_errors.append(float((got.float()-ref.float()).abs().max()))
                if dt==torch.bfloat16:precision_kv_errors.append(float((got.float()-exact_kv[0 if attr=='keys' else 1].float()).abs().max()))
            assert c.length==l+n
            calls.append(dict(request_id=s.request_id,history=l,count=n,total=c.length))
        return out
    def norm_hook(module,inputs,output):norm_calls.append(dict(backend=module.backend,shape=list(inputs[0].shape)))
    def head_hook(module,inputs,output):head_calls.append(dict(input_rows=inputs[0].shape[0],output_shape=list(output.shape),can_sample=sum(bool(item['can_sample']) for item in e.scheduler.scheduled_items)))
    hooks=[mod.register_forward_hook(norm_hook) for mod in e.model.modules() if isinstance(mod,m.RMSNorm)]
    hooks.append(e.model.lm_head.register_forward_hook(head_hook))
    try:
        with torch.inference_mode(),patch.object(e.model,'_forward_append',side_effect=wrapped):verified=generate()
    finally:
        for handle in hooks:handle.remove()
    # Graph replay bypasses Python hooks; model/norm configuration still checked in every mode.
    assert all(mod.backend==a.norm_backend for mod in e.model.modules() if isinstance(mod,m.RMSNorm))
    if a.mode!='graph':assert norm_calls and all(x['backend']==a.norm_backend for x in norm_calls)

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
    workload_records={}
    context='键和值缓存可以复用已经计算的历史信息，避免重复计算。'
    long_prompts=[encode(tokenizer,'请用一句话概括以下内容：'+context*7+str(i)) for i in range(3)]
    workloads={
        'prefill_heavy':[dict(request_id=f'p{i}',prompt_ids=p,max_new_tokens=1) for i,p in enumerate(long_prompts)],
        'decode_heavy':[dict(request_id=f'd{i}',prompt_ids=p,max_new_tokens=32) for i,p in enumerate(prompts)],
    }
    for label,requests in workloads.items():
        assert sum((len(r['prompt_ids'])+r['max_new_tokens']-1+15)//16 for r in requests)<=64
        with torch.inference_mode():
            cold=generate(requests)
            for _ in range(2):assert generate(requests)['results']==cold['results']
            torch.cuda.reset_peak_memory_stats();extra_samples=[]
            for _ in range(a.repeats):
                val=generate(requests);assert val['results']==cold['results'];extra_samples.append(val)
        workload_records[label]=dict(prompt_lengths=[len(r['prompt_ids']) for r in requests],requests=requests,results=cold['results'],output_tokens=sum(map(len,cold['results'].values())),first=cold,steady=summary([v['elapsed_ms'] for v in extra_samples]),ttft=summary([statistics.mean(ts[0] for ts in v['times_ms'].values()) for v in extra_samples]),itl=None if label=='prefill_heavy' else summary([statistics.mean(t1-t0 for ts in v['times_ms'].values() for t0,t1 in zip(ts,ts[1:])) for v in extra_samples]),peak_allocated=torch.cuda.max_memory_allocated(),samples=extra_samples)
    assert file_stats=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in MODEL.iterdir() if p.is_file()} and sha==source_digest()
    itl=[[b-a for ts in s['times_ms'].values() for a,b in zip(ts,ts[1:])] for s in samples]
    ttft=[statistics.mean(ts[0] for ts in s['times_ms'].values()) for s in samples]
    data=dict(workloads=workload_records,version=a.version,norm_backend=a.norm_backend,norm_calls=norm_calls,lm_head_calls=head_calls,measurement_revision=3,reference_outputs_deleted=True,status='measured' if dt==torch.bfloat16 else 'passed',mode=a.mode,source_sha256=sha,source_files=source_files(),torch=torch.__version__,transformers=transformers.__version__,device=torch.cuda.get_device_name(),dtype=a.dtype,memory=memory,hf_results=hf_expected,hf_comparisons=disagreements,precision_logits_max_abs_error=max(precision_errors,default=0),precision_kv_max_abs_error=max(precision_kv_errors,default=0),tf32=False,threads=1,model_dir=str(MODEL),options=opts,questions=questions,prompt_ids=prompts,results=expected,texts={k:tokenizer.decode(v,skip_special_tokens=True) for k,v in expected.items()},tokenizer_load_ms=tokenizer_load_ms,encoding_three_prompts=summary(encoding),model_load_ms=load_ms,first_workload=first,logits_max_abs_error=max(errors),kv_max_abs_error=max(kv_errors),verified_calls=calls,long_eos_result=long_result,steady=summary([s['elapsed_ms'] for s in samples]),ttft_request_mean=summary(ttft),itl_request_mean=summary([statistics.mean(v) for v in itl]),samples=samples,scope='One fresh process per backend. Load excludes tokenizer and pre-initialized CUDA context. First full workload includes first GPU library work/JIT/graph capture; on-disk caches retained. BF16 official comparisons are diagnostic (different rounding policy); BF16 precision-aware dense oracle metrics are recorded for review, not silently accepted with FP32 tolerance. Memory is PyTorch allocated/reserved, not nvidia-smi, reference excluded from first/steady peaks. Reference and full parameter/KV checks are outside timing. Warm benchmark has prefix caching OFF,2 warmups,5 complete workloads by default,3 requests max8 output tokens each. TTFT is request submission to first sampled token (not text streaming), reported mean over requests per run. ITL includes scheduling and other requests/prefill, not pure kernel time. Encoding measured separately; timing not a claim of stable cross-backend speedup. Source files never edited.')
    p=ROOT/'benchmarks/results'/f"step34_compare_v{a.version}_{a.norm_backend}_{a.dtype}_{a.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json";p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');print(p);print(a.mode,'load_ms',load_ms,'steady_ms',data['steady']['median_ms'],'maxerror',max(errors))
if __name__=='__main__':main()
