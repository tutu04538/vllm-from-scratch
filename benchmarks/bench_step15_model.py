"""第十五关：同权重批量连续 decode，扩容缓存与固定缓冲区对照。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from torch.utils.benchmark import Timer
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import step14
import step15
FIXTURE=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/step15_model_contract_20260915T132208.587713Z.json')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def weight_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():h.update(n.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--order',choices=['old-first','new-first'],default='old-first')
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(0)
    sources={n:{'path':str(PROJECT/f'{n}.py'),'sha256':sha(PROJECT/f'{n}.py')} for n in ('step14','step15')}
    fixture=json.loads(FIXTURE.read_text());assert fixture['status']=='passed' and fixture['source_sha256']==sources['step15']['sha256']
    old=step14.TinyCausalLM().eval();new=step15.TinyCausalLM().eval();new.load_state_dict(old.state_dict(),strict=True)
    wh=weight_hash(old);assert weight_hash(new)==wh
    lengths={1:[24],2:[3,1],3:[24,3,1]}[args.batch_size];b=len(lengths)
    prompts=[[(j+i)%5 for j in range(p)] for i,p in enumerate(lengths)]
    padded=torch.tensor([p+[0]*(max(lengths)-len(p)) for p in prompts])
    inputs=[torch.tensor([[(i+j+1)%5] for i in range(b)]) for j in range(4)]
    inputs_before=[x.clone() for x in inputs]
    started=datetime.now(timezone.utc).isoformat()
    with torch.inference_mode():
        initial_logits,initial_old=old.forward_prefill(padded,lengths)
        caches=[{'k':torch.empty(32,8),'v':torch.empty(32,8),'length':0} for _ in lengths]
        initial_new=new.forward_prefill(padded,lengths,caches)
        for i,p in enumerate(lengths):torch.testing.assert_close(initial_logits[i,:p],initial_new[i,:p],atol=1e-5,rtol=1e-5)
        prefixes=[{k:c[k][:p].clone() for k in ('k','v')} for c,p in zip(caches,lengths)]
        tensors=[(c,c['k'],c['v'],c['k'].data_ptr(),c['v'].data_ptr()) for c in caches]
        old_snapshots=[tuple(t.clone() for t in kv) for kv in initial_old]
        def run_old():
            past=initial_old
            for ids in inputs:logits,past=old.forward_decode(ids,past)
            return logits,past
        def run_new():
            # 每个 trial 恢复同一有效前缀。上轮写入的尾部不会被读取，后续会被覆盖。
            # 这几个 Python length 赋值明确计入新版本耗时，不复制/清零整块缓冲区。
            for c,p in zip(caches,lengths):c['length']=p
            for ids in inputs:logits=new.forward_decode(ids,caches)
            return logits,caches
        def check():
            expected,old_final=run_old();actual,new_final=run_new()
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            for i,(c,kv,p) in enumerate(zip(new_final,old_final,lengths)):
                obj,k,v,kptr,vptr=tensors[i]
                assert c is obj and c['k'] is k and c['v'] is v and k.data_ptr()==kptr and v.data_ptr()==vptr
                assert c['length']==p+4
                for key,want in zip(('k','v'),kv):
                    torch.testing.assert_close(c[key][:p+4],want,atol=1e-5,rtol=1e-5)
                    assert torch.equal(c[key][:p],prefixes[i][key])
            assert all(torch.equal(x,y) for kv,oldkv in zip(initial_old,old_snapshots) for x,y in zip(kv,oldkv))
            assert all(torch.equal(a,b) for a,b in zip(inputs,inputs_before))
            assert weight_hash(old)==weight_hash(new)==wh
            return {'max_final_logits_abs_error':(actual-expected).abs().max().item(),
                'final_lengths':[c['length'] for c in caches],'storage_identity_and_prefix_stable':True,'initial_old_cache_unchanged':True}
        traces={};handles=[]
        for mode,model in (('old',old),('new',new)):
            def hook(layer,args,mode=mode):traces.setdefault(mode,[]).append(list(args[0].shape))
            handles.append(model.k_proj.register_forward_pre_hook(hook))
        try:before=check()
        finally:
            for h in handles:h.remove()
        assert all(traces[mode]==[[b,1,8]]*4 for mode in ('old','new'))
        order=['old','new'] if args.order=='old-first' else ['new','old'];measurements=[]
        for mode in order:
            m=Timer(stmt='fn()',globals={'fn':run_old if mode=='old' else run_new},num_threads=1,
                label=f'step15 {mode} 4-step decode',description=f'lengths={lengths}, CPU').blocked_autorange(min_run_time=1.0)
            measurements.append({'mode':mode,'median_us':m.median*1e6,'iqr_us':m.iqr*1e6,'has_warnings':m.has_warnings,
                'number_per_run':m.number_per_run,'raw_times_s':m.raw_times});print(m)
        after=check();assert before==after
    assert all(sha(Path(s['path']))==s['sha256'] for s in sources.values())
    report={'status':'measured','started_at_utc':started,'pid':os.getpid(),'sources':sources,'benchmark_sha256':sha(Path(__file__)),
        'acceptance':str(FIXTURE),'acceptance_sha256':sha(FIXTURE),'python':sys.version,'python_executable':sys.executable,
        'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','dtype':'float32','num_threads':1,'seed':0,
        'parameter_sha256':wh,'model_config':{'vocab_size':5,'d_model':8,'max_seq_len':32},'batch_size':b,'prompt_lengths':lengths,
        'prompts':prompts,'new_tokens':[x.tolist() for x in inputs],'decode_steps':4,
        'boundary':'four batched decode calls, fixed inputs and same prefill prefix; initialized eval models in existing inference_mode; includes all cache append, scratch batching/padding, attention and output computation; new version includes resetting dictionary lengths at each trial, old version starts from immutable original caches; excludes initial prefill/cache preallocation/model initialization/Engine/sampling/checks; no copy or zero-fill of new buffers between trials',
        'before_check':before,'after_check':after,'k_projection_shapes':traces,'measurement_order':order,'min_run_time_s':1.0,'measurements':measurements}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    p=PROJECT/'benchmarks/results'/f'step15_model_B{b}_{stamp}_{os.getpid()}.json'
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Saved:',p)


if __name__=='__main__':main()
