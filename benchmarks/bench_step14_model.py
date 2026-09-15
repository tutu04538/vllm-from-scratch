"""第十四关：仅测单轮 prefill，初始化/输入补齐/Engine/decode 不计时。"""
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
import step13
import step14
FIXTURE=Path('/home/user/proj/vllm-omni/learning_notes/14_vllm_from_scratch/验收记录/step14_model_contract_20260915T111116.735757Z.json')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def weight_hash(model):
    h=hashlib.sha256()
    for n,p in model.state_dict().items():
        h.update(n.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--order',choices=['serial-first','batch-first'],default='serial-first')
    args=parser.parse_args()
    sources={n:{'path':str(PROJECT/f'{n}.py'),'sha256':sha(PROJECT/f'{n}.py')} for n in ('step13','step14')}
    fixture=json.loads(FIXTURE.read_text())
    assert fixture['status']=='passed' and fixture['source_sha256']==sources['step14']['sha256']
    torch.set_num_threads(1);torch.manual_seed(0)
    old=step13.TinyCausalLM().eval();new=step14.TinyCausalLM().eval()
    new.load_state_dict(old.state_dict(),strict=True)
    wh=weight_hash(old);assert weight_hash(new)==wh
    lengths={1:[3],2:[3,1],3:[24,3,1]}[args.batch_size]
    prompts=[((torch.arange(p)+i)%5).reshape(1,p) for i,p in enumerate(lengths)]
    ids=torch.zeros(len(lengths),max(lengths),dtype=torch.long)
    for i,p in enumerate(prompts):ids[i,:p.shape[1]]=p[0]
    ids_before=ids.clone();prompts_before=[p.clone() for p in prompts];lengths_before=list(lengths)
    started=datetime.now(timezone.utc).isoformat()
    with torch.inference_mode():
        def serial():return [old(p) for p in prompts]
        def batch():return new.forward_prefill(ids,lengths)
        def check():
            expected=serial();logits,caches=batch();errors=[]
            assert logits.shape==(len(lengths),max(lengths),5)
            for i,(ref,ref_caches) in enumerate(expected):
                actual=logits[i:i+1,:lengths[i]]
                torch.testing.assert_close(actual,ref,atol=1e-5,rtol=1e-5)
                errors.append((actual-ref).abs().max().item())
                for got,want in zip(caches[i],ref_caches[0]):
                    assert got.shape==(lengths[i],8) and not got.requires_grad
                    torch.testing.assert_close(got,want,atol=1e-5,rtol=1e-5)
            assert torch.equal(ids,ids_before) and lengths==lengths_before
            assert all(torch.equal(a,b) for a,b in zip(prompts,prompts_before))
            assert weight_hash(old)==weight_hash(new)==wh
            return {'max_logits_abs_errors':errors,'cache_lengths':[kv[0].shape[0] for kv in caches],
                    'inputs_parameters_unchanged':True}
        traces={};handles=[]
        for mode,model in (('serial',old),('batch',new)):
            def hook(layer,inputs,mode=mode):traces.setdefault(mode,[]).append(list(inputs[0].shape))
            handles.append(model.k_proj.register_forward_pre_hook(hook))
        try:before=check()
        finally:
            for h in handles:h.remove()
        assert traces['serial']==[[1,p,8] for p in lengths]
        assert traces['batch']==[[len(lengths),max(lengths),8]]
        order=['serial','batch'] if args.order=='serial-first' else ['batch','serial']
        measurements=[]
        for mode in order:
            m=Timer(stmt='fn()',globals={'fn':serial if mode=='serial' else batch},num_threads=1,
                    label=f'step14 {mode} prefill',description=f'lengths={lengths}, CPU').blocked_autorange(min_run_time=1.0)
            measurements.append({'mode':mode,'median_us':m.median*1e6,'iqr_us':m.iqr*1e6,'has_warnings':m.has_warnings,
                                 'number_per_run':m.number_per_run,'raw_times_s':m.raw_times})
            print(m)
        after=check();assert before==after
    assert all(sha(Path(v['path']))==v['sha256'] for v in sources.values())
    report={'status':'measured','started_at_utc':started,'pid':os.getpid(),'sources':sources,
        'benchmark_sha256':sha(Path(__file__)),'acceptance':str(FIXTURE),'acceptance_sha256':sha(FIXTURE),
        'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'platform':platform.platform(),
        'device':'cpu','dtype':'float32','num_threads':1,'seed':0,'parameter_sha256':wh,
        'model_config':{'vocab_size':5,'d_model':8,'max_seq_len':32},'batch_size':len(lengths),'prompt_lengths':lengths,
        'input_ids':ids.tolist(),'prompts':[p.tolist() for p in prompts],
        'boundary':'one prefill for all prompts; preinitialized eval models in existing inference_mode; input tensors/right-padding/lengths prebuilt; includes embedding, projections, causal attention, lm_head and per-request KV construction/slicing (serial Python loop included); excludes imports, initialization, Engine, sampling, subsequent decode and correctness hooks/checks; padded query outputs computed but not compared; returned slices need not own compact storage',
        'min_run_time_s':1.0,'measurement_order':order,'before_check':before,'after_check':after,
        'k_projection_shapes':traces,'projected_input_positions':{'serial':sum(lengths),'batch':len(lengths)*max(lengths)},
        'measurements':measurements}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path=PROJECT/'benchmarks/results'/f'step14_model_B{len(lengths)}_{stamp}_{os.getpid()}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Saved:',path)


if __name__=='__main__':main()
