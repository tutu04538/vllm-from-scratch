"""Register long-input BF16 thresholds from repeated, cross-TP frozen runs only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.benchmark_runner import _atomic_write_json
from benchmarks.step111_matrix import build_cases, compute_pids
from benchmarks.step109_compare import measure
from benchmarks.step111_compare_gate import records


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def first_differences(left, right):
    if left.keys() != right.keys(): raise ValueError('request ID set differs')
    result = []
    for req in sorted(left):
        if len(left[req]) != len(right[req]): raise ValueError(f'{req}: output length differs')
        index = next((i for i,(a,b) in enumerate(zip(left[req],right[req])) if a != b), None)
        if index is not None:
            result.append({'request':req,'position':index,'reference':left[req][index], 'other':right[req][index]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('8b','14b','all'), default='all')
    p.add_argument('--batch', type=int, choices=(1,16))
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.timeout <= 0: p.error('timeout must be positive')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    folder = a.output.parent/(a.output.stem+'_runs'); folder.mkdir(exist_ok=True)
    groups = [(model,batch) for model in ('8b','14b') for batch in (1,16)
              if (a.model=='all' or a.model==model) and (a.batch is None or batch==a.batch)]
    payload = {'status':'running','groups':groups,'results':{},'executions':[],
               'source_sha256':{name:file_sha(ROOT/'benchmarks'/name) for name in
                    ('register_step111_reference.py','step111_matrix.py','step111_parallel_worker.py','step103_hooks.py',
                     'step111_compare_gate.py','step109_compare.py')},
               'prompt_fixture':'step111-v2','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    _atomic_write_json(a.output,payload)
    try:
        for model,batch in groups:
            tps = (1,2,4) if model=='8b' else (2,4)
            cases = [next(c for c in build_cases() if c['model']==model and c['tp']==tp and
                          c['batch']==batch and c['k']==0 and c['mode']=='eager' and
                          (batch==16 or c['lengths']==[1024])) for tp in tps]
            group_name = f'{model}_b{batch}'
            data = {}
            for case in cases:
                tp = case['tp']
                for repeat in (0,1):
                    name=f'{group_name}_tp{tp}_r{repeat}'; path=folder/f'{name}.json'; log=folder/f'{name}.log'
                    before=compute_pids(tp)
                    if before: raise RuntimeError(f'{name}: selected GPUs occupied by {sorted(before)}')
                    path.unlink(missing_ok=True)
                    command=[sys.executable,str(ROOT/'benchmarks/step111_parallel_worker.py'), '--case',json.dumps(case),
                             '--implementation','frozen','--phase','gate','--repeats','3','--output',str(path)]
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VLLM_USE_V2_MODEL_RUNNER='1',
                             VLLM_ENABLE_V1_MULTIPROCESSING='0')
                    started=time.monotonic()
                    with log.open('w') as stream:
                        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        try: code=proc.wait(timeout=a.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try: proc.wait(timeout=10)
                            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                            code=124
                    after=compute_pids(tp); code=125 if after-before else code
                    execution={'name':name,'command':command,'exit_code':code,'seconds':time.monotonic()-started,'log':str(log),'result':str(path),'new_gpu_pids_after_exit':sorted(after-before)}
                    payload['executions'].append(execution); _atomic_write_json(a.output,payload); print(json.dumps(execution),flush=True)
                    if code or not path.is_file(): raise RuntimeError(f'{name} failed; see {log}')
                    data[(tp,repeat)]=(path,json.loads(path.read_text()))
            first_path, first = data[(tps[0],0)]
            expected={f'{req}/{step}' for req in range(batch) for step in range(4)}
            base_records=[records(first_path.with_suffix(''), rank) for rank in range(tps[0])]
            if any(set(r)!=expected for r in base_records): raise ValueError(f'{group_name}: incomplete base records')
            base_tokens=first['samples'][0]['tokens']; prompts=first['prompt_ids']
            thresholds={}; comparisons=[]; hashes={}; history_divergences=[]
            for (tp,repeat),(path,value) in data.items():
                if value['prompt_ids']!=prompts:
                    raise ValueError(f'{group_name} TP{tp} repeat{repeat}: input differs')
                local_baseline = data[(tp,0)][1]['samples'][0]['tokens']
                if first_differences(local_baseline, value['samples'][0]['tokens']):
                    raise ValueError(f'{group_name} TP{tp}: same-TP frozen repeats differ')
                divergences = first_differences(base_tokens, value['samples'][0]['tokens'])
                if any(item['position'] < 4 for item in divergences):
                    raise ValueError(f'{group_name} TP{tp}: history diverges within captured window; fixed-history replay required')
                if divergences:
                    history_divergences.append({'tp':tp,'repeat':repeat,'first_differences':divergences})
                ranks=[records(path.with_suffix(''),rank) for rank in range(tp)]
                if any(set(r)!=expected for r in ranks): raise ValueError(f'{group_name} TP{tp}: incomplete records')
                hashes[str(path)]=file_sha(path)
                for rank in range(tp):
                    rank_path=path.with_suffix('')/f'run0_rank{rank}.pt'; hashes[str(rank_path)]=file_sha(rank_path)
                for key in sorted(expected):
                    for rank in range(tp):
                        if any(ranks[rank][key][field] != base_records[0][key][field] for field in ('position','sequence_length','sampled')):
                            raise ValueError(f'{group_name} TP{tp} rank{rank} {key}: discrete state differs')
                    for field in ('hidden','logits','kv'):
                        left=(torch.cat([r[key][field] for r in base_records],dim=2) if field=='kv' else base_records[0][key][field])
                        right=(torch.cat([r[key][field] for r in ranks],dim=2) if field=='kv' else ranks[0][key][field])
                        metric=measure(left,right); label=f'{key}/{field}'
                        limit=thresholds.setdefault(label,{'max_abs':1e-5,'relative_l2':1e-5})
                        limit['max_abs']=max(limit['max_abs'],2*metric['max_abs'])
                        limit['relative_l2']=max(limit['relative_l2'],2*metric['relative_l2'])
                        comparisons.append({'tp':tp,'repeat':repeat,'label':label,**metric})
            payload['results'][group_name]={'status':'registered','model':model,'batch':batch,'tps':list(tps),
                'prompt_ids':prompts,'thresholds':thresholds,'comparisons':comparisons,'reference_sha256':hashes,
                'cross_tp_later_history_divergences':history_divergences,
                'registration_basis':'twice per TP; identical captured histories; 2x maximum frozen cross-TP/repeat errors, floor 1e-5'}
            _atomic_write_json(a.output,payload); print(json.dumps({'group':group_name,'status':'registered'}),flush=True)
        payload['status']='registered';payload['registered_at_unix']=time.time()
    except Exception as exc:
        payload['status']='failed';payload['error']=f'{type(exc).__name__}: {exc}';print(payload['error'],file=sys.stderr)
    _atomic_write_json(a.output,payload)
    return int(payload['status']!='registered')

if __name__=='__main__': raise SystemExit(main())
