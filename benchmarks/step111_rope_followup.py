"""Diagnose the first-layer KV difference at an already recorded logical position."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/step111/kv_followup_v3'
OUT.mkdir(parents=True,exist_ok=True)
case_id='8b_tp4_b16_n1024_k0_eager'
rows=json.loads((ROOT/'benchmarks/results/step111/full_matrix.json').read_text())['results']
case=next(row['case'] for row in rows if row['case_id']==case_id)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3,0,1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
         VLLM_USE_V2_MODEL_RUNNER='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
results=[]
for implementation in ('frozen','mini'):
    output=OUT/f'{implementation}.json'
    command=[sys.executable,str(ROOT/'benchmarks/step111_rope_diag_probe.py'),
             '--case',json.dumps(case),'--implementation',implementation,'--phase','gate','--repeats','1',
             '--diag-request','12','--diag-positions','850','--output',str(output)]
    print(json.dumps({'implementation':implementation,'status':'running'}),flush=True)
    with (OUT/f'{implementation}.log').open('w') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    results.append({'implementation':implementation,'command':command,'exit_code':result.returncode})
    print(json.dumps(results[-1]),flush=True)
    if result.returncode: break
if len(results)==2 and all(x['exit_code']==0 for x in results):
    command=[sys.executable,str(ROOT/'benchmarks/step111_mixed_diag_compare.py'),
             '--reference',str(OUT/'frozen_layers'),'--mini',str(OUT/'mini_layers'),
             '--position','850','--tp','4','--output',str(OUT/'comparison.json')]
    result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
    results.append({'command':command,'exit_code':result.returncode,'output':result.stdout,'stderr':result.stderr})
    print(json.dumps(results[-1]),flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(int(any(item['exit_code'] for item in results) or len(results)!=3))
