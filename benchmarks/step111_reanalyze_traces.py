"""从已保存的 trace 重建 GEMM/GEMV 分类，保留首次分析。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/results/step111/followup_v2'
reports=sorted(OUT.glob('trace_*_eager.json')) + [OUT/'trace_14b_tp2_k2_graph.json',
        OUT/'numa_same_numa_trace.json',OUT/'numa_cross_numa_trace.json']
results=[]
for report in reports:
    trace=json.loads(report.read_text())
    worker=Path(trace['worker_result']);case=json.loads(worker.read_text())['case']
    target=report.with_name(report.stem+'_analysis_v2.json')
    command=[sys.executable,str(ROOT/'benchmarks/step111_analyze_trace.py'),
             '--sqlite',trace['sqlite'],'--tp',str(case['tp']),'--worker-json',str(worker),'--output',str(target)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=1800)
    item={'report':str(report),'command':command,'exit_code':result.returncode,
          'stdout':result.stdout,'stderr':result.stderr,'analysis':str(target),
          'analyzer_sha256':hashlib.sha256((ROOT/'benchmarks/step111_analyze_trace.py').read_bytes()).hexdigest()}
    results.append(item);print(json.dumps({'report':report.name,'exit_code':result.returncode}),flush=True)
(OUT/'trace_reanalysis_v2.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(int(any(x['exit_code'] for x in results)))
