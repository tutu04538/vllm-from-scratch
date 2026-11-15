"""在修复后的源码上重跑 111 失败配置或完整性能矩阵。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.check_step112_acceptance import selected_gpu_processes
OUT=ROOT/'benchmarks/results/step112/rope_validation'
CAL=ROOT/'benchmarks/results/step111/calibration_full_v3.json'
DRIVER=ROOT/'benchmarks/compare_step111_parallel.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('failed-cases','matrix'),required=True)
    parser.add_argument('--timeout',type=int,default=1800,help='单个子进程时限，完整矩阵自动放宽至 15000 秒')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='从同配置完整矩阵续跑并保留之前的结果')
    args=parser.parse_args()
    if args.timeout<=0:parser.error('timeout must be positive')
    if args.resume and args.case != 'matrix': parser.error('--resume 只支持完整矩阵')
    if args.output.exists() and not args.resume: parser.error('结果文件已存在，请选择新路径')
    run_dir=OUT/args.output.stem
    run_dir.mkdir(parents=True,exist_ok=True)
    if args.resume and args.output.exists():
        prior=run_dir/f'previous_summary_{time.time_ns()}.json'
        with prior.open('xb') as stream: stream.write(args.output.read_bytes())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3,0,1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VLLM_USE_V2_MODEL_RUNNER='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    selection=[('8b',4),('14b',2),('14b',4)] if args.case=='failed-cases' else [(None,None)]
    rows=[]
    for model,tp in selection:
        name=f'{model}_tp{tp}_mixed16' if model else 'full_matrix'
        dest=run_dir/f'{name}.json';log=run_dir/f'{name}.log'
        if dest.exists() and not args.resume: parser.error(f'已有矩阵结果：{dest}')
        if args.resume and not dest.is_file(): parser.error(f'续跑结果不存在：{dest}')
        cmd=[sys.executable,str(DRIVER),'--calibration',str(CAL),'--repeats','3',
             '--timeout',str(args.timeout),'--output',str(dest)]
        if model:cmd.extend(['--model',model,'--tp',str(tp),'--batch','16',
                             '--group','B','--k','0','--mode','eager'])
        if args.resume: cmd.append('--resume')
        print(json.dumps({'case':name,'status':'running'}),flush=True)
        start=time.monotonic()
        selected = ','.join(env['CUDA_VISIBLE_DEVICES'].split(',')[:tp or 4])
        try:
            busy=selected_gpu_processes(selected)
            error=None
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            busy,error=[],str(exc)
        if busy or error:
            row={'case':name,'command':cmd,'exit_code':125,'status':'unrun',
                 'occupied_gpu_pids':busy,'occupancy_error':error,'result_path':str(dest)}
            rows.append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
            continue
        with log.open('a' if args.resume else 'w') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=args.timeout if model else max(args.timeout,15000))
            except subprocess.TimeoutExpired:
                try:
                    parent=psutil.Process(proc.pid)
                    children=parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    children=[]
                for child in reversed(children):
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                try: proc.terminate()
                except ProcessLookupError: pass
                _,alive=psutil.wait_procs(children,timeout=10)
                for child in alive:
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill();proc.wait()
                code=124
        result=json.loads(dest.read_text()) if dest.is_file() else None
        row={'case':name,'command':cmd,'exit_code':code,'elapsed_seconds':time.monotonic()-start,
             'log':str(log),'result_path':str(dest),'status':result.get('status') if result else 'missing',
             'case_counts':result.get('case_counts') if result else None}
        rows.append(row);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
    return int(any(x['exit_code'] for x in rows))

if __name__=='__main__':raise SystemExit(main())
