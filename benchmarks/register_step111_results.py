"""登记 111 的真实矩阵、每 rank 资源与回归；失败不会写成通过。"""
import argparse
from collections import Counter
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--regressions',type=Path,required=True)
    parser.add_argument('--followup',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/results.json')
    args=parser.parse_args()
    old=json.loads(args.output.read_text());matrix=json.loads(args.matrix.read_text())
    regressions=json.loads(args.regressions.read_text());followup=json.loads(args.followup.read_text())
    counts=dict(Counter(row['status'] for row in matrix['results']))
    good={'passed_performance','observed_near_tie'}
    bad=[row['case_id'] for row in matrix['results'] if row['status'] not in good]
    incomplete=[row['group'] for row in regressions if row['exit_code'] or not row['summary']]
    incomplete.extend(sorted({'step111','legacy','parallel'} - {row['group'] for row in regressions}))
    failed_followup=[row['name'] for row in followup if row['exit_code']]
    if len(followup) != 18: failed_followup.append('incomplete followup jobs')
    if len(matrix['cases']) != 50 or len(matrix['results']) != 50:
        incomplete.append('incomplete full matrix')
    trace_path=ROOT/'benchmarks/results/step111/followup_v2/trace_reanalysis_v2.json'
    trace_records=json.loads(trace_path.read_text()) if trace_path.is_file() else []
    if len(trace_records) != 8 or any(item['exit_code'] for item in trace_records):
        failed_followup.append('incomplete trace reanalysis')
    status=('passed: completed performance matrix and trace' if not bad and not incomplete and not failed_followup
            else 'partial: matrix contains failed or unrun cases, or evidence incomplete')
    cases=[]
    for row in matrix['results']:
        item={'case_id':row['case_id'],'case':row['case'],'status':row['status'],
              'strict_equivalent':row.get('strict_equivalent'), 'state_gate':row.get('state_gate'),
              'first_output_divergence':row.get('first_output_divergence')}
        if row['status'] in good:
            item['engines']={}
            for name in ('mini','frozen'):
                result=row['engines'][name]['result']
                item['engines'][name]={'summary':row['engines'][name]['summary'],
                    'rank_before':result['rank_before'], 'rank_after':result['rank_after'],
                    'build_seconds':result['build_seconds'],
                    'capture_seconds_by_rank':result['capture_seconds_by_rank'],
                    'executor_init_including_load_seconds':result['executor_init_including_load_seconds']}
        cases.append(item)
    sources=ROOT/'learning_notes/14_vllm_from_scratch/分布式并行完整需求/参考资料/源码清单.json'
    calibration=Path(matrix['calibration']['path'])
    traces=ROOT/'benchmarks/results/step111/followup_v2/trace_reanalysis_v2.json'
    old['step111']={'results':{'status':status,'package_sha256':matrix['package']['sha256'],
      'source_manifest_sha256':hashlib.sha256(sources.read_bytes()).hexdigest(),
      'baseline_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'environment':old['step110']['results']['environment'],
      'cuda_visible_devices':matrix['cuda_visible_devices'],'planned_cases':len(matrix['cases']),
      'case_counts':counts,'failed_or_unrun':bad,'incomplete':incomplete,'cases':cases,'regressions':regressions,
      'failed_followup':failed_followup,'followup':followup,'trace_reanalysis':str(traces),
      'calibration':{'path':str(calibration),'sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),
                     'prompt_fixture':'step111-v2'},
      'raw_matrix':str(args.matrix),'matrix_sha256':hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
      'limitations':['near-tie cases are not strict token equality',
                     'GPU kernel intervals are inclusive and cannot be added as wall time',
                     '111 numerical failures precede the 112 RoPE fix']},
      'models':old['step110']['models']}
    old['_各关状态']['step111']=status
    args.output.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':status,'case_counts':counts,'failed':bad}))
    return int(bool(incomplete or failed_followup))

if __name__=='__main__':raise SystemExit(main())
