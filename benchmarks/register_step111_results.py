"""登记 111 的真实矩阵、每 rank 资源与回归；失败不会写成通过。"""
import argparse
from collections import Counter
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


REQUIRED_FOLLOWUP = {
    'trace_8b_tp1_k0_eager', 'trace_8b_tp2_k0_eager', 'trace_8b_tp4_k0_eager',
    'trace_14b_tp2_k0_eager', 'trace_14b_tp4_k0_eager', 'trace_14b_tp2_k2_graph',
    'numa_same_numa_gate', 'numa_same_numa_trace',
    'numa_cross_numa_gate', 'numa_cross_numa_trace',
}


def validate_trace_source(report_path, package_sha256):
    """trace 与其门禁须来自相同包，并验证门禁文件未被覆盖。"""
    try:
        report = json.loads(Path(report_path).read_text())
        gate_path = Path(report['gate_report'])
        gate_bytes = gate_path.read_bytes()
        if report.get('status') != 'captured': return False
        if hashlib.sha256(gate_bytes).hexdigest() != report.get('gate_report_sha256'):
            return False
        if json.loads(gate_bytes).get('package', {}).get('sha256') != package_sha256:
            return False
        analyzer = hashlib.sha256((ROOT/'benchmarks/step111_analyze_trace.py').read_bytes()).hexdigest()
        if report.get('harness_sha256', {}).get('step111_analyze_trace.py') == analyzer:
            analysis = Path(report['analysis'])
        else:
            reanalysis = Path(report_path).parent/'trace_reanalysis_v2.json'
            entries = json.loads(reanalysis.read_text())
            matches = [entry for entry in entries
                       if Path(entry['report']).resolve() == Path(report_path).resolve()
                       and entry.get('exit_code') == 0 and entry.get('analyzer_sha256') == analyzer]
            if len(matches) != 1: return False
            analysis = Path(matches[0]['analysis'])
        return json.loads(analysis.read_text()).get('status') == 'passed'
    except (OSError, KeyError, ValueError, TypeError):
        return False


def audit_followup(followup, package_sha256):
    failures = [row['name'] for row in followup if row['exit_code']]
    by_name = {row['name']: row for row in followup}
    failures.extend(sorted(REQUIRED_FOLLOWUP - by_name.keys()))
    for name in sorted(REQUIRED_FOLLOWUP):
        if 'trace' not in name or name not in by_name or by_name[name]['exit_code']:
            continue
        report_path = Path(by_name[name]['log']).with_suffix('.json')
        if not validate_trace_source(report_path, package_sha256):
            failures.append(f'{name}: trace and matrix source do not match')
    return failures


def normalize_regressions(raw, package_sha256, legacy_package_sha256=None):
    """接纳本关旧格式或带包哈希的最终收口回归；明确缺失项。"""
    if isinstance(raw, list):
        required = {'step111', 'legacy', 'parallel'}
        rows = raw
        errors = []
        if package_sha256 != legacy_package_sha256:
            errors.append('legacy regressions do not match matrix source')
    else:
        names = {'pytest': 'step112', 'legacy_final': 'legacy', 'parallel_final': 'parallel'}
        rows = [dict(row, group=names.get(row.get('name'), row.get('name')))
                for row in raw.get('runs', [])]
        required = {'step112', 'legacy', 'parallel'}
        errors = []
        if (raw.get('status') != 'passed_regressions'
                or raw.get('package_sha256') != package_sha256
                or raw.get('package_sha256_end') != package_sha256):
            errors.append('regression source or status does not match matrix')
        if not any(row.get('group') == 'parallel' and
                   'tests/step111' in row.get('command', []) for row in rows):
            errors.append('parallel regressions did not include tests/step111')
    errors.extend(row['group'] for row in rows if row.get('exit_code') or not row.get('summary'))
    errors.extend(sorted(required - {row.get('group') for row in rows}))
    return rows, errors


def retain_history(previous, matrix_sha256):
    history = list(previous.get('history', []))
    result = previous.get('results')
    if result and result.get('matrix_sha256') != matrix_sha256:
        history.append(result)
    return history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--regressions',type=Path,required=True)
    parser.add_argument('--followup',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/results.json')
    args=parser.parse_args()
    old=json.loads(args.output.read_text());matrix=json.loads(args.matrix.read_text())
    regression_raw=json.loads(args.regressions.read_text());followup=json.loads(args.followup.read_text())
    previous=old.get('step111', {})
    first=next(iter(previous.get('history', [])), previous.get('results', {}))
    regressions,incomplete=normalize_regressions(regression_raw, matrix['package']['sha256'],
                                                 first.get('package_sha256'))
    counts=dict(Counter(row['status'] for row in matrix['results']))
    good={'passed_performance','observed_near_tie'}
    bad=[row['case_id'] for row in matrix['results'] if row['status'] not in good]
    failed_followup=audit_followup(followup, matrix['package']['sha256'])
    if len(matrix['cases']) != 50 or len(matrix['results']) != 50:
        incomplete.append('incomplete full matrix')
    trace_path=args.followup.parent/'trace_reanalysis_v2.json'
    if trace_path.is_file():
        trace_records=json.loads(trace_path.read_text())
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
    traces=trace_path if trace_path.is_file() else None
    matrix_sha256=hashlib.sha256(args.matrix.read_bytes()).hexdigest()
    history=retain_history(old.get('step111', {}), matrix_sha256)
    old['step111']={'results':{'status':status,'package_sha256':matrix['package']['sha256'],
      'source_manifest_sha256':hashlib.sha256(sources.read_bytes()).hexdigest(),
      'baseline_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'environment':old['step110']['results']['environment'],
      'cuda_visible_devices':matrix['cuda_visible_devices'],'planned_cases':len(matrix['cases']),
      'case_counts':counts,'failed_or_unrun':bad,'incomplete':incomplete,'cases':cases,'regressions':regressions,
      'failed_followup':failed_followup,'followup':followup,'trace_reanalysis':str(traces) if traces else None,
      'calibration':{'path':str(calibration),'sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),
                     'prompt_fixture':'step111-v2'},
      'raw_matrix':str(args.matrix),'matrix_sha256':matrix_sha256,
      'limitations':['near-tie cases are not strict token equality',
                     'GPU kernel intervals are inclusive and cannot be added as wall time',
                     '111 numerical failures precede the 112 RoPE fix']},
      'models':old['step110']['models'], 'history':history}
    old['_各关状态']['step111']=status
    args.output.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':status,'case_counts':counts,'failed':bad}))
    return int(bool(incomplete or failed_followup))

if __name__=='__main__':raise SystemExit(main())
