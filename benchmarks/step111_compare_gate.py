"""Register frozen-only repeats and compare ordinary TP gate traces per rank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.step109_compare import measure


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path, rank):
    result = {}
    counts = {}
    for item in torch.load(path / f'run0_rank{rank}.pt', map_location='cpu', weights_only=True):
        req = str(item['request_id']).split('-', 1)[0]
        step = counts.get(req, 0); counts[req] = step + 1
        result[f'{req}/{step}'] = item
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--other', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    ref = json.loads(a.reference.with_suffix('.json').read_text())
    other = json.loads(a.other.with_suffix('.json').read_text())
    if ref['implementation'] != 'frozen' or (a.register and other['implementation'] != 'frozen'):
        raise ValueError('registration requires two frozen reference runs')
    if ref['case'] != other['case'] or ref['prompt_ids'] != other['prompt_ids']:
        raise ValueError('case or prompt mismatch')
    tp = ref['case']['tp']; batch = ref['case']['batch']
    reference_hashes = {f'run0_rank{r}.pt':digest(a.reference / f'run0_rank{r}.pt') for r in range(tp)}
    reference_hashes['summary'] = digest(a.reference.with_suffix('.json'))
    if a.register:
        contract = {'reference_hashes':reference_hashes, 'thresholds':{}}
        calibration = None
        if a.calibration is not None:
            source = json.loads(a.calibration.read_text())
            group = f"{ref['case']['model']}_b{batch}"
            calibration = source['results'][group]
            if source.get('status') != 'registered' or calibration['prompt_ids'] != ref['prompt_ids']:
                raise ValueError('cross-TP reference calibration does not match prompt IDs')
            contract['calibration_sha256'] = digest(a.calibration)
    else:
        contract = json.loads(a.contract.read_text())
        if contract['reference_hashes'] != reference_hashes: raise ValueError('reference changed after registration')
        if contract.get('calibration_sha256') and (a.calibration is None or digest(a.calibration) != contract['calibration_sha256']):
            raise ValueError('cross-TP calibration changed after registration')
    report = {'passed':True, 'first_diff':None, 'values':{}, 'top_margins':[],
              'discrete_equal':ref['samples'][0]['tokens'] == other['samples'][0]['tokens']}
    if not report['discrete_equal']: report['passed'] = False; report['first_diff'] = 'tokens'
    rank_discrete = []
    for rank in range(tp):
        left, right = records(a.reference, rank), records(a.other, rank)
        expected = {f'{req}/{step}' for req in range(batch) for step in range(4)}
        if left.keys() != expected or right.keys() != expected:
            report['passed'] = False
            if report['first_diff'] is None: report['first_diff'] = f'rank{rank}: incomplete state records'
        rank_discrete.append({key:(item['position'], item['sequence_length'], item['sampled']) for key,item in right.items()})
        for key in sorted(left.keys() & right.keys()):
            x, y = left[key], right[key]
            for field in ('position', 'sequence_length', 'sampled'):
                if x[field] != y[field]:
                    report['passed'] = False
                    if report['first_diff'] is None: report['first_diff'] = f'rank{rank}/{key}/{field}'
            for field in ('hidden', 'logits', 'kv'):
                label = f'rank{rank}/{key}/{field}'
                try: value = measure(x[field], y[field])
                except (ValueError, TypeError) as exc:
                    report['passed'] = False
                    if report['first_diff'] is None: report['first_diff'] = f'{label}: {exc}'
                    continue
                if a.register:
                    prior = calibration['thresholds'][f'{key}/{field}'] if calibration is not None else {}
                    contract['thresholds'][label] = {'max_abs':max(1e-5, 2*value['max_abs'], prior.get('max_abs', 0)),
                                                     'relative_l2':max(1e-5, 2*value['relative_l2'], prior.get('relative_l2', 0))}
                else:
                    limit = contract['thresholds'][label]; value['threshold'] = limit
                    if value['max_abs'] > limit['max_abs'] or value['relative_l2'] > limit['relative_l2']:
                        report['passed'] = False
                        if report['first_diff'] is None: report['first_diff'] = label
                report['values'][label] = value
                if field == 'logits':
                    vals, ids = x[field].float().topk(2)
                    report['top_margins'].append({'rank':rank, 'event':key, 'top1':int(ids[0]),
                                                  'top2':int(ids[1]), 'margin':float(vals[0]-vals[1])})
    if any(x != rank_discrete[0] for x in rank_discrete[1:]):
        report['passed'] = False
        if report['first_diff'] is None: report['first_diff'] = 'discrete state differs across TP ranks'
    if a.register and report['passed']:
        a.contract.write_text(json.dumps(contract, indent=2)+'\n')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'first_diff':report['first_diff'], 'compared':len(report['values'])}))
    return int(not report['passed'])

if __name__ == '__main__': raise SystemExit(main())
