"""Verify BF16 top-2 ties at the first same-TP greedy divergence."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.step111_compare_gate import records


def verify(gate, reference, mini, stage='gate'):
    case=gate['case'];tp=case['tp'];batch=case['batch']
    fg=gate['gate']['frozen_ref']['result'];mg=gate['gate']['mini']['result']
    if stage == 'performance':
        ref_tokens=gate['engines']['frozen']['result']['samples'][0]['tokens']
        mini_tokens=gate['engines']['mini']['result']['samples'][0]['tokens']
        for req in ref_tokens:
            if (ref_tokens[req][:4] != fg['samples'][0]['tokens'][req][:4] or
                mini_tokens[req][:4] != mg['samples'][0]['tokens'][req][:4]):
                raise ValueError(f'{req}: performance history differs from validated state window')
    else:
        ref_tokens=fg['samples'][0]['tokens'];mini_tokens=mg['samples'][0]['tokens']
    if (reference['tokens']!=ref_tokens or mini['tokens']!=mini_tokens or
        reference['prompt_ids']!=fg['prompt_ids'] or mini['prompt_ids']!=fg['prompt_ids']):
        raise ValueError('diagnostics did not reproduce the gate histories and prompts')
    if len(reference['ranks'])!=tp or len(mini['ranks'])!=tp:
        raise ValueError('diagnostic rank count differs')
    expected = {item['rank']:item['device_uuid'] for item in fg['rank_before']}
    by_rank = []
    for data in (reference,mini):
        if {item['rank']:item['device_uuid'] for item in data['ranks']} != expected:
            raise ValueError('diagnostic GPU set differs from gate')
        by_rank.append({item['rank']:item for item in data['ranks']})
    report=json.loads(Path(gate['gate']['comparison']['report']).read_text())
    values=report['values']
    if len(values)!=batch*4*tp*3: raise ValueError('incomplete numerical comparison')
    for name,value in values.items():
        limit=value['threshold']
        if value['max_abs']>limit['max_abs'] or value['relative_l2']>limit['relative_l2']:
            raise ValueError(f'numerical contract failed: {name}')
    ref_dir=Path(gate['gate']['frozen_ref']['command'][-1]).with_suffix('')
    mini_dir=Path(gate['gate']['mini']['command'][-1]).with_suffix('')
    expected={f'{req}/{step}' for req in range(batch) for step in range(4)}
    first_ref=first_mini=None
    for rank in range(tp):
        x,y=records(ref_dir,rank),records(mini_dir,rank)
        if set(x)!=expected or set(y)!=expected: raise ValueError('incomplete rank state')
        if rank == 0: first_ref,first_mini=x,y
        for key in expected:
            for field in ('position','sequence_length','sampled'):
                if x[key][field]!=y[key][field]: raise ValueError(f'discrete state differs: rank{rank}/{key}/{field}')
                if x[key][field]!=first_ref[key][field] or y[key][field]!=first_mini[key][field]:
                    raise ValueError(f'TP-rank state differs: rank{rank}/{key}/{field}')
    divergences=[]
    if ref_tokens.keys()!=mini_tokens.keys(): raise ValueError('request IDs differ')
    for req in sorted(ref_tokens):
        left,right=ref_tokens[req],mini_tokens[req]
        if len(left)!=len(right): raise ValueError(f'{req}: output lengths differ')
        index=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None)
        if index is None: continue
        if index<4: raise ValueError(f'{req}: divergence precedes validated state window')
        per_rank=[]
        for rank in range(tp):
            events=[]
            for data, token in ((by_rank[0],left[index]),(by_rank[1],right[index])):
                candidates=[e for e in data[rank]['events'] if e['request']==req and e['step']==index]
                if len(candidates)!=1: raise ValueError(f'{req}: missing rank{rank} logit event at {index}')
                event=candidates[0]
                if event.get('dtype') != 'torch.bfloat16':
                    raise ValueError(f'{req}: logits dtype is not verified BF16')
                if event['sampled']!=token or set(event['top_ids'])!={left[index],right[index]}:
                    raise ValueError(f'{req}: divergent IDs are not the two top candidates')
                events.append(event)
            if not any(e['top_values'][0]==e['top_values'][1] for e in events):
                raise ValueError(f'{req}: neither side has an exact BF16 top-2 tie')
            per_rank.append({'rank':rank,'frozen':events[0],'mini':events[1]})
        divergences.append({'request':req,'position':index,'frozen_token':left[index],
                            'mini_token':right[index],'ranks':per_rank})
    if not divergences: raise ValueError('no divergent token to classify')
    return {'passed':True,'strict_equivalent':False,'classification':'exact BF16 top-2 tie in at least one engine',
            'divergences':divergences,'scope':'first divergence after equal request history; first four states and logical KV passed the pre-registered thresholds'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-report',type=Path,required=True);p.add_argument('--case-id')
    p.add_argument('--stage',choices=('gate','performance'),default='gate')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--mini',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cases=json.loads(a.gate_report.read_text())['results']
    matching=[x for x in cases if a.case_id is None or x.get('case_id')==a.case_id]
    if len(matching)!=1:p.error('select exactly one case')
    try: result=verify(matching[0],json.loads(a.reference.read_text()),json.loads(a.mini.read_text()),a.stage)
    except (ValueError,KeyError,OSError) as exc: result={'passed':False,'error':str(exc)}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed']}))
    return int(not result['passed'])

if __name__=='__main__':raise SystemExit(main())
