"""从冻结参考的重复/TP 实测登记数值契约；不读取 minivllm 结果。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def metrics(a, b):
    if a.shape != b.shape: raise ValueError(f'比较形状不同：{a.shape} / {b.shape}')
    a=a.float();b=b.float(); delta=(a-b).abs();flat=delta.reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('非有限数值不能用于目标参考误差登记')
    index=int(torch.argmax(flat)); denominator=float(torch.linalg.vector_norm(b))
    return {'max_abs':float(flat[index]),'relative_l2':float(torch.linalg.vector_norm(a-b))/max(denominator,1e-30),
            'shape':list(a.shape), 'max_error_flat_index':index,
            'max_error_index':[int(x) for x in torch.unravel_index(torch.tensor(index),a.shape)],
            'observed_value':float(a.reshape(-1)[index]),
            'reference_value':float(b.reshape(-1)[index])}


def assemble(directory, iteration, tp, hidden):
    rows=[torch.load(directory/f'iter{iteration}_rank{rank}.pt',map_location='cpu',weights_only=True) for rank in range(tp)]
    for row in rows:
        ids=row['input_ids'].tolist(); positions=row['positions'].tolist()
        if len(ids)!=128 or positions!=list(range(128)) or row['seq_lens'].tolist()!=[128]:
            raise ValueError('固定历史/位置/有效长度不符')
        if row['slot_mapping'].numel()!=128 or (row['slot_mapping']<0).any():
            raise ValueError('有效 slot 不完整')
    for name in ('input_ids','positions','slot_mapping','seq_lens'):
        if any(not torch.equal(rows[0][name],row[name]) for row in rows[1:]):
            raise ValueError(f'TP 组内 {name} 不一致')
    # 只核对有效块：128 token / block size 16；后面的表项是 padding。
    if any(not torch.equal(rows[0]['block_table'][:,:8],row['block_table'][:,:8]) for row in rows[1:]):
        raise ValueError('TP 组内有效 block table 不一致')
    for name in ('attention','hidden','logits'):
        if any(not torch.equal(rows[0][name],row[name]) for row in rows[1:]):
            raise ValueError(f'TP 组内 {name} 不一致')
    q=hidden//tp;k=1024//tp
    parts=[r['qkv'].split([q,k,k],dim=-1) for r in rows]
    qkv=torch.cat([torch.cat([part[i] for part in parts],dim=-1) for i in range(3)],dim=-1)
    kv=torch.cat([r['kv'] for r in rows],dim=1)
    logits=rows[0]['logits'].float()
    values,indices=logits[0].topk(2)
    return {'qkv':qkv,'attention':rows[0]['attention'],'hidden':rows[0]['hidden'],
            'kv':kv,'raw_logits':logits,'probability':torch.softmax(logits,dim=-1),
            'top':{'ids':indices.tolist(),'values':values.tolist(),'margin':float(values[0]-values[1])},
            'input_ids':rows[0]['input_ids']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result={'status':'building','definition':'2×参考最大观察误差，最小绝对门槛1e-5；仅登记本固定输入和所测对象，未使用mini结果',
            'models':{},'unregistered':['EAGLE3映射后logits：090未执行draft前向，113前不得视为已登记']}
    for model,hidden,tps in [('8b',4096,(1,2,4)),('14b',5120,(2,4))]:
        configs={}; baseline=None;prompt=None
        for tp in tps:
            folder=a.reference_root/f'{model}_tp{tp}_kv'
            summary=json.loads((folder/'summary.json').read_text())
            if summary['status']!='passed' or len(summary['iterations'])<3:
                raise ValueError(f'{folder} 参考未完成')
            ranks=summary['ranks']
            if sorted(x['rank'] for x in ranks)!=list(range(tp)): raise ValueError('rank 不完整')
            uuids=[x.get('device_uuid') for x in ranks]
            if any(not x or x=='unavailable' for x in uuids) or len(set(uuids))!=tp:
                raise ValueError('GPU UUID 缺失或重复')
            runs=[assemble(folder,i,tp,hidden) for i in range(3)]
            for i, run in enumerate(runs):
                if run['input_ids'].tolist()!=summary['prompt_token_ids']:
                    raise ValueError('实际 token IDs 与预先固定的历史不一致')
                if summary['iterations'][i]['token_ids'] != [run['top']['ids'][0]]:
                    raise ValueError('greedy token 与参考原始 logits top1 不一致')
            if prompt is None:prompt=runs[0]['input_ids']
            if not torch.equal(prompt,runs[0]['input_ids']):raise ValueError('跨TP token IDs不同')
            if baseline is None: baseline=runs[0]
            keys=('qkv','attention','hidden','kv','raw_logits','probability')
            comparisons={}
            for key in keys:
                comparisons[key]={'repeat': [metrics(runs[i][key],runs[0][key]) for i in (1,2)],
                                  'versus_reference_tp':metrics(runs[0][key],baseline[key])}
            configs[str(tp)]={'comparisons':comparisons,'top':runs[0]['top'],
                              'ranks':ranks,'oracle':summary['oracle'],
                              'local_fp32':summary['local_fp32']}
        thresholds={}
        for key in ('qkv','attention','hidden','kv','raw_logits','probability'):
            comparisons=[v['comparisons'][key] for v in configs.values()]
            rows=[row for c in comparisons for row in [*c['repeat'],c['versus_reference_tp']]]
            thresholds[key]={'max_abs':max(1e-5,2*max(row['max_abs'] for row in rows)),
                             'relative_l2':max(1e-5,2*max(row['relative_l2'] for row in rows))}
        stats=[row for v in configs.values() for row in v['local_fp32']]
        thresholds['local_qkv_fp32']={'max_abs':max(1e-5,2*max(row['max_abs'] for row in stats)),
                                      'relative_l2':max(1e-5,2*max(row['relative_l2'] for row in stats))}
        result['models'][model]={'reference_tp':tps[0],'configs':configs,'thresholds':thresholds,
                                 'prompt_token_ids':prompt.tolist()}
    result['status']='registered_for_fixed_target_fixture_only'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
