"""用固定 Hub revision 的 LFS 元数据核对 090 已审计权重的 SHA256。"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from huggingface_hub import HfApi
from benchmarks.check_step090_models import MODELS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    api=HfApi(); result={'status':'failed','models':{}}
    try:
        for key,(repo,revision,*_) in MODELS.items():
            info=api.model_info(repo,revision=revision,files_metadata=True,timeout=60)
            if info.sha!=revision: raise ValueError(f'{repo} revision不一致')
            expected={x.rfilename:x.lfs.sha256 for x in info.siblings
                      if x.lfs is not None and (x.rfilename.endswith('.safetensors') or x.rfilename=='pytorch_model.bin')}
            if key.endswith('-draft'):
                row=json.loads((args.results/f"convert_{key[:-6]}.json").read_text())
                actual={'pytorch_model.bin':row['conversion']['source_sha256']}
            else:
                row=json.loads((args.results/f'audit_{key}.json').read_text())
                actual=row['audit']['weight_files']
            if row['status']!='passed' or not actual or actual!=expected:
                raise ValueError(f'{repo} LFS SHA256不匹配或文件集合不完整')
            result['models'][key]={'repo':repo,'revision':revision,'files':actual,'passed':True}
        result['status']='passed'
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
