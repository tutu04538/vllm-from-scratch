"""核对公开量化 checkpoint 的 safetensors 索引和张量结构。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from safetensors import safe_open


def check(root):
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    found = {}
    dtypes = Counter()
    for filename in sorted(set(index.values())):
        with safe_open(root / filename, framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                if name in found:
                    raise ValueError(f'duplicate tensor: {name}')
                found[name] = filename
                dtypes[str(tensors.get_slice(name).get_dtype())] += 1
    if found != index:
        raise ValueError('safetensors index does not match tensor names/shards')
    qweight = [n for n in found if n.endswith('.qweight')]
    scales = [n for n in found if n.endswith('.scales')]
    zeros = [n for n in found if n.endswith('.qzeros')]
    if not qweight or len(qweight) != len(scales) or len(qweight) != len(zeros):
        raise ValueError('missing packed weights, scales, or zero points')
    for name in qweight:
        prefix = name.removesuffix('.qweight')
        if prefix + '.scales' not in found or prefix + '.qzeros' not in found:
            raise ValueError(f'incomplete quantized tensor set: {prefix}')
    return {'status': 'passed_tensor_inventory', 'tensor_count': len(found),
            'quantized_projection_count': len(qweight), 'dtypes': dict(dtypes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(check(parser.parse_args().root), indent=2))


if __name__ == '__main__':
    main()
