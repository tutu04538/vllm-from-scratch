"""独立进程调用冻结 vLLM 的 CPU packing 和 AWQ 转换。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from vllm.scalar_type import scalar_types
import vllm.model_executor.parameter as upstream_parameter
import vllm.model_executor.layers.quantization.auto_awq as awq
import vllm.model_executor.layers.quantization.utils.quant_utils as quant

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'learning_notes/14_vllm_from_scratch/量化完整需求/参考资料/vllm_0_28_0/vllm/model_executor/layers/quantization'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_hashes = {}
    for module, relative in [(awq, 'auto_awq.py'), (quant, 'utils/quant_utils.py')]:
        actual, expected = digest(module.__file__), digest(FREEZE / relative)
        if actual != expected:
            raise ValueError(f'安装 vLLM 与冻结源码不一致：{relative}')
        source_hashes[relative] = actual
    # 转换会创建上游 Parameter 并读取 TP 状态；本探针只运行 CPU 单 rank，
    # 不创建 dist/CUDA 进程组，限定替换两个环境查询，不替换被测转换函数。
    upstream_parameter.get_tensor_model_parallel_rank = lambda: 0
    upstream_parameter.get_tensor_model_parallel_world_size = lambda: 1
    data = json.loads(args.input.read_text())
    output = {'source_hashes': source_hashes, 'cases': []}
    for case in data['cases']:
        qweight = torch.tensor(case['qweight'], dtype=torch.int32)
        qzeros = torch.tensor(case['qzeros'], dtype=torch.int32)
        if case.get('format') == 'gptq':
            q = quant.unpack_quantized_values_into_int32(qweight, scalar_types.uint4, packed_dim=0)
            z_raw = quant.unpack_quantized_values_into_int32(qzeros, scalar_types.uint4, packed_dim=1)
            output['cases'].append({'name': case['name'], 'q': q.tolist(), 'z_file_raw': z_raw.tolist()})
            continue
        layer = nn.Module()
        layer.register_parameter('qweight', nn.Parameter(qweight, requires_grad=False))
        layer.register_parameter('qzeros', nn.Parameter(qzeros, requires_grad=False))
        awq._convert_awq_to_standard_format(layer, 'qweight', 'qzeros', 4)
        result = {'name': case['name'], 'qweight': layer.qweight.tolist(),
                  'qzeros': layer.qzeros.tolist(),
                  'qweight_shape': list(layer.qweight.shape),
                  'qzeros_shape': list(layer.qzeros.shape),
                  'q': quant.unpack_quantized_values_into_int32(layer.qweight, scalar_types.uint4, packed_dim=0).tolist(),
                  'z': quant.unpack_quantized_values_into_int32(layer.qzeros, scalar_types.uint4, packed_dim=0).T.tolist(),
                  'qweight_stride': list(layer.qweight.stride()),
                  'qzeros_stride': list(layer.qzeros.stride())}
        if 'scales' in case:
            q = quant.unpack_quantized_values_into_int32(layer.qweight, scalar_types.uint4, packed_dim=0)
            z = quant.unpack_quantized_values_into_int32(layer.qzeros, scalar_types.uint4, packed_dim=0).T
            scale = torch.tensor(case['scales'], dtype=torch.float32)
            group_size = case['group_size']
            output_values = (q.float() - z.repeat_interleave(group_size, dim=0).float()) * scale.repeat_interleave(group_size, dim=0)
            result['dequant'] = output_values.tolist()
        if 'logical' in case:
            logical = torch.tensor(case['logical'], dtype=torch.int32)
            k, n = logical.shape
            result['standard_rows'] = quant.pack_rows(logical, 4, k, n).tolist()
            result['standard_cols'] = quant.pack_cols(logical, 4, k, n).tolist()
            result['awq_packed'] = quant.awq_pack(logical, 4, k, n).tolist()
        output['cases'].append(result)
    args.output.write_text(json.dumps(output) + '\n')


if __name__ == '__main__':
    main()
