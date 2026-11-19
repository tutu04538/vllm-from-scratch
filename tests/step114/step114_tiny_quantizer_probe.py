"""隔离环境中的 tiny Qwen3 CPU 离线量化机制探针；不代表真实模型可用。"""
import argparse
from pathlib import Path
import shutil
import tempfile

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['awq', 'gptq'], required=True)
    args = parser.parse_args()
    torch.manual_seed(114)
    config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                         head_dim=32, max_position_embeddings=128)
    with tempfile.TemporaryDirectory() as tmp:
        target, output = Path(tmp) / 'target', Path(tmp) / args.method
        Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(target)
        source = Path('/data/syh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/'
                      'b968826d9c46dd6066d109eabc6255188de91218')
        for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
            shutil.copy2(source / name, target / name)
        samples = [[1 + (i + j) % 100 for j in range(32)] for i in range(2)]
        if args.method == 'awq':
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_pretrained(str(target), torch_dtype=torch.bfloat16,
                trust_remote_code=False, device_map='cpu', use_cache=False)
            model.quantize(None, quant_config={'zero_point': True, 'q_group_size': 128,
                'w_bit': 4, 'version': 'GEMM'}, calib_data=samples, n_parallel_calib_samples=1,
                max_calib_samples=2, max_calib_seq_len=32)
            model.save_quantized(str(output), safetensors=True)
        else:
            from gptqmodel import GPTQModel, QuantizeConfig
            config = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                                    device='cpu', offload_to_disk=False)
            model = GPTQModel.load(str(target), config, trust_remote_code=False)
            model.quantize([{'input_ids': ids, 'attention_mask': [1] * len(ids)} for ids in samples],
                           batch_size=1, calibration_sort=None)
            model.save(str(output))
        assert (output / 'config.json').is_file()
        assert list(output.glob('*.safetensors'))
        print(f'{args.method}: tiny CPU quantization passed; real checkpoints not tested')


if __name__ == '__main__':
    main()
