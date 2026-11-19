"""独立 vLLM 进程加载量化权重并执行短 teacher-forcing/generate。"""
import argparse
import json
import math
from pathlib import Path


def describe_layers(worker):
    model = worker.model_runner.model
    result = {}
    for name, layer in model.named_modules():
        if name.endswith(('layers.0.self_attn.qkv_proj', 'layers.0.mlp.gate_up_proj')):
            method = getattr(layer, 'quant_method', None)
            kernel = getattr(method, 'fp8_linear', None)
            if kernel is None:
                kernel = getattr(method, 'kernel', None)
            weight = getattr(layer, 'weight', None)
            if weight is None:
                weight = getattr(layer, 'qweight', None)
            result[name] = {'method': type(method).__name__, 'kernel': type(kernel).__name__,
                            'weight_dtype': str(weight.dtype), 'weight_shape': list(weight.shape)}
    return result


def install_capture(worker):
    import torch
    model = worker.model_runner.model
    model._step114_records = {'qkv': [], 'logits': []}
    def hook(name):
        def save(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'{name} output is not a tensor')
            model._step114_records[name].append(value.detach().cpu().clone())
        return save
    model._step114_handles = [
        model.model.layers[0].self_attn.qkv_proj.register_forward_hook(hook('qkv')),
        model.logits_processor.register_forward_hook(hook('logits')),
    ]
    return 'step114 capture installed'


def flush_capture(worker, path):
    import hashlib
    import torch
    model = worker.model_runner.model
    records = model._step114_records
    if not records['qkv'] or not records['logits']:
        raise RuntimeError('step114 did not capture both QKV and logits')
    torch.save(records, path)
    for handle in model._step114_handles:
        handle.remove()
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return {'path': path, 'sha256': digest,
            'qkv_shapes': [list(t.shape) for t in records['qkv']],
            'logits_shapes': [list(t.shape) for t in records['logits']]}


def serialise_logprobs(entries):
    if entries is None:
        return None
    output = []
    for row in entries:
        if row is None:
            output.append(None)
            continue
        values = {}
        for token, value in row.items():
            if not math.isfinite(value.logprob):
                raise ValueError('nonfinite logprob')
            values[str(token)] = {'logprob': float(value.logprob), 'rank': value.rank}
        output.append(values)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--tokens', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--format', choices=['fp8', 'awq', 'gptq'], default='fp8')
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    prompt = json.loads(args.tokens.read_text())[0]['token_ids'][:32]
    engine = LLM(model=str(args.model.resolve()), skip_tokenizer_init=True, dtype='bfloat16',
                 tensor_parallel_size=1, enforce_eager=True, max_model_len=128, max_num_seqs=1,
                 max_num_batched_tokens=128, gpu_memory_utilization=.8,
                 kv_cache_memory_bytes=268435456, attention_config={'backend': 'TRITON_ATTN'},
                 enable_prefix_caching=False, disable_log_stats=True, seed=0)
    layers = engine.collective_rpc(describe_layers)
    expected = {'fp8': ('Fp8LinearMethod', 'CutlassFP8ScaledMMLinearKernel'),
                'awq': ('AutoAWQMarlinLinearMethod', 'MarlinLinearKernel'),
                'gptq': ('AutoGPTQLinearMethod', 'MarlinLinearKernel')}[args.format]
    if not layers or len(layers[0]) != 2 or any(
            (info['method'], info['kernel']) != expected for info in layers[0].values()):
        raise ValueError(f'unexpected {args.format} layer/backend: {layers}')
    if engine.collective_rpc(install_capture) != ['step114 capture installed']:
        raise RuntimeError('step114 capture installation failed')
    outputs = engine.generate([{'prompt_token_ids': prompt}],
        SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, seed=0,
                       logprobs=1, prompt_logprobs=1), use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise ValueError('unexpected number of outputs')
    generated = outputs[0].outputs[0]
    if len(generated.token_ids) != 4:
        raise ValueError('expected 4 generated tokens')
    capture_path = args.output.with_name(args.output.stem + '_tensors.pt').resolve()
    captures = engine.collective_rpc(flush_capture, args=(str(capture_path),))
    result = {'status': 'passed', 'scope': f'frozen vLLM single GPU {args.format} short forward',
              'format': args.format,
              'model': str(args.model.resolve()), 'prompt_token_ids': prompt,
              'generated_token_ids': list(generated.token_ids),
              'prompt_logprobs': serialise_logprobs(outputs[0].prompt_logprobs),
              'generated_logprobs': serialise_logprobs(generated.logprobs), 'layers': layers,
              'captures': captures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'generated_token_ids': result['generated_token_ids']}))


if __name__ == '__main__':
    main()
