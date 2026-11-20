"""Test-only capture of first real Qwen3 prefill at every decoder layer."""
from pathlib import Path


def install_layers(worker):
    runner = worker.model_runner
    model = runner.model.model
    runner._step123_layer_outputs = {}
    runner._step123_handles = []

    def hook(name):
        def capture(_module, _args, output):
            if name not in runner._step123_layer_outputs:
                tensors = output if isinstance(output, tuple) else (output,)
                runner._step123_layer_outputs[name] = tuple(
                    t.detach().cpu().clone() if t is not None else None for t in tensors)
        return capture

    runner._step123_handles.append(model.embed_tokens.register_forward_hook(hook('embedding')))
    for i, layer in enumerate(model.layers):
        runner._step123_handles.append(layer.register_forward_hook(hook(f'layer.{i}')))
    runner._step123_handles.append(model.norm.register_forward_hook(hook('final_norm')))
    return len(model.layers)


def flush_layers(worker, directory):
    import torch
    import torch.distributed as dist
    runner = worker.model_runner
    rank = dist.get_rank() if dist.is_initialized() else 0
    for handle in runner._step123_handles:
        handle.remove()
    path = Path(directory) / f'layers_rank{rank}.pt'
    torch.save(runner._step123_layer_outputs, path)
    return {'rank': rank, 'keys': sorted(runner._step123_layer_outputs), 'path': str(path)}
