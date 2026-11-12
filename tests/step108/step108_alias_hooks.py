"""Test-only inspection of the frozen target's boundary-0 storage alias."""
from pathlib import Path

import torch


def install_alias(worker):
    model = worker.model_runner.model.model
    records = []
    worker.model_runner._step108_alias_records = records

    def snap(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, (tuple, list)):
            return [snap(item) for item in value]
        return value

    def embed_hook(module, args, output):
        records.append({'event': 'embedding', 'value': snap(output)})

    def norm_hook(module, args, output):
        records.append({'event': 'norm', 'value': snap(output)})

    def model_hook(module, args, output):
        records.append({'event': 'model', 'value': snap(output)})

    model.embed_tokens.register_forward_hook(embed_hook)
    model.norm.register_forward_hook(norm_hook)
    model.register_forward_hook(model_hook)
    return True


def flush_alias(worker, directory):
    import torch.distributed as dist
    path = Path(directory)/f'alias_rank{dist.get_rank()}.pt'
    torch.save(worker.model_runner._step108_alias_records, path)
    return str(path)
