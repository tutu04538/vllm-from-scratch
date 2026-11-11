"""106 的共享权重判据；单元测试不替代真实模型验收。"""
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import load_file
import pytest

from minivllm.worker.gpu.spec_decode.eagle import utils
from minivllm.models.qwen3_eagle3 import Eagle3ForCausalLM
from minivllm.testing.tiny_models import tiny_eagle3_dir


class Draft(nn.Module):
    def __init__(self, embed, head, owns_embed=True, owns_head=True):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = embed
        self.lm_head = head
        self.has_own_embed_tokens = owns_embed
        self.has_own_lm_head = owns_head


def _load(monkeypatch, draft, target):
    monkeypatch.setattr(utils, 'get_model', lambda *_: draft)
    config = SimpleNamespace(speculative_config=SimpleNamespace(draft_model_config=object()),
                             device_config=SimpleNamespace(device='cpu'))
    return utils.load_eagle_model(target, config)


def test_equal_embedding_shares_but_distinct_reduced_head_does_not(monkeypatch):
    target = nn.Module()
    target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(8, 3)
    target.lm_head = nn.Linear(3, 8, bias=False)
    draft_embed = nn.Embedding(8, 3)
    draft_embed.weight.data.copy_(target.model.embed_tokens.weight)
    draft = Draft(draft_embed, nn.Linear(3, 4, bias=False))
    head = draft.lm_head
    _load(monkeypatch, draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is head


def test_same_shape_different_embedding_not_shared(monkeypatch):
    target = nn.Module(); target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(8, 3)
    target.lm_head = nn.Linear(3, 8, bias=False)
    embed = nn.Embedding(8, 3)
    with torch.no_grad(): embed.weight.copy_(target.model.embed_tokens.weight + 1)
    draft = Draft(embed, nn.Linear(3, 4, bias=False))
    _load(monkeypatch, draft, target)
    assert draft.model.embed_tokens is embed


def test_absent_weights_share_target(monkeypatch):
    target = nn.Module(); target.model = nn.Module()
    target.model.embed_tokens = nn.Embedding(8, 3)
    target.lm_head = nn.Linear(3, 8, bias=False)
    draft = Draft(nn.Embedding(8, 3), nn.Linear(3, 8, bias=False), False, False)
    _load(monkeypatch, draft, target)
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head


def test_own_flags_follow_checkpoint():
    import json
    from pathlib import Path
    path = Path(tiny_eagle3_dir())
    config = json.loads((path / 'config.json').read_text())
    model = Eagle3ForCausalLM(config)
    weights = load_file(str(path / 'model.safetensors'))
    model.load_weights(iter(sorted(weights.items())))
    assert model.has_own_lm_head
    assert not model.has_own_embed_tokens


@pytest.mark.parametrize('owns,mode,expected', [(False, 'different', True),
                                                (True, 'equal', True),
                                                (True, 'different', False),
                                                (True, 'shape', False)])
def test_share_predicate_matches_frozen(owns, mode, expected):
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share as upstream
    target = nn.Embedding(8, 3)
    draft = nn.Embedding(7 if mode == 'shape' else 8, 3)
    if mode == 'equal':
        with torch.no_grad(): draft.weight.copy_(target.weight)
    model = SimpleNamespace(has_own_embed_tokens=owns)
    assert utils._should_share(model, 'has_own_embed_tokens', draft, target) is expected
    assert upstream(model, 'has_own_embed_tokens', draft, target) is expected
