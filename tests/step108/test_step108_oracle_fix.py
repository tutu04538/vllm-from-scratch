"""The corrected frozen test oracle snapshots boundary zero without changing other boundaries."""
from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys

import torch

from step108_reference_probe import install_boundary_zero_fix


class FakeFrozenModel:
    aux_hidden_state_layers = (0, 1)

    def _maybe_add_hidden_state(self, values, layer, hidden, residual):
        if layer in self.aux_hidden_state_layers:
            values.append(hidden + residual if residual is not None else hidden)
        return values


def test_oracle_correction_snapshots_only_selected_no_residual():
    model = FakeFrozenModel()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(model=model)))
    assert install_boundary_zero_fix(worker) == 'upstream-66546-boundary-zero-snapshot'
    hidden = torch.tensor([1., 2.])
    values = model._maybe_add_hidden_state([], 0, hidden, None)
    hidden.add_(7)
    torch.testing.assert_close(values[0], torch.tensor([1., 2.]), rtol=0, atol=0)
    model._maybe_add_hidden_state(values, 1, hidden, torch.ones_like(hidden))
    torch.testing.assert_close(values[1], torch.tensor([9., 10.]), rtol=0, atol=0)
    assert model._maybe_add_hidden_state([], 2, hidden, None) == []


def test_comparison_rejects_mixed_reference_semantics(tmp_path):
    reference, repeat, candidate = (tmp_path / name for name in ('ref', 'repeat', 'mini'))
    for directory in (reference, repeat, candidate):
        directory.mkdir()
        (directory / 'summary.json').write_text(json.dumps({
            'tp': 1, 'k': 1, 'tokens': [1], 'reference_boundary_zero_fix': True,
        }))
        torch.save([{'event': 'test', 'output': torch.tensor([1.])}], directory / 'rank0.pt')
    script = Path(__file__).with_name('step108_compare_reference.py')
    contract = tmp_path / 'contract.json'

    def run(other, *, register=False):
        command = [sys.executable, str(script), '--reference', str(reference),
                   '--other', str(other), '--contract', str(contract),
                   '--output', str(tmp_path / 'result.json')]
        if register:
            command.append('--register')
        return subprocess.run(command, capture_output=True, text=True, timeout=30)

    assert run(repeat, register=True).returncode == 0
    assert run(candidate).returncode == 0
    summary = json.loads((candidate / 'summary.json').read_text())
    summary['reference_boundary_zero_fix'] = False
    (candidate / 'summary.json').write_text(json.dumps(summary))
    result = run(candidate)
    assert result.returncode != 0
    assert 'different boundary-zero reference semantics' in result.stderr


def test_reference_probe_import_does_not_modify_runner_environment():
    import os
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.pop('VLLM_USE_V2_MODEL_RUNNER', None)
    env.pop('VLLM_ALLOW_INSECURE_SERIALIZATION', None)
    script = ("import os,sys; sys.path.insert(0, 'tests/step108'); "
              "import step108_reference_probe; "
              "assert 'VLLM_USE_V2_MODEL_RUNNER' not in os.environ; "
              "assert 'VLLM_ALLOW_INSECURE_SERIALIZATION' not in os.environ")
    subprocess.run([sys.executable, '-c', script], cwd=root, env=env, check=True, timeout=60)
