"""单卡真实量化 checkpoint 加载探针；避免与其他计算进程争卡。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmark_step113_quant_capability import gpu_occupancy

ROOT = Path(__file__).resolve().parents[1]


def check_public_artifact(artifact, manifest_path, method, expected_tokenizer_sha, target_name):
    from check_step114_quant_data import sha
    from step114_download_public import MODELS
    manifest = json.loads(manifest_path.read_text())
    repo = manifest['repo']
    if (repo not in MODELS or manifest['revision'] != MODELS[repo]
            or Path(manifest['snapshot']).resolve() != artifact.resolve()
            or manifest['status'] != 'download_and_integrity_passed_gpu_not_validated'):
        raise ValueError('public checkpoint identity mismatch')
    name = repo.split('/')[1]
    expected_name = target_name + ('-AWQ' if method == 'awq' else '-GPTQ-Int4')
    if method not in ('awq', 'gptq') or name != expected_name:
        raise ValueError('public checkpoint model or format mismatch')
    files = manifest['files']
    for name, entry in files.items():
        path = artifact / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'public checkpoint file mismatch: {name}')
        if entry['hub_lfs_sha256'] and entry['sha256'] != entry['hub_lfs_sha256']:
            raise ValueError(f'public checkpoint LFS mismatch: {name}')
    if files['tokenizer.json']['sha256'] != expected_tokenizer_sha:
        raise ValueError('public checkpoint tokenizer differs from held-out tokenizer')
    from check_step114_public_tensors import check as check_tensors
    inventory = check_tensors(artifact)
    return {'status': 'passed_file_integrity', 'inventory': inventory, 'repo': repo, 'revision': manifest['revision'],
            'source_bf16_revision': 'unknown', 'calibration_leakage': 'unknown',
            'tokenizer_json_sha256': expected_tokenizer_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--public-manifest', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--format', choices=['fp8', 'awq', 'gptq'], default='fp8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error('device must be non-negative and timeout positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {'occupancy': gpu_occupancy(args.device), 'artifact': str(args.artifact.resolve())}
    if state['occupancy']['status'] != 'idle':
        state.update(status='not_run', reason='GPU not confirmed idle')
        args.output.write_text(json.dumps(state, indent=2) + '\n')
        return 2
    # 验证输入文件身份，再次检查 GPU；校验不应让模型加载绕过忙卡保护。
    try:
        from check_step114_quant_data import check as check_data, sha
        dataset_root = args.tokens.parent.parent
        data_audit = check_data(dataset_root)
        data_manifest = json.loads((dataset_root / 'manifest.json').read_text())
        target_name = args.tokens.parent.name
        if (args.tokens.name != 'heldout.json' or target_name not in data_manifest['targets']
                or sha(args.tokens) != data_manifest['targets'][target_name]['files']['heldout.json']):
            raise ValueError('held-out token identity mismatch')
        state['data_audit'] = data_audit
        if args.public_manifest:
            state['artifact_audit'] = check_public_artifact(
                args.artifact, args.public_manifest, args.format,
                data_manifest['targets'][target_name]['tokenizer_files']['tokenizer.json'],
                target_name)
        else:
            artifact_manifest = json.loads((args.artifact / 'conversion_manifest.json').read_text())
            if artifact_manifest['source_revision'] != data_manifest['targets'][target_name]['source_revision']:
                raise ValueError('checkpoint and held-out tokenizer source revisions differ')
            if args.format == 'fp8':
                from check_step114_fp8_artifact import check as check_fp8
                state['artifact_audit'] = check_fp8(args.artifact)
        state['occupancy_recheck'] = gpu_occupancy(args.device)
        if state['occupancy_recheck']['status'] != 'idle':
            state.update(status='not_run', reason='GPU became busy during input validation')
            args.output.write_text(json.dumps(state, indent=2) + '\n')
            return 2
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        state.update(status='failed', reason=f'input audit failed: {type(exc).__name__}: {exc}')
        args.output.write_text(json.dumps(state, indent=2) + '\n')
        return 1
    probe_output = args.output.with_name(args.output.stem + '_reference.json')
    logfile = args.output.with_suffix('.log')
    command = [sys.executable, str(ROOT / 'tests/step114/step114_vllm_load_probe.py'),
               '--model', str(args.artifact.resolve()), '--tokens', str(args.tokens.resolve()),
               '--output', str(probe_output.resolve()), '--format', args.format]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), VLLM_USE_V2_MODEL_RUNNER='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_BATCH_INVARIANT='0',
               VLLM_ALLOW_INSECURE_SERIALIZATION='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    state.update(status='running', command=command, log=str(logfile), probe_output=str(probe_output))
    args.output.write_text(json.dumps(state, indent=2) + '\n')
    with logfile.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    state['exit_code'] = code
    if probe_output.is_file():
        state['reference'] = json.loads(probe_output.read_text())
    state['status'] = 'passed' if code == 0 and state.get('reference', {}).get('status') == 'passed' else 'failed'
    args.output.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'status': state['status'], 'exit_code': code, 'log': str(logfile)}))
    return 0 if state['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
