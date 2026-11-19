"""离线固定 step114 校准/质量数据；不生成或验收任何量化模型。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
HUB = Path('/data/syh/.cache/huggingface/hub')
WIKI_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
GSM_REV = '740312add88f781978c0658806c59bc2815b9866'
TARGETS = {
    'Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
    'Qwen3-14B': '40c069824f4251a91eefaf281ebe4c544efd3e18',
}
WIKI = HUB / 'datasets--Salesforce--wikitext' / 'snapshots' / WIKI_REV
GSM = HUB / 'datasets--openai--gsm8k' / 'snapshots' / GSM_REV


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(ids):
    return hashlib.sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()


def windows(token_ids, split, max_windows, width=2048):
    """Non-overlapping consecutive windows, retain exact IDs and content hash."""
    result = []
    for offset in range(0, len(token_ids) - width + 1, width):
        ids = token_ids[offset:offset + width]
        result.append({'id': f'{split}:tokens:{offset}:{offset + width}', 'token_start': offset,
                       'token_ids': ids, 'sha256_uint32_le': token_hash(ids)})
        if len(result) == max_windows:
            break
    return result


def check_no_leakage(train, test):
    train_hashes = {x['sha256_uint32_le'] for x in train}
    test_hashes = {x['sha256_uint32_le'] for x in test}
    if train_hashes & test_hashes:
        raise ValueError('train/test token window overlap')
    if len(test_hashes) != len(test):
        raise ValueError('duplicate test token windows')


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def wiki_tokens(rows, tokenizer):
    if tokenizer.eos_token_id is None:
        raise ValueError('tokenizer lacks eos_token_id')
    result = []
    for row in rows:
        text = row['text']
        if not isinstance(text, str):
            raise ValueError('wikitext row is not text')
        if text:
            result.extend(tokenizer.encode(text, add_special_tokens=False))
            result.append(tokenizer.eos_token_id)
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrow-python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite existing dataset: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = {'wikitext_train': WIKI / 'wikitext-2-raw-v1/train-00000-of-00001.parquet',
             'wikitext_test': WIKI / 'wikitext-2-raw-v1/test-00000-of-00001.parquet',
             'gsm8k_test': GSM / 'main/test-00000-of-00001.parquet',
             'wikitext_card': WIKI / 'README.md', 'gsm8k_card': GSM / 'README.md'}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory(prefix='.step114-data-', dir=destination.parent) as tmp:
        tmpdir = Path(tmp)
        rows = {}
        parquet_metadata = {}
        for name in ('wikitext_train', 'wikitext_test', 'gsm8k_test'):
            out = tmpdir / f'{name}.jsonl'
            command = [str(args.arrow_python), str(ROOT / 'benchmarks/step114_read_parquet.py'),
                       str(paths[name]), str(out)]
            child = subprocess.run(command, text=True, capture_output=True, check=True, timeout=180)
            parquet_metadata[name] = json.loads(child.stdout)
            rows[name] = read_rows(out)
        if len(rows['gsm8k_test']) < 256:
            raise ValueError('GSM8K test must contain at least 256 rows')
        manifest = {'kind': 'data_preparation_only', 'status': 'prepared_not_model_verified',
                    'sources': {name: {'path': str(path), 'sha256': sha_file(path),
                                       'bytes': path.stat().st_size} for name, path in paths.items()},
                    'wikitext_repo': 'Salesforce/wikitext', 'wikitext_revision': WIKI_REV,
                    'wikitext_license': ['cc-by-sa-3.0', 'gfdl'],
                    'gsm8k_repo': 'openai/gsm8k', 'gsm8k_revision': GSM_REV, 'gsm8k_license': ['mit'],
                    'pyarrow_executable': str(args.arrow_python.resolve()), 'parquet': parquet_metadata,
                    'transformers_version': importlib.metadata.version('transformers'),
                    'preparation_script_sha256': sha_file(Path(__file__)),
                    'parquet_reader_sha256': sha_file(ROOT / 'benchmarks/step114_read_parquet.py'),
                    'recipe': {'calibration': 'first 128 non-overlapping 2048-token windows from train',
                               'quality': 'up to first 128 non-overlapping 2048-token windows from test',
                               'tokenization': 'each nonempty raw row without added special tokens; append EOS per row',
                               'seed': 0, 'sampling': 'deterministic corpus order',
                               'gsm8k': 'first 256 test rows, zero-shot, enable_thinking=false, max_tokens=512',
                               'gsm8k_parser': 'last numeric value after ####; unanswered/invalid count incorrect'},
                    'targets': {}}
        for name, revision in TARGETS.items():
            model = HUB / f'models--Qwen--{name}' / 'snapshots' / revision
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
            train = windows(wiki_tokens(rows['wikitext_train'], tokenizer), 'train', 128)
            test = windows(wiki_tokens(rows['wikitext_test'], tokenizer), 'test', 128)
            if len(train) != 128 or not test:
                raise ValueError(f'{name}: insufficient train/test windows: {len(train)}/{len(test)}')
            check_no_leakage(train, test)
            gsm = []
            for index, row in enumerate(rows['gsm8k_test'][:256]):
                prompt = row['question'] + '\nGive the final answer after ####.'
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                gsm.append({'id': f'main/test:{index}', 'question': row['question'],
                            'answer': row['answer'], 'prompt_token_ids': ids,
                            'prompt_sha256_uint32_le': token_hash(ids)})
            folder = tmpdir / name
            folder.mkdir()
            for filename, value in (('calibration.json', train), ('heldout.json', test), ('gsm8k.json', gsm)):
                write_json(folder / filename, value)
            tokenizer_files = [p for p in model.iterdir() if (p.name.startswith('tokenizer') or p.name in ('vocab.json', 'merges.txt', 'config.json')) and p.is_file()]
            manifest['targets'][name] = {'source_revision': revision, 'tokenizer_directory': str(model),
                'tokenizer_files': {p.name: sha_file(p) for p in sorted(tokenizer_files)},
                'eos_token_id': tokenizer.eos_token_id, 'calibration_windows': len(train),
                'heldout_windows': len(test), 'gsm8k_rows': len(gsm),
                'calibration_test_intersection': 0,
                'files': {filename: sha_file(folder / filename)
                          for filename in ('calibration.json', 'heldout.json', 'gsm8k.json')}}
        for name in rows:
            (tmpdir / f'{name}.jsonl').unlink()
        write_json(tmpdir / 'manifest.json', manifest)
        tmpdir.rename(destination)
    print(json.dumps({'status': manifest['status'], 'output': str(destination),
                      'counts': {name: [entry['calibration_windows'], entry['heldout_windows'], entry['gsm8k_rows']]
                                 for name, entry in manifest['targets'].items()}}))


if __name__ == '__main__':
    main()
