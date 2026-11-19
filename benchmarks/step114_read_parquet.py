"""在隔离 pyarrow 环境中将冻结的 Parquet 行原样转为 JSONL。"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    table = pq.read_table(args.input)
    with args.output.open('w', encoding='utf-8') as stream:
        for row in table.to_pylist():
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    print(json.dumps({'rows': table.num_rows, 'columns': table.column_names,
                      'pyarrow': pa.__version__}))


if __name__ == '__main__':
    main()
