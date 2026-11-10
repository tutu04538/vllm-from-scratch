#!/usr/bin/env python3
"""106 CPU tiny 阶段切分验收，结果不代表多进程 PP。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "partitions", "forward", "residual"), default="all")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selectors = {"all": "", "partitions": "partition", "forward": "split_matches_full",
                 "residual": "residual_and_local_missing"}
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "pytest", "tests/step106", "-q", "-ra"]
    if selectors[args.case]:
        command += ["-k", selectors[args.case]]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                capture_output=True, timeout=args.timeout)
        data = {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        data = {"command": command, "exit_code": 124, "error": str(exc)}
    data.update(case=args.case, elapsed_seconds=time.monotonic() - start,
                scope="CPU tiny sequential stages; no PP transport")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
