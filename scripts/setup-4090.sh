#!/usr/bin/env bash
# 独立 Python 环境 + 私有 CUDA 工具链；不改驱动和其他项目环境。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
UV="${UV:-$HOME/.local/bin/uv}"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
CUDA_ROOT="${MINIVLLM_CUDA_HOME:-$HOME/infra/cuda-12.9}"
WHEELS="${MINIVLLM_WHEEL_DIR:-$HOME/infra/.wheels}"
mkdir -p benchmarks/results/parallel_setup
"$UV" python install 3.12.13
if [[ ! -x .venv/bin/python ]]; then
    "$UV" venv --managed-python --python 3.12.13 .venv
fi
.venv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
assert (Path(sysconfig.get_path('include')) / 'Python.h').is_file(), (
    'Python 缺开发头文件。使用 uv managed Python 创建 .venv，不以缺头文件的系统 Python 构建。')
PY
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
    "$CONDA" create -y -p "$CUDA_ROOT" --override-channels -c nvidia -c conda-forge \
        'cuda-nvcc=12.9' 'cuda-cudart-dev=12.9' 'cuda-cccl=12.9'
fi
source scripts/activate-4090.sh
# Arctic 的隔离构建依赖另一套 protobuf。先装运行环境，再单独构建 Arctic。
runtime_requirements=requirements/4090-dev.txt
if [[ -f requirements/4090-dev.lock.txt ]]; then
    runtime_requirements=requirements/4090-dev.lock.txt
fi
grep -v '^arctic-inference' "$runtime_requirements" > benchmarks/results/parallel_setup/base-requirements.txt
wheel_args=()
build_args=(--build-constraint requirements/4090-build.txt)
torch_wheel="$WHEELS/torch-2.13.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
triton_wheel="$WHEELS/pytorch/triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
if [[ -f "$torch_wheel" && -f "$triton_wheel" ]]; then
    # 官方 PyTorch index 中发布的摘要，避免使用中断下载或混装同名 wheel。
    printf '%s  %s\n' \
        df28741fcd89e3da7cce2d48cbe5299d6732d510ac20f5d422d0b85edf18c327 "$torch_wheel" \
        225910e79149807de74a0ca63160aa285956a8d64a2951d558997c57e2584ae1 "$triton_wheel" | sha256sum -c -
    wheel_args=("$torch_wheel" "$triton_wheel")
    .venv/bin/python - "$torch_wheel" "$triton_wheel" <<'PY'
import sys
from pathlib import Path
text = ''.join(f'{name} @ {Path(path).resolve().as_uri()}\n'
               for name, path in zip(('torch', 'triton'), sys.argv[1:]))
Path('benchmarks/results/parallel_setup/local-build-constraints.txt').write_text(text)
PY
    build_args+=(--build-constraint benchmarks/results/parallel_setup/local-build-constraints.txt)
fi
"$UV" pip install --python .venv/bin/python "${wheel_args[@]}" \
    -r benchmarks/results/parallel_setup/base-requirements.txt --torch-backend=cu129
CMAKE_BUILD_PARALLEL_LEVEL=4 "$UV" pip install --python .venv/bin/python \
    arctic-inference==0.3.0 "${build_args[@]}" --torch-backend=cu129
"$UV" pip check --python .venv/bin/python
printf '\n环境安装完成；随后运行 docs/parallel_001_4090_setup.md 中的功能验收。\n'
