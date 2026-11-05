# 使用：source scripts/activate-4090.sh（在仓库根目录，bash/zsh 均可）
if [ ! -f .venv/bin/activate ]; then
    echo '请在 vllm-from-scratch 仓库根目录 source 此文件。' >&2
    return 1
fi
source .venv/bin/activate
export CUDA_HOME="${MINIVLLM_CUDA_HOME:-$HOME/infra/cuda-12.9}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "缺少开发工具链：$CUDA_HOME/bin/nvcc" >&2
    return 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
# NVIDIA conda 工具链的头文件位于 targets/，由 Torch C++ 扩展显式读取。
export CUDA_INC_PATH="$CUDA_HOME/targets/x86_64-linux/include"
# Triton 也使用这套 CUDA 12.9 汇编器，避免混入包内默认的 CUDA 13 工具。
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="8.9"
export MINIVLLM_UVA_BACKEND=cpp
# 不设置 CUDA_VISIBLE_DEVICES/NCCL_P2P_DISABLE；默认保留四卡与 NCCL 自动选路。
