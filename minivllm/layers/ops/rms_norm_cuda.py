"""BF16 RMSNorm CUDA reduction for the frozen vLLM CUDA execution path.

Compilation is lazy, per CUDA architecture, and serialized across workers.  The
extension contains only local kernels; the installed vLLM is never imported.
"""
from __future__ import annotations

import ctypes
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

_SOURCE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include <stdint.h>

__device__ __forceinline__ int64_t row_offset(int row, int height, int64_t s0, int64_t s1) {
  return height ? (int64_t)(row / height) * s0 + (int64_t)(row % height) * s1 : (int64_t)row * s0;
}

__global__ void plain(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
                      int d, float eps, int height, int64_t s0, int64_t s1) {
  int row = blockIdx.x, lane = threadIdx.x;
  int64_t offset = row_offset(row, height, s0, s1);
  float sum = 0.f;
  for (int chunk = lane; chunk < d / 8; chunk += blockDim.x) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = __bfloat162float(x[offset + chunk * 8 + j]);
      sum += v * v;
    }
  }
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ Reduce::TempStorage scratch;
  __shared__ float inv;
  float total = Reduce(scratch).Reduce(sum, cub::Sum(), blockDim.x);
  if (lane == 0) inv = rsqrtf(total / (float)d + eps);
  __syncthreads();
  for (int chunk = lane; chunk < d / 8; chunk += blockDim.x) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int col = chunk * 8 + j;
      float v = __bfloat162float(x[offset + col]);
      out[(int64_t)row * d + col] = __float2bfloat16_rn(v * inv * __bfloat162float(w[col]));
    }
  }
}

__global__ void fused(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                      const __nv_bfloat16* w, __nv_bfloat16* out, __nv_bfloat16* residual_out,
                      int d, float eps, int height, int64_t xs0, int64_t xs1,
                      int64_t rs0, int64_t rs1) {
  int row = blockIdx.x, lane = threadIdx.x;
  int64_t xo = row_offset(row, height, xs0, xs1), ro = row_offset(row, height, rs0, rs1);
  float sum = 0.f;
  for (int chunk = lane; chunk < d / 8; chunk += blockDim.x) {
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int col = chunk * 8 + j;
      __nv_bfloat16 rounded = __float2bfloat16_rn(__bfloat162float(x[xo + col]) + __bfloat162float(residual[ro + col]));
      v[j] = __bfloat162float(rounded);
      residual_out[(int64_t)row * d + col] = rounded;
    }
    #pragma unroll
    for (int j = 0; j < 8; j += 2) sum += v[j] * v[j] + v[j + 1] * v[j + 1];
  }
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ Reduce::TempStorage scratch;
  __shared__ float inv;
  float total = Reduce(scratch).Reduce(sum, cub::Sum(), blockDim.x);
  if (lane == 0) inv = rsqrtf(total / (float)d + eps);
  __syncthreads();
  for (int chunk = lane; chunk < d / 8; chunk += blockDim.x) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int col = chunk * 8 + j;
      float v = __bfloat162float(residual_out[(int64_t)row * d + col]);
      out[(int64_t)row * d + col] = __float2bfloat16_rn(v * inv * __bfloat162float(w[col]));
    }
  }
}

extern "C" int minivllm_rms_plain(void* x, void* w, void* out, int rows, int d, float eps,
                                  int height, int64_t s0, int64_t s1, cudaStream_t stream) {
  int threads = d / 8 < 1024 ? d / 8 : 1024;
  plain<<<rows, threads, 0, stream>>>((__nv_bfloat16*)x, (__nv_bfloat16*)w, (__nv_bfloat16*)out,
                                      d, eps, height, s0, s1);
  return (int)cudaGetLastError();
}
extern "C" int minivllm_rms_fused(void* x, void* r, void* w, void* out, void* ro,
                                  int rows, int d, float eps, int height,
                                  int64_t xs0, int64_t xs1, int64_t rs0, int64_t rs1,
                                  cudaStream_t stream) {
  int threads = d < 1024 ? d : 1024;
  fused<<<rows, threads, 0, stream>>>((__nv_bfloat16*)x, (__nv_bfloat16*)r, (__nv_bfloat16*)w,
      (__nv_bfloat16*)out, (__nv_bfloat16*)ro, d, eps, height, xs0, xs1, rs0, rs1);
  return (int)cudaGetLastError();
}
'''


@functools.lru_cache(maxsize=None)
def _load(major: int, minor: int):
    compiler = shutil.which('nvcc')
    if compiler is None:
        raise RuntimeError('BF16 CUDA RMSNorm 需要 nvcc；请激活项目 CUDA 工具链')
    key = hashlib.sha256((_SOURCE + f'|{major}.{minor}|{compiler}').encode()).hexdigest()[:24]
    directory = Path(tempfile.gettempdir()) / f'minivllm-rms-{os.getuid()}'
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    source, library, lock_path = directory / f'{key}.cu', directory / f'{key}.so', directory / f'{key}.lock'
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            source.write_text(_SOURCE)
            temporary = directory / f'{key}.{os.getpid()}.so'
            result = subprocess.run([compiler, '-std=c++17', '-O3', '-shared', '-Xcompiler', '-fPIC',
                                     f'-arch=sm_{major}{minor}', str(source), '-o', str(temporary)],
                                    capture_output=True, text=True, check=False)
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f'编译 BF16 CUDA RMSNorm 失败：{result.stderr}')
            temporary.replace(library)
    handle = ctypes.CDLL(str(library))
    ptr, integer, stride = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    handle.minivllm_rms_plain.argtypes = [ptr, ptr, ptr, integer, integer, ctypes.c_float,
                                         integer, stride, stride, ptr]
    handle.minivllm_rms_plain.restype = integer
    handle.minivllm_rms_fused.argtypes = [ptr, ptr, ptr, ptr, ptr, integer, integer, ctypes.c_float,
                                         integer, stride, stride, stride, stride, ptr]
    handle.minivllm_rms_fused.restype = integer
    return handle


def supports(x: torch.Tensor, weight: torch.Tensor | None,
             residual: torch.Tensor | None, add_one: bool) -> bool:
    """Select the aligned BF16 CUDA path; other layouts use the existing Triton path."""
    if (add_one or x.dtype != torch.bfloat16 or weight is None or weight.dtype != x.dtype
            or x.shape[-1] % 8 or not weight.is_contiguous()):
        return False
    tensors = (x, weight) if residual is None else (x, weight, residual)
    if any(t.data_ptr() % 16 for t in tensors):
        return False
    for t in (x,) if residual is None else (x, residual):
        if any(t.stride(i) % 8 for i in range(t.ndim - 1)):
            return False
    return True


def run(x: torch.Tensor, weight: torch.Tensor, eps: float,
        residual: torch.Tensor | None, rows: int, *, inplace: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out = x if inplace else torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows == 0:
        return out if residual is None else (out, residual if inplace else torch.empty_like(out))
    height = x.shape[1] if x.ndim == 3 else 0
    xs0 = x.stride(0) if x.ndim > 1 else 0
    xs1 = x.stride(1) if x.ndim == 3 else 0
    with torch.cuda.device(x.device):
        lib = _load(*torch.cuda.get_device_capability(x.device))
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if residual is None:
            error = lib.minivllm_rms_plain(x.data_ptr(), weight.data_ptr(), out.data_ptr(),
                                           rows, x.shape[-1], eps, height, xs0, xs1, stream)
            if error:
                raise RuntimeError(f'BF16 CUDA RMSNorm 启动失败，CUDA error={error}')
            return out
        residual_out = residual if inplace else torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
        rs0 = residual.stride(0) if residual.ndim > 1 else 0
        rs1 = residual.stride(1) if residual.ndim == 3 else 0
        error = lib.minivllm_rms_fused(x.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                                       out.data_ptr(), residual_out.data_ptr(), rows, x.shape[-1],
                                       eps, height, xs0, xs1, rs0, rs1, stream)
        if error:
            raise RuntimeError(f'BF16 CUDA fused RMSNorm 启动失败，CUDA error={error}')
        return out, residual_out
