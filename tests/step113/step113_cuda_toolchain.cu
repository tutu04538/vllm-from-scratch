// 仅验证本地 CUDA 12.9 可为 SM89 编译 FP8 类型，不是量化 GEMM kernel。
#include <cuda_fp8.h>

extern "C" __global__ void step113_fp8_cast(const float *input, __nv_fp8_e4m3 *output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = __nv_fp8_e4m3(input[i]);
}
