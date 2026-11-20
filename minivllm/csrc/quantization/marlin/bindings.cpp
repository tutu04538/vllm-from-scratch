// SPDX-License-Identifier: Apache-2.0
// 独立命名空间；schema 同构冻结 vLLM，算子不注册到上游 _C。
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(minivllm_quant, m) {
  m.def("gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  m.def("marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? b_bias_or_none, Tensor b_scales, Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
}
