// SPDX-License-Identifier: Apache-2.0
// Q8_0 CUDA encoder for omk_qad: llama.cpp's quantize_row_q8_0_ref (ggml-quants.c), operation by operation in
// float32, so the payload is byte-identical to libggml's (omk_qad/q8.py has the torch reference). llama-quantize
// ignores the imatrix for Q8_0, so there is no importance argument.
//
// One thread per 32-value block: amax over the block in order, d = amax / 127, id = d ? 1 / d : 0, codes
// roundf(x * id) as int8 (round half away from zero, as roundf). The fp16 scale is written as ONE aligned 16-bit
// store: ptxas (12.6, 12.8) compiles a byte store of a cvt.f16 result to F2I.U8.F16 -- the half's VALUE truncated to
// an integer -- which zeroes the scale's low byte (bug-832 / bug-966; gate the built .so with
// `cuobjdump -sass | grep -c F2I.U8.F16` == 0).

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

namespace {

constexpr int kBlock = 32;
constexpr int kBytes = 34;  // fp16 d + 32 int8 codes
constexpr int kThreads = 256;

__global__ void q8_0_encode(const float *__restrict__ x, int64_t nblocks, uint8_t *__restrict__ out) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const float *xb = x + b * kBlock;
  float amax = 0.f;
  for (int j = 0; j < kBlock; ++j) amax = fmaxf(amax, fabsf(xb[j]));  // MAX(amax, fabsf(v))
  const float d = __fdiv_rn(amax, 127.f);                                // amax / ((1 << 7) - 1)
  const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;                  // d ? 1.0f/d : 0.0f
  uint8_t *p = out + b * kBytes;
  *reinterpret_cast<uint16_t *>(p) = __half_as_ushort(__float2half_rn(d));  // GGML_FP32_TO_FP16, one u16 store
  for (int j = 0; j < kBlock; ++j) {
    const float x0 = __fmul_rn(xb[j], id);
    p[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(roundf(x0)));
  }
}

}  // namespace

at::Tensor q8_0_pack_cuda(at::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous(),
              "q8_0_pack: contiguous float32 CUDA [rows, cols] expected");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % kBlock == 0, "q8_0_pack: cols must be a multiple of 32, got ", cols);
  const c10::cuda::CUDAGuard guard(input.device());
  const int64_t nblocks = rows * (cols / kBlock);
  auto out = at::empty({rows, cols / kBlock, kBytes}, input.options().dtype(at::kByte));
  if (nblocks == 0) return out;
  const int64_t grid = (nblocks + kThreads - 1) / kThreads;
  q8_0_encode<<<static_cast<unsigned>(grid), kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), nblocks, out.data_ptr<uint8_t>());
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}
