// SPDX-License-Identifier: Apache-2.0
// IQ4_XS / IQ4_NL CUDA encoder for omk_qad: llama.cpp's quantize_row_iq4_nl_impl (ggml-quants.c), operation by
// operation in float32, so the payload is byte-identical to libggml's (omk_qad/iq4.py has the derivation and the
// torch reference). Every product, sum, quotient and square root goes through an explicit round-to-nearest
// intrinsic (no fused multiply-add; the extension is also built with --fmad=false), in llama.cpp's order.
//
// IQ4_XS: 8 threads per 256-value super-block, one per 32-value sub-block (scale search), then lane 0 of the
// group sets the super-block scale and the 6-bit sub-block scales. IQ4_NL: one thread per 32-value block.
// importance (optional, float32 [cols]) is llama-quantize's E[x^2] per input column.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <optional>

namespace {

__constant__ float kValues[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                  1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};
constexpr int kSub = 32;
constexpr int kNtry = 7;
constexpr float kGroupMaxEps = 1e-15f;
constexpr int kXsBlock = 256;
constexpr int kXsBytes = 136;
constexpr int kNlBytes = 18;
constexpr int kThreads = 256;

__device__ __forceinline__ int best_index(float x) {
  if (x <= kValues[0]) return 0;
  if (x >= kValues[15]) return 15;
  int ml = 0, mu = 15;
  while (mu - ml > 1) {
    const int mav = (ml + mu) / 2;
    if (x < kValues[mav]) mu = mav; else ml = mav;
  }
  return __fsub_rn(x, kValues[mu - 1]) < __fsub_rn(kValues[mu], x) ? mu - 1 : mu;
}

// sumqx / sumq2 of a sub-block at inverse scale `inv`, accumulated j = 0..31 in order.
__device__ __forceinline__ void sums(const float *xb, const float *w, float inv, float &sumqx, float &sumq2) {
  sumqx = 0.f;
  sumq2 = 0.f;
  for (int j = 0; j < kSub; ++j) {
    const float q = kValues[best_index(__fmul_rn(inv, xb[j]))];
    const float wq = __fmul_rn(w[j], q);
    sumqx = __fadd_rn(sumqx, __fmul_rn(wq, xb[j]));
    sumq2 = __fadd_rn(sumq2, __fmul_rn(wq, q));
  }
}

// The scale of one 32-value sub-block (0 when max |x| < GROUP_MAX_EPS).
__device__ float search(const float *xb, const float *w) {
  float amax = 0.f, mx = 0.f;
  for (int j = 0; j < kSub; ++j) {
    const float ax = fabsf(xb[j]);
    if (ax > amax) { amax = ax; mx = xb[j]; }
  }
  if (amax < kGroupMaxEps) return 0.f;
  float d = __fdiv_rn(-mx, kValues[0]);  // ntry > 0
  float sumqx, sumq2;
  sums(xb, w, __fdiv_rn(1.f, d), sumqx, sumq2);
  d = sumq2 > 0.f ? __fdiv_rn(sumqx, sumq2) : 0.f;
  float best = __fmul_rn(d, sumqx);
  for (int itry = -kNtry; itry <= kNtry; ++itry) {
    sums(xb, w, __fdiv_rn(static_cast<float>(itry + static_cast<int>(kValues[0])), mx), sumqx, sumq2);
    if (sumq2 > 0.f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
      d = __fdiv_rn(sumqx, sumq2);
      best = __fmul_rn(d, sumqx);
    }
  }
  return d;
}

// Weights of the 32 values xb of a block whose (super-)block starts at blk (n values): imatrix or x^2.
__device__ __forceinline__ void weights(const float *blk, int n, const float *xb, const float *qw, float *w) {
  if (qw == nullptr) {
    for (int j = 0; j < kSub; ++j) w[j] = __fmul_rn(xb[j], xb[j]);
    return;
  }
  float sigma2 = 0.f;
  for (int j = 0; j < n; ++j) sigma2 = __fadd_rn(sigma2, __fmul_rn(blk[j], blk[j]));
  sigma2 = __fmul_rn(sigma2, 2.f / static_cast<float>(n));
  for (int j = 0; j < kSub; ++j) w[j] = __fmul_rn(qw[j], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[j], xb[j]))));
}

// One 16-bit store (p is 2-byte aligned: blocks are 136 / 18 bytes). NOT two byte stores: ptxas 12.6 (sm_86)
// compiles `st.u8 [p], <cvt.rn.f16.f32 result>` to F2I.U8.F16 -- the half's VALUE truncated to an int (0), not its
// low byte -- so every scale's low byte came out 0 (omk_qad_work/iq4/sass).
__device__ __forceinline__ void store_half(uint8_t *p, float v) {
  *reinterpret_cast<uint16_t *>(p) = __half_as_ushort(__float2half_rn(v));
}

__global__ void iq4_xs_encode(const float *x, int64_t nblocks, int64_t blocks_per_row, const float *importance,
                              uint8_t *out) {
  __shared__ float s_scale[kThreads];
  __shared__ int s_l[kThreads];
  __shared__ float s_d[kThreads / 8];
  __shared__ float s_id[kThreads / 8];
  const int t = threadIdx.x, g = t / 8, ib = t % 8;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * (kThreads / 8) + g;
  const bool valid = sb < nblocks;
  float xb[kSub], w[kSub];
  if (valid) {
    const float *blk = x + sb * kXsBlock;
    for (int j = 0; j < kSub; ++j) xb[j] = blk[ib * kSub + j];
    const float *qw = importance ? importance + (sb % blocks_per_row) * kXsBlock + ib * kSub : nullptr;
    weights(blk, kXsBlock, xb, qw, w);
    s_scale[t] = search(xb, w);
  }
  __syncthreads();
  if (valid && ib == 0) {
    float max_scale = 0.f, amax_scale = 0.f;
    for (int k = 0; k < 8; ++k) {
      const float a = fabsf(s_scale[t + k]);
      if (a > amax_scale) { amax_scale = a; max_scale = s_scale[t + k]; }
    }
    const float d = __fdiv_rn(-max_scale, 32.f);
    s_d[g] = d;
    s_id[g] = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
  }
  __syncthreads();
  if (valid) {
    const float d = s_d[g];
    int l = __float2int_rn(__fmul_rn(s_id[g], s_scale[t]));  // nearest_int: round half to even
    l = max(-32, min(31, l));
    const float dl = __fmul_rn(d, static_cast<float>(l));
    const float idl = dl != 0.f ? __fdiv_rn(1.f, dl) : 0.f;
    uint8_t lev[kSub];
    for (int j = 0; j < kSub; ++j) lev[j] = static_cast<uint8_t>(best_index(__fmul_rn(idl, xb[j])));
    uint8_t *qs = out + sb * kXsBytes + 8 + 16 * ib;
    for (int j = 0; j < 16; ++j) qs[j] = static_cast<uint8_t>(lev[j] | (lev[16 + j] << 4));
    s_l[t] = l + 32;
  }
  __syncthreads();
  if (valid && ib == 0) {
    uint8_t *p = out + sb * kXsBytes;
    store_half(p, s_d[g]);
    uint16_t sh = 0;
    for (int k = 0; k < 8; ++k) sh |= static_cast<uint16_t>((s_l[t + k] >> 4) << (2 * k));
    p[2] = static_cast<uint8_t>(sh);
    p[3] = static_cast<uint8_t>(sh >> 8);
    for (int k = 0; k < 4; ++k)
      p[4 + k] = static_cast<uint8_t>((s_l[t + 2 * k] & 0xF) | ((s_l[t + 2 * k + 1] & 0xF) << 4));
  }
}

__global__ void iq4_nl_encode(const float *x, int64_t nblocks, int64_t blocks_per_row, const float *importance,
                              uint8_t *out) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (b >= nblocks) return;
  float xb[kSub], w[kSub];
  for (int j = 0; j < kSub; ++j) xb[j] = x[b * kSub + j];
  const float *qw = importance ? importance + (b % blocks_per_row) * kSub : nullptr;
  weights(xb, kSub, xb, qw, w);
  const float s = search(xb, w);
  uint8_t *p = out + b * kNlBytes;
  store_half(p, s);
  const float inv = s != 0.f ? __fdiv_rn(1.f, s) : 0.f;
  uint8_t lev[kSub];
  for (int j = 0; j < kSub; ++j) lev[j] = static_cast<uint8_t>(best_index(__fmul_rn(inv, xb[j])));
  for (int j = 0; j < 16; ++j) p[2 + j] = static_cast<uint8_t>(lev[j] | (lev[16 + j] << 4));
}

at::Tensor pack(const char *name, int block, int nbytes, bool xs, at::Tensor input,
                std::optional<at::Tensor> importance) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous(),
              name, ": input must be a contiguous float32 CUDA [rows, cols] tensor");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % block == 0, name, ": cols must be a multiple of ", block);
  const float *imp = nullptr;
  if (importance.has_value()) {
    const auto &i = *importance;
    TORCH_CHECK(i.is_cuda() && i.scalar_type() == at::kFloat && i.dim() == 1 && i.numel() == cols &&
                    i.is_contiguous() && i.device() == input.device(),
                name, ": importance must be a contiguous float32 [cols] tensor on the input's device");
    imp = i.data_ptr<float>();
  }
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t bpr = cols / block, nblocks = rows * bpr;
  auto out = at::empty({rows, bpr, nbytes}, input.options().dtype(at::kByte));
  if (nblocks == 0) return out;
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (xs) {
    const int64_t grid = (nblocks + kThreads / 8 - 1) / (kThreads / 8);
    iq4_xs_encode<<<static_cast<unsigned>(grid), kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp,
                                                                         out.data_ptr<uint8_t>());
  } else {
    const int64_t grid = (nblocks + kThreads - 1) / kThreads;
    iq4_nl_encode<<<static_cast<unsigned>(grid), kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp,
                                                                         out.data_ptr<uint8_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

at::Tensor iq4_xs_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("IQ4_XS", kXsBlock, kXsBytes, true, input, importance);
}

at::Tensor iq4_nl_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("IQ4_NL", kSub, kNlBytes, false, input, importance);
}
