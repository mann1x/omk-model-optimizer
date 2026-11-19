// SPDX-License-Identifier: Apache-2.0
// K-quant (Q2_K..Q6_K) and legacy block-32 (Q4_0, Q4_1, Q5_0, Q5_1) CUDA encoders for omk_qad (bm-901): llama.cpp's
// quantize_row_<type>_impl (imatrix) and quantize_row_<type>_ref (no imatrix) from ggml-quants.c, operation by
// operation in float32, so the payload is
// byte-identical to libggml's (omk_qad/kquant.py has the derivation; libggml is the oracle). Every product, sum,
// quotient and square root goes through an explicit round-to-nearest intrinsic in llama.cpp's order (no fused
// multiply-add; the extension is also built with --fmad=false). libggml itself carries no FMA in these functions
// (objdump: 0 vfmadd in libggml-base.so on both hosts, 2026-11-19).
//
// Thread mapping: Q4_K / Q5_K -- 8 threads per 256-value super-block, one per 32-value sub-block (make_qkx3_quants
// with the imatrix, make_qkx2_quants without); lane 0 of the group runs make_qp_quants (imatrix) or the 63-level
// max rule (no imatrix) over the 8 scales and 8 mins, packs the 12 scale bytes and writes d / dmin; every thread
// then re-quantizes its sub-block with the fp16-rounded super-block scales and the pair-even threads pack the nibbles.
// Q6_K -- 16 threads per super-block, one per 16-value sub-block (make_qx_quants, rmse_type 1); lane 0 picks the
// signed max scale and writes d, every thread writes its int8 scale and re-quantizes, then packs 4 of the 32 columns
// of its half of the super-block. Q2_K / Q3_K -- 16 threads per super-block likewise (make_qkx3 + make_qp_quants /
// make_qx_quants over the 16 sub-block scales with the imatrix; make_qkx2 with |x| weights / make_q3_quants without).
// Legacy Q4_0 / Q4_1 / Q5_0 / Q5_1 -- one thread per 32-value block; with an imatrix the weights use the ROW's
// sigma2 (sum x^2 / n_per_row in llama.cpp's order), computed once per row by row_sigma2 before the encode kernel.
//
// fp16 fields are written as ONE aligned 16-bit store each (never two byte stores): ptxas 12.6 / 12.8 compiles a byte
// store of a cvt.f16 result to F2I.U8.F16, the half's VALUE truncated to an integer, zeroing the low byte of every scale
// (bug-832 / bug-966; gate the built .so with `cuobjdump -sass | grep -c F2I.U8.F16` == 0).

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <optional>

namespace {

constexpr int kQK = 256;         // QK_K
constexpr float kEps = 1e-15f;   // GROUP_MAX_EPS
constexpr int kThreads = 256;
constexpr int kQ4Bytes = 144;    // d, dmin, scales[12], qs[128]
constexpr int kQ5Bytes = 176;    // d, dmin, scales[12], qh[32], qs[128]
constexpr int kQ6Bytes = 210;    // ql[128], qh[64], scales[16], d
constexpr int kQ2Bytes = 84;     // scales[16], qs[64], d, dmin
constexpr int kQ3Bytes = 110;    // hmask[32], qs[64], scales[12], d
constexpr int kLeg = 32;         // legacy block
constexpr int kQ40Bytes = 18;    // d, qs[16]
constexpr int kQ41Bytes = 20;    // d, m, qs[16]
constexpr int kQ50Bytes = 22;    // d, qh[4], qs[16]
constexpr int kQ51Bytes = 24;    // d, m, qh[4], qs[16]

__device__ __forceinline__ int nearest_int(float f) { return __float2int_rn(f); }  // (f + 12582912.f) trick: RNE

// One 16-bit store of a float rounded to fp16 (GGML_FP32_TO_FP16), p 2-byte aligned.
__device__ __forceinline__ void store_half(uint8_t *p, float v) {
  *reinterpret_cast<uint16_t *>(p) = __half_as_ushort(__float2half_rn(v));
}
__device__ __forceinline__ float half_round(float v) { return __half2float(__float2half_rn(v)); }

// make_qkx2_quants (no imatrix) / make_qkx3_quants (imatrix): the least-squares scale + min of one sub-block.
// kx3 differs from kx2 in the degenerate test (max <= min vs max == min) and in the sum_w seed (weights[0] in both
// here: the callers always pass weights). Returns scale, sets the_min and L[n] (0..nmax).
template <bool kx3>
__device__ float make_qkx(int n, int nmax, const float *x, const float *weights, uint8_t *L, float &the_min,
                          float rmin, float rdelta, int nstep, bool use_mad = false) {
  float mn = x[0], mx = x[0];
  float sum_w = weights[0];
  float sum_x = __fmul_rn(sum_w, x[0]);
  for (int i = 1; i < n; ++i) {
    if (x[i] < mn) mn = x[i];
    if (x[i] > mx) mx = x[i];
    const float w = weights[i];
    sum_w = __fadd_rn(sum_w, w);
    sum_x = __fadd_rn(sum_x, __fmul_rn(w, x[i]));
  }
  if (mn > 0.f) mn = 0.f;
  if (kx3 ? (mx <= mn) : (mx == mn)) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    the_min = -mn;
    return 0.f;
  }
  float iscale = __fdiv_rn(static_cast<float>(nmax), __fsub_rn(mx, mn));
  float scale = __fdiv_rn(1.f, iscale);
  float best = 0.f;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(__fmul_rn(iscale, __fsub_rn(x[i], mn)));
    l = max(0, min(nmax, l));
    L[i] = static_cast<uint8_t>(l);
    float diff = __fsub_rn(__fadd_rn(__fmul_rn(scale, static_cast<float>(l)), mn), x[i]);
    diff = use_mad ? fabsf(diff) : __fmul_rn(diff, diff);
    best = __fadd_rn(best, __fmul_rn(weights[i], diff));
  }
  uint8_t Laux[32];
  for (int is = 0; is <= nstep; ++is) {
    iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, static_cast<float>(is))), static_cast<float>(nmax)),
                       __fsub_rn(mx, mn));
    float sum_l = 0.f, sum_l2 = 0.f, sum_xl = 0.f;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(__fmul_rn(iscale, __fsub_rn(x[i], mn)));
      l = max(0, min(nmax, l));
      Laux[i] = static_cast<uint8_t>(l);
      const float w = weights[i], lf = static_cast<float>(l);
      const float wl = __fmul_rn(w, lf);
      sum_l = __fadd_rn(sum_l, wl);
      sum_l2 = __fadd_rn(sum_l2, __fmul_rn(wl, lf));
      sum_xl = __fadd_rn(sum_xl, __fmul_rn(wl, x[i]));
    }
    const float D = __fsub_rn(__fmul_rn(sum_w, sum_l2), __fmul_rn(sum_l, sum_l));
    if (D > 0.f) {
      float this_scale = __fdiv_rn(__fsub_rn(__fmul_rn(sum_w, sum_xl), __fmul_rn(sum_x, sum_l)), D);
      float this_min = __fdiv_rn(__fsub_rn(__fmul_rn(sum_l2, sum_x), __fmul_rn(sum_l, sum_xl)), D);
      if (this_min > 0.f) {
        this_min = 0.f;
        this_scale = __fdiv_rn(sum_xl, sum_l2);
      }
      float err = 0.f;
      for (int i = 0; i < n; ++i) {
        float diff = __fsub_rn(__fadd_rn(__fmul_rn(this_scale, static_cast<float>(Laux[i])), this_min), x[i]);
        diff = use_mad ? fabsf(diff) : __fmul_rn(diff, diff);
        err = __fadd_rn(err, __fmul_rn(weights[i], diff));
      }
      if (err < best) {
        for (int i = 0; i < n; ++i) L[i] = Laux[i];
        best = err;
        scale = this_scale;
        mn = this_min;
      }
    }
  }
  the_min = -mn;
  return scale;
}

// make_qp_quants: the 6-bit super-block quantization of 8 positive scales (or mins), weights sw. L is uint8_t as in
// C: a negative nearest_int wraps, and the wrapped value is what the refinement loop reads back.
__device__ float make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *qw) {
  float mx = 0.f;
  for (int i = 0; i < n; ++i) mx = x[i] > mx ? x[i] : mx;  // MAX(max, x[i])
  if (mx < kEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  float iscale = __fdiv_rn(static_cast<float>(nmax), mx);
  for (int i = 0; i < n; ++i) L[i] = static_cast<uint8_t>(nearest_int(__fmul_rn(iscale, x[i])));
  const float scale = __fdiv_rn(1.f, iscale);
  float best_mse = 0.f;
  for (int i = 0; i < n; ++i) {
    const float diff = __fsub_rn(x[i], __fmul_rn(scale, static_cast<float>(L[i])));
    best_mse = __fadd_rn(best_mse, __fmul_rn(__fmul_rn(qw[i], diff), diff));
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0) continue;
    const float iscale_is = __fdiv_rn(__fadd_rn(__fmul_rn(0.1f, static_cast<float>(is)), static_cast<float>(nmax)), mx);
    const float scale_is = __fdiv_rn(1.f, iscale_is);
    float mse = 0.f;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(__fmul_rn(iscale_is, x[i]));
      l = min(nmax, l);
      const float diff = __fsub_rn(x[i], __fmul_rn(scale_is, static_cast<float>(l)));
      mse = __fadd_rn(mse, __fmul_rn(__fmul_rn(qw[i], diff), diff));
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0.f, suml2 = 0.f;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(__fmul_rn(iscale, x[i]));
    l = min(nmax, l);
    L[i] = static_cast<uint8_t>(l);
    const float w = qw[i], lf = static_cast<float>(l);
    sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), lf));
    suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, lf), lf));
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      const float w = qw[i], Lf = static_cast<float>(L[i]);
      float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), Lf));
      float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, Lf), Lf));
      if (slx > 0.f && sl2 > 0.f) {
        int new_l = nearest_int(__fdiv_rn(__fmul_rn(x[i], sl2), slx));
        new_l = min(nmax, new_l);
        if (new_l != static_cast<int>(L[i])) {
          const float nf = static_cast<float>(new_l);
          slx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), nf));
          sl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, nf), nf));
          if (__fmul_rn(__fmul_rn(slx, slx), suml2) > __fmul_rn(__fmul_rn(sumlx, sumlx), sl2)) {
            L[i] = static_cast<uint8_t>(new_l);
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) break;
  }
  return suml2 > 0.f ? __fdiv_rn(sumlx, suml2) : 0.f;
}

// make_qx_quants(n, nmax, x, L, rmse_type = 1, qw): the signed scale of a Q6_K 16-value sub-block, L in 0..2*nmax-1.
__device__ float make_qx_quants(int n, int nmax, const float *x, uint8_t *L, const float *qw) {
  float amax = 0.f, mx = 0.f;
  for (int i = 0; i < n; ++i) {
    const float ax = fabsf(x[i]);
    if (ax > amax) { amax = ax; mx = x[i]; }
  }
  if (amax < kEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  float iscale = __fdiv_rn(static_cast<float>(-nmax), mx);
  float sumlx = 0.f, suml2 = 0.f;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(__fmul_rn(iscale, x[i]));
    l = max(-nmax, min(nmax - 1, l));
    L[i] = static_cast<uint8_t>(l + nmax);
    const float w = qw ? qw[i] : __fmul_rn(x[i], x[i]), lf = static_cast<float>(l);
    sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), lf));
    suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, lf), lf));
  }
  float scale = suml2 != 0.f ? __fdiv_rn(sumlx, suml2) : 0.f;
  float best = __fmul_rn(scale, sumlx);
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) continue;
    iscale = __fdiv_rn(-__fadd_rn(static_cast<float>(nmax), __fmul_rn(0.1f, static_cast<float>(is))), mx);
    sumlx = 0.f;
    suml2 = 0.f;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(__fmul_rn(iscale, x[i]));
      l = max(-nmax, min(nmax - 1, l));
      const float w = qw ? qw[i] : __fmul_rn(x[i], x[i]), lf = static_cast<float>(l);
      sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), lf));
      suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, lf), lf));
    }
    if (suml2 > 0.f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
      for (int i = 0; i < n; ++i) {
        int l = nearest_int(__fmul_rn(iscale, x[i]));
        L[i] = static_cast<uint8_t>(nmax + max(-nmax, min(nmax - 1, l)));
      }
      scale = __fdiv_rn(sumlx, suml2);
      best = __fmul_rn(scale, sumlx);
    }
  }
  return scale;
}

// get_scale_min_k4
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// The 12 packed scale bytes from 8 six-bit scales ls and mins lm (all 12 bytes are assigned before any |=).
__device__ __forceinline__ void pack_scales(const uint8_t *ls, const uint8_t *lm, uint8_t *sc) {
  for (int j = 0; j < 8; ++j) {
    if (j < 4) {
      sc[j] = ls[j];
      sc[j + 4] = lm[j];
    } else {
      sc[j + 4] = static_cast<uint8_t>((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
      sc[j - 4] = static_cast<uint8_t>(sc[j - 4] | ((ls[j] >> 4) << 6));
      sc[j] = static_cast<uint8_t>(sc[j] | ((lm[j] >> 4) << 6));
    }
  }
}

// Q4_K (kFive = false) and Q5_K (kFive = true): 8 threads per super-block.
template <bool kFive>
__global__ void qk45_encode(const float *__restrict__ x, int64_t nblocks, int64_t blocks_per_row,
                            const float *__restrict__ importance, uint8_t *__restrict__ out) {
  constexpr int nmax = kFive ? 31 : 15;
  constexpr int nbytes = kFive ? kQ5Bytes : kQ4Bytes;
  constexpr int kGroups = kThreads / 8;
  __shared__ float s_scale[kThreads], s_min[kThreads], s_sw[kThreads];
  __shared__ uint8_t s_L[kGroups][kQK];
  __shared__ uint8_t s_sc[kGroups][12];
  __shared__ float s_d[kGroups], s_dmin[kGroups];
  const int t = threadIdx.x, g = t / 8, j = t % 8;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  float xb[32], w[32];
  uint8_t L[32];
  if (valid) {
    const float *blk = x + sb * kQK;
    for (int l = 0; l < 32; ++l) xb[l] = blk[32 * j + l];
    float sumw = 0.f, mn;
    if (importance) {
      float sum_x2 = 0.f;
      for (int l = 0; l < kQK; ++l) sum_x2 = __fadd_rn(sum_x2, __fmul_rn(blk[l], blk[l]));
      const float sigma2 = __fdiv_rn(__fmul_rn(2.f, sum_x2), static_cast<float>(kQK));
      const float *qw = importance + (sb % blocks_per_row) * kQK + 32 * j;
      for (int l = 0; l < 32; ++l) w[l] = __fmul_rn(qw[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
      for (int l = 0; l < 32; ++l) sumw = __fadd_rn(sumw, w[l]);
      s_scale[t] = make_qkx<true>(32, nmax, xb, w, L, mn, -0.9f, 0.05f, 36);
    } else {
      float sum_x2 = 0.f;
      for (int l = 0; l < 32; ++l) sum_x2 = __fadd_rn(sum_x2, __fmul_rn(xb[l], xb[l]));
      const float av_x = __fsqrt_rn(__fdiv_rn(sum_x2, 32.f));
      for (int l = 0; l < 32; ++l) w[l] = __fadd_rn(av_x, fabsf(xb[l]));
      s_scale[t] = kFive ? make_qkx<false>(32, nmax, xb, w, L, mn, -0.5f, 0.1f, 15)
                         : make_qkx<false>(32, nmax, xb, w, L, mn, -1.f, 0.1f, 20);
    }
    s_min[t] = mn;
    s_sw[t] = sumw;
  }
  __syncthreads();
  if (valid && j == 0) {
    uint8_t ls[8], lm[8];
    float d_block, m_block;
    if (importance) {
      d_block = make_qp_quants(8, 63, s_scale + t, ls, s_sw + t);
      m_block = make_qp_quants(8, 63, s_min + t, lm, s_sw + t);
      if (kFive) {
        for (int k = 0; k < 8; ++k) { ls[k] = min(63, static_cast<int>(ls[k])); lm[k] = min(63, static_cast<int>(lm[k])); }
      }
    } else {
      float max_scale = 0.f, max_min = 0.f;
      for (int k = 0; k < 8; ++k) {
        if (s_scale[t + k] > max_scale) max_scale = s_scale[t + k];
        if (s_min[t + k] > max_min) max_min = s_min[t + k];
      }
      const float inv_scale = max_scale > 0.f ? __fdiv_rn(63.f, max_scale) : 0.f;
      const float inv_min = max_min > 0.f ? __fdiv_rn(63.f, max_min) : 0.f;
      for (int k = 0; k < 8; ++k) {
        ls[k] = min(63, static_cast<int>(static_cast<uint8_t>(nearest_int(__fmul_rn(inv_scale, s_scale[t + k])))));
        lm[k] = min(63, static_cast<int>(static_cast<uint8_t>(nearest_int(__fmul_rn(inv_min, s_min[t + k])))));
      }
      d_block = __fdiv_rn(max_scale, 63.f);
      m_block = __fdiv_rn(max_min, 63.f);
    }
    pack_scales(ls, lm, s_sc[g]);
    uint8_t *p = out + sb * nbytes;
    store_half(p, d_block);
    store_half(p + 2, m_block);
    for (int k = 0; k < 12; ++k) p[4 + k] = s_sc[g][k];
    s_d[g] = half_round(d_block);
    s_dmin[g] = half_round(m_block);
  }
  __syncthreads();
  if (valid) {
    int sc, m;
    scale_min_k4(j, s_sc[g], sc, m);
    const float d = __fmul_rn(s_d[g], static_cast<float>(sc));
    if (d != 0.f) {
      const float dm = __fmul_rn(s_dmin[g], static_cast<float>(m));
      for (int l = 0; l < 32; ++l) {
        int q = nearest_int(__fdiv_rn(__fadd_rn(xb[l], dm), d));
        L[l] = static_cast<uint8_t>(max(0, min(nmax, q)));
      }
    }
    for (int l = 0; l < 32; ++l) s_L[g][32 * j + l] = L[l];
  }
  __syncthreads();
  if (valid) {
    uint8_t *p = out + sb * nbytes;
    const uint8_t *Lg = s_L[g];
    if (kFive) {
      // qh[32]: bit 2k from sub-block 2k, bit 2k+1 from sub-block 2k+1 (values > 15); thread j writes columns 4j..4j+3
      for (int c = 4 * j; c < 4 * j + 4; ++c) {
        int h = 0;
        for (int k = 0; k < 4; ++k) {
          if (Lg[64 * k + c] > 15) h |= 1 << (2 * k);
          if (Lg[64 * k + c + 32] > 15) h |= 2 << (2 * k);
        }
        p[16 + c] = static_cast<uint8_t>(h);
      }
      if ((j & 1) == 0) {
        const int k = j / 2;
        for (int c = 0; c < 32; ++c)
          p[48 + 32 * k + c] = static_cast<uint8_t>((Lg[64 * k + c] & 0xF) | ((Lg[64 * k + c + 32] & 0xF) << 4));
      }
    } else if ((j & 1) == 0) {
      const int k = j / 2;
      for (int c = 0; c < 32; ++c) p[16 + 32 * k + c] = static_cast<uint8_t>(Lg[64 * k + c] | (Lg[64 * k + c + 32] << 4));
    }
  }
}

// Q6_K: 16 threads per super-block, one per 16-value sub-block.
__global__ void q6k_encode(const float *__restrict__ x, int64_t nblocks, int64_t blocks_per_row,
                           const float *__restrict__ importance, uint8_t *__restrict__ out) {
  constexpr int kGroups = kThreads / 16;
  __shared__ float s_scale[kThreads];
  __shared__ uint8_t s_L[kGroups][kQK];
  __shared__ float s_iscale[kGroups], s_d[kGroups];
  __shared__ int s_zero[kGroups];
  const int t = threadIdx.x, g = t / 16, ib = t % 16;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  float xb[16];
  uint8_t L[16];
  if (valid) {
    const float *blk = x + sb * kQK;
    for (int l = 0; l < 16; ++l) xb[l] = blk[16 * ib + l];
    const float *qw = importance ? importance + (sb % blocks_per_row) * kQK + 16 * ib : nullptr;
    s_scale[t] = make_qx_quants(16, 32, xb, L, qw);
  }
  __syncthreads();
  if (valid && ib == 0) {
    float max_scale = 0.f, max_abs_scale = 0.f;
    for (int k = 0; k < 16; ++k) {
      const float a = fabsf(s_scale[t + k]);
      if (a > max_abs_scale) { max_abs_scale = a; max_scale = s_scale[t + k]; }
    }
    uint8_t *p = out + sb * kQ6Bytes;
    if (max_abs_scale < kEps) {
      s_zero[g] = 1;
      for (int k = 0; k < kQ6Bytes; ++k) p[k] = 0;  // memset(&y[i], 0, sizeof(block_q6_K)); d = fp16(0) = 0x0000
    } else {
      s_zero[g] = 0;
      const float iscale = __fdiv_rn(-128.f, max_scale);
      const float d = __fdiv_rn(1.f, iscale);
      store_half(p + 208, d);
      s_iscale[g] = iscale;
      s_d[g] = half_round(d);
    }
  }
  __syncthreads();
  if (valid && !s_zero[g]) {
    uint8_t *p = out + sb * kQ6Bytes;
    const int sc = min(127, nearest_int(__fmul_rn(s_iscale[g], s_scale[t])));
    p[192 + ib] = static_cast<uint8_t>(static_cast<int8_t>(sc));
    const float d = __fmul_rn(s_d[g], static_cast<float>(static_cast<int8_t>(sc)));
    if (d != 0.f) {
      for (int l = 0; l < 16; ++l) {
        int q = nearest_int(__fdiv_rn(xb[l], d));
        q = max(-32, min(31, q));
        L[l] = static_cast<uint8_t>(q + 32);
      }
    }
    for (int l = 0; l < 16; ++l) s_L[g][16 * ib + l] = L[l];
  }
  __syncthreads();
  if (valid && !s_zero[g]) {
    uint8_t *p = out + sb * kQ6Bytes;
    const int h = ib / 8;  // half of the super-block (128 values)
    const uint8_t *Lh = s_L[g] + 128 * h;
    for (int c = (ib % 8) * 4; c < (ib % 8) * 4 + 4; ++c) {
      const uint8_t q1 = Lh[c] & 0xF, q2 = Lh[c + 32] & 0xF, q3 = Lh[c + 64] & 0xF, q4 = Lh[c + 96] & 0xF;
      p[64 * h + c] = static_cast<uint8_t>(q1 | (q3 << 4));
      p[64 * h + 32 + c] = static_cast<uint8_t>(q2 | (q4 << 4));
      p[128 + 32 * h + c] = static_cast<uint8_t>((Lh[c] >> 4) | ((Lh[c + 32] >> 4) << 2) | ((Lh[c + 64] >> 4) << 4) |
                                                 ((Lh[c + 96] >> 4) << 6));
    }
  }
}


// make_q3_quants(n, nmax, x, L, do_rmse = true): the Q3_K no-imatrix sub-block scale; L signed during the refinement
// (int8_t in C), nmax added at the end. Weights are x^2.
__device__ float make_q3_quants(int n, int nmax, const float *x, uint8_t *L) {
  float amax = 0.f, mx = 0.f;
  for (int i = 0; i < n; ++i) {
    const float ax = fabsf(x[i]);
    if (ax > amax) { amax = ax; mx = x[i]; }
  }
  if (amax < kEps) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  const float iscale = __fdiv_rn(static_cast<float>(-nmax), mx);
  int Ls[16];
  float sumlx = 0.f, suml2 = 0.f;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(__fmul_rn(iscale, x[i]));
    l = max(-nmax, min(nmax - 1, l));
    Ls[i] = l;
    const float w = __fmul_rn(x[i], x[i]), lf = static_cast<float>(l);
    sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), lf));
    suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, lf), lf));
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      const float w = __fmul_rn(x[i], x[i]), Lf = static_cast<float>(Ls[i]);
      float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), Lf));
      if (slx > 0.f) {
        float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, Lf), Lf));
        int new_l = nearest_int(__fdiv_rn(__fmul_rn(x[i], sl2), slx));
        new_l = max(-nmax, min(nmax - 1, new_l));
        if (new_l != Ls[i]) {
          const float nf = static_cast<float>(new_l);
          slx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), nf));
          sl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, nf), nf));
          if (sl2 > 0.f && __fmul_rn(__fmul_rn(slx, slx), suml2) > __fmul_rn(__fmul_rn(sumlx, sumlx), sl2)) {
            Ls[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) break;
  }
  for (int i = 0; i < n; ++i) L[i] = static_cast<uint8_t>(Ls[i] + nmax);
  return suml2 > 0.f ? __fdiv_rn(sumlx, suml2) : 0.f;
}

// The Q3_K 12 scale bytes from 16 six-bit values v (zeroed first; j < 8 low nibbles, j >= 8 high nibbles, top 2 bits
// in bytes 8..11).
__device__ __forceinline__ void pack_q3_scales(const int *v, uint8_t *sc) {
  for (int k = 0; k < 12; ++k) sc[k] = 0;
  for (int j = 0; j < 16; ++j) {
    int l = v[j];
    if (j < 8) sc[j] = static_cast<uint8_t>(l & 0xF);
    else sc[j - 8] = static_cast<uint8_t>(sc[j - 8] | ((l & 0xF) << 4));
    l >>= 4;
    sc[j % 4 + 8] = static_cast<uint8_t>(sc[j % 4 + 8] | (l << (2 * (j / 4))));
  }
}

__device__ __forceinline__ int q3_scale(int j, const uint8_t *sc) {
  int s = j < 8 ? (sc[j] & 0xF) : (sc[j - 8] >> 4);
  return static_cast<int>(static_cast<int8_t>(s | (((sc[8 + j % 4] >> (2 * (j / 4))) & 3) << 4))) - 32;
}

// Q2_K: 16 threads per super-block, one per 16-value sub-block.
__global__ void q2k_encode(const float *__restrict__ x, int64_t nblocks, int64_t blocks_per_row,
                           const float *__restrict__ importance, uint8_t *__restrict__ out) {
  constexpr int kGroups = kThreads / 16;
  __shared__ float s_scale[kThreads], s_min[kThreads], s_sw[kThreads];
  __shared__ uint8_t s_L[kGroups][kQK];
  __shared__ uint8_t s_sc[kGroups][16];
  __shared__ float s_d[kGroups], s_dmin[kGroups];
  const int t = threadIdx.x, g = t / 16, ib = t % 16;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  float xb[16], w[16];
  uint8_t L[16];
  if (valid) {
    const float *blk = x + sb * kQK;
    for (int l = 0; l < 16; ++l) xb[l] = blk[16 * ib + l];
    float mn, sumw = 0.f;
    if (importance) {
      float sumx2 = 0.f;
      for (int l = 0; l < kQK; ++l) sumx2 = __fadd_rn(sumx2, __fmul_rn(blk[l], blk[l]));
      const float sigma2 = __fdiv_rn(sumx2, static_cast<float>(kQK));
      const float *qw = importance + (sb % blocks_per_row) * kQK + 16 * ib;
      for (int l = 0; l < 16; ++l) w[l] = __fmul_rn(qw[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
      for (int l = 0; l < 16; ++l) sumw = __fadd_rn(sumw, w[l]);
      s_scale[t] = make_qkx<true>(16, 3, xb, w, L, mn, -0.9f, 0.05f, 36, false);
    } else {
      for (int l = 0; l < 16; ++l) w[l] = fabsf(xb[l]);
      s_scale[t] = make_qkx<false>(16, 3, xb, w, L, mn, -0.5f, 0.1f, 15, true);
    }
    s_min[t] = mn;
    s_sw[t] = sumw;
  }
  __syncthreads();
  if (valid && ib == 0) {
    uint8_t *p = out + sb * kQ2Bytes;
    float d_block, m_block;
    if (importance) {
      uint8_t Ls[16], Lm[16];
      d_block = make_qp_quants(16, 15, s_scale + t, Ls, s_sw + t);
      m_block = make_qp_quants(16, 15, s_min + t, Lm, s_sw + t);
      for (int j = 0; j < 16; ++j) s_sc[g][j] = static_cast<uint8_t>(Ls[j] | (Lm[j] << 4));
    } else {
      float max_scale = 0.f, max_min = 0.f;
      for (int k = 0; k < 16; ++k) {
        if (s_scale[t + k] > max_scale) max_scale = s_scale[t + k];
        if (s_min[t + k] > max_min) max_min = s_min[t + k];
      }
      for (int j = 0; j < 16; ++j) s_sc[g][j] = 0;
      if (max_scale > 0.f) {
        const float iscale = __fdiv_rn(15.f, max_scale);
        for (int j = 0; j < 16; ++j) s_sc[g][j] = static_cast<uint8_t>(nearest_int(__fmul_rn(iscale, s_scale[t + j])));
        d_block = __fdiv_rn(max_scale, 15.f);
      } else {
        d_block = 0.f;
      }
      if (max_min > 0.f) {
        const float iscale = __fdiv_rn(15.f, max_min);
        for (int j = 0; j < 16; ++j)
          s_sc[g][j] = static_cast<uint8_t>(s_sc[g][j] | (nearest_int(__fmul_rn(iscale, s_min[t + j])) << 4));
        m_block = __fdiv_rn(max_min, 15.f);
      } else {
        m_block = 0.f;
      }
    }
    store_half(p + 80, d_block);
    store_half(p + 82, m_block);
    for (int j = 0; j < 16; ++j) p[j] = s_sc[g][j];
    s_d[g] = half_round(d_block);
    s_dmin[g] = half_round(m_block);
  }
  __syncthreads();
  if (valid) {
    const int sc = s_sc[g][ib];
    const float d = __fmul_rn(s_d[g], static_cast<float>(sc & 0xF));
    if (d != 0.f) {
      const float m = __fmul_rn(s_dmin[g], static_cast<float>(sc >> 4));
      for (int l = 0; l < 16; ++l) {
        int q = nearest_int(__fdiv_rn(__fadd_rn(xb[l], m), d));
        L[l] = static_cast<uint8_t>(max(0, min(3, q)));
      }
    }
    for (int l = 0; l < 16; ++l) s_L[g][16 * ib + l] = L[l];
  }
  __syncthreads();
  if (valid) {
    uint8_t *p = out + sb * kQ2Bytes;
    const int h = ib / 8;
    const uint8_t *Lh = s_L[g] + 128 * h;
    for (int c = (ib % 8) * 4; c < (ib % 8) * 4 + 4; ++c)
      p[16 + 32 * h + c] = static_cast<uint8_t>(Lh[c] | (Lh[c + 32] << 2) | (Lh[c + 64] << 4) | (Lh[c + 96] << 6));
  }
}

// Q3_K: 16 threads per super-block, one per 16-value sub-block.
__global__ void q3k_encode(const float *__restrict__ x, int64_t nblocks, int64_t blocks_per_row,
                           const float *__restrict__ importance, uint8_t *__restrict__ out) {
  constexpr int kGroups = kThreads / 16;
  __shared__ float s_scale[kThreads], s_sw[kThreads];
  __shared__ uint8_t s_L[kGroups][kQK];
  __shared__ uint8_t s_sc[kGroups][12];
  __shared__ float s_d[kGroups];
  const int t = threadIdx.x, g = t / 16, ib = t % 16;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  float xb[16], w[16];
  uint8_t L[16];
  if (valid) {
    const float *blk = x + sb * kQK;
    for (int l = 0; l < 16; ++l) xb[l] = blk[16 * ib + l];
    if (importance) {
      float sumx2 = 0.f;
      for (int l = 0; l < kQK; ++l) sumx2 = __fadd_rn(sumx2, __fmul_rn(blk[l], blk[l]));
      const float sigma2 = __fdiv_rn(__fmul_rn(2.f, sumx2), static_cast<float>(kQK));
      const float *qw = importance + (sb % blocks_per_row) * kQK + 16 * ib;
      for (int l = 0; l < 16; ++l) w[l] = __fmul_rn(qw[l], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[l], xb[l]))));
      float sumw = 0.f;
      for (int l = 0; l < 16; ++l) sumw = __fadd_rn(sumw, w[l]);
      s_sw[t] = sumw;
      s_scale[t] = make_qx_quants(16, 4, xb, L, w);
    } else {
      s_scale[t] = make_q3_quants(16, 4, xb, L);
    }
  }
  __syncthreads();
  if (valid && ib == 0) {
    uint8_t *p = out + sb * kQ3Bytes;
    int v[16];
    float d_block;
    if (importance) {
      uint8_t Ls[16];
      d_block = make_qx_quants(16, 32, s_scale + t, Ls, s_sw + t);
      for (int j = 0; j < 16; ++j) v[j] = Ls[j];
    } else {
      float max_scale = 0.f, amax = 0.f;
      for (int k = 0; k < 16; ++k) {
        const float a = fabsf(s_scale[t + k]);
        if (a > amax) { amax = a; max_scale = s_scale[t + k]; }
      }
      if (max_scale != 0.f) {
        const float iscale = __fdiv_rn(-32.f, max_scale);
        for (int j = 0; j < 16; ++j) {
          int l = static_cast<int8_t>(nearest_int(__fmul_rn(iscale, s_scale[t + j])));
          v[j] = max(-32, min(31, l)) + 32;
        }
        d_block = __fdiv_rn(1.f, iscale);
      } else {
        for (int j = 0; j < 16; ++j) v[j] = 0;
        d_block = 0.f;
      }
    }
    pack_q3_scales(v, s_sc[g]);
    store_half(p + 108, d_block);
    for (int k = 0; k < 12; ++k) p[96 + k] = s_sc[g][k];
    s_d[g] = half_round(d_block);
  }
  __syncthreads();
  if (valid) {
    const int sc = q3_scale(ib, s_sc[g]);
    const float d = __fmul_rn(s_d[g], static_cast<float>(sc));
    if (d != 0.f) {
      for (int l = 0; l < 16; ++l) {
        int q = nearest_int(__fdiv_rn(xb[l], d));
        L[l] = static_cast<uint8_t>(max(-4, min(3, q)) + 4);
      }
    }
    for (int l = 0; l < 16; ++l) s_L[g][16 * ib + l] = L[l];
  }
  __syncthreads();
  if (valid) {
    uint8_t *p = out + sb * kQ3Bytes;
    const uint8_t *Lg = s_L[g];
    // hmask[m] bit b = value 32 b + m is > 3; thread ib owns m = 2 ib, 2 ib + 1
    for (int m = 2 * ib; m < 2 * ib + 2; ++m) {
      int hm = 0;
      for (int b = 0; b < 8; ++b)
        if (Lg[32 * b + m] > 3) hm |= 1 << b;
      p[m] = static_cast<uint8_t>(hm);
    }
    const int h = ib / 8;
    const uint8_t *Lh = Lg + 128 * h;
    for (int c = (ib % 8) * 4; c < (ib % 8) * 4 + 4; ++c)
      p[32 + 32 * h + c] = static_cast<uint8_t>((Lh[c] & 3) | ((Lh[c + 32] & 3) << 2) | ((Lh[c + 64] & 3) << 4) |
                                                 ((Lh[c + 96] & 3) << 6));
  }
}

// sum x^2 / n of every row, accumulated in order (the legacy _impl paths' sigma2).
__global__ void row_sigma2(const float *__restrict__ x, int64_t rows, int64_t cols, float *__restrict__ sig) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float *xr = x + r * cols;
  float s = 0.f;
  for (int64_t j = 0; j < cols; ++j) s = __fadd_rn(s, __fmul_rn(xr[j], xr[j]));
  sig[r] = __fdiv_rn(s, static_cast<float>(cols));
}

// float -> int8_t -> uint8_t as the legacy _ref paths cast ((int8_t)(x0 + 8.5f) etc.): truncation toward zero.
__device__ __forceinline__ int trunc_i8(float f) { return static_cast<int>(static_cast<int8_t>(__float2int_rz(f))); }

// Legacy block-32 types, one thread per block. kType: 40, 41, 50, 51.
template <int kType>
__global__ void legacy_encode(const float *__restrict__ x, int64_t nblocks, int64_t blocks_per_row,
                              const float *__restrict__ importance, const float *__restrict__ sigma2,
                              uint8_t *__restrict__ out) {
  constexpr bool has_min = kType == 41 || kType == 51;
  constexpr bool five = kType == 50 || kType == 51;
  constexpr int nbytes = kType == 40 ? kQ40Bytes : kType == 41 ? kQ41Bytes : kType == 50 ? kQ50Bytes : kQ51Bytes;
  constexpr int qs_off = 2 + (has_min ? 2 : 0) + (five ? 4 : 0);
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const float *xb = x + b * kLeg;
  uint8_t *p = out + b * nbytes;
  uint8_t L[kLeg];
  if (importance) {
    const float s2 = sigma2[b / blocks_per_row];
    const float *qw = importance + (b % blocks_per_row) * kLeg;
    float w[kLeg];
    for (int j = 0; j < kLeg; ++j) w[j] = __fmul_rn(qw[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    if (has_min) {
      float mn;
      const float d = make_qkx<true>(kLeg, five ? 31 : 15, xb, w, L, mn, -0.9f, 0.05f, 36, false);
      store_half(p, d);
      store_half(p + 2, -mn);
    } else {
      const float d = make_qx_quants(kLeg, five ? 16 : 8, xb, L, w);
      store_half(p, d);
    }
  } else if (has_min) {
    float mn = 3.402823466e+38f, mx = -3.402823466e+38f;  // FLT_MAX / -FLT_MAX
    for (int j = 0; j < kLeg; ++j) {
      if (xb[j] < mn) mn = xb[j];
      if (xb[j] > mx) mx = xb[j];
    }
    const float d = __fdiv_rn(__fsub_rn(mx, mn), five ? 31.f : 15.f);
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    store_half(p, d);
    store_half(p + 2, mn);
    for (int j = 0; j < kLeg; ++j) {
      const float x0 = __fmul_rn(__fsub_rn(xb[j], mn), id);
      const int q = trunc_i8(__fadd_rn(x0, 0.5f));
      L[j] = static_cast<uint8_t>(five ? q : min(15, q));  // Q5_1_ref casts to uint8_t without the MIN
    }
  } else {
    float amax = 0.f, mx = 0.f;
    for (int j = 0; j < kLeg; ++j) {
      if (amax < fabsf(xb[j])) { amax = fabsf(xb[j]); mx = xb[j]; }
    }
    const float d = __fdiv_rn(mx, five ? -16.f : -8.f);
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    store_half(p, d);
    for (int j = 0; j < kLeg; ++j) {
      const float x0 = __fmul_rn(xb[j], id);
      L[j] = static_cast<uint8_t>(min(five ? 31 : 15, trunc_i8(__fadd_rn(x0, five ? 16.5f : 8.5f))));
    }
  }
  if (five) {
    uint32_t qh = 0;
    for (int j = 0; j < 16; ++j) {
      p[qs_off + j] = static_cast<uint8_t>((L[j] & 0x0F) | ((L[j + 16] & 0x0F) << 4));
      qh |= static_cast<uint32_t>((L[j] & 0x10u) >> 4) << j;
      qh |= static_cast<uint32_t>((L[j + 16] & 0x10u) >> 4) << (j + 16);
    }
    for (int k = 0; k < 4; ++k) p[qs_off - 4 + k] = static_cast<uint8_t>(qh >> (8 * k));  // little-endian memcpy
  } else {
    for (int j = 0; j < 16; ++j) p[qs_off + j] = static_cast<uint8_t>(L[j] | (L[j + 16] << 4));
  }
}

at::Tensor pack(const char *name, int block, int nbytes, int which, at::Tensor input,
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
  const float *xp = input.data_ptr<float>();
  uint8_t *op = out.data_ptr<uint8_t>();
  if (which >= 40) {  // legacy block-32
    at::Tensor sig;
    const float *sp = nullptr;
    if (imp) {
      sig = at::empty({rows}, input.options());
      row_sigma2<<<static_cast<unsigned>((rows + kThreads - 1) / kThreads), kThreads, 0, stream>>>(xp, rows, cols,
                                                                                                    sig.data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      sp = sig.data_ptr<float>();
    }
    const unsigned grid = static_cast<unsigned>((nblocks + kThreads - 1) / kThreads);
    switch (which) {
      case 40: legacy_encode<40><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, sp, op); break;
      case 41: legacy_encode<41><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, sp, op); break;
      case 50: legacy_encode<50><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, sp, op); break;
      default: legacy_encode<51><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, sp, op); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  const int per_cta = (which == 4 || which == 5) ? kThreads / 8 : kThreads / 16;
  const unsigned grid = static_cast<unsigned>((nblocks + per_cta - 1) / per_cta);
  switch (which) {
    case 2: q2k_encode<<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, op); break;
    case 3: q3k_encode<<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, op); break;
    case 4: qk45_encode<false><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, op); break;
    case 5: qk45_encode<true><<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, op); break;
    default: q6k_encode<<<grid, kThreads, 0, stream>>>(xp, nblocks, bpr, imp, op); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

at::Tensor q2_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q2_K", kQK, kQ2Bytes, 2, input, importance);
}

at::Tensor q3_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q3_K", kQK, kQ3Bytes, 3, input, importance);
}

at::Tensor q4_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q4_K", kQK, kQ4Bytes, 4, input, importance);
}

at::Tensor q5_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q5_K", kQK, kQ5Bytes, 5, input, importance);
}

at::Tensor q6_k_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q6_K", kQK, kQ6Bytes, 6, input, importance);
}

at::Tensor q4_0_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q4_0", kLeg, kQ40Bytes, 40, input, importance);
}

at::Tensor q4_1_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q4_1", kLeg, kQ41Bytes, 41, input, importance);
}

at::Tensor q5_0_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q5_0", kLeg, kQ50Bytes, 50, input, importance);
}

at::Tensor q5_1_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance) {
  return pack("Q5_1", kLeg, kQ51Bytes, 51, input, importance);
}
