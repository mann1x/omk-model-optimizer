// SPDX-License-Identifier: Apache-2.0
// IQ3_XXS / IQ3_S CUDA encoder for omk_qad: llama.cpp's quantize_row_iq3_xxs_impl / quantize_row_iq3_s_impl
// (ggml-quants.c), operation by operation in float32, so the payload is byte-identical to libggml's (omk_qad/iq3.py
// has the derivation, the lattice tables and the torch reference). Every product, sum, quotient and square root goes
// through an explicit round-to-nearest intrinsic (no fused multiply-add; the extension is also built with
// --fmad=false), in llama.cpp's order.
//
// 8 threads per 256-value super-block, one per 32-value sub-block (signs, 31 / 19 trial scales with grid snapping,
// re-snap), then the group sets the fp16 super-block scale and the 4-bit sub-block scales. The tables (packed grid,
// kmap, neighbour lists) come from omk_qad.iq3.tables, built exactly as iq3xs_init_impl builds them.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <optional>

namespace {

constexpr int kSub = 32;
constexpr int kMaxQ = 8;
constexpr int kThreads = 256;
constexpr int kGroups = kThreads / 8;  // super-blocks per CUDA block
constexpr int kXxsBytes = 98;
constexpr int kSBytes = 110;
constexpr float kEpsXxs = 1e-8f;  // GROUP_MAX_EPS_IQ3_XXS

struct Tables {
  const uint32_t *grid;  // packed int8 odd levels (iq3_data.grid)
  const int *kmap;       // 4096: grid index, or -(neighbour offset + 1)
  const int *neigh;      // [count, index...] per off-grid code
};

__device__ __forceinline__ int level(float id, float x) {  // nearest_int(0.5f*(id*x - 1)), clamped to 0..kMaxQ-1
  const int l = __float2int_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, x), 1.f)));
  return max(0, min(kMaxQ - 1, l));
}

__device__ __forceinline__ int grid_level(uint32_t g, int i) { return (static_cast<int>((g >> (8 * i)) & 0xFF) - 1) / 2; }

// iq3_find_best_neighbour: the best neighbour of off-grid code u at `scale` (levels written to L), its grid index.
__device__ int best_neighbour(const Tables &t, int u, const float *xval, const float *w, float scale, int8_t *L) {
  const int *nb = t.neigh + (-t.kmap[u] - 1);
  const int n = nb[0];
  float best_d2 = FLT_MAX;
  int gi = -1;
  for (int j = 1; j <= n; ++j) {
    const uint32_t g = t.grid[nb[j]];
    float d2 = 0.f;
    for (int i = 0; i < 4; ++i) {
      const float q = static_cast<float>((g >> (8 * i)) & 0xFF);
      const float diff = __fsub_rn(__fmul_rn(scale, q), xval[i]);
      d2 = __fadd_rn(d2, __fmul_rn(__fmul_rn(w[i], diff), diff));
    }
    if (d2 < best_d2) { best_d2 = d2; gi = nb[j]; }
  }
  for (int i = 0; i < 4; ++i) L[i] = static_cast<int8_t>(grid_level(t.grid[gi], i));
  return gi;
}

__device__ __forceinline__ int code(const int8_t *L) { return L[0] | (L[1] << 3) | (L[2] << 6) | (L[3] << 9); }

__device__ __forceinline__ void sums(const float *w, const float *xval, const int8_t *L, float &sumqx, float &sumq2) {
  sumqx = 0.f;
  sumq2 = 0.f;
  for (int i = 0; i < kSub; ++i) {
    const float q = static_cast<float>(2 * L[i] + 1);
    sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w[i], xval[i]), q));
    sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w[i], q), q));
  }
}

__device__ __forceinline__ void store_half(uint8_t *p, float v) {
  // one 16-bit store: ptxas 12.6 miscompiles a byte store of a cvt.f16 result (see iq4.cu)
  *reinterpret_cast<uint16_t *>(p) = __half_as_ushort(__float2half_rn(v));
}

template <bool kS>
__global__ void iq3_encode(const float *x, int64_t nblocks, int64_t blocks_per_row, const float *importance, Tables t,
                           uint8_t *out) {
  __shared__ float s_scale[kThreads];
  __shared__ bool s_skip[kThreads];
  __shared__ float s_id[kGroups];
  __shared__ bool s_live[kGroups];
  const int th = threadIdx.x, g = th / 8, ib = th % 8;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  constexpr int kNbytes = kS ? kSBytes : kXxsBytes;
  constexpr float kFudge = kS ? 1.033f : 1.0125f;
  float xb[kSub], xval[kSub], weight[kSub], waux[kSub];
  int8_t L[kSub], Laux[kSub];
  bool on[8], on_aux[8];
  uint8_t sgn[4] = {0, 0, 0, 0};
  int gidx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float scale = 0.f;
  bool skip = true;
  if (valid) {
    const float *blk = x + sb * 256;
    float sumx2 = 0.f;
    for (int i = 0; i < 256; ++i) sumx2 = __fadd_rn(sumx2, __fmul_rn(blk[i], blk[i]));
    const float sigma2 = __fdiv_rn(__fmul_rn(2.f, sumx2), 256.f);
    for (int i = 0; i < kSub; ++i) xb[i] = blk[ib * kSub + i];
    const float *qw = importance ? importance + (sb % blocks_per_row) * 256 + ib * kSub : nullptr;
    for (int i = 0; i < kSub; ++i)
      weight[i] = qw ? __fmul_rn(qw[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))))
                     : __fmul_rn(xb[i], xb[i]);
    for (int i = 0; i < kSub; ++i) waux[i] = __fsqrt_rn(weight[i]);
    for (int k = 0; k < 4; ++k) {
      int nflip = 0;
      uint8_t s = 0;
      for (int i = 0; i < 8; ++i) {
        const float v = xb[8 * k + i];
        if (v >= 0) {
          xval[8 * k + i] = v;
        } else {
          xval[8 * k + i] = -v;
          ++nflip;
          s |= static_cast<uint8_t>(1 << i);
        }
      }
      if (!kS && (nflip % 2)) {  // IQ3_XXS: an even number of flips
        int imin = 0;
        float mn = __fmul_rn(__fmul_rn(weight[8 * k], xb[8 * k]), xb[8 * k]);
        for (int i = 1; i < 8; ++i) {
          const float ax = __fmul_rn(__fmul_rn(weight[8 * k + i], xb[8 * k + i]), xb[8 * k + i]);
          if (ax < mn) { mn = ax; imin = i; }
        }
        xval[8 * k + imin] = -xval[8 * k + imin];
        s ^= static_cast<uint8_t>(1 << imin);
      }
      sgn[k] = kS ? s : static_cast<uint8_t>(s & 127);
    }
    float mx = xval[0];
    for (int i = 1; i < kSub; ++i) mx = mx > xval[i] ? mx : xval[i];
    for (int i = 0; i < kSub; ++i) L[i] = 0;
    skip = kS ? !(mx != 0.f) : mx < kEpsXxs;
    if (!skip) {
      float best = 0.f;
      scale = __fdiv_rn(mx, static_cast<float>(2 * kMaxQ - 1));
      for (int k = 0; k < 8; ++k) on[k] = !kS;
      const int lo = kS ? -9 : -15;
      for (int is = lo; is <= -lo; ++is) {
        const float id = __fdiv_rn(__fadd_rn(static_cast<float>(2 * kMaxQ - 1), __fmul_rn(static_cast<float>(is), 0.2f)),
                                   mx);
        const float this_scale = __fdiv_rn(1.f, id);
        for (int k = 0; k < 8; ++k) {
          for (int i = 0; i < 4; ++i) Laux[4 * k + i] = static_cast<int8_t>(level(id, xval[4 * k + i]));
          const int u = code(Laux + 4 * k);
          on_aux[k] = t.kmap[u] >= 0;
          if (!on_aux[k]) best_neighbour(t, u, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
        }
        float sumqx, sumq2;
        sums(weight, xval, Laux, sumqx, sumq2);
        if (sumq2 > 0.f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
          scale = __fdiv_rn(sumqx, sumq2);
          best = __fmul_rn(scale, sumqx);
          for (int i = 0; i < kSub; ++i) L[i] = Laux[i];
          for (int k = 0; k < 8; ++k) on[k] = on_aux[k];
        }
      }
      int n_off = 0;
      for (int k = 0; k < 8; ++k) n_off += on[k] ? 0 : 1;
      if (n_off > 0 && scale > 0.f) {
        const float id = __fdiv_rn(1.f, scale);
        for (int k = 0; k < 8; ++k) {
          if (!kS && on[k]) continue;  // IQ3_S re-snaps every group
          int8_t l4[4];
          for (int i = 0; i < 4; ++i) l4[i] = static_cast<int8_t>(level(id, xval[4 * k + i]));
          const int u = code(l4);
          int gi = t.kmap[u];
          if (gi < 0) gi = best_neighbour(t, u, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
          for (int i = 0; i < 4; ++i) L[4 * k + i] = static_cast<int8_t>(grid_level(t.grid[gi], i));
        }
        float sumqx, sumq2;
        sums(weight, xval, L, sumqx, sumq2);
        if (sumq2 > 0.f) scale = __fdiv_rn(sumqx, sumq2);
      }
      if (scale < 0.f) {
        scale = -scale;
        for (int k = 0; k < 4; ++k) sgn[k] = kS ? static_cast<uint8_t>(~sgn[k]) : static_cast<uint8_t>(~sgn[k] & 127);
      }
      for (int k = 0; k < 8; ++k) gidx[k] = t.kmap[code(L + 4 * k)];
    }
    s_scale[th] = skip ? 0.f : scale;
    s_skip[th] = skip;
  }
  __syncthreads();
  if (valid && ib == 0) {
    float max_scale = 0.f;
    for (int k = 0; k < 8; ++k) max_scale = max_scale > s_scale[th + k] ? max_scale : s_scale[th + k];
    s_live[g] = max_scale != 0.f;
    if (s_live[g]) {
      const float d = __fdiv_rn(max_scale, 31.f);
      store_half(out + sb * kNbytes, __fmul_rn(d, kFudge));
      s_id[g] = __fdiv_rn(1.f, d);
    }
  }
  __syncthreads();
  if (!valid) return;
  uint8_t *p = out + sb * kNbytes;
  const bool live = s_live[g];
  auto sub_scale = [&](int k) {  // the 4-bit scale of sub-block k of this super-block: nearest_int(0.5f*(id*s - 1))
    return max(0, min(15, __float2int_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(s_id[g], s_scale[th - ib + k]), 1.f)))));
  };
  if (!kS) {
    const int l = live ? sub_scale(ib) : 0;
    if (!live) return;  // an all-zero super-block is all zero bytes
    for (int k = 0; k < 8; ++k) p[2 + 8 * ib + k] = skip ? 0 : static_cast<uint8_t>(gidx[k]);
    uint32_t sas = static_cast<uint32_t>(l) << 28;
    if (!skip) sas |= sgn[0] | (sgn[1] << 7) | (sgn[2] << 14) | (sgn[3] << 21);
    for (int b = 0; b < 4; ++b) p[66 + 4 * ib + b] = static_cast<uint8_t>(sas >> (8 * b));
    return;
  }
  if (!skip) {  // qs / signs at the running cursor (a skipped sub-block does not advance it), qh at ib
    int rank = 0;
    for (int k = 0; k < ib; ++k) rank += s_skip[th - ib + k] ? 0 : 1;
    uint8_t qh = 0;
    for (int k = 0; k < 8; ++k) {
      p[2 + 8 * rank + k] = static_cast<uint8_t>(gidx[k] & 255);
      qh |= static_cast<uint8_t>(((gidx[k] >> 8) & 1) << k);
    }
    p[66 + ib] = qh;
    for (int k = 0; k < 4; ++k) p[74 + 4 * rank + k] = sgn[k];
  }
  if (live && ib == 0)
    for (int k = 0; k < 4; ++k) p[106 + k] = static_cast<uint8_t>(sub_scale(2 * k) | (sub_scale(2 * k + 1) << 4));
}

}  // namespace

at::Tensor iq3_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance, bool s, at::Tensor grid,
                         at::Tensor kmap, at::Tensor neigh) {
  const char *name = s ? "IQ3_S" : "IQ3_XXS";
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous(),
              name, ": input must be a contiguous float32 CUDA [rows, cols] tensor");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % 256 == 0, name, ": cols must be a multiple of 256");
  for (const auto *tab : {&grid, &kmap, &neigh})
    TORCH_CHECK(tab->is_cuda() && tab->scalar_type() == at::kInt && tab->is_contiguous() &&
                    tab->device() == input.device(),
                name, ": tables must be contiguous int32 tensors on the input's device");
  TORCH_CHECK(grid.numel() == (s ? 512 : 256) && kmap.numel() == 4096, name, ": wrong table sizes");
  const float *imp = nullptr;
  if (importance.has_value()) {
    const auto &i = *importance;
    TORCH_CHECK(i.is_cuda() && i.scalar_type() == at::kFloat && i.dim() == 1 && i.numel() == cols &&
                    i.is_contiguous() && i.device() == input.device(),
                name, ": importance must be a contiguous float32 [cols] tensor on the input's device");
    imp = i.data_ptr<float>();
  }
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t bpr = cols / 256, nblocks = rows * bpr;
  const int nbytes = s ? kSBytes : kXxsBytes;
  auto out = at::zeros({rows, bpr, nbytes}, input.options().dtype(at::kByte));
  if (nblocks == 0) return out;
  const Tables t{reinterpret_cast<const uint32_t *>(grid.data_ptr<int>()), kmap.data_ptr<int>(), neigh.data_ptr<int>()};
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const unsigned nblk = static_cast<unsigned>((nblocks + kGroups - 1) / kGroups);
  if (s)
    iq3_encode<true><<<nblk, kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp, t,
                                                     out.data_ptr<uint8_t>());
  else
    iq3_encode<false><<<nblk, kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp, t,
                                                      out.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
