// SPDX-License-Identifier: Apache-2.0
// IQ2_XXS / IQ2_S CUDA encoder for omk_qad: llama.cpp's quantize_row_iq2_xxs_impl / quantize_row_iq2_s_impl
// (ggml-quants.c), operation by operation in float32, so the payload is byte-identical to libggml's (omk_qad/iq2.py
// has the derivation, the lattice tables and the torch reference). Every product, sum, quotient and square root goes
// through an explicit round-to-nearest intrinsic (no fused multiply-add; the extension is also built with
// --fmad=false), in llama.cpp's order.
//
// One thread per sub-block: IQ2_XXS 8 threads x 32 values per 256-value super-block (4 groups of 8 with parity
// signs, make_qp_quants seed scale, 13 trial scales, every group re-snapped), IQ2_S 16 threads x 16 values
// (2 groups of 8 with full sign bytes, 19 trial scales, off-grid groups re-snapped). Then the super-block sets the
// fp16 scale (IQ2_S x 0.9875) and the 4-bit sub-block scales. The tables (grid of odd levels, kmap over the 43692
// 2-bit codes, neighbour lists) come from omk_qad.iq2.tables, built exactly as iq2xs_init_impl builds them.

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <optional>

namespace {

constexpr int kMaxQ = 3;
constexpr int kThreads = 256;
constexpr int kXxsBytes = 66;
constexpr int kSBytes = 82;
constexpr int kKmap = 43692;

struct Tables {
  const uint8_t *grid;  // [n, 8] odd levels 1 / 3 / 5 (iq2_data.grid as bytes)
  const int *kmap;      // 43692: grid index, or -(neighbour offset + 1)
  const int *neigh;     // [count, index...] per off-grid code
};

__device__ __forceinline__ int level(float id, float x) {  // nearest_int(0.5f*(id*x - 1)), clamped to 0..kMaxQ-1
  const int l = __float2int_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(id, x), 1.f)));
  return max(0, min(kMaxQ - 1, l));
}

__device__ __forceinline__ int code(const int8_t *L) {
  int u = 0;
  for (int i = 0; i < 8; ++i) u |= L[i] << (2 * i);
  return u;
}

// iq2_find_best_neighbour: the best neighbour of off-grid code u at `scale` (levels written to L), its grid index.
__device__ int best_neighbour(const Tables &t, int u, const float *xval, const float *w, float scale, int8_t *L) {
  const int *nb = t.neigh + (-t.kmap[u] - 1);
  const int n = nb[0];
  float best_d2 = FLT_MAX;
  int gi = -1;
  for (int j = 1; j <= n; ++j) {
    const uint8_t *pg = t.grid + 8 * nb[j];
    float d2 = 0.f;
    for (int i = 0; i < 8; ++i) {
      const float diff = __fsub_rn(__fmul_rn(scale, static_cast<float>(pg[i])), xval[i]);
      d2 = __fadd_rn(d2, __fmul_rn(__fmul_rn(w[i], diff), diff));
    }
    if (d2 < best_d2) { best_d2 = d2; gi = nb[j]; }
  }
  for (int i = 0; i < 8; ++i) L[i] = static_cast<int8_t>((t.grid[8 * gi + i] - 1) / 2);
  return gi;
}

template <int kSub>
__device__ __forceinline__ void sums(const float *w, const float *xval, const int8_t *L, float &sumqx, float &sumq2) {
  sumqx = 0.f;
  sumq2 = 0.f;
  for (int i = 0; i < kSub; ++i) {
    const float q = static_cast<float>(2 * L[i] + 1);
    sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w[i], xval[i]), q));
    sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w[i], q), q));
  }
}

// make_qp_quants(32, 4, x, L, w): the seed scale of an IQ2_XXS sub-block. L is a uint8_t array in llama.cpp, so a
// negative level (one value of an odd group is negated by the parity flip) wraps to 255 wherever L[i] is read back,
// while the `int l` temporaries stay negative -- reproduced here.
__device__ float make_qp_quants(const float *x, const float *w, uint8_t *L) {
  constexpr int n = 32, nmax = kMaxQ + 1;
  float mx = 0.f;
  for (int i = 0; i < n; ++i) mx = mx > x[i] ? mx : x[i];
  if (mx < 1e-15f) {
    for (int i = 0; i < n; ++i) L[i] = 0;
    return 0.f;
  }
  float iscale = __fdiv_rn(static_cast<float>(nmax), mx);
  for (int i = 0; i < n; ++i) L[i] = static_cast<uint8_t>(__float2int_rn(__fmul_rn(iscale, x[i])));
  const float scale = __fdiv_rn(1.f, iscale);
  float best_mse = 0.f;
  for (int i = 0; i < n; ++i) {
    const float diff = __fsub_rn(x[i], __fmul_rn(scale, static_cast<float>(L[i])));
    best_mse = __fadd_rn(best_mse, __fmul_rn(__fmul_rn(w[i], diff), diff));
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0) continue;
    const float iscale_is = __fdiv_rn(__fadd_rn(__fmul_rn(0.1f, static_cast<float>(is)), static_cast<float>(nmax)), mx);
    const float scale_is = __fdiv_rn(1.f, iscale_is);
    float mse = 0.f;
    for (int i = 0; i < n; ++i) {
      const int l = min(nmax, __float2int_rn(__fmul_rn(iscale_is, x[i])));
      const float diff = __fsub_rn(x[i], __fmul_rn(scale_is, static_cast<float>(l)));
      mse = __fadd_rn(mse, __fmul_rn(__fmul_rn(w[i], diff), diff));
    }
    if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
  }
  float sumlx = 0.f, suml2 = 0.f;
  for (int i = 0; i < n; ++i) {
    const int l = min(nmax, __float2int_rn(__fmul_rn(iscale, x[i])));
    L[i] = static_cast<uint8_t>(l);
    const float lf = static_cast<float>(l);
    sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w[i], x[i]), lf));
    suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w[i], lf), lf));
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      const float li = static_cast<float>(L[i]);  // uint8_t read-back
      float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w[i], x[i]), li));
      float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w[i], li), li));
      if (slx > 0.f && sl2 > 0.f) {
        const int new_l = min(nmax, __float2int_rn(__fdiv_rn(__fmul_rn(x[i], sl2), slx)));
        if (new_l != static_cast<int>(L[i])) {
          const float nl = static_cast<float>(new_l);
          slx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w[i], x[i]), nl));
          sl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w[i], nl), nl));
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

__device__ __forceinline__ void store_half(uint8_t *p, float v) {
  // one 16-bit store: ptxas 12.6 miscompiles a byte store of a cvt.f16 result (see iq4.cu)
  *reinterpret_cast<uint16_t *>(p) = __half_as_ushort(__float2half_rn(v));
}

template <bool kS>
__global__ void iq2_encode(const float *x, int64_t nblocks, int64_t blocks_per_row, const float *importance, Tables t,
                           uint8_t *out) {
  constexpr int kSub = kS ? 16 : 32;      // values per sub-block (one thread)
  constexpr int kNsub = 256 / kSub;       // threads per super-block
  constexpr int kNg = kSub / 8;           // 8-value groups per sub-block
  constexpr int kGroups = kThreads / kNsub;  // super-blocks per CUDA block
  constexpr int kNbytes = kS ? kSBytes : kXxsBytes;
  constexpr float kFudge = kS ? 0.9875f : 1.f;
  constexpr float kEps = kS ? 1e-8f : 1e-15f;  // GROUP_MAX_EPS_IQ2_S / GROUP_MAX_EPS
  __shared__ float s_scale[kThreads];
  __shared__ uint8_t s_hi[kThreads];  // IQ2_S: the two grid-index high bits of each group, 2 bits each
  __shared__ float s_id[kGroups];
  __shared__ bool s_live[kGroups];
  const int th = threadIdx.x, g = th / kNsub, ib = th % kNsub;
  const int64_t sb = static_cast<int64_t>(blockIdx.x) * kGroups + g;
  const bool valid = sb < nblocks;
  float xb[kSub], xval[kSub], weight[kSub], waux[kSub];
  int8_t L[kSub], Laux[kSub];
  bool on[kNg], on_aux[kNg];
  uint8_t sgn[kNg];
  int gidx[kNg];
  for (int k = 0; k < kNg; ++k) { sgn[k] = 0; gidx[k] = 0; }
  float scale = 0.f;
  bool skip = true;
  if (valid) {
    const float *blk = x + sb * 256;
    float sumx2 = 0.f;
    for (int i = 0; i < 256; ++i) sumx2 = __fadd_rn(sumx2, __fmul_rn(blk[i], blk[i]));
    const float sigma2 = kS ? __fdiv_rn(__fmul_rn(2.f, sumx2), 256.f) : __fdiv_rn(sumx2, 256.f);
    for (int i = 0; i < kSub; ++i) xb[i] = blk[ib * kSub + i];
    const float *qw = importance ? importance + (sb % blocks_per_row) * 256 + ib * kSub : nullptr;
    for (int i = 0; i < kSub; ++i)  // IQ2_XXS always has importance (the host refuses to encode without it)
      weight[i] = qw ? __fmul_rn(qw[i], __fsqrt_rn(__fadd_rn(sigma2, __fmul_rn(xb[i], xb[i]))))
                     : __fadd_rn(__fmul_rn(0.25f, sigma2), __fmul_rn(xb[i], xb[i]));
    for (int i = 0; i < kSub; ++i) waux[i] = __fsqrt_rn(weight[i]);
    for (int k = 0; k < kNg; ++k) {
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
      if (!kS && (nflip % 2)) {  // IQ2_XXS: an even number of flips
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
    skip = mx < kEps;
    float denom = mx;
    if (!skip && !kS) {
      uint8_t Lu[32];
      scale = make_qp_quants(xval, weight, Lu);
      denom = __fmul_rn(scale, static_cast<float>(kMaxQ));  // eff_max
      if (denom <= 0.f) skip = true;
    }
    if (!skip) {
      float best = 0.f;
      if (kS) scale = __fdiv_rn(mx, static_cast<float>(2 * kMaxQ - 1));
      for (int k = 0; k < kNg; ++k) on[k] = true;
      constexpr int lo = kS ? -9 : -6;
      for (int is = lo; is <= -lo; ++is) {
        const float id =
            __fdiv_rn(__fadd_rn(static_cast<float>(2 * kMaxQ - 1), __fmul_rn(static_cast<float>(is), 0.1f)), denom);
        const float this_scale = __fdiv_rn(1.f, id);
        for (int k = 0; k < kNg; ++k) {
          for (int i = 0; i < 8; ++i) Laux[8 * k + i] = static_cast<int8_t>(level(id, xval[8 * k + i]));
          const int u = code(Laux + 8 * k);
          on_aux[k] = t.kmap[u] >= 0;
          if (!on_aux[k]) best_neighbour(t, u, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
        }
        float sumqx, sumq2;
        sums<kSub>(weight, xval, Laux, sumqx, sumq2);
        if (sumq2 > 0.f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
          scale = __fdiv_rn(sumqx, sumq2);
          best = __fmul_rn(scale, sumqx);
          for (int i = 0; i < kSub; ++i) L[i] = Laux[i];
          for (int k = 0; k < kNg; ++k) on[k] = on_aux[k];
        }
      }
      int n_off = 0;
      for (int k = 0; k < kNg; ++k) n_off += on[k] ? 0 : 1;
      if (scale > 0.f && (!kS || n_off > 0)) {  // IQ2_XXS re-snaps every group, IQ2_S only the off-grid ones
        const float id = __fdiv_rn(1.f, scale);
        for (int k = 0; k < kNg; ++k) {
          if (kS && on[k]) continue;
          int8_t l8[8];
          for (int i = 0; i < 8; ++i) l8[i] = static_cast<int8_t>(level(id, xval[8 * k + i]));
          const int u = code(l8);
          int gi = t.kmap[u];
          if (gi < 0) gi = best_neighbour(t, u, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
          for (int i = 0; i < 8; ++i) L[8 * k + i] = static_cast<int8_t>((t.grid[8 * gi + i] - 1) / 2);
        }
        float sumqx, sumq2;
        sums<kSub>(weight, xval, L, sumqx, sumq2);
        if (sumq2 > 0.f) scale = __fdiv_rn(sumqx, sumq2);
      }
      if (scale < 0.f) {
        scale = -scale;
        for (int k = 0; k < kNg; ++k) sgn[k] = kS ? static_cast<uint8_t>(~sgn[k]) : static_cast<uint8_t>(~sgn[k] & 127);
      }
      for (int k = 0; k < kNg; ++k) gidx[k] = t.kmap[code(L + 8 * k)];
    }
    s_scale[th] = skip ? 0.f : scale;
    s_hi[th] = kS && !skip ? static_cast<uint8_t>(((gidx[0] >> 8) & 3) | (((gidx[1] >> 8) & 3) << 2)) : 0;
  }
  __syncthreads();
  if (valid && ib == 0) {
    float max_scale = 0.f;
    for (int k = 0; k < kNsub; ++k) max_scale = max_scale > s_scale[th + k] ? max_scale : s_scale[th + k];
    s_live[g] = max_scale != 0.f;
    if (s_live[g]) {
      const float d = __fdiv_rn(max_scale, 31.f);
      store_half(out + sb * kNbytes, __fmul_rn(d, kFudge));
      s_id[g] = __fdiv_rn(1.f, d);
    }
  }
  __syncthreads();
  if (!valid || !s_live[g]) return;  // an all-zero super-block is all zero bytes
  uint8_t *p = out + sb * kNbytes;
  auto sub_scale = [&](int k) {  // the 4-bit scale of sub-block k of this super-block: nearest_int(0.5f*(id*s - 1))
    return max(0, min(15, __float2int_rn(__fmul_rn(0.5f, __fsub_rn(__fmul_rn(s_id[g], s_scale[th - ib + k]), 1.f)))));
  };
  if (!kS) {  // q2[2 ib] = grid indices, q2[2 ib + 1] = 7-bit signs | 4-bit scale << 28
    uint32_t w0 = 0, w1 = static_cast<uint32_t>(sub_scale(ib)) << 28;
    if (!skip) {
      w0 = gidx[0] | (gidx[1] << 8) | (gidx[2] << 16) | (gidx[3] << 24);
      w1 |= sgn[0] | (sgn[1] << 7) | (sgn[2] << 14) | (sgn[3] << 21);
    }
    for (int b = 0; b < 4; ++b) {
      p[2 + 8 * ib + b] = static_cast<uint8_t>(w0 >> (8 * b));
      p[6 + 8 * ib + b] = static_cast<uint8_t>(w1 >> (8 * b));
    }
    return;
  }
  // IQ2_S: i8 = 2 ib + k: qs[i8] low index byte, qs[32 + i8] signs, qh[i8 / 4] 2 high bits each, scales nibbles
  if (!skip) {
    for (int k = 0; k < 2; ++k) {
      p[2 + 2 * ib + k] = static_cast<uint8_t>(gidx[k] & 255);
      p[34 + 2 * ib + k] = sgn[k];
    }
  }
  if (ib % 2 == 0) {
    p[66 + ib / 2] = static_cast<uint8_t>(s_hi[th] | (s_hi[th + 1] << 4));
    p[74 + ib / 2] = static_cast<uint8_t>(sub_scale(ib) | (sub_scale(ib + 1) << 4));
  }
}

}  // namespace

at::Tensor iq2_pack_cuda(at::Tensor input, std::optional<at::Tensor> importance, bool s, at::Tensor grid,
                         at::Tensor kmap, at::Tensor neigh) {
  const char *name = s ? "IQ2_S" : "IQ2_XXS";
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kFloat && input.dim() == 2 && input.is_contiguous(),
              name, ": input must be a contiguous float32 CUDA [rows, cols] tensor");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % 256 == 0, name, ": cols must be a multiple of 256");
  TORCH_CHECK(grid.is_cuda() && grid.scalar_type() == at::kByte && grid.is_contiguous() &&
                  grid.device() == input.device() && grid.numel() == (s ? 1024 : 256) * 8,
              name, ": grid must be a contiguous uint8 [n, 8] tensor on the input's device");
  for (const auto *tab : {&kmap, &neigh})
    TORCH_CHECK(tab->is_cuda() && tab->scalar_type() == at::kInt && tab->is_contiguous() &&
                    tab->device() == input.device(),
                name, ": kmap / neigh must be contiguous int32 tensors on the input's device");
  TORCH_CHECK(kmap.numel() == kKmap, name, ": kmap must have ", kKmap, " entries");
  const float *imp = nullptr;
  if (importance.has_value()) {
    const auto &i = *importance;
    TORCH_CHECK(i.is_cuda() && i.scalar_type() == at::kFloat && i.dim() == 1 && i.numel() == cols &&
                    i.is_contiguous() && i.device() == input.device(),
                name, ": importance must be a contiguous float32 [cols] tensor on the input's device");
    imp = i.data_ptr<float>();
  }
  TORCH_CHECK(s || imp, name, ": requires an importance matrix (llama.cpp: 'missing quantization weights')");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t bpr = cols / 256, nblocks = rows * bpr;
  const int nbytes = s ? kSBytes : kXxsBytes;
  auto out = at::zeros({rows, bpr, nbytes}, input.options().dtype(at::kByte));
  if (nblocks == 0) return out;
  const Tables t{grid.data_ptr<uint8_t>(), kmap.data_ptr<int>(), neigh.data_ptr<int>()};
  const auto stream = c10::cuda::getCurrentCUDAStream();
  if (s) {
    constexpr int kGroups = kThreads / 16;
    const unsigned nblk = static_cast<unsigned>((nblocks + kGroups - 1) / kGroups);
    iq2_encode<true><<<nblk, kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp, t,
                                                     out.data_ptr<uint8_t>());
  } else {
    constexpr int kGroups = kThreads / 8;
    const unsigned nblk = static_cast<unsigned>((nblocks + kGroups - 1) / kGroups);
    iq2_encode<false><<<nblk, kThreads, 0, stream>>>(input.data_ptr<float>(), nblocks, bpr, imp, t,
                                                      out.data_ptr<uint8_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
