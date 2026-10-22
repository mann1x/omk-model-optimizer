# SPDX-License-Identifier: Apache-2.0
"""IQ4_XS / IQ4_NL as native QAD formats: a CUDA encoder that reproduces llama.cpp's, and their decoder.

The encoder is llama.cpp's ``quantize_row_iq4_nl_impl`` (ggml-quants.c) ported operation by operation, so a
student trained at IQ4_XS is served -- and exported as -- the blocks ``llama-quantize`` itself would write:

  * per 32-value sub-block: weights ``qw * sqrt(sigma2 + x^2)`` (imatrix, E[x^2] like llama-quantize) or ``x^2``;
    a least-squares scale from ``max/127``, then 15 trial scales ``(t - 127) / max``, t in [-7, 7], each
    re-assigning every value to its nearest ``kvalues_iq4nl`` level; the best weighted fit wins;
  * IQ4_XS (256-value super-block, 136 bytes): ``d = -max_scale / 32`` (fp16), 6-bit sub-block scales
    ``nearest_int(scale / d) + 32``, values re-assigned with the quantized scale;
  * IQ4_NL (32-value block, 18 bytes, llama.cpp's fallback when a row is not a multiple of 256): fp16 block scale.

Every sum is accumulated in llama.cpp's order in float32 without fused multiply-adds (the kernel uses explicit
``__fmul_rn``/``__fadd_rn`` and is built with ``--fmad=false``), so the bytes match libggml's bit for bit.

On a CUDA device encoding ALWAYS uses the CUDA kernel: there is no CPU fallback, and an extension that cannot
be built raises. :func:`encode_reference` is the same algorithm in torch, for CPU devices (the CPU test models)
and as the kernel's oracle; ``omk_qad.ggml_lib`` (libggml, the code llama-quantize runs) is the oracle of both.
"""

from __future__ import annotations

from pathlib import Path

import torch

VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)  # kvalues_iq4nl
GROUP_MAX_EPS = 1e-15
NTRY = 7
SPECS = {"IQ4_XS": (256, 136), "IQ4_NL": (32, 18)}  # type: (values per block, bytes per block)
TYPES = frozenset(SPECS)
SUB = 32  # sub-block / IQ4_NL block


def _ext():
    """The CUDA encoder (built on first use); raises if it cannot be built -- there is no CPU fallback."""
    if not hasattr(_ext, "mod"):
        from modelopt.torch.utils.cpp_extension import load_cpp_extension

        here = Path(__file__).parent / "kernels"
        _ext.mod = load_cpp_extension(
            name="omk_cuda_ext_iq4",
            sources=[here / "iq4.cpp", here / "iq4.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="omk_qad IQ4_XS/IQ4_NL CUDA encoder is unavailable (no CPU fallback on a CUDA device).",
            extra_cuda_cflags=["-O3", "--fmad=false"],
            raise_if_failed=True,
        )
    return _ext.mod


def _check(qtype: str, x: torch.Tensor, importance: torch.Tensor | None) -> int:
    if qtype not in SPECS:
        raise ValueError(f"not an IQ4 type: {qtype}")
    block = SPECS[qtype][0]
    if x.dim() != 2 or x.dtype != torch.float32 or x.shape[1] % block:
        raise ValueError(f"{qtype} needs float32 [rows, cols] with cols a multiple of {block}, got "
                         f"{x.dtype} {tuple(x.shape)}")
    if importance is not None and (importance.dim() != 1 or importance.numel() != x.shape[1]):
        raise ValueError(f"importance must be [cols={x.shape[1]}], got {tuple(importance.shape)}")
    return block


def encode(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 payload [rows, cols / block, block bytes] of float32 ``x`` [rows, cols]; ``importance`` = E[x^2] [cols]."""
    _check(qtype, x, importance)
    if not x.is_cuda:
        return encode_reference(qtype, x, importance)
    imp = None if importance is None else importance.to(device=x.device, dtype=torch.float32).contiguous()
    fn = _ext().iq4_xs_pack if qtype == "IQ4_XS" else _ext().iq4_nl_pack
    return fn(x.contiguous(), imp)


def decode(qtype: str, packed: torch.Tensor, rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[rows, cols] values of a payload (llama.cpp's dequantize_row_iq4_xs / _nl), on the payload's device."""
    block, nbytes = SPECS[qtype]
    p = packed.reshape(-1, nbytes)
    vals = torch.tensor(VALUES, dtype=torch.float32, device=p.device)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    if qtype == "IQ4_NL":
        qs = p[:, 2:].long()  # [B, 16]
        q = torch.cat([qs & 0xF, qs >> 4], 1)  # values j and j + 16
        return (d * vals[q]).reshape(rows, cols).to(dtype)
    sh = (p[:, 2].long() | (p[:, 3].long() << 8))[:, None]
    sl = p[:, 4:8].long()
    ib = torch.arange(8, device=p.device)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)  # [B, 8]
    dl = d * (ls - 32).float()  # [B, 8]
    qs = p[:, 8:].long().reshape(-1, 8, 16)
    q = torch.cat([qs & 0xF, qs >> 4], 2)  # [B, 8, 32]
    return (dl[:, :, None] * vals[q]).reshape(rows, cols).to(dtype)


# ------------------------------------------------------------------------------------------- torch reference
def _best_index(x: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """best_index_int8: nearest level, ties as llama.cpp's binary search resolves them."""
    mu = torch.searchsorted(vals, x.contiguous(), right=True).clamp(1, 15)
    idx = torch.where((x - vals[mu - 1]) < (vals[mu] - x), mu - 1, mu)
    idx = torch.where(x <= vals[0], torch.zeros_like(idx), idx)
    return torch.where(x >= vals[15], torch.full_like(idx, 15), idx)


def _sums(xs: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, vals: torch.Tensor):
    """sumqx, sumq2 accumulated over j = 0..31 in order (no reduction tree)."""
    sumqx = torch.zeros(xs.shape[0], dtype=torch.float32, device=xs.device)
    sumq2 = torch.zeros_like(sumqx)
    for j in range(SUB):
        q = vals[_best_index(scale * xs[:, j], vals)]
        wq = w[:, j] * q
        sumqx = sumqx + wq * xs[:, j]
        sumq2 = sumq2 + wq * q
    return sumqx, sumq2


def _search(xs: torch.Tensor, w: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
    """Best scale of every 32-value sub-block [S, 32] (0 where max |x| < GROUP_MAX_EPS)."""
    ax = xs.abs()
    first = ax.argmax(1, keepdim=True)  # the first maximal |x| wins, as `if (ax > amax)`
    amax, mx = ax.gather(1, first)[:, 0], xs.gather(1, first)[:, 0]
    zero = amax < GROUP_MAX_EPS
    mx = torch.where(zero, torch.ones_like(mx), mx)  # keeps the divisions finite; zero blocks are masked below
    d = -mx / torch.full_like(mx, VALUES[0])  # a tensor divisor: torch turns `/ scalar` into `* (1/scalar)`
    sumqx, sumq2 = _sums(xs, w, 1 / d, vals)
    d = torch.where(sumq2 > 0, sumqx / sumq2, torch.zeros_like(sumqx))
    best = d * sumqx
    for itry in range(-NTRY, NTRY + 1):
        inv = torch.tensor(float(itry + VALUES[0]), dtype=torch.float32, device=xs.device) / mx
        sumqx, sumq2 = _sums(xs, w, inv, vals)
        better = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        nd = sumqx / sumq2
        d = torch.where(better, nd, d)
        best = torch.where(better, nd * sumqx, best)
    return torch.where(zero, torch.zeros_like(d), d)


def _weights(x: torch.Tensor, block: int, imp: torch.Tensor | None) -> torch.Tensor:
    """Per-value weights of blocks x [B, block] (imp: [B, block] importance columns or None)."""
    if imp is None:
        return x * x
    sigma2 = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
    for j in range(block):
        sigma2 = sigma2 + x[:, j] * x[:, j]
    sigma2 = sigma2 * (2.0 / block)
    # sqrtf is correctly rounded; torch's float32 CPU sqrt (SLEEF u05) is not always. A float64 sqrt rounded once
    # to float32 is (53 >= 2*24 + 2 bits: double rounding is exact for sqrt).
    return imp * torch.sqrt((sigma2[:, None] + x * x).double()).float()


def encode_reference(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """The CUDA kernel's algorithm in torch (any device), byte-identical to it and to libggml."""
    block = _check(qtype, x, importance)
    rows, cols = x.shape
    nbytes = SPECS[qtype][1]
    vals = torch.tensor(VALUES, dtype=torch.float32, device=x.device)
    xb = x.reshape(-1, block)  # flattened blocks, row-major
    imp = None
    if importance is not None:
        imp = importance.to(device=x.device, dtype=torch.float32).reshape(1, cols // block, block)
        imp = imp.expand(rows, -1, -1).reshape(-1, block)
    w = _weights(xb, block, imp)
    nsub = block // SUB
    xs, ws = xb.reshape(-1, SUB), w.reshape(-1, SUB)
    scales = _search(xs, ws, vals).reshape(-1, nsub)  # [B, nsub]
    out = torch.zeros(xb.shape[0], nbytes, dtype=torch.uint8, device=x.device)
    if qtype == "IQ4_NL":
        s = scales[:, 0]
        out[:, :2] = s.to(torch.float16).reshape(-1, 1).view(torch.uint8)
        inv = torch.where(s != 0, 1 / s, torch.zeros_like(s))
        lev = _best_index(inv[:, None] * xb, vals)
        out[:, 2:] = (lev[:, :16] | (lev[:, 16:] << 4)).to(torch.uint8)
        return out.reshape(rows, cols // block, nbytes)
    absd = scales.abs()
    first = absd.argmax(1, keepdim=True)
    max_scale = torch.where(absd.gather(1, first)[:, 0] > 0, scales.gather(1, first)[:, 0],
                            torch.zeros_like(scales[:, 0]))  # `if (abs_d > amax_scale)` from 0: +0.0 if none
    d = -max_scale / torch.full_like(max_scale, 32.0)
    out[:, :2] = d.to(torch.float16).reshape(-1, 1).view(torch.uint8)
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    lq = torch.round(inv[:, None] * scales).clamp(-32, 31)  # nearest_int: round half to even
    dl = d[:, None] * lq
    idl = torch.where(dl != 0, 1 / dl, torch.zeros_like(dl))
    lev = _best_index(idl[:, :, None] * xb.reshape(-1, nsub, SUB), vals)  # [B, 8, 32]
    li = lq.long() + 32
    sh = ((li >> 4) << (2 * torch.arange(nsub, device=x.device))).sum(1)
    out[:, 2] = (sh & 0xFF).to(torch.uint8)
    out[:, 3] = (sh >> 8).to(torch.uint8)
    out[:, 4:8] = ((li[:, 0::2] & 0xF) | ((li[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    out[:, 8:] = (lev[:, :, :16] | (lev[:, :, 16:] << 4)).reshape(-1, 128).to(torch.uint8)
    return out.reshape(rows, cols // block, nbytes)
