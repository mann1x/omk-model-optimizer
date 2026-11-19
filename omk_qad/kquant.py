# SPDX-License-Identifier: Apache-2.0
"""K-quants and legacy block-32 types as native QAD formats: CUDA encoders byte-identical to llama.cpp's (bm-901).

Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q4_0, Q4_1, Q5_0, Q5_1, with their torch decoders.

The encoders are llama.cpp's ``quantize_row_<type>_impl`` (with an imatrix) and ``quantize_row_<type>_ref`` (without
one) from ggml-quants.c, ported operation by operation in float32 -- the same source ``llama-quantize`` runs, so a
student trained at Q4_K is served and exported as the blocks llama-quantize itself writes:

  * Q4_K / Q5_K (256-value super-block, 144 / 176 bytes): per 32-value sub-block a least-squares scale + min
    (``make_qkx3_quants``, 37 trial scales, weights ``qw * sqrt(sigma2 + x^2)`` with the imatrix; ``make_qkx2_quants``
    with weights ``av_x + |x|`` without), then the 8 scales and 8 mins quantized to 6 bits by ``make_qp_quants`` (a
    weighted search plus a greedy refinement; the plain 63-level max rule without an imatrix), fp16 ``d`` / ``dmin``,
    and every value re-quantized against the fp16-rounded super-block scales. Q5_K keeps a fifth bit per value in
    ``qh``.
  * Q6_K (210 bytes): per 16-value sub-block a signed scale from ``make_qx_quants`` (rmse_type 1: weights = imatrix
    or x^2, 19 trial scales), the 16 scales as int8 against ``d = -max_scale / 128`` (fp16), values re-quantized to
    6 bits (``ql`` low nibbles, ``qh`` top 2 bits).
  * Q2_K (84 bytes) / Q3_K (110 bytes): 16 sub-blocks of 16; Q2_K as Q4_K with 2-bit values and 4-bit scales/mins
    (``make_qkx2_quants`` with |x| weights and the MAD criterion without an imatrix), Q3_K as Q6_K with 3-bit values
    and 6-bit scales (``make_q3_quants`` without an imatrix, ``make_qx_quants`` over the 16 scales with one).
  * Q4_0 / Q4_1 / Q5_0 / Q5_1 (32-value blocks, 18 / 20 / 22 / 24 bytes): with an imatrix ``make_qx_quants`` /
    ``make_qkx3_quants`` weighted with the ROW's sigma2 (sum x^2 / n_per_row -- a GPTQ group slice therefore carries
    the group-local sigma2, as with libggml); without one the plain max / min-max rules of the ``_ref`` encoders.
    llama-quantize falls back to these for rows that are not a multiple of 256 (Gemma-4's 1408-wide expert down
    projections: Q5_0 in Q4_K_M, Q5_1 in Q5_K_M, Q4_0 in Q3_K_M).

Every sum is accumulated in llama.cpp's order in float32 without fused multiply-adds (the kernel uses explicit
``__fmul_rn`` / ``__fadd_rn`` / ``__fdiv_rn`` / ``__fsqrt_rn`` and is built with ``--fmad=false``; libggml carries
no FMA in these functions), so the bytes match libggml's bit for bit.

On a CUDA device encoding ALWAYS uses the CUDA kernel (no CPU fallback; a build failure raises). On the CPU
:func:`encode` is libggml itself (``omk_qad.ggml_lib``, the code llama-quantize runs), which is also the oracle of
the kernel: unlike the IQ / Q8_0 ports there is no torch reference -- ``make_qp_quants`` and the trial-scale loops are
sequential, data-dependent greedy searches whose torch port would be as slow as the CPU paths the test-suite rework
(bm-902) is removing, and libggml is already the CPU encoder of every K-quant in the trainer and the rounder.
:func:`decode` is torch on the payload's device (``dequantize_row_q{4,5,6}_K``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

SPECS = {"Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
         "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24)}  # type: (values per block, bytes)
TYPES = frozenset(SPECS)
KQUANTS = frozenset(t for t in SPECS if t.endswith("_K"))
LLAMA_CPP_CANDIDATES = ("/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")  # after $OMK_LLAMA_CPP


def _ext():
    """The CUDA encoder (built on first use); raises if it cannot be built -- there is no CPU fallback."""
    if not hasattr(_ext, "mod"):
        from modelopt.torch.utils.cpp_extension import load_cpp_extension

        here = Path(__file__).parent / "kernels"
        _ext.mod = load_cpp_extension(
            name="omk_cuda_ext_kquant",
            sources=[here / "kquant.cpp", here / "kquant.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="omk_qad K-quant / legacy CUDA encoder is unavailable (no CPU fallback on a CUDA device).",
            extra_cuda_cflags=["-O3", "--fmad=false"],
            raise_if_failed=True,
        )
    return _ext.mod


def llama_cpp_tree() -> Path:
    """The llama.cpp tree whose libggml encodes on the CPU: ``$OMK_LLAMA_CPP`` or the first known build."""
    for p in (os.environ.get("OMK_LLAMA_CPP"), *LLAMA_CPP_CANDIDATES):
        if p and (Path(p) / "build/bin/libggml-base.so").is_file():
            return Path(p)
    raise RuntimeError("K-quant CPU encode needs libggml: set OMK_LLAMA_CPP to a llama.cpp tree with "
                       "build/bin/libggml-base.so (the CUDA kernel has no CPU port)")


def set_llama_cpp(tree: str | Path) -> None:
    """Pin the libggml tree of the CPU path (kfac_round --llama-cpp, the trainer's Codec) instead of the defaults."""
    os.environ["OMK_LLAMA_CPP"] = str(tree)
    if hasattr(_ggml, "lib") and Path(getattr(_ggml, "tree", "")) != Path(tree):
        del _ggml.lib


def _ggml():
    if not hasattr(_ggml, "lib"):
        from .ggml_lib import GGMLLib

        _ggml.tree = llama_cpp_tree()
        _ggml.lib = GGMLLib(_ggml.tree)
    return _ggml.lib


# ggml_type enum values (ggml.h; stable since 2023 -- the GGUF file format depends on them). Looked up here rather
# than through `import gguf`: importing a pip gguf before the Codec imports the llama.cpp tree's gguf-py makes
# gguf_export._import_llama_cpp refuse ("a different gguf is already imported", bug-973).
GGML_TYPE_ID = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14}


def _ggml_id(qtype: str) -> int:
    return GGML_TYPE_ID[qtype]


def _check(qtype: str, x: torch.Tensor, importance: torch.Tensor | None) -> int:
    if qtype not in SPECS:
        raise ValueError(f"not a K-quant / legacy type with a kernel: {qtype} (one of {', '.join(sorted(SPECS))})")
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
        return encode_ggml(qtype, x, importance)
    imp = None if importance is None else importance.to(device=x.device, dtype=torch.float32).contiguous()
    return getattr(_ext(), f"{qtype.lower()}_pack")(x.contiguous(), imp)


def encode_ggml(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """The bytes of libggml's ``ggml_quantize_chunk`` (what llama-quantize runs): CPU path and kernel oracle."""
    block = _check(qtype, x, importance)
    rows, cols = x.shape
    imp = None if importance is None else importance.detach().float().cpu().numpy()
    p = _ggml().quantize(_ggml_id(qtype), x.detach().float().cpu().numpy(), imp)
    return torch.from_numpy(p).reshape(rows, cols // block, SPECS[qtype][1]).to(x.device)


def decode(qtype: str, packed: torch.Tensor, rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[rows, cols] values of a payload (llama.cpp's dequantize_row_<type>), on the payload's device."""
    if qtype not in SPECS:
        raise ValueError(f"not a K-quant / legacy type with a kernel: {qtype}")
    p = packed.reshape(-1, SPECS[qtype][1])
    if qtype not in KQUANTS:
        return _decode_legacy(qtype, p).reshape(rows, cols).to(dtype)
    if qtype == "Q6_K":
        return _decode_q6(p).reshape(rows, cols).to(dtype)
    if qtype == "Q2_K":
        return _decode_q2(p).reshape(rows, cols).to(dtype)
    if qtype == "Q3_K":
        return _decode_q3(p).reshape(rows, cols).to(dtype)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    dmin = p[:, 2:4].contiguous().view(torch.float16).float()
    sc, mn = _scale_min_k4(p[:, 4:16].long())  # [B, 8] each
    if qtype == "Q4_K":
        qs = p[:, 16:].long().reshape(-1, 4, 32)
    else:
        qh = p[:, 16:48].long()  # [B, 32]
        qs = p[:, 48:].long().reshape(-1, 4, 32)
    lo, hi = qs & 0xF, qs >> 4  # [B, 4, 32]: sub-blocks 2k and 2k + 1
    if qtype == "Q5_K":
        k = torch.arange(4, device=p.device)[None, :, None]
        lo = lo + (((qh[:, None, :] >> (2 * k)) & 1) << 4)
        hi = hi + (((qh[:, None, :] >> (2 * k + 1)) & 1) << 4)
    q = torch.stack([lo, hi], 2).reshape(-1, 8, 32).float()  # [B, 8 sub-blocks, 32]
    d1 = d * sc.float()  # [B, 8]
    m1 = dmin * mn.float()
    return (d1[:, :, None] * q - m1[:, :, None]).reshape(rows, cols).to(dtype)


def _scale_min_k4(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """get_scale_min_k4 for j = 0..7 on the 12 packed scale bytes s [B, 12] (long)."""
    sc_lo = s[:, 0:4] & 63
    mn_lo = s[:, 4:8] & 63
    sc_hi = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn_hi = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return torch.cat([sc_lo, sc_hi], 1), torch.cat([mn_lo, mn_hi], 1)


def _decode_q6(p: torch.Tensor) -> torch.Tensor:
    """dequantize_row_q6_K on payload rows p [B, 210]."""
    d = p[:, 208:210].contiguous().view(torch.float16).float()  # [B, 1]
    sc = p[:, 192:208].contiguous().view(torch.int8).float()  # [B, 16]
    ql = p[:, :128].long().reshape(-1, 2, 64)  # [B, half, 64]
    qh = p[:, 128:192].long().reshape(-1, 2, 32)  # [B, half, 32]
    q1 = (ql[:, :, :32] & 0xF) | (((qh >> 0) & 3) << 4)
    q2 = (ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
    q3 = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
    q = (torch.stack([q1, q2, q3, q4], 2) - 32).float()  # [B, half, 4 (m), 32 (l)]: value 128*half + 32*m + l
    sub = (sc.reshape(-1, 2, 8)[:, :, :, None].expand(-1, -1, -1, 16).reshape(-1, 2, 128))  # scale of each value
    dsc = (d[:, :, None] * sub).reshape(-1, 2, 4, 32)  # d * sc[is] first, then * q, as dequantize_row_q6_K
    return (dsc * q).reshape(p.shape[0], 256)


def _decode_q2(p: torch.Tensor) -> torch.Tensor:
    """dequantize_row_q2_K on rows p [B, 84]: value v is in sub-block v // 16, bits at qs[32 (v // 128) + v % 32]."""
    d = p[:, 80:82].contiguous().view(torch.float16).float()  # [B, 1]
    dmin = p[:, 82:84].contiguous().view(torch.float16).float()
    sc = p[:, :16].long()  # [B, 16]
    qs = p[:, 16:80].long().reshape(-1, 2, 32)  # [B, half, 32]
    shift = (2 * torch.arange(4, device=p.device))[None, None, :, None]  # [1, 1, m, 1]
    q = ((qs[:, :, None, :] >> shift) & 3).float()  # [B, half, m, l]: value 128 half + 32 m + l
    sub = (sc.reshape(-1, 2, 8)[:, :, :, None].expand(-1, -1, -1, 16).reshape(-1, 2, 4, 32))  # scale byte per value
    dl = d[:, :, None, None] * (sub & 0xF).float()
    ml = dmin[:, :, None, None] * (sub >> 4).float()
    return (dl * q - ml).reshape(p.shape[0], 256)


def _decode_q3(p: torch.Tensor) -> torch.Tensor:
    """dequantize_row_q3_K on rows p [B, 110]: 6-bit scales - 32, 2-bit values - 4 unless the hmask bit is set."""
    d = p[:, 108:110].contiguous().view(torch.float16).float()  # [B, 1]
    b = p[:, 96:108].long()  # [B, 12]
    j = torch.arange(16, device=p.device)
    lo = torch.where(j < 8, b[:, j.clamp(max=7)] & 0xF, b[:, (j - 8).clamp(min=0)] >> 4)
    hi = (b[:, 8 + j % 4] >> (2 * (j // 4))) & 3
    sc = ((lo | (hi << 4)) - 32).float()  # [B, 16]
    hm = p[:, :32].long()  # [B, 32]: bit b of column m = value 32 b + m
    qs = p[:, 32:96].long().reshape(-1, 2, 32)
    shift = (2 * torch.arange(4, device=p.device))[None, None, :, None]
    q = ((qs[:, :, None, :] >> shift) & 3).float()  # [B, half, m, l]: value v = 128 half + 32 m + l
    half4 = (4 * torch.arange(2, device=p.device))[:, None, None]
    bit = half4 + torch.arange(4, device=p.device)[None, :, None]  # v // 32 of value v = 128 half + 32 m + l
    high = ((hm[:, None, None, :] >> bit[None]) & 1).float()  # [B, half, m, l]
    q = q - 4.0 * (1.0 - high)
    sub = sc.reshape(-1, 2, 8)[:, :, :, None].expand(-1, -1, -1, 16).reshape(-1, 2, 4, 32)
    dl = d[:, :, None, None] * sub
    return (dl * q).reshape(p.shape[0], 256)


def _decode_legacy(qtype: str, p: torch.Tensor) -> torch.Tensor:
    """dequantize_row_q4_0 / q4_1 / q5_0 / q5_1 on payload rows p [B, nbytes]."""
    has_min, five = qtype in ("Q4_1", "Q5_1"), qtype in ("Q5_0", "Q5_1")
    off = 2 + (2 if has_min else 0)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    qs = p[:, off + (4 if five else 0):].long()  # [B, 16]
    q = torch.cat([qs & 0xF, qs >> 4], 1)  # [B, 32]: values j and j + 16
    if five:
        qh = p[:, off:off + 4].long()
        qh = qh[:, 0] | (qh[:, 1] << 8) | (qh[:, 2] << 16) | (qh[:, 3] << 24)  # little-endian uint32
        j = torch.arange(32, device=p.device)
        q = q | (((qh[:, None] >> j) & 1) << 4)
    q = q.float()
    if has_min:
        m = p[:, 2:4].contiguous().view(torch.float16).float()
        return q * d + m
    return (q - (16.0 if five else 8.0)) * d
