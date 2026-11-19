# SPDX-License-Identifier: Apache-2.0
"""Q8_0 as a native QAD format: a CUDA encoder that reproduces llama.cpp's, and its decoder.

The encoder is llama.cpp's ``quantize_row_q8_0_ref`` (ggml-quants.c): per 32-value block ``amax = max |x|``,
``d = amax / 127`` (stored as fp16, round to nearest even, as ``GGML_FP32_TO_FP16``), ``id = d ? 1 / d : 0`` on the
float32 ``d``, codes ``roundf(x * id)`` as int8 -- ``roundf`` rounds half AWAY from zero (``torch.round`` rounds half to
even, so the reference does not use it). ``llama-quantize`` ignores the imatrix for Q8_0; ``importance`` is accepted
and ignored so the type shares the ``encode`` contract of :mod:`omk_qad.iq4` / :mod:`omk_qad.iq2`.

Why a kernel of our own (2026-11-19, bm-900): NVIDIA ModelOpt's CUDA Q8_0 packer zeroed the low byte of every block's
fp16 scale on bs2 (99.4 % of blocks differ from llama-quantize, decode error ~300x; its CPU path is byte-identical),
the F2I.U8.F16 byte-store miscompile of bug-832 -- and the QAD rule is that the encoder in the loop must be the one
that ships. The kernel writes the scale as one aligned 16-bit store and is built with ``--fmad=false``.

On a CUDA device encoding ALWAYS uses the CUDA kernel (no CPU fallback; a build failure raises).
:func:`encode_reference` is the same algorithm in torch, for CPU devices and as the kernel's oracle; libggml
(``omk_qad.ggml_lib``, the code llama-quantize runs) is the oracle of both.
"""

from __future__ import annotations

from pathlib import Path

import torch

SPECS = {"Q8_0": (32, 34)}  # type: (values per block, bytes per block)
TYPES = frozenset(SPECS)
BLOCK, NBYTES = SPECS["Q8_0"]


def _ext():
    """The CUDA encoder (built on first use); raises if it cannot be built -- there is no CPU fallback."""
    if not hasattr(_ext, "mod"):
        from modelopt.torch.utils.cpp_extension import load_cpp_extension

        here = Path(__file__).parent / "kernels"
        _ext.mod = load_cpp_extension(
            name="omk_cuda_ext_q8",
            sources=[here / "q8.cpp", here / "q8.cu"],
            cuda_version_specifiers=">=11.8",
            fail_msg="omk_qad Q8_0 CUDA encoder is unavailable (no CPU fallback on a CUDA device).",
            extra_cuda_cflags=["-O3", "--fmad=false"],
            raise_if_failed=True,
        )
    return _ext.mod


def _check(qtype: str, x: torch.Tensor, importance: torch.Tensor | None) -> int:
    if qtype not in SPECS:
        raise ValueError(f"not a Q8 type: {qtype}")
    if x.dim() != 2 or x.dtype != torch.float32 or x.shape[1] % BLOCK:
        raise ValueError(f"{qtype} needs float32 [rows, cols] with cols a multiple of {BLOCK}, got "
                         f"{x.dtype} {tuple(x.shape)}")
    if importance is not None and (importance.dim() != 1 or importance.numel() != x.shape[1]):
        raise ValueError(f"importance must be [cols={x.shape[1]}], got {tuple(importance.shape)}")
    return BLOCK


def encode(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """uint8 payload [rows, cols / 32, 34] of float32 ``x`` [rows, cols]; ``importance`` ignored, as llama-quantize."""
    _check(qtype, x, importance)
    if not x.is_cuda:
        return encode_reference(qtype, x, importance)
    return _ext().q8_0_pack(x.contiguous())


def decode(qtype: str, packed: torch.Tensor, rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[rows, cols] values of a payload (llama.cpp's dequantize_row_q8_0: ``qs[j] * d``), on the payload's device."""
    _check_type(qtype)
    p = packed.reshape(-1, NBYTES)
    d = p[:, :2].contiguous().view(torch.float16).float()  # [B, 1]
    qs = p[:, 2:].contiguous().view(torch.int8).float()  # [B, 32]
    return (d * qs).reshape(rows, cols).to(dtype)


def _check_type(qtype: str) -> None:
    if qtype not in SPECS:
        raise ValueError(f"not a Q8 type: {qtype}")


def _roundf(x: torch.Tensor) -> torch.Tensor:
    """C ``roundf``: half away from zero. In float64 so that ``x + 0.5`` cannot round up a float32 x just below .5."""
    xd = x.double()
    return torch.trunc(xd + torch.copysign(torch.full_like(xd, 0.5), xd))


def encode_reference(qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
    """The CUDA kernel's algorithm in torch (any device), byte-identical to it and to libggml."""
    _check(qtype, x, importance)
    rows, cols = x.shape
    xb = x.reshape(-1, BLOCK)  # flattened blocks, row-major
    amax = xb.abs().amax(1)  # exact: a maximum has no rounding
    d = amax / torch.full_like(amax, 127.0)  # a tensor divisor: torch turns `/ scalar` into `* (1/scalar)`
    inv = torch.where(d != 0, torch.ones_like(d) / d, torch.zeros_like(d))
    q = _roundf(xb * inv[:, None]).to(torch.int8)
    out = torch.empty(xb.shape[0], NBYTES, dtype=torch.uint8, device=x.device)
    out[:, :2] = d.to(torch.float16).reshape(-1, 1).view(torch.uint8)
    out[:, 2:] = q.view(torch.uint8)
    return out.reshape(rows, cols // BLOCK, NBYTES)
