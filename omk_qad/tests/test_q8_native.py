# SPDX-License-Identifier: Apache-2.0
"""omk_qad.q8: the Q8_0 reference (and kernel, on CUDA) against libggml, llama-quantize's own code (bm-900)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from omk_qad import q8

LLAMA = next((p for p in (os.environ.get("OMK_LLAMA_CPP"), "/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")
              if p and (Path(p) / "build/bin/libggml-base.so").is_file()), None)
needs_lib = pytest.mark.skipif(LLAMA is None, reason="no llama.cpp tree with build/bin/libggml-base.so")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _lib():
    from omk_qad.kfac_round import GgmlNative

    return GgmlNative(LLAMA)


def _cases():
    g = torch.Generator().manual_seed(0)
    yield "gauss", torch.randn(16, 256, generator=g) * 0.03
    x = torch.randn(8, 128, generator=g)
    x[0, :32] = 0.0  # a zero block: d = 0, id = 0, codes 0
    x[1, 32:64] = 1e-30  # sub-eps block: d is a float32 denormal, fp16 d = 0, codes = roundf(x / d) nonzero
    x[2] = x[2].abs()  # one-signed block
    yield "degenerate", x
    # exact halves: amax = 127 so d = 1 and id = 1; roundf(0.5) = 1, roundf(-0.5) = -1, roundf(2.5) = 3 (torch.round
    # would give 0, 0, 2)
    h = torch.zeros(1, 32)
    h[0, 0] = 127.0
    h[0, 1:9] = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5])
    yield "halves", h
    yield "large", torch.randn(4, 64, generator=g) * 3000.0  # d around 100: fp16 rounding of d visible in the decode


def test_roundf_is_half_away_from_zero():
    """C roundf semantics, including a float32 just below .5 that ``x + 0.5`` would round up in float32."""
    x = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, 0.49999997, -0.49999997, 126.5])
    assert q8._roundf(x).tolist() == [1.0, -1.0, 2.0, -2.0, 3.0, 0.0, -0.0, 127.0]


@needs_lib
@pytest.mark.parametrize(("name", "x"), list(_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_reference_matches_libggml(name, x):
    """The torch reference writes libggml's bytes and decodes to libggml's values."""
    lib = _lib()
    ref = lib.encode("Q8_0", x, None)
    got = q8.encode_reference("Q8_0", x, None)
    assert got.shape == ref.shape == (x.shape[0], x.shape[1] // 32, 34)
    assert torch.equal(got, ref), f"{name}: {(got != ref).reshape(-1, 34).any(-1).sum().item()} blocks differ"
    d_lib = lib.decode("Q8_0", ref, *x.shape)
    assert torch.equal(q8.decode("Q8_0", got, *x.shape), d_lib)


@needs_lib
def test_importance_is_ignored_like_llama_quantize():
    """llama-quantize's quantize_q8_0 drops quant_weights; so does encode (same bytes with and without)."""
    lib = _lib()
    x = torch.randn(4, 96, generator=torch.Generator().manual_seed(1))
    imp = torch.rand(96) * 5 + 0.01
    assert torch.equal(q8.encode("Q8_0", x, imp), lib.encode("Q8_0", x, imp))
    assert torch.equal(q8.encode("Q8_0", x, imp), q8.encode("Q8_0", x, None))


@needs_cuda
@pytest.mark.parametrize(("name", "x"), list(_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_kernel_matches_reference(name, x):
    """The CUDA kernel writes the reference's bytes on every case, degenerate blocks included."""
    ref = q8.encode_reference("Q8_0", x, None)
    got = q8.encode("Q8_0", x.cuda(), None).cpu()
    assert torch.equal(got, ref), f"{name}: {(got != ref).reshape(-1, 34).any(-1).sum().item()} blocks differ"


@needs_cuda
def test_kernel_scale_low_byte_survives():
    """The bug-832 / bug-966 signature: every block's fp16 scale low byte 0 while the codes are exact."""
    x = torch.randn(64, 2816, generator=torch.Generator().manual_seed(2)) * 0.03
    got = q8.encode("Q8_0", x.cuda(), None).cpu().reshape(-1, 34)
    assert int((got[:, 0] != 0).sum()) > got.shape[0] // 2


def test_rejects_bad_shapes():
    """Wrong block multiple, dtype or type name are refused before any encode."""
    with pytest.raises(ValueError):
        q8.encode("Q8_0", torch.zeros(2, 33), None)
    with pytest.raises(ValueError):
        q8.encode("Q8_0", torch.zeros(2, 64, dtype=torch.bfloat16), None)
    with pytest.raises(ValueError):
        q8.encode("IQ4_XS", torch.zeros(2, 64), None)
