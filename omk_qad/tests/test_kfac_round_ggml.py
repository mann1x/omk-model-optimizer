"""kfac_round's IQ2 path: libggml encode/decode behind the iq3/iq4 contract (bm-895, 2-bit arm)."""

import os
from pathlib import Path

import pytest
import torch

from omk_qad import kfac_round

LLAMA = next((p for p in (os.environ.get("OMK_LLAMA_CPP"), "/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")
              if p and (Path(p) / "build/bin/libggml-base.so").is_file()), None)
needs_lib = pytest.mark.skipif(LLAMA is None, reason="no llama.cpp tree with build/bin/libggml-base.so")


def test_iq2_without_llama_cpp_is_refused(monkeypatch):
    """An IQ2 type is only encodable through libggml; without --llama-cpp the module must say so, early."""
    monkeypatch.setitem(kfac_round._GGML, "mod", None)
    with pytest.raises(ValueError, match="--llama-cpp"):
        kfac_round.native("IQ2_XXS")
    with pytest.raises(ValueError, match="not a supported type"):
        kfac_round.native("Q4_K")


@needs_lib
@pytest.mark.parametrize("qtype", ["IQ2_XXS", "IQ2_S"])
def test_iq2_encode_decode_contract(qtype):
    """Payload [rows, cols/256, block bytes], decode back to [rows, cols], deterministic, and a real 2-bit fit."""
    m = kfac_round.GgmlNative(LLAMA)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 512, generator=g)
    imp = torch.rand(512, generator=g) + 0.5
    p = m.encode(qtype, x, imp)
    assert p.dtype == torch.uint8 and p.shape[:2] == (8, 2)
    assert p.shape[2] == m.lib.row_size(m._id(qtype), 512) // 2
    q = m.decode(qtype, p, 8, 512)
    assert q.shape == (8, 512) and q.dtype == torch.float32
    corr = torch.corrcoef(torch.stack([x.flatten(), q.flatten()]))[0, 1]
    assert corr > 0.8, float(corr)  # a 2-bit lattice fit of Gaussian data, not garbage
    assert torch.equal(m.encode(qtype, x, imp), p)


@needs_lib
def test_iq2_gptq_runs_at_block_granularity():
    """gptq_expert with the libggml encoder: the error feedback keeps the curvature-weighted error at or below RTN."""
    kfac_round._GGML["mod"] = kfac_round.GgmlNative(LLAMA)
    try:
        g = torch.Generator().manual_seed(1)
        w0 = torch.randn(16, 512, generator=g)
        mix = torch.eye(512) + 0.3 * torch.randn(512, 512, generator=g) / 512 ** 0.5
        xs = torch.randn(4096, 512, generator=g) @ mix
        a = xs.T @ xs / 4096
        imp = a.diagonal().clone()
        gm = torch.eye(16)
        _, q_rtn, _ = kfac_round.rtn_expert(w0, imp, "IQ2_XXS")
        p, q, latent, _ = kfac_round.gptq_expert(w0, a, imp, "IQ2_XXS", 0.01)
        assert p.shape == (16, 2, p.shape[2]) and latent.shape == w0.shape
        l_rtn = kfac_round.loss_terms(q_rtn - w0, a, gm)[0]
        l_gptq = kfac_round.loss_terms(q - w0, a, gm)[0]
        assert l_gptq <= l_rtn * 1.02, (l_gptq, l_rtn)
    finally:
        kfac_round._GGML["mod"] = None
