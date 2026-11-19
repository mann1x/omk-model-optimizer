"""kfac_round's IQ2 path: libggml encode/decode behind the iq3/iq4 contract (bm-895, 2-bit arm)."""

import os
from pathlib import Path

import pytest
import torch

from omk_qad import kfac_round

LLAMA = next((p for p in (os.environ.get("OMK_LLAMA_CPP"), "/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")
              if p and (Path(p) / "build/bin/libggml-base.so").is_file()), None)
needs_lib = pytest.mark.skipif(LLAMA is None, reason="no llama.cpp tree with build/bin/libggml-base.so")


def test_ggml_type_without_llama_cpp_is_refused(monkeypatch):
    """A libggml-only type needs --llama-cpp, said early; an unknown type is refused by name.

    The example is the first entry of ``kfac_round.GGML_TYPES`` rather than a literal: IQ2_XS was the literal until
    its kernel landed (bm-900, 2026-11-19) and the test then asserted a refusal that no longer exists (bug-972). When
    every type has a kernel the first half of the test has nothing to check and is skipped.
    """
    monkeypatch.setitem(kfac_round._GGML, "mod", None)
    if kfac_round.GGML_TYPES:
        with pytest.raises(ValueError, match="--llama-cpp"):
            kfac_round.native(kfac_round.GGML_TYPES[0])
    for kernel_type in kfac_round.KERNEL_TYPES:
        assert kfac_round.native(kernel_type) is not None  # a kernel type never needs libggml
    with pytest.raises(ValueError, match="not a supported type"):
        kfac_round.native("BF16")


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


@needs_lib
def test_iq2_gptq_fp16_latent_reencodes_to_its_own_payload():
    """With latent_dtype=fp16 the returned W_adj is fp16-exact and its block-wise re-encode IS the payload (bm-892)."""
    kfac_round._GGML["mod"] = kfac_round.GgmlNative(LLAMA)
    try:
        g = torch.Generator().manual_seed(2)
        w0 = torch.randn(16, 512, generator=g) * 0.05
        mix = torch.eye(512) + 0.3 * torch.randn(512, 512, generator=g) / 512 ** 0.5
        xs = torch.randn(4096, 512, generator=g) @ mix
        a = xs.T @ xs / 4096
        imp = a.diagonal().clone()
        p, _, latent, _ = kfac_round.gptq_expert(w0, a, imp, "IQ2_XXS", 0.01, torch.float16)
        assert torch.equal(latent, latent.half().float()), "latent is not fp16-exact"
        stored = latent.half().float()  # what the trainer holds after --init-latents
        again = torch.cat([kfac_round.encode_decode("IQ2_XXS", stored[:, s:s + 256], imp[s:s + 256])[0]
                           for s in range(0, 512, 256)], 1)
        assert torch.equal(again, p), "re-encoding the stored fp16 seed does not give the saved payload"
        p32, _, lat32, _ = kfac_round.gptq_expert(w0, a, imp, "IQ2_XXS", 0.01)
        assert not torch.equal(lat32, lat32.half().float())  # the fp32 path is what the old files were encoded from
    finally:
        kfac_round._GGML["mod"] = None


@needs_lib
@pytest.mark.parametrize("qtype", ["Q4_K", "Q6_K", "Q3_K", "Q5_0", "IQ4_NL", "Q8_0"])
def test_kquant_types_round_through_libggml(qtype):
    """The ladder's K-quant and block-32 types keep the encode/decode contract; GPTQ never worse than RTN.

    On the CPU the K-quants and legacy types reach libggml through omk_qad.kquant (bm-901), IQ4_NL / Q8_0 their torch
    references.

    IQ4_NL and Q8_0 blocks are row-independent (a 64-column slice encodes to the row's bytes); Q4_0/Q5_0 weight
    each block with the ROW's sigma2 in libggml, so a GPTQ group of them carries the group-local sigma2 instead.
    """
    kfac_round._GGML["mod"] = kfac_round.GgmlNative(LLAMA)
    try:
        g = torch.Generator().manual_seed(3)
        w0 = torch.randn(16, 512, generator=g) * 0.05
        mix = torch.eye(512) + 0.3 * torch.randn(512, 512, generator=g) / 512 ** 0.5
        xs = torch.randn(4096, 512, generator=g) @ mix
        a = xs.T @ xs / 4096
        imp = a.diagonal().clone()
        p, q, _ = kfac_round.rtn_expert(w0, imp, qtype)
        assert p.shape[:2] == (16, 2 if qtype.endswith("_K") else 16) and q.shape == w0.shape  # 256- / 32-blocks
        # a block-32 type encodes a 64-column slice to the same bytes as the full row (independent blocks)
        if qtype in ("IQ4_NL", "Q8_0"):
            sl = torch.cat([kfac_round.encode_decode(qtype, w0[:, s:s + 64], imp[s:s + 64])[0]
                            for s in range(0, 512, 64)], 1)
            assert torch.equal(sl, p)
        group = kfac_round.group_of(w0.shape[1])
        corr = torch.corrcoef(torch.stack([w0.flatten(), q.flatten()]))[0, 1]
        assert corr > 0.95, float(corr)
        pg, qg, latent, _ = kfac_round.gptq_expert(w0, a, imp, qtype, 0.01, torch.float16, group)
        gm = torch.eye(16)
        assert kfac_round.loss_terms(qg - w0, a, gm)[0] <= kfac_round.loss_terms(q - w0, a, gm)[0] * 1.02
        assert torch.equal(latent, latent.half().float())
    finally:
        kfac_round._GGML["mod"] = None


def test_layer_qtypes_auto_needs_ref_and_known_types():
    """--qtype auto follows the --ref tensor types and refuses unknown ones."""
    with pytest.raises(SystemExit, match="needs --ref"):
        kfac_round.layer_qtypes("auto", None, [0])

    class T:
        def __init__(self, n):
            self.tensor_type = type("TT", (), {"name": n})()

    class Ref:
        t = {"blk.0.ffn_gate_up_exps.weight": T("Q4_K"), "blk.1.ffn_gate_up_exps.weight": T("IQ2_XXS")}

    gu = "ffn_gate_up_exps"
    assert kfac_round.layer_qtypes("auto", Ref(), [0, 1]) == {(0, gu): "Q4_K", (1, gu): "IQ2_XXS"}
    assert kfac_round.layer_qtypes("IQ4_XS", None, [0, 1]) == {(0, gu): "IQ4_XS", (1, gu): "IQ4_XS"}
    with pytest.raises(SystemExit, match="needs --qtype auto"):
        kfac_round.layer_qtypes("IQ4_XS", None, [0], ["ffn_gate_up_exps", "ffn_down_exps"])
    Ref.t["blk.0.ffn_down_exps.weight"] = T("IQ4_NL")
    Ref.t["blk.1.ffn_down_exps.weight"] = T("Q8_0")
    assert kfac_round.layer_qtypes("auto", Ref(), [0, 1], kfac_round.STEM_SETS["routed"]) == {
        (0, gu): "Q4_K", (1, gu): "IQ2_XXS", (0, "ffn_down_exps"): "IQ4_NL", (1, "ffn_down_exps"): "Q8_0"}
    assert kfac_round.group_of(2816) == 256 and kfac_round.group_of(704) == 64 and kfac_round.group_of(2112) == 64
    Ref.t["blk.1.ffn_gate_up_exps.weight"] = T("BF16")
    with pytest.raises(SystemExit, match="not a type this rounder encodes"):
        kfac_round.layer_qtypes("auto", Ref(), [0, 1])


def test_write_payload_opens_the_output_once_and_flushes_on_demand(tmp_path):
    """2026-11-19: re-opening the output GGUF per tensor cost ~17 s x 150 per tier (42 of 67 min). One reader per
    run, payload lands, flush(close=True) makes it visible to a fresh reader and drops the mapping.
    """  # noqa: D205
    import numpy as np

    from omk_qad import kfac_round as R
    from omk_qad.gguf_export import _import_llama_cpp
    from omk_qad.tests.test_qad_cpu import LLAMA_CPP

    gguf = _import_llama_cpp(LLAMA_CPP)[0]
    out = tmp_path / "tiny.gguf"
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_block_count(1)
    a = np.zeros((2, 4, 8), dtype=np.float32)
    b = np.zeros((4, 8), dtype=np.float32)
    w.add_tensor("blk.0.ffn_gate_up_exps.weight", a)
    w.add_tensor("blk.0.ffn_gate.weight", b)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    pa = np.arange(a.nbytes, dtype=np.uint8)
    pb = np.arange(b.nbytes, dtype=np.uint8)[::-1].copy()
    R.write_payload(str(out), 0, pa, "ffn_gate_up_exps")
    r1 = R._OUT[str(out)][0]
    R.write_payload(str(out), 0, pb, "ffn_gate")
    assert R._OUT[str(out)][0] is r1  # one reader for the run, not one per tensor
    with pytest.raises(SystemExit, match="payload 3 bytes"):
        R.write_payload(str(out), 0, np.zeros(3, dtype=np.uint8), "ffn_gate")
    R.flush_payloads(str(out))
    assert str(out) in R._OUT
    R.flush_payloads(str(out), close=True)
    assert str(out) not in R._OUT
    rd = gguf.GGUFReader(str(out))
    t = {x.name: x for x in rd.tensors}
    assert np.array_equal(np.asarray(t["blk.0.ffn_gate_up_exps.weight"].data).reshape(-1).view(np.uint8), pa)
    assert np.array_equal(np.asarray(t["blk.0.ffn_gate.weight"].data).reshape(-1).view(np.uint8), pb)
