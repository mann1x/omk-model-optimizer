# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103, F811
"""IQ2_XXS / IQ2_S as native QAD formats (omk_qad.iq2, bm-896).

Bytes == libggml (llama-quantize) for the torch reference and the CUDA kernel, decode == gguf-py, no CPU fallback on
a CUDA device, the codec and the K-FAC rounder route the types to the kernel, and a student trained at each type
ships exactly the blocks it was served. Needs a built llama.cpp (libggml-base.so); artifacts go to $OMK_QAD_WORK.
"""

import numpy as np
import pytest
import torch

from omk_qad.tests.test_native_quants import (  # noqa: F401
    CUDA,
    LLAMA_CPP,
    _flat,
    _gguf,
    _inputs,
    _mod,
    lib,
    setup,
)
from omk_qad.tests.test_native_quants import (
    test_train_export_ships_the_trained_blocks as _train_export,
)

QTYPES = ("IQ2_XXS", "IQ2_S")


def _imats(qtype, imats):
    """IQ2_XXS cannot be encoded without an imatrix (llama.cpp aborts); IQ2_S has a no-imatrix weighting."""
    return [i for i in imats if i is not None or qtype == "IQ2_S"]


def test_tables_match_iq2xs_init_impl():
    from omk_qad import iq2

    for qtype, n, nwant in (("IQ2_XXS", 256, 2), ("IQ2_S", 1024, 1)):
        t = iq2.tables(qtype)
        assert t["grid"].shape == (n, 8) and set(np.unique(t["grid"])) == {1, 3, 5}
        assert t["kmap"].shape == (iq2.KMAP,) and (t["kmap"] >= 0).sum() == n  # every grid point maps to itself
        on = np.flatnonzero(t["kmap"] >= 0)
        codes = [sum(((v - 1) // 2) << 2 * i for i, v in enumerate(p)) for p in t["grid"]]
        assert sorted(codes) == sorted(on.tolist())
        # an off-grid code's list holds the nearest `nwant` distance shells, nearest first, by (d2, index)
        u = int(np.flatnonzero(t["kmap"] < 0)[1234])
        off = -t["kmap"][u] - 1
        cnt = t["neighbours"][off]
        lst = t["neighbours"][off + 1 : off + 1 + cnt]
        pos = np.array([2 * ((u >> 2 * k) & 3) + 1 for k in range(8)])
        d2 = ((t["grid"] - pos) ** 2).sum(1)
        shells = sorted(set(d2.tolist()))[:nwant]
        assert sorted(lst.tolist()) == sorted(np.flatnonzero(np.isin(d2, shells)).tolist())
        assert list(lst) == sorted(lst, key=lambda j: (d2[j], j))


@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_bytes_equal_libggml(lib, qtype, seed):
    qt = int(_gguf().GGMLQuantizationType[qtype])
    x, imats = _inputs(seed)
    for imat in _imats(qtype, imats):
        want = lib.quantize(qt, x, imat).reshape(-1)
        got = _mod(qtype).encode_reference(qtype, torch.from_numpy(x),
                                           None if imat is None else torch.from_numpy(imat))
        assert np.array_equal(_flat(got), want), (qtype, seed, imat is None)


def test_iq2_xxs_requires_an_imatrix():
    with pytest.raises(ValueError, match="requires an importance matrix"):
        _mod("IQ2_XXS").encode("IQ2_XXS", torch.randn(2, 256))


@CUDA
@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cuda_bytes_equal_libggml(lib, qtype, seed):
    mod = _mod(qtype)
    qt = int(_gguf().GGMLQuantizationType[qtype])
    x, imats = _inputs(seed, rows=96, cols=2048)
    xc = torch.from_numpy(x).cuda()
    for imat in _imats(qtype, imats):
        want = lib.quantize(qt, x, imat).reshape(-1)
        got = mod.encode(qtype, xc, None if imat is None else torch.from_numpy(imat).cuda())
        assert got.is_cuda
        nb = mod.SPECS[qtype][1]
        bad = np.flatnonzero((_flat(got).reshape(-1, nb) != want.reshape(-1, nb)).any(1))
        assert bad.size == 0, (qtype, seed, imat is None, f"{bad.size} blocks differ", bad[:8])


@pytest.mark.parametrize("qtype", QTYPES)
def test_decode_equals_gguf_py(lib, qtype):
    gguf = _gguf()
    x, imats = _inputs(5)
    q = lib.quantize(int(gguf.GGMLQuantizationType[qtype]), x, imats[1])
    want = gguf.quants.dequantize(q, gguf.GGMLQuantizationType[qtype]).reshape(x.shape)
    got = _mod(qtype).decode(qtype, torch.from_numpy(q), *x.shape)
    assert np.array_equal(got.numpy(), want)


@CUDA
@pytest.mark.parametrize("qtype", QTYPES)
def test_cuda_never_falls_back(monkeypatch, qtype):
    """A CUDA input with no buildable extension raises; it is never encoded on the CPU behind the trainer's back."""
    mod = _mod(qtype)

    def broken():
        raise RuntimeError("omk_qad CUDA encoder is unavailable")

    monkeypatch.setattr(mod, "_ext", broken)
    monkeypatch.setattr(mod, "encode_reference", lambda *a, **k: pytest.fail("CPU fallback taken"))
    with pytest.raises(RuntimeError, match="unavailable"):
        mod.encode(qtype, torch.randn(4, 256, device="cuda"), torch.ones(256, device="cuda"))


@CUDA
@pytest.mark.parametrize("qtype", QTYPES)
def test_codec_routes_native_types_to_the_gpu(qtype):
    from omk_qad.fakequant import GPU_TYPES, MODELOPT_TYPES, Codec

    assert qtype in GPU_TYPES and qtype not in MODELOPT_TYPES  # the kernel, at the refresh cadence; not ModelOpt
    codec = Codec(LLAMA_CPP)
    x = torch.randn(32, 512, device="cuda") * 0.02
    payload = codec.encode(qtype, x, torch.ones(512, device="cuda"))
    assert payload.is_cuda
    d = codec.decode(qtype, payload, 32, 512, "cpu")
    assert d.device.type == "cpu" and tuple(d.shape) == (32, 512)


def test_codec_cpu_path_is_the_reference(lib):
    """On a CPU device the codec encodes with the torch reference and the importance is llama.cpp's qw (E[x^2])."""
    from omk_qad.fakequant import Codec
    from omk_qad.imatrix import importance_for

    codec = Codec(LLAMA_CPP)
    x, imats = _inputs(7, rows=8)  # 24 x 1024: _inputs' edge blocks need 1024 columns
    msq = torch.from_numpy(imats[1])
    assert torch.equal(importance_for("IQ2_XXS", msq), msq.float())
    for qtype in QTYPES:
        got = codec.encode(qtype, torch.from_numpy(x), importance_for(qtype, msq))
        want = lib.quantize(int(_gguf().GGMLQuantizationType[qtype]), x, imats[1]).reshape(-1)
        assert np.array_equal(_flat(got), want), qtype
        assert np.array_equal(codec.to_bytes(qtype, got, x.shape[0]).reshape(-1), want), qtype


def test_codec_decodes_a_libggml_payload_of_a_native_type(lib):
    """A checkpoint written while IQ2 went through --ggml-encode holds numpy payloads: the resume must serve them."""
    from omk_qad.fakequant import Codec

    codec = Codec(LLAMA_CPP)
    x, imats = _inputs(3, rows=8)
    for qtype in QTYPES:
        q = lib.quantize(int(_gguf().GGMLQuantizationType[qtype]), x, imats[1])  # numpy [rows, row bytes]
        want = codec.decode(qtype, codec.encode(qtype, torch.from_numpy(x), torch.from_numpy(imats[1])), *x.shape,
                            "cpu")
        got = codec.decode(qtype, q, *x.shape, "cpu")
        assert torch.equal(got, want), qtype


def test_force_ggml_takes_a_native_type_off_the_gpu(monkeypatch):
    from omk_qad import fakequant as F

    monkeypatch.setattr(F, "MODELOPT_TYPES", set(F.MODELOPT_TYPES))
    monkeypatch.setattr(F, "NATIVE_TYPES", dict(F.NATIVE_TYPES))
    monkeypatch.setattr(F, "GPU_TYPES", set(F.GPU_TYPES))
    assert F.force_ggml(["iq2_xxs"]) == {"IQ2_XXS"}
    assert "IQ2_XXS" not in F.NATIVE_TYPES and "IQ2_XXS" not in F.GPU_TYPES and "IQ2_S" in F.NATIVE_TYPES
    with pytest.raises(ValueError, match="not GPU-encoded"):
        F.force_ggml(["Q4_K"])


def test_kfac_rounder_routes_iq2_to_the_kernel():
    from omk_qad import iq2, kfac_round

    for qtype in QTYPES:
        assert qtype in kfac_round.KERNEL_TYPES and qtype not in kfac_round.GGML_TYPES
        assert kfac_round.native(qtype) is iq2


def test_census_formats_price_the_gguf_bytes():
    from omk_qad.census import FORMATS

    for t in ("iq2_xxs", "iq2_s"):
        block, bits = FORMATS[t]
        assert bits * block / 8 == _mod(t.upper()).SPECS[t.upper()][1], t


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("qtype", QTYPES)
def test_train_export_ships_the_trained_blocks(setup, qtype, device):
    """test_native_quants' end-to-end check (2 QAD steps, export, GGUF bytes == served payloads) at the IQ2 types."""
    _train_export(setup, qtype, device)
