# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""IQ4_XS / IQ4_NL / IQ3_XXS / IQ3_S as native QAD formats (omk_qad.iq4, omk_qad.iq3).

Bytes == libggml (llama-quantize), decode == gguf-py, no CPU fallback on a CUDA device, and a student trained at
each type ships exactly the blocks it was served.
Needs a built llama.cpp (libggml-base.so, llama-quantize, llama-perplexity); artifacts go to $OMK_QAD_WORK.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from omk_qad.tests.test_qad_cpu import (  # noqa: F401
    LLAMA_CPP,
    TRAIN,
    WORK,
    _args,
    _flat,
    _run,
    setup,
)

QTYPES = ("IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_S")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _mod(qtype: str):
    from omk_qad.fakequant import _native

    return _native(qtype)


def _gguf():
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    return gguf


def _inputs(seed: int, rows: int = 48, cols: int = 1024):
    """Gaussian, heavy-tailed and outlier rows, plus all-zero, sub-GROUP_MAX_EPS and denormal-product blocks."""
    g = np.random.default_rng(seed)
    x = np.concatenate([g.standard_normal((rows, cols)) * 0.02, g.standard_t(2, (rows, cols)) * 0.01,
                        g.standard_normal((rows, cols)) * 0.02 * (g.random((rows, cols)) < 0.01) * 50], 0)
    x[0, :256] = 0.0
    x[1, 256:512] = 1e-17
    x[2, 512:768] = g.standard_normal(256) * 2e-14  # sum(w q x) ~ x^3 is float32-denormal: no flush-to-zero anywhere
    x[3, 256:288] = 0.0  # one zero sub-block in a live super-block (IQ3_S then stores the rest one slot early)
    x = x.astype(np.float32)
    imats = (None, (g.random(cols) * 3).astype(np.float32) ** 2, np.ones(cols, np.float32))
    return x, imats


@pytest.fixture(scope="module")
def lib():
    from omk_qad.ggml_lib import GGMLLib

    return GGMLLib(LLAMA_CPP)


@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_bytes_equal_libggml(lib, qtype, seed):
    qt = int(_gguf().GGMLQuantizationType[qtype])
    x, imats = _inputs(seed)
    for imat in imats:
        want = lib.quantize(qt, x, imat).reshape(-1)
        got = _mod(qtype).encode_reference(qtype, torch.from_numpy(x),
                                           None if imat is None else torch.from_numpy(imat))
        assert np.array_equal(_flat(got), want), (qtype, seed, imat is None)


@CUDA
@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cuda_bytes_equal_libggml(lib, qtype, seed):
    mod = _mod(qtype)
    qt = int(_gguf().GGMLQuantizationType[qtype])
    x, imats = _inputs(seed, rows=96, cols=2048)
    xc = torch.from_numpy(x).cuda()
    for imat in imats:
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
        mod.encode(qtype, torch.randn(4, 256, device="cuda"))


@CUDA
@pytest.mark.parametrize("qtype", QTYPES)
def test_codec_routes_native_types_to_the_gpu(qtype):
    from omk_qad.fakequant import GPU_TYPES, Codec

    assert qtype in GPU_TYPES  # re-encoded at the GPU cadence (refresh), not ggml_refresh
    codec = Codec(LLAMA_CPP)
    x = torch.randn(32, 512, device="cuda") * 0.02
    payload = codec.encode(qtype, x, None)
    assert payload.is_cuda
    d = codec.decode(qtype, payload, 32, 512, "cpu")
    assert d.device.type == "cpu" and tuple(d.shape) == (32, 512)


@pytest.mark.parametrize("qtype", QTYPES)
def test_ptq_refuses_native_types(qtype):
    from omk_qad.census import _numerics

    with pytest.raises(ValueError, match="not a ModelOpt ggml format"):
        _numerics(qtype.lower())


def _run_on(device: str, *args: str) -> None:
    """``python -m args`` on ``device`` (test_qad_cpu._run always hides the GPUs)."""
    env = {**os.environ, "WANDB_MODE": "disabled", **({"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else {})}
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]


def _cache_on(device: str, setup) -> Path:  # noqa: F811
    """The teacher cache for ``device``.

    A CPU-built cache fails the lossless gate on the GPU (klk 0.079 on the sharp tiny teacher, with the IQ2_XXS
    layout too), so the CUDA run gets its own, built on the GPU.
    """
    if device == "cpu":
        return setup["cache"]
    model, lay = setup["moe"]
    cache = WORK / "cache_moe_cuda"
    if not all((cache / f).exists() for f in ("cache.json", "imatrix.pt")):
        _run_on("cuda", "omk_qad.train", "--phase", "cache", "--model", str(model), "--cache", str(cache), "--layout",
                str(lay), "--device", "cuda", "--data", str(setup["text"]), "--eval-data", str(setup["text"]),
                "--n-seq", "8", "--n-eval", "2", "--seq", "64", "--shard", "4", "--llama-cpp", str(LLAMA_CPP))
    return cache


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("qtype", QTYPES)
def test_train_export_ships_the_trained_blocks(setup, qtype, device):  # noqa: F811
    """Experts fixed at the type: 2 steps, export, and every GGUF tensor is byte-identical to what training served.

    cuda: trained and exported through the CUDA kernel, checked against payloads rebuilt on the CPU (torch
    reference), so the kernel and the reference agree on real trained latents, not only on synthetic matrices.
    """
    gguf = _gguf()
    from omk_qad import train as T
    from omk_qad.layout import build_layout

    model, _ = setup["moe"]
    t = qtype.lower()
    lay = WORK / f"layout2_moe_{t}.pt"
    if not lay.exists():
        torch.save(build_layout(model, LLAMA_CPP, "IQ2_M", fixed={"routed_experts": t, "shared_experts": t},
                                work=WORK / f"lay_moe_{t}"), lay)
    run, out = WORK / f"run_moe_{t}_{device}", WORK / f"qad_moe_{t}_{device}.gguf"
    if run.exists():
        shutil.rmtree(run)  # this test's own scratch
    cache = _cache_on(device, setup)
    train = [a if a != "cpu" else device for a in TRAIN]
    _run_on(device, "omk_qad.train", *train, "--model", str(model), "--layout", str(lay), "--cache", str(cache),
            "--out", str(run), "--steps", "2", "--eval-every", "2", "--save-every", "2")
    s1 = run / "stage1"
    assert "QAD_DONE" in (s1 / "log.txt").read_text()
    _run_on(device, "omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device",
            device, "--load", str(s1 / "final.pt"), "--gguf", str(out), "--latent-dtype", "fp16", "--latent-min",
            "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    planned = torch.load(lay, weights_only=False)["targets"]
    # the 4 layers x (gate/up/down exps + shexp) we fixed, plus whatever llama.cpp's own rules put at this type
    # (IQ2_M rest: token_embd at IQ3_S)
    assert rep["type_counts"][qtype] == sum(t["type"] == qtype for t in planned.values()) >= 4 * 6

    imat = str(cache / "imatrix.pt")
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp16", imatrix=imat),
                                             torch.load(lay, weights_only=False))
    T.load_state(str(s1 / "final.pt"), lats, student, None)
    tensors = {x.name: x for x in gguf.GGUFReader(out).tensors}
    checked = 0
    for lat in lats.values():
        for g, chunks in lat.payloads().items():
            if tensors[g].tensor_type.name != qtype or not g.endswith(("_exps.weight", "_shexp.weight")):
                continue
            want = np.concatenate([_flat(c) for c in chunks])
            assert np.array_equal(np.asarray(tensors[g].data).reshape(-1), want), g
            checked += 1
    assert checked == 4 * 6

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(out), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]


def test_census_formats_and_llama_cpp_fallbacks():
    from omk_qad.census import FALLBACK, FORMATS

    assert FORMATS["iq4_xs"] == (256, 4.25) and FORMATS["iq4_nl"] == (32, 4.5)
    assert FORMATS["iq3_xxs"] == (256, 3.0625) and FORMATS["iq3_s"] == (256, 3.4375)
    for t in ("iq3_xxs", "iq3_s", "iq4_xs"):  # llama-quantize: rows not a multiple of 256 -> IQ4_NL
        assert FALLBACK[t] == "iq4_nl", t
    assert FALLBACK["iq4_nl"] == "q8_0"
    for t in ("iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs"):  # the byte sizes the census prices are the GGUF's
        block, bits = FORMATS[t]
        assert bits * block / 8 == _mod(t.upper()).SPECS[t.upper()][1], t
