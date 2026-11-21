# SPDX-License-Identifier: Apache-2.0
"""omk_qad.kquant: Q2_K..Q6_K and Q4_0/Q4_1/Q5_0/Q5_1 CUDA encoders against libggml, llama-quantize's code (bm-901)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from omk_qad import kquant
from omk_qad.tests.test_qad_cpu import setup  # noqa: F401  (the tiny teachers + caches fixture of the QAD smoke)

LLAMA = next((p for p in (os.environ.get("OMK_LLAMA_CPP"), "/opt/llama.cpp", "/mnt/sdc/ml/llama.cpp-b12386")
              if p and (Path(p) / "build/bin/libggml-base.so").is_file()), None)
needs_lib = pytest.mark.skipif(LLAMA is None, reason="no llama.cpp tree with build/bin/libggml-base.so")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
TYPES = sorted(kquant.SPECS)


def _lib():
    from omk_qad.kfac_round import GgmlNative

    return GgmlNative(LLAMA)


def _cases():
    g = torch.Generator().manual_seed(0)
    yield "gauss", torch.randn(16, 512, generator=g) * 0.03
    yield "wide", torch.randn(4, 2816, generator=g) * 0.5
    x = torch.randn(8, 512, generator=g)
    x[0, :256] = 0.0  # a zero super-block: every sub-block scale 0 (Q6_K: the memset path)
    x[1, 32:64] = 1e-30  # a sub-eps sub-block
    x[2, :256] = x[2, :256].abs()  # one-signed: min clamps to 0
    x[3, :256] = -x[3, :256].abs()  # all negative
    x[4, 64:96] = 0.0125  # a constant positive sub-block (max > min = 0)
    x[5, 96:128] = -0.0125  # a constant negative sub-block (max == min after the scan: the kx2/kx3 degenerate branch)
    x[6, 7] = 30.0  # one outlier dominating its super-block
    x[7, :16] = 0.0  # a zero Q6_K sub-block inside a live super-block (make_qx_quants all-zero branch, L = 0)
    yield "degenerate", x
    yield "large", torch.randn(4, 256, generator=g) * 3000.0  # fp16 rounding of d / dmin visible
    yield "thin", torch.randn(8, 256, generator=g) * 0.02  # one super-block per row: the legacy row sigma2 = 8 blocks


def _imats(cols):
    g = torch.Generator().manual_seed(7)
    yield "none", None
    yield "ones", torch.ones(cols)
    yield "rand", torch.rand(cols, generator=g) * 4 + 0.01
    z = torch.rand(cols, generator=g) * 2
    z[::3] = 0.0  # zero importance columns: zero weights, D = 0 paths in make_qkx3 / make_qp_quants
    yield "sparse", z


def _differ(a, b, nbytes):
    return int((a != b).reshape(-1, nbytes).any(-1).sum())


@needs_lib
@needs_cuda
@pytest.mark.parametrize("qtype", TYPES)
@pytest.mark.parametrize(("name", "x"), list(_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_kernel_matches_libggml(qtype, name, x):
    """The CUDA kernel writes libggml's bytes with every imatrix variant, degenerate blocks included."""
    lib = _lib()
    nbytes = kquant.SPECS[qtype][1]
    for iname, imat in _imats(x.shape[1]):
        ref = lib.encode(qtype, x, imat)
        got = kquant.encode(qtype, x.cuda(), None if imat is None else imat.cuda()).cpu()
        assert got.shape == ref.shape == (x.shape[0], x.shape[1] // kquant.SPECS[qtype][0], nbytes)
        nblk = ref.numel() // nbytes
        bad = _differ(got, ref, nbytes)
        assert torch.equal(got, ref), f"{qtype} {name}/{iname}: {bad} of {nblk} super-blocks differ"


@needs_lib
@pytest.mark.parametrize("qtype", TYPES)
def test_decode_matches_libggml(qtype):
    """The torch decoder reproduces libggml's dequantize_row on libggml's own bytes (both imatrix paths)."""
    lib = _lib()
    x = torch.randn(6, 768, generator=torch.Generator().manual_seed(3)) * 0.1
    for imat in (None, torch.rand(768) + 0.1):
        ref = lib.encode(qtype, x, imat)
        assert torch.equal(kquant.decode(qtype, ref, *x.shape), lib.decode(qtype, ref, *x.shape))


@needs_lib
def test_cpu_encode_is_libggml():
    """On the CPU encode is libggml itself (no torch port of the greedy searches)."""
    x = torch.randn(2, 512, generator=torch.Generator().manual_seed(5))
    imat = torch.rand(512) + 0.5
    for qtype in TYPES:
        assert torch.equal(kquant.encode(qtype, x, imat), _lib().encode(qtype, x, imat))
        assert torch.equal(kquant.encode(qtype, x, None), _lib().encode(qtype, x, None))


@needs_cuda
def test_kernel_scale_low_byte_survives():
    """The bug-832 / bug-966 signature (every fp16 scale's low byte 0) on d, dmin and Q6_K's trailing d."""
    x = torch.randn(32, 2816, generator=torch.Generator().manual_seed(2)) * 0.03
    for qtype, offs in (("Q4_K", (0, 2)), ("Q5_K", (0, 2)), ("Q6_K", (208,))):
        got = kquant.encode(qtype, x.cuda(), None).cpu().reshape(-1, kquant.SPECS[qtype][1])
        for o in offs:
            assert int((got[:, o] != 0).sum()) > got.shape[0] // 2, f"{qtype} byte {o}"


@needs_lib
@needs_cuda
@pytest.mark.parametrize("qtype", ["Q4_0", "Q4_1", "Q5_0", "Q5_1"])
def test_legacy_row_sigma2_is_the_rows(qtype):
    """The legacy imatrix weights use the whole row's sigma2; the kernel follows libggml on the row and on a slice.

    A 64-column slice encodes to different bytes than the row (the libggml behaviour kfac_round's GPTQ groups rely on).
    """
    lib = _lib()
    x = torch.randn(4, 512, generator=torch.Generator().manual_seed(9)) * 0.05
    imat = torch.rand(512) + 0.2
    full = lib.encode(qtype, x, imat)
    assert torch.equal(kquant.encode(qtype, x.cuda(), imat.cuda()).cpu(), full)
    sl = kquant.encode(qtype, x[:, :64].cuda().contiguous(), imat[:64].cuda()).cpu()
    assert torch.equal(sl, lib.encode(qtype, x[:, :64].contiguous(), imat[:64]))
    assert not torch.equal(sl, full[:, :2])


def test_rejects_bad_shapes():
    """Wrong block multiple, dtype, importance length or type name are refused before any encode."""
    with pytest.raises(ValueError):
        kquant.encode("Q4_K", torch.zeros(2, 288), None)
    with pytest.raises(ValueError):
        kquant.encode("Q4_0", torch.zeros(2, 48), None)
    with pytest.raises(ValueError):
        kquant.encode("Q6_K", torch.zeros(2, 256, dtype=torch.bfloat16), None)
    with pytest.raises(ValueError):
        kquant.encode("Q5_K", torch.zeros(2, 256), torch.ones(255))
    with pytest.raises(ValueError):
        kquant.encode("Q8_0", torch.zeros(2, 256), None)


@needs_lib
@needs_cuda
@pytest.mark.parametrize("qtype", ["Q4_K", "Q6_K", "Q5_0"])
def test_qad_student_ships_the_kernel_blocks(setup, qtype):  # noqa: F811
    """QAD smoke on the GPU path (bm-901 acceptance): experts fixed at a K-quant / legacy type, 2 steps at the kernel
    cadence, export, and every expert tensor of the GGUF is byte-identical to the payload training served; llama.cpp
    loads the file. The tiny MoE, cache and TRAIN args are test_native_quants' / test_qad_cpu's.
    """  # noqa: D205
    import json
    import os
    import shutil
    import subprocess

    import numpy as np

    from omk_qad import train as T
    from omk_qad.layout import build_layout
    from omk_qad.tests import test_native_quants as NQ
    from omk_qad.tests.test_qad_cpu import LLAMA_CPP, TRAIN, WORK, _args, _flat

    gguf = NQ._gguf()
    model, _ = setup["moe"]
    t = qtype.lower()
    lay = WORK / f"layout2_moe_{t}.pt"
    if not lay.exists():
        torch.save(build_layout(model, LLAMA_CPP, "IQ2_M", fixed={"routed_experts": t, "shared_experts": t},
                                work=WORK / f"lay_moe_{t}"), lay)
    run, out = WORK / f"run_moe_{t}_cuda", WORK / f"qad_moe_{t}_cuda.gguf"
    if run.exists():
        shutil.rmtree(run)
    cache = NQ._cache_on("cuda", setup)
    train = [a if a != "cpu" else "cuda" for a in TRAIN]
    NQ._run_on("cuda", "omk_qad.train", *train, "--model", str(model), "--layout", str(lay), "--cache", str(cache),
               "--out", str(run), "--steps", "2", "--eval-every", "2", "--save-every", "2")
    s1 = run / "stage1"
    assert "QAD_DONE" in (s1 / "log.txt").read_text()
    NQ._run_on("cuda", "omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device",
               "cuda", "--load", str(s1 / "final.pt"), "--gguf", str(out), "--latent-dtype", "fp16", "--latent-min",
               "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    assert rep["type_counts"].get(qtype, 0) >= 1, rep["type_counts"]
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp16", imatrix=str(cache / "imatrix.pt")),
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
    assert checked >= 4, checked
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(out), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]
