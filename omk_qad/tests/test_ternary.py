# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""T1 ternary codec: bit-exact against ports of the C reference and against a shipped PrismML GGUF; ternarizers.

The C reference is ``opencoti/vendors/patches/llamafile/0590-bonsai2-types.patch`` (quantize_row_pq2_0_ref,
dequantize_row_pq2_0, quantize_row_ptq1_0_ref, dequantize_row_ptq1_0), ported loop for loop below.
The PrismML file check runs when $OMK_BONSAI_GGUF (default: the opencoti-bench copy) exists.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from omk_qad.ternary import (
    BLOCK_BYTES,
    G,
    TernaryCodec,
    decode,
    gguf_raw_rows,
    gguf_tensor_index,
    pack,
    ternarize,
    unpack,
)

BONSAI = Path(os.environ.get(
    "OMK_BONSAI_GGUF",
    "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/opencoti-bench/bonsai2/"
    "Ternary-Bonsai-2-27B-PTQ1_0-MTP.gguf"))


# --- C reference, ported loop for loop ---------------------------------------------------------------------------
def _roundf(v: float) -> int:  # C roundf / lroundf: half away from zero
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def _f16(v: float) -> bytes:
    return np.float16(np.float32(v)).tobytes()


def c_quantize_pq2_0(x: np.ndarray) -> bytes:
    out = bytearray()
    for i in range(len(x) // G):
        blk = x[i * G:(i + 1) * G].astype(np.float32)
        d = np.float32(np.abs(blk).max())
        idd = np.float32(1.0) / d if d > 0 else np.float32(0.0)
        qs = [0] * (G // 4)
        for j in range(G):
            q = min(max(_roundf(float(np.float32(blk[j] * idd))) + 1, 0), 3)
            qs[j // 4] |= q << ((j % 4) * 2)
        out += _f16(d) + bytes(qs)
    return bytes(out)


def c_dequantize_pq2_0(raw: bytes) -> np.ndarray:
    y = []
    for i in range(len(raw) // 34):
        b = raw[i * 34:(i + 1) * 34]
        d = np.frombuffer(b[:2], np.float16)[0].astype(np.float32)
        y += [np.float32(((b[2 + j // 4] >> ((j % 4) * 2)) & 3) - 1) * d for j in range(G)]
    return np.array(y, np.float32)


STAGES = (32, 16, 8)


def c_quantize_ptq1_0(x: np.ndarray) -> bytes:
    out = bytearray()
    x = x.astype(np.float32)
    pos = 0
    for _ in range(len(x) // G):
        d = np.float32(np.abs(x[pos:pos + G]).max())
        idd = np.float32(1.0) / d if d else np.float32(0.0)
        qs, qh = [0] * 24, [0] * 2
        j = 0
        for c in STAGES:
            while j + c <= 24:
                for m in range(c):
                    q = 0
                    for n in range(5):
                        q = q * 3 + _roundf(float(np.float32(x[pos + m + n * c] * idd))) + 1
                    qs[j + m] = (q * 256 + 242) // 243
                pos += 5 * c
                j += c
        for h in range(2):
            q = 0
            for m in range(4):
                q = q * 3 + _roundf(float(np.float32(x[pos + h + m * 2] * idd))) + 1
            qh[h] = (q * 3 * 256 + 242) // 243
        pos += 8
        out += bytes(qs) + bytes(qh) + _f16(d)
    return bytes(out)


def c_dequantize_ptq1_0(raw: bytes) -> np.ndarray:
    pow3 = (1, 3, 9, 27, 81, 243)
    y = []
    for i in range(len(raw) // 28):
        b = raw[i * 28:(i + 1) * 28]
        qs, qh = b[:24], b[24:26]
        d = np.frombuffer(b[26:28], np.float16)[0].astype(np.float32)
        j = 0
        for c in STAGES:
            while j + c <= 24:
                for n in range(5):
                    for m in range(c):
                        q = (qs[j + m] * pow3[n]) & 0xFF
                        y.append(np.float32(((q * 3) >> 8) - 1) * d)
                j += c
        for n in range(4):
            for h in range(2):
                q = (qh[h] * pow3[n]) & 0xFF
                y.append(np.float32(((q * 3) >> 8) - 1) * d)
    return np.array(y, np.float32)


C_REF = {"PQ2_0": (c_quantize_pq2_0, c_dequantize_pq2_0), "PTQ1_0": (c_quantize_ptq1_0, c_dequantize_ptq1_0)}


def _weights(rows=6, n=3 * G, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, n, generator=g) * 0.02
    x[0, :G] = 0.0  # an all-zero group
    x[1, G:2 * G] *= 1e-6  # a group whose scale is an fp16 subnormal
    x[2] = torch.randn(n, generator=g).sign() * 0.01  # massive ties
    return x


METHODS = [("top:86", False), ("top:128", False), ("top:1", False), ("opt", False), ("opt", True), ("top:86", True)]


@pytest.mark.parametrize("fmt", ["PQ2_0", "PTQ1_0"])
@pytest.mark.parametrize(("method", "weighted"), METHODS)
def test_pack_is_the_c_reference(fmt, method, weighted):
    """Pack == C quantize(decode); unpack inverts pack; C dequantize(pack) == decode (the served values)."""
    x = _weights()
    imp = torch.rand(x.shape[1], generator=torch.Generator().manual_seed(1)) ** 4 if weighted else None
    trits, d = ternarize(x, method, imp)
    served = decode(trits, d)
    raw = pack(fmt, trits, d)
    assert raw.shape == (x.shape[0], x.shape[1] // G * BLOCK_BYTES[fmt])
    cq, cdq = C_REF[fmt]
    for r in range(x.shape[0]):
        assert bytes(raw[r].numpy()) == cq(served[r].numpy()), (fmt, method, r)
        assert np.array_equal(cdq(bytes(raw[r].numpy())), served[r].numpy()), (fmt, method, r)
    t2, d2 = unpack(fmt, raw, x.shape[1])
    assert torch.equal(t2, trits) and torch.equal(d2, d)


def test_pq2_0_unpack_keeps_code_3():
    raw = pack("PQ2_0", torch.zeros(1, G, dtype=torch.int8), torch.ones(1, 1))
    raw[0, 2] |= 3  # element 0 -> code 3 (= +2, valid PQ2_0, not ternary)
    t, _ = unpack("PQ2_0", raw, G)
    assert int(t[0, 0]) == 2
    assert np.array_equal(c_dequantize_pq2_0(bytes(raw[0].numpy()))[:1], np.array([2.0], np.float32))


def _err(x, trits, d, imp):
    return float(((decode(trits, d) - x).square() * imp).sum())


@pytest.mark.parametrize("weighted", [False, True])
def test_opt_is_the_l2_optimum(weighted):
    """``opt`` (fp16 scale, no bf16 snap) is at least as good as every fixed-K choice, per group, under the metric."""
    x = torch.randn(8, 2 * G, generator=torch.Generator().manual_seed(2)) * 0.02
    imp = torch.rand(x.shape[1], generator=torch.Generator().manual_seed(3)) ** 3 if weighted else None
    metric = imp if weighted else torch.ones(x.shape[1])
    t_opt, d_opt = ternarize(x, "opt", imp, scale="fp16")
    for r in range(x.shape[0]):
        for gi in range(2):
            sl = slice(gi * G, (gi + 1) * G)
            xs, ms = x[r:r + 1, sl], metric[sl]
            best = _err(xs, t_opt[r:r + 1, sl], d_opt[r:r + 1, gi:gi + 1], ms)
            for k in range(1, G + 1):
                t, d = ternarize(xs, f"top:{k}", imp[sl] if weighted else None, scale="fp16")
                assert best <= _err(xs, t, d, ms) * (1 + 2e-3) + 1e-12, (r, gi, k)  # fp16 scale rounding


def test_top_k_counts_and_scale():
    x = torch.randn(4, 4 * G, generator=torch.Generator().manual_seed(4))
    t, d = ternarize(x, "top:86")
    nnz = t.reshape(4, 4, G).ne(0).sum(-1)
    assert bool((nnz == 86).all())  # no ties in random floats
    kept = x.reshape(4, 4, G).abs() * t.reshape(4, 4, G).ne(0)
    assert torch.equal(d, (kept.sum(-1) / 86).half().float())  # fp16 scale (PrismML)
    _, db = ternarize(x, "top:86", scale="bf16")
    assert torch.equal(db, db.to(torch.bfloat16).float())


def test_top_k_keeps_ties():
    x = torch.zeros(1, G)
    x[0, :90] = 0.01  # 90 equal magnitudes: top:86 keeps all of them
    x[0, :45] *= -1
    t, d = ternarize(x, "top:86")
    assert int(t.ne(0).sum()) == 90 and float(d) == float(torch.tensor(0.01).half())


def test_importance_moves_the_solution():
    """Heavy importance on a few small columns: the weighted optimum keeps them, the unweighted one need not."""
    x = torch.randn(1, G, generator=torch.Generator().manual_seed(5)) * 0.02
    small = x.abs().argsort(-1)[0, :8]
    imp = torch.full((G,), 1e-3)
    imp[small] = 1e3
    tw, dw = ternarize(x, "opt", imp, scale="fp16")
    tu, du = ternarize(x, "opt", None, scale="fp16")
    assert _err(x, tw, dw, imp) < _err(x, tu, du, imp)


def test_codec_interface():
    x = _weights(rows=3)
    codec = TernaryCodec("top:86")
    payload = codec.encode("PTQ1_0", x)
    assert torch.equal(codec.decode("PTQ1_0", payload, 3, x.shape[1]), decode(*payload))
    for fmt in ("PQ2_0", "PTQ1_0"):
        flat = codec.to_bytes(fmt, payload, 3)
        assert flat.dtype == np.uint8 and flat.size == 3 * x.shape[1] // G * BLOCK_BYTES[fmt]


@pytest.mark.skipif(not BONSAI.exists(), reason=f"no PrismML GGUF at {BONSAI}")
def test_prismml_file_round_trips_bitwise():
    """Real PrismML PTQ1_0 tensors: unpack -> pack reproduces their bytes; the trits/scales are PrismML's shape."""
    infos, _ = gguf_tensor_index(BONSAI)
    tern = {n: i for n, i in infos.items() if i["type"] == 143}
    assert tern, "no PTQ1_0 tensor in the file"
    picks = [n for n in tern if n.startswith("blk.3.")][:6] or list(tern)[:6]
    k86, groups = 0, 0
    for name in picks:
        info = tern[name]
        n = info["shape"][0]
        raw = gguf_raw_rows(BONSAI, info, "PTQ1_0", slice(0, 64))
        trits, d = unpack("PTQ1_0", raw, n)
        assert int(trits.abs().max()) <= 1
        assert np.array_equal(pack("PTQ1_0", trits, d).numpy(), raw), name
        assert np.array_equal(c_dequantize_ptq1_0(bytes(raw[0, :4 * 28])), decode(trits, d)[0, :4 * G].numpy()), name
        nnz = trits.reshape(trits.shape[0], -1, G).ne(0).sum(-1)
        assert int(nnz.min()) >= 86, name  # top-86 with ties kept
        k86 += int((nnz == 86).sum())
        groups += nnz.numel()
        # our default ternarizer (top:86, ties kept, fp16 scale) is a fixed point on every shipped group
        t_re, d_re = ternarize(decode(trits, d), "top:86")
        assert torch.equal(t_re, trits) and torch.equal(d_re, d), name
    assert k86 / groups > 0.9, f"only {k86}/{groups} groups carry exactly 86 nonzeros"


def test_trainer_codec_ptq1_0_repacks_the_same_trits():
    from omk_qad.fakequant import Codec

    llama_cpp = Path(os.environ.get("OMK_LLAMA_CPP", "/opt/llama.cpp"))
    codec = Codec(llama_cpp)
    x = _weights(rows=4)
    payload = codec.encode("PTQ1_0", x)  # held as PQ2_0 bytes
    t, d = ternarize(x)
    assert np.array_equal(codec.to_bytes("PQ2_0", payload, 4), pack("PQ2_0", t, d).numpy())
    assert np.array_equal(codec.to_bytes("PTQ1_0", payload, 4), pack("PTQ1_0", t, d).numpy())
    assert torch.equal(codec.decode("PTQ1_0", payload, 4, x.shape[1], "cpu"), decode(t, d))
