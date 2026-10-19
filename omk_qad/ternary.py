# SPDX-License-Identifier: Apache-2.0
"""Ternary codecs of PrismML's Bonsai 2 format: ternarizers + bit-exact PQ2_0 / PTQ1_0 packing (plan T1).

Both formats hold one fp16 scale ``d`` per group of 128 weights and a trit ``t`` in {-1, 0, +1} per weight
(value ``t * d``); they decode to the same trits and differ only in packing:

  PQ2_0   type 142, 34 B / 128 w (2.125 bpw): ``d`` then 32 B of 2-bit codes ``t + 1``, element j at byte j // 4,
          bits 2 * (j % 4) (code 3 = +2 is never produced here).
  PTQ1_0  type 143, 28 B / 128 w (1.75 bpw): ``qs`` 24 B = elements 0..79 as 16 bytes x 5 trits plus 80..119 as
          8 bytes x 5 trits (byte m of a stage of width c holds elements m + n * c, n = 0..4, first trit most
          significant), ``qh`` 2 B = elements 120..127 (byte h holds 120 + h + 2 m, m = 0..3, times 3), then ``d``.
          A byte holding base-3 value v stores ceil(v * 256 / 243).

The layouts are those of the opencoti patch ``0590-bonsai2-types.patch`` (PrismML fork / upstream PR #29672);
``tests/test_ternary.py`` checks them against line-by-line ports of its C reference and against a shipped
PrismML GGUF.

The formats' own quantizer (``d = max|w|, t = round(w / d)``) is a PACKER, not a ternarizer: on real weights it
zeroes most of them. A ternarizer chooses the support and the scale:

  ``top:K``  the K largest |w| of each group, ties at the K-th kept (PrismML ships K = 86: measured on its
             27B PTQ1_0 file, every group has >= 86 nonzeros and ~97.6 % exactly 86), d = mean of the kept |w|;
  ``opt``    the exact L2 optimum per group. For a fixed support S the best scale is d = sum_S a|w| / sum_S a and the
             error drops by (sum_S a|w|)^2 / sum_S a; for a fixed d an element belongs in S iff |w| > d / 2 whatever
             its weight a, so the optimal S is a prefix of the |w|-sorted group and one sort + cumsum finds it.

``importance`` (per input column, E[x^2] from the imatrix) weights both the scale and the ``opt`` support: the
output error of a column-wise weight change is sum_j E[x_j^2] dw_j^2 (diagonal approximation). Without it a = 1.
The scale is stored as fp16 (``scale="fp16"``, default). PrismML's shipped scales are full fp16 values: only
~12.5 % are bf16-exact on its 27B file, the chance rate. ``scale="bf16"`` snaps first, as
omnimergekit's bonsai pipeline must (it writes +-d into a bf16 GGUF before packing). The served value is always
the stored fp16 number, so ``decode(encode(x))`` is exactly what the GGUF holds.
"""

from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

G = 128
TYPES = {"PQ2_0": 142, "PTQ1_0": 143}
BLOCK_BYTES = {"PQ2_0": 34, "PTQ1_0": 28}
IMPORTANCE_FLOOR = 1e-4  # of the group-wide mean; zero-activation columns must not zero a denominator


def _snap_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()  # round-to-nearest-even, as PrismML's snap


def ternarize(x: torch.Tensor, method: str = "top:86", importance: torch.Tensor | None = None,
              scale: str = "fp16") -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [rows, n] -> (trits int8 [rows, n], d float32 [rows, n // 128]; d is an fp16 value)."""
    rows, n = x.shape
    if n % G:
        raise ValueError(f"row length {n} is not a multiple of {G}")
    w = x.detach().float().reshape(rows, n // G, G)
    a = w.abs()
    if importance is None:
        imp = torch.ones_like(a)
    else:
        imp = importance.detach().float().to(w.device).reshape(1, n // G, G).expand_as(w)
        imp = imp.clamp_min(IMPORTANCE_FLOOR * float(imp.mean().clamp_min(torch.finfo(torch.float32).tiny)))
    kind, *arg = method.split(":")
    if kind == "top":
        k = int(arg[0])
        if not 0 < k <= G:
            raise ValueError(f"top:K needs 0 < K <= {G}")
        mask = a >= a.topk(k, dim=-1).values[..., -1:]  # ties at the threshold are kept (PrismML)
    elif kind == "opt":
        srt, order = a.sort(dim=-1, descending=True)
        ws = imp.gather(-1, order)
        num, den = (ws * srt).cumsum(-1), ws.cumsum(-1)
        keep = (num.square() / den).argmax(-1, keepdim=True)  # last kept rank
        ranks = torch.arange(G, device=w.device).expand_as(a) <= keep
        mask = torch.zeros_like(ranks).scatter_(-1, order, ranks)
    else:
        raise ValueError(f"unknown ternarizer {method!r} (top:K | opt)")
    m = mask.float()
    d = (imp * a * m).sum(-1) / (imp * m).sum(-1).clamp_min(torch.finfo(torch.float32).tiny)
    if scale == "bf16":
        d = _snap_bf16(d)
    elif scale != "fp16":
        raise ValueError(f"unknown scale rounding {scale!r} (bf16 | fp16)")
    d = d.half().float()
    trits = (torch.sign(w) * m).to(torch.int8)
    trits[d == 0] = 0  # a zero scale decodes to zeros; keep the trits consistent with what the packer writes
    return trits.reshape(rows, n), d


def decode(trits: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Served values t * d, float32 [rows, n]."""
    rows, n = trits.shape
    return (trits.float().reshape(rows, n // G, G) * d.float()[..., None]).reshape(rows, n)


def _d_bytes(d: torch.Tensor) -> torch.Tensor:
    return d.half().contiguous().view(torch.uint8).reshape(*d.shape, 2)  # little-endian fp16


def _d_from(raw: torch.Tensor) -> torch.Tensor:
    return raw.contiguous().view(torch.float16).reshape(raw.shape[:-1]).float()


def pack(fmt: str, trits: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(trits [rows, n], d [rows, n // 128]) -> GGUF row bytes uint8 [rows, n // 128 * block bytes]."""
    rows, n = trits.shape
    nb = n // G
    t = (trits.to(torch.int32) + 1).reshape(rows, nb, G)  # 0, 1, 2
    if fmt == "PQ2_0":
        c = t.reshape(rows, nb, G // 4, 4)
        qs = c[..., 0] | c[..., 1] << 2 | c[..., 2] << 4 | c[..., 3] << 6
        blk = torch.cat([_d_bytes(d), qs.to(torch.uint8)], -1)
    elif fmt == "PTQ1_0":
        p5 = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=t.device)
        a = (t[..., :80].reshape(rows, nb, 5, 16) * p5[:, None]).sum(-2)
        b = (t[..., 80:120].reshape(rows, nb, 5, 8) * p5[:, None]).sum(-2)
        h = (t[..., 120:].reshape(rows, nb, 4, 2) * p5[1:, None]).sum(-2) * 3
        qs = (torch.cat([a, b, h], -1) * 256 + 242) // 243  # ceil(v * 256 / 243)
        blk = torch.cat([qs.to(torch.uint8), _d_bytes(d)], -1)
    else:
        raise ValueError(f"unknown ternary format {fmt!r}")
    return blk.reshape(rows, nb * BLOCK_BYTES[fmt])


def unpack(fmt: str, raw: torch.Tensor | np.ndarray, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """GGUF row bytes [rows, n // 128 * block bytes] -> (trits int8 [rows, n], d float32 [rows, n // 128]).

    PQ2_0 code 3 (= +2) is returned as 2: it is valid PQ2_0 but not ternary, and the caller decides.
    """
    raw = torch.as_tensor(np.ascontiguousarray(raw) if isinstance(raw, np.ndarray) else raw.contiguous())
    nb = n // G
    blk = raw.reshape(-1, nb, BLOCK_BYTES[fmt])
    rows = blk.shape[0]
    if fmt == "PQ2_0":
        d = _d_from(blk[..., :2])
        q = blk[..., 2:].to(torch.int32)
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=raw.device)
        t = ((q[..., None] >> shifts) & 3).reshape(rows, nb, G)
    elif fmt == "PTQ1_0":
        d = _d_from(blk[..., 26:])
        q = blk[..., :26].to(torch.int32)
        p5 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=raw.device)
        trit = (((q[..., None] * p5) & 0xFF) * 3) >> 8  # [rows, nb, 26 bytes, 5 trits]: trit n of each byte
        a = trit[..., :16, :].transpose(-1, -2).reshape(rows, nb, 80)
        b = trit[..., 16:24, :].transpose(-1, -2).reshape(rows, nb, 40)
        h = trit[..., 24:, :4].transpose(-1, -2).reshape(rows, nb, 8)
        t = torch.cat([a, b, h], -1)
    else:
        raise ValueError(f"unknown ternary format {fmt!r}")
    return (t - 1).to(torch.int8).reshape(rows, n), d


class TernaryCodec:
    """The trainer-facing codec shape (encode / decode / to_bytes) for PQ2_0 and PTQ1_0; payload = (trits, d)."""

    def __init__(self, method: str = "top:86", scale: str = "fp16"):
        """``method`` / ``scale`` as :func:`ternarize`."""
        self.method, self.scale = method, scale

    def encode(self, fmt: str, x: torch.Tensor, importance: torch.Tensor | None = None):
        """Ternarize ``x`` [rows, n] (``importance`` = E[x^2] per column, optional)."""
        del fmt  # both formats hold the same trits
        return ternarize(x, self.method, importance, self.scale)

    @staticmethod
    def decode(fmt: str, payload, rows: int, n: int, device="cpu") -> torch.Tensor:
        """Served float32 [rows, n]."""
        del fmt, rows, n
        return decode(*payload).to(device)

    @staticmethod
    def to_bytes(fmt: str, payload, rows: int) -> np.ndarray:
        """GGUF bytes of ``payload`` (flat uint8)."""
        del rows
        trits, d = payload
        return pack(fmt, trits.cpu(), d.cpu()).numpy().reshape(-1)


# --- raw GGUF tensor access (gguf-py cannot open files carrying types 142 / 143) -------------------------------

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_tensor_index(path: Path) -> tuple[dict[str, dict], int]:
    """{name: {"type", "shape" (ne0 first), "offset" (absolute)}} and the file's alignment; KV values skipped."""
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(rd("Q")).decode("utf-8")

        def value(vt):
            if vt == 8:
                return string()
            if vt == 9:
                it, cnt = rd("I"), rd("Q")
                return [value(it) for _ in range(cnt)]
            return rd(_SCALARS[vt])

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        rd("I")
        n_tensors, n_kv = rd("Q"), rd("Q")
        align = 32
        for _ in range(n_kv):
            key, vt = string(), rd("I")
            v = value(vt)
            if key == "general.alignment":
                align = int(v)
        infos = {}
        for _ in range(n_tensors):
            name = string()
            nd = rd("I")
            shape = [rd("Q") for _ in range(nd)]
            infos[name] = {"type": rd("I"), "shape": shape, "offset": rd("Q")}
        start = (f.tell() + align - 1) // align * align
    for v in infos.values():
        v["offset"] += start
    return infos, align


def gguf_raw_rows(path: Path, info: dict, fmt: str, rows: slice | None = None) -> np.ndarray:
    """Raw bytes of a ternary tensor as [rows, row bytes] (memory-mapped; ``rows`` selects a row range)."""
    n = info["shape"][0]
    nrows = int(np.prod(info["shape"][1:], dtype=np.int64)) if len(info["shape"]) > 1 else 1
    rb = n // G * BLOCK_BYTES[fmt]
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=info["offset"], shape=(nrows, rb))
    return np.array(mm[rows] if rows is not None else mm)


# --- Hadamard basis (the Bonsai 2 runtime contract: y = W_s . H(s * x), H blockwise normalized Sylvester WHT) ------

def fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform over the last axis in independent blocks (its own inverse)."""
    shape = x.shape
    x = x.reshape(-1, shape[-1] // block, block)
    h = 1
    while h < block:
        x = x.reshape(x.shape[0], x.shape[1], block // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return x.reshape(shape) / math.sqrt(block)


def hadamard_block(width: int) -> int:
    """Block for an input width: 1024 where it divides, else 512 (PrismML / bonsai rule), else 256 (tiny models)."""
    for b in (1024, 512, 256):
        if width % b == 0:
            return b
    raise ValueError(f"input width {width} is not a multiple of 256: no Hadamard block fits")


def seeded_signs(width: int, device=None) -> torch.Tensor:
    """+-1 per input column, from a fixed generator per width (omnimergekit bonsai_tern_lib's seeded basis)."""
    g = torch.Generator().manual_seed(1234 + width)
    return (torch.randint(0, 2, (width,), generator=g) * 2 - 1).float().to(device)


class HadamardBasis:
    """fold / unfold along the input (last) axis of [rows, width] tensors in GGUF column order.

    fold(w) = H(w * s) is the latent (ternary) basis; unfold(y) = H(y) * s maps back. Both are orthogonal, so the
    gradient of a folded latent is fold(gradient of the served weight).
    """

    def __init__(self, width: int, device=None, signs: torch.Tensor | None = None, block: int | None = None):
        """Seeded signs and the standard block unless given (a PrismML reference supplies its own)."""
        self.width = width
        self.block = block or hadamard_block(width)
        self.signs = (signs if signs is not None else seeded_signs(width)).float().to(device)

    def fold(self, w: torch.Tensor) -> torch.Tensor:
        """Model basis -> latent basis (float32)."""
        return fwht(w.float() * self.signs.to(w.device), self.block)

    def unfold(self, y: torch.Tensor) -> torch.Tensor:
        """Latent basis -> model basis (float32)."""
        return fwht(y.float(), self.block) * self.signs.to(y.device)


# --- runtime contract (opencoti 0588 / upstream #29600 loader) -----------------------------------------------------
_FOLDABLE_KINDS = ("attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output", "ffn_gate", "ffn_up",
                   "ffn_down", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps", "ffn_gate_shexp",
                   "ffn_up_shexp", "ffn_down_shexp", "ssm_out")


def foldable(gguf_name: str) -> bool:
    """Whether the runtime applies the activation transform for this weight (its loader refuses any other).

    ``output.weight``, ``token_embd.weight`` (as an inverse lookup) and ``blk.<n>.<kind>.weight`` of the kinds above.
    """
    if gguf_name in ("output.weight", "token_embd.weight"):
        return True
    parts = gguf_name.split(".")
    return len(parts) == 4 and parts[0] == "blk" and parts[1].isdigit() and parts[2] in _FOLDABLE_KINDS \
        and parts[3] == "weight"


def model_block(widths) -> int:
    """The one ``prism.hadamard.block_size`` of a model.

    The largest of 1024 / 512 / 256 dividing every folded input width: the runtime has a single block for all.
    """
    widths = set(widths)
    for b in (1024, 512, 256):
        if all(w % b == 0 for w in widths):
            return b
    raise ValueError(f"no Hadamard block of 256..1024 divides every folded input width {sorted(widths)}")


def register_gguf_types(gguf) -> None:
    """Add PQ2_0 (142) / PTQ1_0 (143) to gguf-py's type enum and size table, in-process (idempotent)."""
    qt = gguf.GGMLQuantizationType
    for name, val in TYPES.items():
        if val in qt._value2member_map_:
            continue
        member = int.__new__(qt, val)
        member._name_, member._value_ = name, val
        qt._member_map_[name] = member
        qt._value2member_map_[val] = member
        qt._member_names_.append(name)
        gguf.GGML_QUANT_SIZES[member] = (G, BLOCK_BYTES[name])
