# SPDX-License-Identifier: Apache-2.0
"""Training-time fake quantization driven by a layout (omk_qad.layout), with the update done in backward.

Every large weight is a *latent* (float16 on GPU, 2 bytes/weight; stochastic rounding on write). Its *served*
value is built when its module runs: for each GGUF tensor the layout maps to it, the latent rows the GGUF
tensor holds are gathered in GGUF column order (so the GGML blocks are exactly the GGUF's), encoded to the
planned type, decoded and scattered back. Encoders: ModelOpt (torch/CUDA) for IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/
IQ2_S/Q8_0, omk_qad.iq4 / omk_qad.iq3 (CUDA, byte-identical to llama-quantize) for IQ4_XS/IQ4_NL/IQ3_XXS/
IQ3_S, libggml (the code llama-quantize runs) for everything else (K-quants, ...). Export
re-encodes the final latent with the same encoder, so the GGUF holds the blocks training served.

The gradient of the served weight is the latent's gradient (straight-through). It is captured in the
backward of ``_Served`` and applied at once (momentum-free Adafactor, per expert for 3-D tensors), so no
gradient of the whole model is ever held -- the ternary bonsai27 recipe, made model-agnostic.

Served weights are swapped into ``module._parameters`` by a forward pre-hook and restored by a forward
hook, so they exist for one module call at a time (gradient-checkpoint recompute runs the hooks again;
the packed encoding is cached per latent version, so a recompute only decodes). The version is the
latent's own counter, bumped by every write (``Latent.touch``): writes go through ``.data``, whose version
counter is separate from the parameter's. ``Control.refresh`` / ``ggml_refresh`` let an optimizer update
keep a payload for N steps (re-encoded at most every N steps); a write outside the optimizer
(``touch``) always drops it. Checkpoints carry the payloads, so a resume serves what the straight run did.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELOPT_TYPES = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q8_0"}
# omk_qad's own CUDA encoders, byte-identical to llama-quantize (no CPU fallback on a CUDA device)
NATIVE_TYPES = {"IQ4_XS": "iq4", "IQ4_NL": "iq4", "IQ3_XXS": "iq3", "IQ3_S": "iq3"}  # type -> omk_qad module
GPU_TYPES = MODELOPT_TYPES | set(NATIVE_TYPES)  # encoded on the training device (``refresh``); the rest by libggml
FLOAT_TYPES = {"F32", "F16", "BF16"}
TERNARY_TYPES = {"PQ2_0", "PTQ1_0"}  # PrismML Bonsai 2 (omk_qad.ternary); the latent lives in the folded basis
CHUNK_ELEMS = 64 * 2**20  # rows are encoded / updated in chunks of about this many values


def _native(qtype: str):
    """The omk_qad module (iq4 / iq3) that encodes and decodes a native type."""
    import importlib

    return importlib.import_module(f".{NATIVE_TYPES[qtype]}", __package__)


class Codec:
    """Encode/decode 2-D float32 row blocks to one GGML type with the encoder export will also use."""

    def __init__(self, llama_cpp: Path, ternary_method: str = "top:86"):
        """``llama_cpp`` provides libggml (non-ModelOpt types) and gguf-py's type enum.

        Ternary types (PQ2_0 / PTQ1_0, omk_qad.ternary) use ``ternary_method``; their payload is always the PQ2_0
        packing (2.125 bpw, the same trits as PTQ1_0), repacked at :meth:`to_bytes` when the target is PTQ1_0.
        """
        from .gguf_export import _import_llama_cpp

        self.gguf = _import_llama_cpp(Path(llama_cpp))[0]
        self.llama_cpp = Path(llama_cpp)
        self._ggml = None
        self.ternary_method = ternary_method

    @property
    def ggml(self):
        """The libggml binding, loaded on first use."""
        if self._ggml is None:
            from .ggml_lib import GGMLLib

            self._ggml = GGMLLib(self.llama_cpp)
        return self._ggml

    def qtype(self, name: str):
        """gguf-py GGMLQuantizationType of ``name``."""
        return self.gguf.GGMLQuantizationType[name]

    def encode(self, qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None):
        """Packed payload of float32 ``x`` [rows, cols]: uint8 torch tensor (ModelOpt) or numpy array (ggml).

        ``importance`` is the encoder-convention vector from :func:`omk_qad.imatrix.importance_for` (or None).
        """
        if qtype in TERNARY_TYPES:
            from .ternary import pack, ternarize

            return pack("PQ2_0", *ternarize(x.float(), self.ternary_method, importance))
        if qtype in MODELOPT_TYPES:
            from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

            kw = {} if importance is None else {"importance": importance.to(x.device)}
            packed, _ = GGML_FORMAT_REGISTRY[qtype.lower()].quantize(x.float().contiguous(), **kw)
            return packed
        if qtype in NATIVE_TYPES:  # CUDA kernel on a CUDA device (raises rather than fall back), torch on CPU
            return _native(qtype).encode(qtype, x.float().contiguous(), importance)
        imat = None if importance is None else importance.detach().float().cpu().numpy()
        return self.ggml.quantize(int(self.qtype(qtype)), x.detach().float().cpu().numpy(), imat)

    def decode(self, qtype: str, packed, rows: int, cols: int, device, dtype=torch.float32) -> torch.Tensor:
        """Inverse of :meth:`encode`, as a [rows, cols] tensor on ``device``."""
        if qtype in TERNARY_TYPES:
            from .ternary import decode, unpack

            return decode(*unpack("PQ2_0", packed, cols)).to(device=device, dtype=dtype)
        if qtype in MODELOPT_TYPES:
            from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

            shape = torch.tensor([rows, cols], dtype=torch.int64)
            # ModelOpt dequantizes on the payload's device; honour ``device`` like the other branches
            # (export encodes on the GPU and decodes to CPU for its numpy verification).
            out = GGML_FORMAT_REGISTRY[qtype.lower()].dequantize(packed, shape, dtype=dtype)
            return out.reshape(rows, cols).to(device=device)
        if qtype in NATIVE_TYPES:
            return _native(qtype).decode(qtype, packed, rows, cols, dtype).to(device=device)
        out = self.ggml.dequantize(int(self.qtype(qtype)), packed, cols)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    def to_bytes(self, qtype: str, packed, rows: int) -> np.ndarray:
        """Payload as numpy uint8 [rows, row_bytes] (GGUF row layout)."""
        if qtype == "PTQ1_0":
            from .ternary import pack, unpack

            packed = pack("PTQ1_0", *unpack("PQ2_0", packed.cpu(), packed.shape[-1] // 34 * 128))
        arr = packed.detach().cpu().numpy() if isinstance(packed, torch.Tensor) else packed
        return np.ascontiguousarray(arr, dtype=np.uint8).reshape(rows, -1)


@dataclass
class Segment:
    """Rows of a latent's 2-D view that one GGUF tensor holds, in GGUF column order, at one type."""

    gguf: str
    qtype: str
    rows: slice | torch.Tensor
    colperm: torch.Tensor | None
    inv: torch.Tensor | None
    n_rows: int
    chunks: list = field(default_factory=list)  # (rows, n, expert): never crosses an expert boundary


def _as_slice(rows: torch.Tensor) -> slice | torch.Tensor:
    r0 = int(rows[0])
    if torch.equal(rows, torch.arange(r0, r0 + len(rows), dtype=rows.dtype)):
        return slice(r0, r0 + len(rows))
    return rows


def _chunk(rows: slice | torch.Tensor, n: int, cols: int) -> list[slice | torch.Tensor]:
    step = max(1, CHUNK_ELEMS // max(1, cols))
    if isinstance(rows, slice):
        return [slice(r, min(r + step, rows.stop)) for r in range(rows.start, rows.stop, step)]
    return [rows[i : i + step] for i in range(0, n, step)]


def sr_cast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """float32 -> ``dtype`` with stochastic rounding (fp16/bf16), so sub-ulp updates survive on average."""
    if dtype == torch.float32:
        return x
    if dtype == torch.float16:  # dithered round-to-nearest == stochastic rounding (bonsai_tern_lib.sr16)
        _, e = torch.frexp(x.abs().clamp(min=6.2e-5))
        ulp = torch.ldexp(torch.ones_like(x), e - 11)
        return (x + (torch.rand_like(x) - 0.5) * ulp).to(torch.float16)
    if dtype == torch.bfloat16:  # add 16 random low bits, truncate
        bits = x.contiguous().view(torch.int32)
        noise = torch.randint_like(bits, 0, 1 << 16)
        return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    raise ValueError(f"no stochastic rounding for {dtype}")


@dataclass
class Control:
    """Run-wide switches shared by every latent."""

    compute_dtype: torch.dtype = torch.float32
    lr: float = 0.0
    beta2: float = 0.999
    clip: float = 1.0
    eps: float = 1e-30
    bypass: bool = False  # serve the latent unquantized (lossless gate)
    step: int = 0
    ggml_refresh: int = 1  # re-encode libggml segments every N steps (1 = every change)
    refresh: int = 1  # re-encode ModelOpt segments (IQ1/IQ2/Q8_0) every N steps (1 = every change)
    hadamard_block: int | None = None  # the model's one Hadamard block (layout["hadamard_block"]) for ternary latents
    accum: int = 1  # micro-batches per update: their gradients sum in Latent.gacc, one update on the last
    apply: bool = True  # this micro-batch ends an update (False: gradients only accumulate)
    beta1: float = 0.0  # Adafactor momentum on the normalised, clipped update (0 = momentum-free)
    state_dtype: torch.dtype = torch.float32  # storage of the accumulation buffer and the momentum
    dummy: torch.Tensor = field(default_factory=lambda: torch.zeros((), requires_grad=True))
    stats: dict = field(default_factory=lambda: {"encodes": 0, "updates": 0, "undelivered": 0})
    opt: dict = field(default_factory=dict)  # per weight class: device sums of this step's updates (opt_metrics)


def weight_class(name: str) -> str:
    """Telemetry class of a trained weight: experts | router | attn | mlp | head | other."""
    n = name.lower()
    if "experts" in n:
        return "experts"
    if "router" in n or "gate_inp" in n or re.search(r"(^|\.)gate\.weight$", n):
        return "router"
    if "attn" in n or "attention" in n:
        return "attn"
    if "mlp" in n or "ffn" in n:
        return "mlp"
    if "embed" in n or "lm_head" in n:
        return "head"
    return "other"


def opt_metrics(ctl: Control) -> dict:
    """This step's gradient / update telemetry per weight class (one device sync), then reset the sums.

    Per class ``c`` (units = matrices, or experts of a 3-D weight; only active units count, i.e. units whose
    gradient is not all zero): ``opt/c/grad_rms`` raw gradient RMS, ``opt/c/update_rms`` RMS of the applied step
    (lr x normalised update), ``opt/c/update_ratio`` ||step|| / ||weight||, ``opt/c/clip_frac`` share of active
    units whose normalised update hit the RMS clip, ``opt/c/u_rms`` their mean pre-clip RMS, ``opt/c/active_frac``
    active units / units. Totals: ``opt/grad_norm`` (L2 of every latent gradient) and ``opt/update_ratio``.
    """
    out, tot = {}, {"gsq": 0.0, "dsq": 0.0, "wsq": 0.0}
    for c, d in sorted(ctl.opt.items()):
        v = {k: float(x) for k, x in d.items()}
        n, act = max(v.get("n", 0.0), 1.0), max(v.get("active", 0.0), 1.0)
        out[f"opt/{c}/grad_rms"] = math.sqrt(v.get("gsq", 0.0) / n)
        out[f"opt/{c}/update_rms"] = math.sqrt(v.get("dsq", 0.0) / n)
        out[f"opt/{c}/update_ratio"] = math.sqrt(v.get("dsq", 0.0) / max(v.get("wsq", 0.0), 1e-30))
        out[f"opt/{c}/clip_frac"] = v.get("clipped", 0.0) / act
        out[f"opt/{c}/u_rms"] = v.get("u_rms", 0.0) / act
        out[f"opt/{c}/active_frac"] = v.get("active", 0.0) / max(v.get("units", 0.0), 1.0)
        for k in tot:
            tot[k] += v.get(k, 0.0)
    if ctl.opt:
        out["opt/grad_norm"] = math.sqrt(tot["gsq"])
        out["opt/update_ratio"] = math.sqrt(tot["dsq"] / max(tot["wsq"], 1e-30))
    ctl.opt = {}
    return out


class Latent:
    """One trained weight: latent storage, its layout segments and its Adafactor state."""

    def __init__(self, name: str, param: nn.Parameter, segments: list[Segment], ctl: Control, codec: Codec,
                 latent_dtype: torch.dtype):
        """Convert ``param`` in place to ``latent_dtype`` storage; it no longer requires grad."""
        self.name, self.param, self.segments, self.ctl, self.codec = name, param, segments, ctl, codec
        self.kind = weight_class(name)
        param.requires_grad_(False)
        self.shape = tuple(param.shape)
        self.cols = self.shape[-1]
        self.rows = param.numel() // self.cols
        self.latent_dtype = latent_dtype
        self.basis = None  # ternary: the latent is H(s * W) in GGUF column order (omk_qad.ternary.HadamardBasis)
        self.colperm = self.inv = None
        if any(seg.qtype in TERNARY_TYPES for seg in segments):
            self._init_folded()
        else:
            param.data = param.data.to(latent_dtype)
        b = self.shape[0] if param.ndim == 3 else 1
        dev = param.device
        self.r = torch.zeros(b, self.rows // b, device=dev)
        self.c = torch.zeros(b, self.cols, device=dev)
        self.t = torch.zeros(b, device=dev)
        self.uses = 0
        self.version = 0  # bumped by every write; param._version cannot be used: updates go through .data
        self.gbuf: torch.Tensor | None = None
        self.gacc: torch.Tensor | None = None  # gradient sum over the micro-batches of one update (accum > 1)
        self.m: torch.Tensor | None = None  # momentum of the update (beta1 > 0), created at the first update
        self._cache: dict[int, tuple] = {}  # chunk id -> (latent version, step encoded, payload)
        self.msq: torch.Tensor | None = None  # imatrix E[x^2] per expert [E, C] (set_imatrix)
        self._imp: dict = {}

    def _init_folded(self) -> None:
        """Ternary weight: every row is in a ternary segment sharing one column order; store the folded latent."""
        from .ternary import HadamardBasis

        segs = self.segments
        if any(seg.qtype not in TERNARY_TYPES for seg in segs):
            raise ValueError(f"{self.name}: ternary and non-ternary segments in one weight are not supported")
        perms = {None if seg.colperm is None else tuple(seg.colperm.tolist()) for seg in segs}
        if len(perms) != 1:
            raise ValueError(f"{self.name}: ternary segments with different column orders")
        if sum(seg.n_rows for seg in segs) != self.rows:
            raise ValueError(f"{self.name}: a ternary weight must be covered by its segments row for row")
        if not self.ctl.hadamard_block:  # a per-width block would train a basis the runtime cannot serve
            raise ValueError(f"{self.name}: ternary layout without hadamard_block (plan it with plan_layout / retype)")
        self.colperm, self.inv = segs[0].colperm, segs[0].inv
        self.basis = HadamardBasis(self.cols, self.param.device, block=self.ctl.hadamard_block)
        w = self.param.data.float().reshape(self.rows, self.cols)
        if self.colperm is not None:
            w = w[:, self.colperm.to(w.device)]
        self.param.data = self.basis.fold(w).reshape(self.shape).to(self.latent_dtype)

    def _to_model(self, y: torch.Tensor) -> torch.Tensor:
        """Folded rows [n, cols] (GGUF column order) -> model basis in the model's column order (float32)."""
        u = self.basis.unfold(y)
        return u[:, self.inv.to(u.device)] if self.inv is not None else u

    def _to_latent(self, g: torch.Tensor) -> torch.Tensor:
        """Model-basis gradient [rows, cols] -> folded basis (orthogonal: the exact latent gradient)."""
        g = g.float()
        if self.colperm is not None:
            g = g[:, self.colperm.to(g.device)]
        return self.basis.fold(g)

    @torch.no_grad()
    def set_model_weight(self, w: torch.Tensor) -> None:
        """Write a model-basis weight (e.g. another format's checkpoint) into this latent, folding if ternary."""
        w = w.to(self.param.device).float().reshape(self.rows, self.cols)
        if self.basis is not None:
            if self.colperm is not None:
                w = w[:, self.colperm.to(w.device)]
            w = self.basis.fold(w)
        self.param.data.copy_(w.reshape(self.shape).to(self.param.dtype))
        self.touch()

    @torch.no_grad()
    def model_weight(self) -> torch.Tensor:
        """The latent in the model's basis and layout, float32 (what a converter / checkpoint writer must see)."""
        if self.basis is None:
            return self.param.data.float()
        return self._to_model(self.param.data.reshape(self.rows, self.cols)).reshape(self.shape)

    @torch.no_grad()
    def _served_folded(self, dtype: torch.dtype) -> torch.Tensor:
        w = self.param.data.reshape(self.rows, self.cols)
        if self.ctl.bypass:
            return self._to_model(w).to(dtype).reshape(self.shape)
        out = torch.empty(self.rows, self.cols, dtype=dtype, device=w.device)
        k = 0
        for seg in self.segments:
            for rows, n, expert in seg.chunks:
                hit = self._cache.get(k)
                if hit is not None and hit[0] == self.version:
                    payload = hit[2]
                else:
                    payload = self.codec.encode(seg.qtype, w[rows].float(), self.importance(seg, expert))
                    self._cache[k] = (self.version, self.ctl.step, payload)
                    self.ctl.stats["encodes"] += 1
                out[rows] = self._to_model(self.codec.decode(seg.qtype, payload, n, self.cols, w.device)).to(dtype)
                k += 1
        return out.reshape(self.shape)

    def set_imatrix(self, msq: torch.Tensor | None, folded: bool = False) -> None:
        """E[x^2] [E, C] of this weight's inputs (omk_qad.imatrix.mean_sq); invalidates cached encodings.

        A ternary (folded) latent takes only the folded statistic (``folded=True``: E[(H s x)^2], GGUF column
        order); the model-basis diagonal is not the folded one (the rotation mixes columns) and is ignored.
        """
        if msq is not None and msq.shape[-1] != self.cols:
            raise ValueError(f"{self.name}: imatrix width {msq.shape[-1]} != {self.cols}")
        if folded and self.basis is None:
            raise ValueError(f"{self.name}: a folded-basis imatrix for a model-basis latent")
        if self.basis is not None and not folded:
            msq = None
        self.msq, self._imp, self._cache = msq, {}, {}

    def importance(self, seg: Segment, expert: int) -> torch.Tensor | None:
        """Encoder importance for one expert's rows of ``seg`` (columns in GGUF order)."""
        if self.msq is None:
            return None
        key = (seg.gguf, expert)
        if key not in self._imp:
            from .imatrix import importance_for

            m = self.msq[expert if self.msq.shape[0] > 1 else 0]
            if seg.colperm is not None and self.basis is None:  # a folded statistic is already in GGUF order
                m = m[seg.colperm]
            self._imp[key] = importance_for(seg.qtype, m)
        return self._imp[key]

    # ---------------------------------------------------------------- served weight
    @torch.no_grad()
    def served(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The weight the model runs with: latent with every quantized segment encoded/decoded."""
        dtype = dtype or self.ctl.compute_dtype
        if self.basis is not None:
            return self._served_folded(dtype)
        lat = self.param.data
        if self.ctl.bypass or not self.segments:
            return lat.to(dtype)
        out = lat.to(dtype).clone().reshape(self.rows, self.cols)
        w = lat.reshape(self.rows, self.cols)
        version = self.version
        k = 0
        for seg in self.segments:
            for rows, _, expert in seg.chunks:
                x = w[rows].float()
                if seg.colperm is not None:
                    x = x[:, seg.colperm.to(x.device)]
                hit = self._cache.get(k)
                every = self.ctl.refresh if seg.qtype in GPU_TYPES else self.ctl.ggml_refresh
                fresh = hit is not None and (hit[0] == version or self.ctl.step - hit[1] < every)
                if fresh:
                    payload = hit[2]
                else:
                    payload = self.codec.encode(seg.qtype, x, self.importance(seg, expert))
                    self._cache[k] = (version, self.ctl.step, payload)
                    self.ctl.stats["encodes"] += 1
                d = self.codec.decode(seg.qtype, payload, x.shape[0], self.cols, x.device)
                if seg.inv is not None:
                    d = d[:, seg.inv.to(d.device)]
                out[rows] = d.to(dtype)
                k += 1
        return out.reshape(self.shape)

    def payloads(self) -> dict[str, list]:
        """Current encoded payload chunks per GGUF tensor (re-encoded if stale); for tests/inspection."""
        self.served()
        out, k = {}, 0
        for seg in self.segments:
            for _ in seg.chunks:
                out.setdefault(seg.gguf, []).append(self._cache[k][2])
                k += 1
        return out

    def touch(self) -> None:
        """Mark the latent as written (any write outside :meth:`update`, e.g. a checkpoint load).

        Drops every cached payload: a refresh window only defers re-encoding across optimizer steps.
        """
        self.version += 1
        self._cache = {}

    def cache_state(self) -> dict:
        """Cached payloads {chunk: (step encoded, payload on CPU, encoded from the current latent)}."""
        return {k: (s, p.cpu() if isinstance(p, torch.Tensor) else p, v == self.version)
                for k, (v, s, p) in self._cache.items()}

    def load_cache(self, state: dict) -> None:
        """Restore :meth:`cache_state` after the latent itself was loaded (and touched).

        Only a payload that was current stays current; a stale one gets a version no write reaches, so the
        refresh window alone decides when it is re-encoded, as it did before the checkpoint.
        """
        dev = self.param.device
        self._cache = {k: (self.version if cur else -1, s, p.to(dev) if isinstance(p, torch.Tensor) else p)
                       for k, (s, p, cur) in state.items()}

    # ---------------------------------------------------------------- gradient delivery + update
    def add_use(self) -> None:
        """One more pending gradient this step."""
        self.uses += 1

    def on_grad(self, g: torch.Tensor) -> None:
        """Gradient of one use of the served weight; the update runs when the last pending use arrives."""
        g = g.detach().float().reshape(self.shape)
        if self.basis is not None:
            g = self._to_latent(g.reshape(self.rows, self.cols)).reshape(self.shape)
        self.gbuf = g if self.gbuf is None else self.gbuf.add_(g)
        self.uses -= 1
        if self.uses <= 0:
            self.uses = 0
            self._deliver(self.gbuf)
            self.gbuf = None

    def flush(self) -> None:
        """End of micro-batch: deliver what is pending, and apply an accumulated update on its last micro-batch.

        A partially delivered gradient is a use whose output never reached the loss.
        """
        if self.uses:
            self.ctl.stats["undelivered"] += self.uses
            self.uses = 0
        if self.gbuf is not None:
            self._deliver(self.gbuf)
            self.gbuf = None
        if self.ctl.apply and self.gacc is not None:
            g = self.gacc.float().div_(self.ctl.accum)
            self.gacc = None
            self.update(g)

    def _deliver(self, g: torch.Tensor) -> None:
        """A finished gradient of this micro-batch: update at once (accum 1), else add it to the sum."""
        if self.ctl.accum <= 1:
            self.update(g)
        elif self.gacc is None:
            self.gacc = g.to(self.ctl.state_dtype)
        else:
            self.gacc.add_(g)

    def _momentum(self, u: torch.Tensor, rows: tuple, active: torch.Tensor | None = None) -> torch.Tensor:
        """``u`` through the momentum EMA (beta1 > 0) at ``rows`` of the [B, R/B, C] view.

        Inactive experts keep their momentum and do not move.
        """
        ctl = self.ctl
        if not ctl.beta1:
            return u
        if self.m is None:
            self.m = torch.zeros(self.shape, dtype=ctl.state_dtype, device=self.param.device)
        b = self.r.shape[0]
        m = self.m.view(b, self.rows // b, self.cols)[rows]
        mu = ctl.beta1 * m.float() + (1 - ctl.beta1) * u
        if active is not None:
            mu = torch.where(active[:, None, None], mu, m.float())
        m.copy_(sr_cast(mu, m.dtype))
        return mu

    def _opt(self, **sums) -> None:
        """Add this update's telemetry sums (device scalars or numbers) to its weight class (``opt_metrics``)."""
        d = self.ctl.opt.setdefault(self.kind, {})
        for k, v in sums.items():
            d[k] = d[k] + v if k in d else v

    @torch.no_grad()
    def update(self, grad: torch.Tensor) -> None:
        """Momentum-free Adafactor step (factored 2nd moments, unit-RMS clip), per expert for 3-D weights."""
        ctl = self.ctl
        b = self.r.shape[0]
        g3 = grad.reshape(b, self.rows // b, self.cols)
        lat3 = self.param.data.reshape(b, self.rows // b, self.cols)
        if g3[0].numel() > CHUNK_ELEMS:  # one matrix above a chunk (a 2-D embedding/head): rows in chunks
            for e in range(b):
                self._update_big(e, g3[e], lat3[e])
            self.version += 1
            ctl.stats["updates"] += 1
            return
        per = max(1, CHUNK_ELEMS // max(1, g3[0].numel()))
        for s0 in range(0, b, per):
            sl = slice(s0, min(s0 + per, b))
            g = g3[sl]
            active = g.flatten(1).abs().amax(1) > 0
            self._opt(units=active.numel())
            if not active.any():
                continue
            g2 = g.square() + ctl.eps
            a = active.float()
            b2 = ctl.beta2
            self.r[sl] = torch.where(active[:, None], b2 * self.r[sl] + (1 - b2) * g2.mean(-1), self.r[sl])
            self.c[sl] = torch.where(active[:, None], b2 * self.c[sl] + (1 - b2) * g2.mean(-2), self.c[sl])
            self.t[sl] += a
            bc = (1 - ctl.beta2 ** self.t[sl].clamp(min=1))[:, None]
            rh, ch = self.r[sl] / bc, self.c[sl] / bc
            v = rh[:, :, None] * ch[:, None, :] / rh.mean(-1).clamp(min=1e-30)[:, None, None]
            u = g / v.sqrt().clamp(min=1e-15)
            rms = u.square().mean((1, 2)).sqrt()
            u /= (rms / ctl.clip).clamp(min=1.0)[:, None, None]
            u = self._momentum(u, (sl,), active)
            w = lat3[sl].float()
            step = ctl.lr * u * a[:, None, None]
            self._opt(gsq=g.square().sum(), dsq=step.square().sum(), wsq=(w.square() * a[:, None, None]).sum(),
                      n=a.sum() * g[0].numel(), active=a.sum(), clipped=((rms > ctl.clip) & active).sum(),
                      u_rms=(rms * a).sum())
            lat3[sl] = sr_cast(w - step, self.param.dtype)
        self.version += 1
        ctl.stats["updates"] += 1

    def _update_big(self, e: int, g: torch.Tensor, lat: torch.Tensor) -> None:
        """The :meth:`update` step of one [R, C] matrix, CHUNK_ELEMS rows at a time (bonsai27 ``Store.update``).

        Three passes keep the result the whole-matrix step: second moments, then the unit-RMS norm of the
        update over every chunk, then the write. Done whole, a 636M-weight tied head held ~10 float32 copies
        at once (46 GiB peak for a 4B model).
        """
        ctl = self.ctl
        n_r = g.shape[0]
        step = max(1, CHUNK_ELEMS // self.cols)
        spans = [(r0, min(r0 + step, n_r)) for r0 in range(0, n_r, step)]
        self._opt(units=1)
        if not any(bool(g[r0:r1].abs().amax() > 0) for r0, r1 in spans):
            return
        b2 = ctl.beta2
        csum = torch.zeros_like(self.c[e])
        for r0, r1 in spans:
            g2 = g[r0:r1].square() + ctl.eps
            self.r[e, r0:r1] = b2 * self.r[e, r0:r1] + (1 - b2) * g2.mean(-1)
            csum += g2.sum(0)
        self.c[e] = b2 * self.c[e] + (1 - b2) * csum / n_r
        self.t[e] += 1
        bc = 1 - ctl.beta2 ** self.t[e].clamp(min=1)
        rh, ch = self.r[e] / bc, self.c[e] / bc
        den = rh.mean().clamp(min=1e-30)

        def u_of(r0: int, r1: int) -> torch.Tensor:
            return g[r0:r1] / (rh[r0:r1, None] * ch[None, :] / den).sqrt().clamp(min=1e-15)

        sq = torch.zeros((), device=g.device)
        for r0, r1 in spans:
            sq += u_of(r0, r1).square().sum()
        rms = (sq / g.numel()).sqrt()
        scale = (rms / ctl.clip).clamp(min=1.0)
        gsq, dsq, wsq = (torch.zeros((), device=g.device) for _ in range(3))
        for r0, r1 in spans:
            u = self._momentum((u_of(r0, r1) / scale)[None], (slice(e, e + 1), slice(r0, r1)))[0]
            w = lat[r0:r1].float()
            gsq += g[r0:r1].float().square().sum()
            dsq += (ctl.lr * u).square().sum()
            wsq += w.square().sum()
            lat[r0:r1] = sr_cast(w - ctl.lr * u, self.param.dtype)
        self._opt(gsq=gsq, dsq=dsq, wsq=wsq, n=g.numel(), active=1, clipped=(rms > ctl.clip).float(), u_rms=rms)


class _Served(torch.autograd.Function):
    """Served weight of a latent; backward hands the gradient to the latent and passes nothing on."""

    @staticmethod
    def forward(ctx, dummy, lat):
        ctx.lat = lat
        return lat.served()

    @staticmethod
    def backward(ctx, grad):
        ctx.lat.on_grad(grad)
        return torch.zeros_like(ctx.lat.ctl.dummy), None


def served_tensor(lat: Latent) -> torch.Tensor:
    """Served weight inside the autograd graph; counts a pending gradient when called in a forward pass."""
    if torch.is_grad_enabled() and torch._C._current_graph_task_id() == -1:
        lat.add_use()  # recompute during backward (checkpointing) is not a new use
    return _Served.apply(lat.ctl.dummy, lat)


def build_segments(layout: dict, name: str, train_types: bool = True) -> list[Segment]:
    """Quantized segments of parameter ``name`` from the layout (none for float targets)."""
    segs = []
    for g in layout["by_param"].get(name, []):
        t = layout["targets"][g]
        if t["type"] in FLOAT_TYPES or not train_types:
            continue
        rows = t["rows"].long()
        perm = t["colperm"].long() if t["colperm"] is not None else None
        inv = torch.argsort(perm) if perm is not None else None
        seg = Segment(g, t["type"], _as_slice(rows), perm, inv, len(rows))
        seg.chunks = expert_chunks(rows, layout["params"][name])
        segs.append(seg)
    return segs


def expert_chunks(rows: torch.Tensor, shape: list[int]) -> list[tuple]:
    """(rows, n, expert) chunks of a segment: split where the expert changes (3-D weights) and by size."""
    cols = shape[-1]
    per_expert = shape[1] if len(shape) == 3 else None
    eid = rows // per_expert if per_expert else torch.zeros_like(rows)
    cut = [0, *(torch.nonzero(eid[1:] != eid[:-1]).flatten() + 1).tolist(), len(rows)]
    out = []
    for a, b in itertools.pairwise(cut):
        part = _as_slice(rows[a:b])
        out.extend((r, _len(r), int(eid[a])) for r in _chunk(part, b - a, cols))
    return out


def _len(rows: slice | torch.Tensor) -> int:
    return rows.stop - rows.start if isinstance(rows, slice) else len(rows)


def install(model: nn.Module, latents: dict[str, Latent], skip: set[str] = frozenset()) -> list:
    """Serve every latent (except ``skip``) through forward hooks on its owning module; returns the handles."""
    owners: dict[nn.Module, list[tuple[str, Latent]]] = {}
    for name, lat in latents.items():
        if name in skip:
            continue
        mod_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        owners.setdefault(model.get_submodule(mod_name), []).append((attr, lat))

    handles = []
    for mod, items in owners.items():
        def pre(module, args, items=items):
            del args
            for attr, lat in items:
                module._parameters[attr] = served_tensor(lat)

        def post(module, args, output, items=items):
            del args
            for attr, lat in items:
                module._parameters[attr] = lat.param
            return output

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    return handles


def adafactor_state(latents: dict[str, Latent]) -> dict:
    """Optimizer state of every latent (CPU), for checkpoints."""
    return {n: (lat.r.cpu(), lat.c.cpu(), lat.t.cpu()) + (() if lat.m is None else (lat.m.cpu(),))
            for n, lat in latents.items()}


def load_adafactor_state(latents: dict[str, Latent], state: dict) -> None:
    """Inverse of :func:`adafactor_state`."""
    for n, (r, c, t, *m) in state.items():
        lat = latents[n]
        lat.r.copy_(r)
        lat.c.copy_(c)
        lat.t.copy_(t)
        if m:
            lat.m = m[0].to(lat.param.device)


def cosine_lr(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup then cosine from ``lr`` to ``min_lr`` over ``total`` steps (Bolt: 1e-5 -> 1e-7)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
