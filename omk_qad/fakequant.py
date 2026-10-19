# SPDX-License-Identifier: Apache-2.0
"""Training-time fake quantization driven by a layout (omk_qad.layout), with the update done in backward.

Every large weight is a *latent* (float16 on GPU, 2 bytes/weight; stochastic rounding on write). Its *served*
value is built when its module runs: for each GGUF tensor the layout maps to it, the latent rows the GGUF
tensor holds are gathered in GGUF column order (so the GGML blocks are exactly the GGUF's), encoded to the
planned type, decoded and scattered back. Encoders: ModelOpt (torch/CUDA) for IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/
IQ2_S/Q8_0, libggml (the code llama-quantize runs) for everything else (IQ3_S, K-quants, ...). Export
re-encodes the final latent with the same encoder, so the GGUF holds the blocks training served.

The gradient of the served weight is the latent's gradient (straight-through). It is captured in the
backward of ``_Served`` and applied at once (momentum-free Adafactor, per expert for 3-D tensors), so no
gradient of the whole model is ever held -- the ternary bonsai27 recipe, made model-agnostic.

Served weights are swapped into ``module._parameters`` by a forward pre-hook and restored by a forward
hook, so they exist for one module call at a time (gradient-checkpoint recompute runs the hooks again;
the packed encoding is cached per latent version, so a recompute only decodes). The version is the
latent's own counter, bumped by every write (``Latent.touch``): writes go through ``.data``, whose version
counter is separate from the parameter's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELOPT_TYPES = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q8_0"}
FLOAT_TYPES = {"F32", "F16", "BF16"}
CHUNK_ELEMS = 64 * 2**20  # rows are encoded / updated in chunks of about this many values


class Codec:
    """Encode/decode 2-D float32 row blocks to one GGML type with the encoder export will also use."""

    def __init__(self, llama_cpp: Path):
        """``llama_cpp`` provides libggml (non-ModelOpt types) and gguf-py's type enum."""
        from .gguf_export import _import_llama_cpp

        self.gguf = _import_llama_cpp(Path(llama_cpp))[0]
        self.llama_cpp = Path(llama_cpp)
        self._ggml = None

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

    def encode(self, qtype: str, x: torch.Tensor):
        """Packed payload of float32 ``x`` [rows, cols]: uint8 torch tensor (ModelOpt) or numpy array (ggml)."""
        if qtype in MODELOPT_TYPES:
            from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

            packed, _ = GGML_FORMAT_REGISTRY[qtype.lower()].quantize(x.float().contiguous())
            return packed
        return self.ggml.quantize(int(self.qtype(qtype)), x.detach().float().cpu().numpy())

    def decode(self, qtype: str, packed, rows: int, cols: int, device, dtype=torch.float32) -> torch.Tensor:
        """Inverse of :meth:`encode`, as a [rows, cols] tensor on ``device``."""
        if qtype in MODELOPT_TYPES:
            from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY

            shape = torch.tensor([rows, cols], dtype=torch.int64)
            return GGML_FORMAT_REGISTRY[qtype.lower()].dequantize(packed, shape, dtype=dtype).reshape(rows, cols)
        out = self.ggml.dequantize(int(self.qtype(qtype)), packed, cols)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    def to_bytes(self, qtype: str, packed, rows: int) -> np.ndarray:
        """Payload as numpy uint8 [rows, row_bytes] (GGUF row layout)."""
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
    dummy: torch.Tensor = field(default_factory=lambda: torch.zeros((), requires_grad=True))
    stats: dict = field(default_factory=lambda: {"encodes": 0, "updates": 0, "undelivered": 0})


class Latent:
    """One trained weight: latent storage, its layout segments and its Adafactor state."""

    def __init__(self, name: str, param: nn.Parameter, segments: list[Segment], ctl: Control, codec: Codec,
                 latent_dtype: torch.dtype):
        """Convert ``param`` in place to ``latent_dtype`` storage; it no longer requires grad."""
        self.name, self.param, self.segments, self.ctl, self.codec = name, param, segments, ctl, codec
        param.requires_grad_(False)
        param.data = param.data.to(latent_dtype)
        self.shape = tuple(param.shape)
        self.cols = self.shape[-1]
        self.rows = param.numel() // self.cols
        b = self.shape[0] if param.ndim == 3 else 1
        dev = param.device
        self.r = torch.zeros(b, self.rows // b, device=dev)
        self.c = torch.zeros(b, self.cols, device=dev)
        self.t = torch.zeros(b, device=dev)
        self.uses = 0
        self.version = 0  # bumped by every write; param._version cannot be used: updates go through .data
        self.gbuf: torch.Tensor | None = None
        self._cache: dict[int, tuple] = {}  # chunk id -> (latent version, step encoded, payload)

    # ---------------------------------------------------------------- served weight
    @torch.no_grad()
    def served(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The weight the model runs with: latent with every quantized segment encoded/decoded."""
        dtype = dtype or self.ctl.compute_dtype
        lat = self.param.data
        if self.ctl.bypass or not self.segments:
            return lat.to(dtype)
        out = lat.to(dtype).clone().reshape(self.rows, self.cols)
        w = lat.reshape(self.rows, self.cols)
        version = self.version
        k = 0
        for seg in self.segments:
            for rows in _chunk(seg.rows, seg.n_rows, self.cols):
                x = w[rows].float()
                if seg.colperm is not None:
                    x = x[:, seg.colperm.to(x.device)]
                hit = self._cache.get(k)
                fresh = hit is not None and (
                    hit[0] == version
                    or (seg.qtype not in MODELOPT_TYPES and self.ctl.step - hit[1] < self.ctl.ggml_refresh)
                )
                if fresh:
                    payload = hit[2]
                else:
                    payload = self.codec.encode(seg.qtype, x)
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
            for _ in _chunk(seg.rows, seg.n_rows, self.cols):
                out.setdefault(seg.gguf, []).append(self._cache[k][2])
                k += 1
        return out

    def touch(self) -> None:
        """Mark the latent as written (any write outside :meth:`update`, e.g. a checkpoint load)."""
        self.version += 1

    # ---------------------------------------------------------------- gradient delivery + update
    def add_use(self) -> None:
        """One more pending gradient this step."""
        self.uses += 1

    def on_grad(self, g: torch.Tensor) -> None:
        """Gradient of one use of the served weight; the update runs when the last pending use arrives."""
        g = g.detach().float().reshape(self.shape)
        self.gbuf = g if self.gbuf is None else self.gbuf.add_(g)
        self.uses -= 1
        if self.uses <= 0:
            self.uses = 0
            self.update(self.gbuf)
            self.gbuf = None

    def flush(self) -> None:
        """End of step: apply a partially delivered gradient (a use whose output never reached the loss)."""
        if self.uses:
            self.ctl.stats["undelivered"] += self.uses
            self.uses = 0
        if self.gbuf is not None:
            self.update(self.gbuf)
            self.gbuf = None

    @torch.no_grad()
    def update(self, grad: torch.Tensor) -> None:
        """Momentum-free Adafactor step (factored 2nd moments, unit-RMS clip), per expert for 3-D weights."""
        ctl = self.ctl
        b = self.r.shape[0]
        g3 = grad.reshape(b, self.rows // b, self.cols)
        lat3 = self.param.data.reshape(b, self.rows // b, self.cols)
        per = max(1, CHUNK_ELEMS // max(1, g3[0].numel()))
        for s0 in range(0, b, per):
            sl = slice(s0, min(s0 + per, b))
            g = g3[sl]
            active = g.flatten(1).abs().amax(1) > 0
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
            new = lat3[sl].float() - ctl.lr * u * a[:, None, None]
            lat3[sl] = sr_cast(new, self.param.dtype)
        self.version += 1
        ctl.stats["updates"] += 1


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
        segs.append(Segment(g, t["type"], _as_slice(rows), perm, inv, len(rows)))
    return segs


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
    return {n: (lat.r.cpu(), lat.c.cpu(), lat.t.cpu()) for n, lat in latents.items()}


def load_adafactor_state(latents: dict[str, Latent], state: dict) -> None:
    """Inverse of :func:`adafactor_state`."""
    for n, (r, c, t) in state.items():
        lat = latents[n]
        lat.r.copy_(r)
        lat.c.copy_(c)
        lat.t.copy_(t)


def cosine_lr(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup then cosine from ``lr`` to ``min_lr`` over ``total`` steps (Bolt: 1e-5 -> 1e-7)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
