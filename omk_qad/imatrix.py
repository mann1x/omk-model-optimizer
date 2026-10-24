# SPDX-License-Identifier: Apache-2.0
"""Importance matrix (llama.cpp imatrix): collect from the teacher, save in llama.cpp's format, feed the encoders.

Collected per HF weight as the per-input-column sum of x^2 and the number of rows seen -- per expert for fused
3-D expert weights (llama.cpp's ``nmat``). Linear layers are hooked on their input; fused experts by
intercepting ``F.linear`` inside the experts module and recovering the expert from the weight slice's storage
offset (the HF fused-expert loop calls F.linear on ``param[expert]`` views).

Saved twice, always (the imatrix is part of every quant's provenance):
  * ``imatrix.pt``   -- by HF parameter name: {"sums": [E, C] float64, "counts": [E] int64}
  * ``imatrix.gguf`` -- llama.cpp's GGUF imatrix (``<gguf name>.in_sum2`` [E, C], ``<gguf name>.counts`` [E, 1],
    ``imatrix.datasets/chunk_count/chunk_size``), keyed by GGUF tensor names through the layout (column
    permutation applied), so ``llama-quantize --imatrix`` reads it too.

Encoder conventions: libggml takes E[x^2] (sum / count; 1 where count is 0, as llama-quantize); ModelOpt's
weighted IQ encoders take sqrt(E[x^2]) floored at 1e-4 of its mean (ModelOpt's measured choice).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Collector:
    """Accumulates sum(x^2) per input column and row counts for every 2-D+ weight of ``model``.

    ``folded`` {weight name: GGUF column order or None} also accumulates the ternary students' statistic,
    sum((H s x)^2) with x in GGUF column order (omk_qad.ternary.HadamardBasis, seeded signs): a ternary latent
    lives in that basis, and the rotation mixes columns, so the model-basis diagonal does not carry over.
    """

    def __init__(self, model: nn.Module, names: set[str] | None = None,
                 folded: dict[str, torch.Tensor | None] | None = None, block: int | None = None):
        """Hook every Linear (and fused-experts module) whose weight name is in ``names`` (default: all)."""
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.fsums: dict[str, torch.Tensor] = {}
        self.folded = {n: (None if perm is None else perm.long(), None) for n, perm in (folded or {}).items()}
        self._bases: dict = {}  # input width -> omk_qad.ternary.HadamardBasis
        self.block = block  # the model's one Hadamard block (omk_qad.ternary.model_block)
        self.handles = []
        # transformers' fused experts default to grouped_mm (torch._grouped_mm, no F.linear): the per-expert
        # eager loop computes the same thing and is what F.linear interception sees; restored by close()
        self._config = model.config
        self._impl = getattr(model.config, "_experts_implementation", None)
        self.n_fused = 0
        named = dict(model.named_parameters())
        want = names if names is not None else set(named)
        # by identity, not by module path: a tied head's weight is registered once, under the embedding's name
        # (lm_head.weight is model.embed_tokens.weight), and its input -- the final hidden state -- belongs there
        name_of = {id(p): n for n, p in named.items()}
        for mod_name, mod in model.named_modules():
            prefix = f"{mod_name}." if mod_name else ""
            wname = name_of.get(id(mod.weight)) if isinstance(mod, nn.Linear) else None
            if wname is not None and wname in want:
                self.handles.append(mod.register_forward_pre_hook(self._linear_hook(wname)))
            fused = [(f"{prefix}{p}", getattr(mod, p)) for p in ("gate_up_proj", "up_proj", "down_proj")
                     if isinstance(getattr(mod, p, None), torch.Tensor) and getattr(mod, p).ndim == 3]
            fused = [(n, p) for n, p in fused if n in want]
            if fused and not isinstance(mod, nn.Linear):
                self._wrap_experts(mod, fused)
                self.n_fused += 1
        if self.n_fused and self._impl not in (None, "eager"):
            model.config._experts_implementation = "eager"

    def basis(self, width: int):
        """The seeded Hadamard basis of an input width (shared by every weight of that width)."""
        from .ternary import HadamardBasis

        if width not in self._bases:
            self._bases[width] = HadamardBasis(width, block=self.block)
        return self._bases[width]

    def _add(self, name: str, x: torch.Tensor, expert: int, n_expert: int) -> None:
        x = x.detach().reshape(-1, x.shape[-1])
        xd = x.double()
        if name not in self.sums:
            self.sums[name] = torch.zeros(n_expert, x.shape[-1], dtype=torch.float64, device=x.device)
            self.counts[name] = torch.zeros(n_expert, dtype=torch.int64, device=x.device)
            if name in self.folded:
                self.fsums[name] = torch.zeros(n_expert, x.shape[-1], dtype=torch.float64, device=x.device)
        self.sums[name][expert] += xd.square().sum(0)
        self.counts[name][expert] += x.shape[0]
        if name in self.folded:
            perm = self.folded[name][0]
            xf = x.float()[:, perm.to(x.device)] if perm is not None else x.float()
            self.fsums[name][expert] += self.basis(x.shape[-1]).fold(xf).double().square().sum(0)

    def _linear_hook(self, name: str):
        def hook(module, args):
            del module
            self._add(name, args[0], 0, 1)

        return hook

    def _wrap_experts(self, mod: nn.Module, fused: list[tuple[str, torch.Tensor]]) -> None:
        collector = self
        real_forward = mod.forward

        def forward(*args, **kwargs):
            def linear(inp, weight, bias=None):
                for name, param in fused:
                    if weight.untyped_storage().data_ptr() == param.untyped_storage().data_ptr():
                        e = (weight.storage_offset() - param.storage_offset()) // param.stride(0)
                        collector._add(name, inp, int(e), param.shape[0])
                        break
                return real_linear(inp, weight, bias)

            real_linear = F.linear
            F.linear = linear
            try:
                return real_forward(*args, **kwargs)
            finally:
                F.linear = real_linear

        mod.forward = forward
        self.handles.append(_Restore(mod))

    def close(self) -> None:
        """Remove the hooks and restore the experts implementation."""
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.n_fused and self._impl is not None:
            self._config._experts_implementation = self._impl

    def state(self) -> dict:
        """{param name: {"sums", "counts"}} on CPU."""
        return {n: {"sums": self.sums[n].cpu(), "counts": self.counts[n].cpu()} for n in self.sums}

    def folded_state(self) -> dict:
        """{param name: {"sums" (folded basis, GGUF column order), "counts", "block"}} on CPU."""
        return {n: {"sums": self.fsums[n].cpu(), "counts": self.counts[n].cpu(),
                    "block": self.basis(self.fsums[n].shape[-1]).block} for n in self.fsums}


class _Restore:
    """Handle that drops an instance-level ``forward`` override (back to the class forward)."""

    def __init__(self, mod: nn.Module):
        self.mod = mod

    def remove(self) -> None:
        self.mod.__dict__.pop("forward", None)


def mean_sq(entry: dict) -> torch.Tensor:
    """E[x^2] per expert and column [E, C] (float32); 1 where an expert saw no rows (as llama-quantize)."""
    c = entry["counts"].double().clamp(min=1)[:, None]
    m = (entry["sums"] / c).float()
    m[entry["counts"] == 0] = 1.0
    return m


def importance_for(qtype: str, msq: torch.Tensor | None) -> torch.Tensor | None:
    """Importance vector an encoder of ``qtype`` takes from E[x^2] ``msq`` [C] (see module doc)."""
    if msq is None:
        return None
    from .fakequant import MODELOPT_TYPES

    if qtype in MODELOPT_TYPES:
        from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
        from modelopt.torch.quantization.ggml.common import floored_second_moments

        if not GGML_FORMAT_REGISTRY[qtype.lower()].weighted:
            return None
        floored = floored_second_moments(msq)
        return None if floored is None else floored.sqrt()
    return msq.float()


def per_gguf(imat: dict, layout: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """llama.cpp imatrix entries by GGUF name: (in_sum2 [E, C] float32, counts [E] float32)."""
    out = {}
    for g, t in layout["targets"].items():
        p = t["src"]
        if p is None or p not in imat or len(t["shape"]) < 2:
            continue
        sums, counts = imat[p]["sums"], imat[p]["counts"]
        if t["colperm"] is not None:
            sums = sums[:, t["colperm"].long()]
        out[g] = (sums.float().numpy(), counts.float().numpy())
    return out


def save(imat: dict, layout: dict | None, out_dir: Path, datasets: list[str], chunk_size: int, chunks: int,
         llama_cpp: Path) -> list[Path]:
    """Write ``imatrix.pt`` and (with a layout) llama.cpp's ``imatrix.gguf`` into ``out_dir``.

    Both atomically, the GGUF first: ``imatrix.pt`` is the completion marker callers test for.
    """
    from .runctl import atomic_torch_save

    out_dir = Path(out_dir)
    paths = [out_dir / "imatrix.pt"]
    if layout is not None:
        from .gguf_export import _import_llama_cpp

        gguf = _import_llama_cpp(Path(llama_cpp))[0]
        part = out_dir / f"imatrix.gguf.{os.getpid()}.part"
        w = gguf.GGUFWriter(str(part), arch="imatrix")
        w.add_string("general.type", "imatrix")
        w.add_array("imatrix.datasets", datasets)
        w.add_uint32("imatrix.chunk_count", chunks)
        w.add_uint32("imatrix.chunk_size", chunk_size)
        for g, (s, c) in sorted(per_gguf(imat, layout).items()):
            w.add_tensor(f"{g}.in_sum2", np.ascontiguousarray(s, dtype=np.float32))
            w.add_tensor(f"{g}.counts", np.ascontiguousarray(c.reshape(-1, 1), dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        os.replace(part, out_dir / "imatrix.gguf")
        paths.append(out_dir / "imatrix.gguf")
    atomic_torch_save({"entries": imat, "datasets": datasets, "chunk_size": chunk_size, "chunk_count": chunks},
                      paths[0])
    return paths


def load(path: Path) -> dict:
    """``imatrix.pt`` record: {"entries": by HF parameter name, "datasets", "chunk_size", "chunk_count"}."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["path"] = str(path)
    return rec


def folded_candidates(layout: dict) -> tuple[dict[str, torch.Tensor | None], int | None]:
    """Weights a ternary student could fold, and the model's Hadamard block.

    2-D+ sources whose every GGUF target the runtime folds (omk_qad.ternary.foldable), with one GGUF column order;
    the block is the one ``prism.hadamard.block_size`` that divides all their input widths.
    """
    from .ternary import foldable, model_block

    out: dict[str, torch.Tensor | None] = {}
    for p, gs in layout["by_param"].items():
        shape = layout["params"][p]
        if len(shape) < 2 or shape[-1] % 256 or not all(foldable(g) for g in gs):
            continue
        perms = {None if layout["targets"][g]["colperm"] is None else tuple(layout["targets"][g]["colperm"].tolist())
                 for g in gs}
        if len(perms) == 1:
            perm = next(iter(perms))
            out[p] = None if perm is None else torch.tensor(perm)
    block = model_block(layout["params"][p][-1] for p in out) if out else None
    return out, block


def save_folded(fimat: dict, out_dir: Path, datasets: list[str], chunk_size: int, chunks: int) -> Path:
    """``imatrix.folded.pt``: the ternary students' statistic, with the basis it was computed in."""
    from .runctl import atomic_torch_save

    path = Path(out_dir) / "imatrix.folded.pt"
    atomic_torch_save({"entries": fimat, "datasets": datasets, "chunk_size": chunk_size, "chunk_count": chunks,
                       "basis": "seeded"}, path)
    return path
