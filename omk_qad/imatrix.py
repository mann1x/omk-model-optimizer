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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Collector:
    """Accumulates sum(x^2) per input column and row counts for every 2-D+ weight of ``model``."""

    def __init__(self, model: nn.Module, names: set[str] | None = None):
        """Hook every Linear (and fused-experts module) whose weight name is in ``names`` (default: all)."""
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
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

    def _add(self, name: str, x: torch.Tensor, expert: int, n_expert: int) -> None:
        x = x.detach().reshape(-1, x.shape[-1]).double()
        if name not in self.sums:
            self.sums[name] = torch.zeros(n_expert, x.shape[-1], dtype=torch.float64, device=x.device)
            self.counts[name] = torch.zeros(n_expert, dtype=torch.int64, device=x.device)
        self.sums[name][expert] += x.square().sum(0)
        self.counts[name][expert] += x.shape[0]

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
    """Write ``imatrix.pt`` and (with a layout) llama.cpp's ``imatrix.gguf`` into ``out_dir``."""
    out_dir = Path(out_dir)
    paths = [out_dir / "imatrix.pt"]
    torch.save({"entries": imat, "datasets": datasets, "chunk_size": chunk_size, "chunk_count": chunks}, paths[0])
    if layout is not None:
        from .gguf_export import _import_llama_cpp

        gguf = _import_llama_cpp(Path(llama_cpp))[0]
        w = gguf.GGUFWriter(str(out_dir / "imatrix.gguf"), arch="imatrix")
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
        paths.append(out_dir / "imatrix.gguf")
    return paths


def load(path: Path) -> dict:
    """``imatrix.pt`` record: {"entries": by HF parameter name, "datasets", "chunk_size", "chunk_count"}."""
    rec = torch.load(path, map_location="cpu", weights_only=False)
    rec["path"] = str(path)
    return rec
