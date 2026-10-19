# SPDX-License-Identifier: Apache-2.0
"""Engine-gate controls for a ternary GGUF (plan M4 / T6): does the runtime apply the basis we trained in?

The engine computes ``W_s . H(s * x)`` from the folded file. Two controls built from the same file isolate that
contract:

  ``unfold``  every ternary tensor decoded and unfolded to the model basis (``H(t * d) * s``), written F32, and the
              ``prism.hadamard.*`` keys dropped. The engine then runs a plain model with the very weights the
              ternary file serves, so its perplexity must match the ternary file's up to float error.
  ``flip``    the ternary file with every sign negated. That is still a valid file, and its perplexity must
              differ: a gate that cannot tell it apart from the ternary file proves nothing.

    python -m omk_qad.ternary_gate unfold tern.gguf tern.unfold.gguf [--llama-cpp /opt/llama.cpp]
    python -m omk_qad.ternary_gate flip tern.gguf tern.flip.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .ternary import TYPES, HadamardBasis, decode, register_gguf_types, unpack

PRISM = "prism.hadamard."


def _sign_table(reader) -> dict[int, torch.Tensor]:
    widths = [int(w) for w in reader.fields[PRISM + "sign_widths"].contents()]
    values = np.asarray(reader.fields[PRISM + "sign_values"].contents(), dtype=np.float32)
    out, at = {}, 0
    for w in widths:
        out[w] = torch.from_numpy(values[at:at + w].copy())
        at += w
    if at != len(values):
        raise SystemExit(f"sign_values holds {len(values)} values, sign_widths sum to {at}")
    return out


def build(mode: str, src: Path, dst: Path, llama_cpp: Path = Path("/opt/llama.cpp")) -> dict:
    """Write the ``mode`` control of ``src`` to ``dst``; returns a short report."""
    from .gguf_export import _import_llama_cpp

    gguf = _import_llama_cpp(Path(llama_cpp))[0]  # llama.cpp's gguf-py, as the exporter
    register_gguf_types(gguf)
    reader = gguf.GGUFReader(src)
    if PRISM + "block_size" not in reader.fields:
        raise SystemExit(f"{src} has no prism.hadamard metadata")
    block = int(reader.fields[PRISM + "block_size"].contents())
    signs = _sign_table(reader)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if mode == "unfold" and field.name.startswith(PRISM):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if mode == "flip" and field.name == PRISM + "sign_values":
            value = [-int(v) for v in value]
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    data, unfolded = [], 0
    for t in reader.tensors:
        if mode == "unfold" and t.tensor_type.name in TYPES:
            n = int(t.shape[0])
            raw = np.asarray(t.data).reshape(-1, t.data.shape[-1])
            folded = decode(*unpack(t.tensor_type.name, raw, n))
            w = HadamardBasis(n, signs=signs[n], block=block).unfold(folded).numpy()
            w = w.reshape(*reversed([int(s) for s in t.shape]))
            writer.add_tensor_info(t.name, w.shape, w.dtype, w.nbytes, gguf.GGMLQuantizationType.F32)
            data.append(w)
            unfolded += 1
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            data.append(t.data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in data:
        writer.write_tensor_data(d, tensor_endianess=reader.endianess)
    writer.close()
    return {"mode": mode, "src": str(src), "dst": str(dst), "block": block, "sign_widths": sorted(signs),
            "unfolded": unfolded, "tensors": len(reader.tensors)}


def main(argv=None) -> None:
    """CLI: ``unfold`` / ``flip`` SRC DST."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("unfold", "flip"))
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--llama-cpp", type=Path, default=Path("/opt/llama.cpp"))
    a = ap.parse_args(argv)
    print(build(a.mode, a.src, a.dst, a.llama_cpp))


if __name__ == "__main__":
    main()
