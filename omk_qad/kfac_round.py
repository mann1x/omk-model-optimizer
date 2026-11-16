# SPDX-License-Identifier: Apache-2.0
"""H-aware rounding of the routed experts' gate_up with the K-FAC factors (bm-895 step 3).

For every decoder layer and expert e, the gate_up weight W0_e [out, in] (read from the BF16 base GGUF, the tensor
the factors were collected on) is rounded to a GGUF type so that the curvature-weighted error

    L_e(Δ) = ½ tr(Δ A_e Δᵀ G_e),   Δ = Q(W) − W0_e

is small, where A_e = E[x xᵀ] (the layer input) and G_e = E[g gᵀ] (the sampled-label Fisher of the gate_up output)
come from :mod:`omk_qad.kfac` (``factors_w512_s2048_g10``). The encoders are the project's own IQ3_XXS / IQ4_XS
kernels (:mod:`omk_qad.iq3`, :mod:`omk_qad.iq4`), byte-identical to llama-quantize's, and for the IQ2 family
(IQ2_XXS / IQ2_XS / IQ2_S) libggml itself through :mod:`omk_qad.ggml_lib` (``--llama-cpp``: the exact code
llama-quantize runs, on the CPU), all fed the same imatrix (``E[x²]`` per column, 1 where an expert saw no token), so

  * ``rtn``  reproduces llama-quantize's PTQ codes exactly (the ``--ref`` GGUF's payload is compared byte for byte:
    ENCODER_MATCH); it is the floor every other method has to beat;
  * ``gptq`` is GPTQ at the encoder's block granularity (one 256-column super-block of every row quantized jointly
    with the real encoder, its error folded into the later columns through the Cholesky factor of the damped
    inverse of A_e -- the OBQ update is the same for any G, see the plan), the standard A-only error feedback;
  * ``--ste K`` refines the latent after GPTQ with K straight-through steps on the full quadratic (gradient
    G_e Δ A_e, the only place the output-side curvature G enters), keeping the best iterate.

Only the gate_up payloads are written: ``--out`` is a byte copy of ``--ref`` (a llama-quantize PTQ GGUF of the same
layout) with each ``blk.N.ffn_gate_up_exps.weight`` replaced in place, so the GGUF-side score differs from the
PTQ arm's by the rounding alone. Per layer and method the three predictions of :mod:`omk_qad.kfac_validate`
(kfac / diag / plain Σ Δ²), as per-scored-token means, go to ``--report`` (comparable to validate_arms_*.json).

Usage:
  python -m omk_qad.kfac_round --factors DIR --base BF16.gguf --imatrix imatrix.gguf --qtype IQ4_XS
         [--ref PTQ.gguf --out OUT.gguf] [--report R.json] [--layers all|0,15-17] [--method gptq|rtn]
         [--damp 0.01] [--ste 0 --lr 0.2] [--experts N] [--device cuda:0] [--llama-cpp DIR  (IQ2 types)]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .kfac_validate import Gguf, quad

STEM = "ffn_gate_up_exps"
BLOCK = 256  # super-block of IQ2_* / IQ3_XXS / IQ4_XS: the encoder's unit along the input dimension
METHODS = ("rtn", "gptq")
GGML_TYPES = ("IQ2_XXS", "IQ2_XS", "IQ2_S")  # encoded by libggml (no project kernel); need --llama-cpp
QTYPES = ("IQ4_XS", "IQ3_XXS", *GGML_TYPES)
_GGML: dict[str, GgmlNative | None] = {"mod": None}


class GgmlNative:
    """The IQ2 family with the same ``encode`` / ``decode`` contract as :mod:`omk_qad.iq3`, through libggml.

    ``ggml_quantize_chunk`` is the code llama-quantize runs, so ``rtn`` stays byte-identical to the ``--ref`` GGUF
    (the ENCODER_MATCH check keeps its meaning). It runs on the CPU, row-parallel; tensors are moved per call and the
    payload comes back on the input's device, so the GPTQ loop is unchanged.
    """

    def __init__(self, llama_cpp: str | Path):
        """Bind ``<llama_cpp>/build/bin/libggml-base.so``."""
        from .ggml_lib import GGMLLib

        self.lib = GGMLLib(Path(llama_cpp))

    @staticmethod
    def _id(qtype: str) -> int:
        import gguf

        return int(gguf.GGMLQuantizationType[qtype])

    def encode(self, qtype: str, x: torch.Tensor, importance: torch.Tensor | None = None) -> torch.Tensor:
        """uint8 payload [rows, cols / 256, block bytes] of float32 ``x`` [rows, cols]; ``importance`` = E[x²] [cols]."""  # noqa: E501
        if x.dim() != 2 or x.shape[1] % BLOCK:
            raise ValueError(f"{qtype} needs [rows, cols] with cols a multiple of {BLOCK}, got {tuple(x.shape)}")
        if importance is not None and (importance.dim() != 1 or importance.numel() != x.shape[1]):
            raise ValueError(f"importance must be [cols={x.shape[1]}], got {tuple(importance.shape)}")
        rows, cols = x.shape
        imp = None if importance is None else importance.detach().float().cpu().numpy()
        p = self.lib.quantize(self._id(qtype), x.detach().float().cpu().numpy(), imp)  # [rows, row bytes]
        return torch.from_numpy(p).reshape(rows, cols // BLOCK, -1).to(x.device)

    def decode(self, qtype: str, packed: torch.Tensor, rows: int, cols: int,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[rows, cols] values of a payload (the type's ``to_float``), on the payload's device."""
        p = packed.detach().cpu().numpy().reshape(rows, -1)
        return torch.from_numpy(self.lib.dequantize(self._id(qtype), p, cols)).to(device=packed.device, dtype=dtype)


def native(qtype: str):
    """The encoder/decoder module of ``qtype`` (an IQ2 type needs ``--llama-cpp`` to have been given)."""
    if qtype in ("IQ4_XS", "IQ4_NL"):
        from . import iq4 as m
    elif qtype in ("IQ3_XXS", "IQ3_S"):
        from . import iq3 as m
    elif qtype in GGML_TYPES:
        if _GGML["mod"] is None:
            raise ValueError(f"{qtype} is encoded by libggml: pass --llama-cpp <llama.cpp tree with "
                             "build/bin/libggml-base.so>")
        return _GGML["mod"]
    else:
        raise ValueError(f"{qtype}: not a supported type ({', '.join(QTYPES)})")
    return m


def load_imatrix(path: str, layers: list[int]) -> dict[int, torch.Tensor]:
    """E[x²] per expert and input column [E, C] of the gate_up of each layer (1 where an expert saw no token)."""
    import gguf

    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    out = {}
    for j in layers:
        s = np.asarray(t[f"blk.{j}.{STEM}.weight.in_sum2"].data, dtype=np.float64)  # [E, C]
        c = np.asarray(t[f"blk.{j}.{STEM}.weight.counts"].data, dtype=np.float64).reshape(-1, 1)  # [E, 1]
        m = s / np.maximum(c, 1.0)
        m[(c == 0)[:, 0]] = 1.0
        out[j] = torch.from_numpy(m.astype(np.float32))
    return out


def factors_gu(path: Path, device) -> dict:
    """gu_A [E, C, C], gu_G [E, R, R] (fp32, scale applied) on ``device``; n [E]; the layer meta."""
    f = torch.load(path, map_location="cpu", weights_only=False)
    out = {"n": f["n"].double(), "meta": f["meta"]}
    for k in ("gu_A", "gu_G"):
        s = f[k + "_scale"].view(-1, 1, 1).to(device)
        out[k] = f[k].to(device).float() * s
    return out


def encode_decode(qtype: str, x: torch.Tensor, imp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Payload uint8 [R, cols/256, nbytes] and dequantised float32 [R, cols] of ``x`` [R, cols] (cols % 256 == 0)."""
    m = native(qtype)
    p = m.encode(qtype, x.contiguous(), imp)
    return p, m.decode(qtype, p, x.shape[0], x.shape[1])


def rtn_expert(w0: torch.Tensor, imp: torch.Tensor, qtype: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """llama-quantize's rounding of one expert: payload, Q, and the latent that encodes to it (W0 itself)."""
    p, q = encode_decode(qtype, w0, imp)
    return p, q, w0


def gptq_expert(w0: torch.Tensor, a: torch.Tensor, imp: torch.Tensor, qtype: str, damp: float, latent_dtype=None):
    """GPTQ with the encoder's 256-column block as the group: payload, Q, and the error-fed latent W_adj.

    ``a`` is the per-token input second moment E[x xᵀ] [C, C] (the GPTQ Hessian up to a factor). Columns the data
    never excites (zero diagonal) get a unit diagonal and keep their weights. The latent returned is the weight each
    block was quantised from (W0 plus the folded errors of the earlier blocks): encoding it block by block gives
    exactly this payload, so it is the start point of the STE refinement.

    ``latent_dtype`` (e.g. torch.float16): every block is rounded to that storage type BEFORE it is encoded, and
    the fed-back error is taken against the rounded block, so the returned latent is exactly representable in that
    dtype and a trainer storing it so (omk_qad.train --latent-dtype fp16 --init-latents) re-encodes to this very
    payload. Without it the fp16 copy of an fp32-encoded W_adj flips ~0.3 % of the IQ2_XXS bytes (bm-892).
    """
    rows, cols = w0.shape
    u, floored = inverse_cholesky(a, damp)
    w = w0.clone()
    adj = w0.clone()
    pay = []
    for s0 in range(0, cols, BLOCK):
        s1 = s0 + BLOCK
        blk = w[:, s0:s1] if latent_dtype is None else w[:, s0:s1].to(latent_dtype).float()
        adj[:, s0:s1] = blk
        p, q = encode_decode(qtype, blk, imp[s0:s1])
        pay.append(p)
        if s1 < cols:
            err = torch.linalg.solve_triangular(u[s0:s1, s0:s1], blk - q, upper=True, left=False)
            w[:, s1:].addmm_(err, u[s0:s1, s1:], alpha=-1.0)
        w[:, s0:s1] = q
    return torch.cat(pay, 1), w, adj, floored


def inverse_cholesky(a: torch.Tensor, damp: float, floor: float = 1e-6) -> tuple[torch.Tensor, int]:
    """U (float32) with Uᵀ U = (A + damp·mean(diag A)·I)⁻¹, and the number of eigenvalues that had to be floored.

    The factors are stored in fp16 (per-expert scale), and Gemma's hidden states have a few huge dimensions, so the
    fp16 rounding of the large entries leaves A indefinite by ~1e-3 of its top eigenvalue: a plain Cholesky fails
    (layer 0, leading minor 922). A is symmetrised and its eigenvalues floored at ``floor``·λ_max before the damping.
    Columns the data never excites (zero diagonal) get a unit diagonal and keep their weights.
    """
    h = a.double()
    h = 0.5 * (h + h.T)
    d = h.diagonal().clone()
    dead = d == 0
    if dead.any():
        h = h.clone()
        h.diagonal()[dead] = 1.0
        d[dead] = 1.0
    ev, vec = torch.linalg.eigh(h)
    lo = floor * ev[-1].clamp_min(1e-300)
    floored = int((ev < lo).sum())
    if floored:
        h = (vec * ev.clamp_min(lo)) @ vec.T
        h = 0.5 * (h + h.T)
    h.diagonal().add_(damp * d.mean())
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(hinv, upper=True).float(), floored


def loss_terms(delta: torch.Tensor, a: torch.Tensor, g: torch.Tensor) -> tuple[float, float, float]:
    """tr(Δ A Δᵀ G), Σ Δ² G_ii A_jj, Σ Δ² of one expert (float64 Python floats)."""
    full, dg, sq = quad(delta[None], a[None], g[None])
    return full.double().item(), dg.double().item(), sq.double().item()


def ste_refine(latent: torch.Tensor, w0: torch.Tensor, a: torch.Tensor, g: torch.Tensor, imp: torch.Tensor,
               qtype: str, steps: int, lr: float):
    """Straight-through descent of ½ tr(Δ A Δᵀ G) on the latent; returns the best (payload, Q, loss, step)."""
    p, q = encode_decode(qtype, latent, imp)
    best = (p, q, loss_terms(q - w0, a, g)[0], 0)
    rms_q = (q - w0).square().mean().sqrt().clamp_min(1e-12)  # the quantisation error scale sets the step
    x = latent.clone()
    for it in range(1, steps + 1):
        delta = q - w0
        grad = g @ delta @ a  # ∂/∂Q of ½ tr(Δ A Δᵀ G), through the STE = ∂/∂latent
        x.add_(grad, alpha=-(lr * rms_q / grad.square().mean().sqrt().clamp_min(1e-30)).item())
        p, q = encode_decode(qtype, x, imp)
        full = loss_terms(q - w0, a, g)[0]
        if full < best[2]:
            best = (p, q, full, it)
    return best


def ref_payload(ref: Gguf, layer: int, qtype: str) -> np.ndarray:
    """The ``--ref`` GGUF's gate_up payload of ``layer`` as flat uint8 (its type must be ``qtype``)."""
    t = ref.t[f"blk.{layer}.{STEM}.weight"]
    if t.tensor_type.name != qtype:
        raise SystemExit(f"KFACR_FAIL --ref layer {layer} {STEM} is {t.tensor_type.name}, not {qtype}")
    return np.asarray(t.data).reshape(-1).view(np.uint8)


def write_payload(out_path: str, layer: int, payload: np.ndarray) -> None:
    """Replace ``blk.<layer>.ffn_gate_up_exps.weight`` in the (already copied) output GGUF, in place."""
    import gguf

    r = gguf.GGUFReader(out_path, "r+")
    t = {x.name: x for x in r.tensors}[f"blk.{layer}.{STEM}.weight"]
    view = np.asarray(t.data).reshape(-1).view(np.uint8)
    if view.size != payload.size:
        raise SystemExit(f"KFACR_FAIL layer {layer}: payload {payload.size} bytes, GGUF tensor {view.size}")
    view[:] = payload
    r.data.flush()
    del r


def parse_layers(spec: str, avail: list[int]) -> list[int]:
    """``all`` or ``0,3,15-17`` → the layers to process, restricted to those with a factor file."""
    if spec == "all":
        return avail
    keep = {j for part in spec.split(",") for j in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)}
    return [j for j in avail if j in keep]


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--base", required=True, help="BF16 GGUF the factors were collected on (W0)")
    ap.add_argument("--imatrix", required=True, help="llama.cpp imatrix GGUF the PTQ used (E[x²] per expert column)")
    ap.add_argument("--qtype", required=True, choices=QTYPES)
    ap.add_argument("--llama-cpp", help="llama.cpp tree whose build/bin/libggml-base.so encodes the IQ2 types "
                                        "(llama-quantize's own encoder; required for them, unused otherwise)")
    ap.add_argument("--ref", help="llama-quantize PTQ GGUF of the target layout (byte check + the output's template)")
    ap.add_argument("--out", help="output GGUF: a copy of --ref with the gate_up payloads replaced (never overwritten)")
    ap.add_argument("--report", help="JSON report path")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--method", default="gptq", choices=METHODS)
    ap.add_argument("--damp", type=float, default=0.01, help="GPTQ damping, fraction of mean diag(A)")
    ap.add_argument("--ste", type=int, default=0, help="STE refinement steps after the method (0 = none)")
    ap.add_argument("--lr", type=float, default=0.2, help="STE step as a fraction of the quantisation-error RMS")
    ap.add_argument("--experts", type=int, default=0, help="only the first N experts of each layer (smoke)")
    ap.add_argument("--save-latents", default="", help="directory for the per-layer error-fed latents W_adj (fp16 "
                    "[E, R, C] in GGUF tensor order, layer_NN.pt): the weight each block was encoded from, so "
                    "re-encoding it block by block reproduces the payload -- the seed of a QAD that starts at this "
                    "rounding instead of W0 (bm-892)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    if a.llama_cpp:
        _GGML["mod"] = GgmlNative(a.llama_cpp)
    native(a.qtype)  # fail early on an unsupported type (or an IQ2 type without --llama-cpp)
    fdir = Path(a.factors)
    avail = sorted(int(p.stem.split("_")[1]) for p in fdir.glob("layer_*.pt"))
    layers = parse_layers(a.layers, avail)
    if not layers:
        raise SystemExit(f"KFACR_FAIL no layers in {fdir} match {a.layers}")
    base = Gguf(a.base)
    ref = Gguf(a.ref) if a.ref else None
    if a.out:
        if not a.ref:
            raise SystemExit("KFACR_FAIL --out needs --ref (the GGUF whose gate_up payloads are replaced)")
        if Path(a.out).exists():
            raise SystemExit(f"KFACR_FAIL {a.out} exists (never overwritten)")
        if a.experts:
            raise SystemExit("KFACR_FAIL --experts is a smoke option; no --out with it")
    imat = load_imatrix(a.imatrix, layers)
    meta0 = torch.load(fdir / f"layer_{layers[0]:02d}.pt", map_location="cpu", weights_only=False)["meta"]
    n_scored = float(meta0["scored"])
    rep = {"factors": str(fdir), "base": a.base, "imatrix": a.imatrix, "qtype": a.qtype, "method": a.method,
           "damp": a.damp, "ste": a.ste, "lr": a.lr, "n_scored": n_scored, "layers": layers, "per_layer": {}}
    print(f"KFACR_START {a.qtype} {a.method}{f'+ste{a.ste}' if a.ste else ''} layers {layers} factors {fdir} "
          f"ref {a.ref} out {a.out}", flush=True)
    if a.out:
        t0 = time.time()
        shutil.copyfile(a.ref, a.out)
        print(f"copied {a.ref} -> {a.out} in {time.time() - t0:.0f}s", flush=True)
    t_all = time.time()
    tot = {"rtn": [0.0, 0.0, 0.0], a.method: [0.0, 0.0, 0.0], "final": [0.0, 0.0, 0.0]}
    for j in layers:
        t0 = time.time()
        fac = factors_gu(fdir / f"layer_{j:02d}.pt", device)
        if int(fac["meta"]["scored"]) != int(n_scored):
            raise SystemExit(f"KFACR_FAIL layer {j} scored {fac['meta']['scored']} != {int(n_scored)}")
        w0_all = torch.from_numpy(base.dequant(j, STEM))  # [E, R, C] float32
        n_exp = w0_all.shape[0] if not a.experts else min(a.experts, w0_all.shape[0])
        w_e = (fac["n"] / n_scored).tolist()  # weight of each expert's quadratic in the per-scored-token mean
        acc = {k: [0.0, 0.0, 0.0] for k in ("rtn", a.method, "final")}
        payloads, match, ste_steps, ste_gain, floored = [], 0, [], 0.0, 0
        refp = ref_payload(ref, j, a.qtype) if ref is not None else None
        per_exp = refp.size // w0_all.shape[0] if refp is not None else 0
        lat_out = [] if a.save_latents else None
        latent_dtype = torch.float16 if a.save_latents else None  # a saved seed must re-encode to this very payload
        for e in range(n_exp):
            w0 = w0_all[e].to(device)
            a_e, g_e = fac["gu_A"][e], fac["gu_G"][e]
            imp = imat[j][e].to(device)
            p_rtn, q_rtn, _ = rtn_expert(w0, imp, a.qtype)
            terms = loss_terms(q_rtn - w0, a_e, g_e)
            for i in range(3):
                acc["rtn"][i] += terms[i] * (w_e[e] if i < 2 else 1.0)
            if refp is not None:
                mine = p_rtn.reshape(-1).cpu().numpy()
                match += int(np.array_equal(mine, refp[e * per_exp:(e + 1) * per_exp]))
            if a.method == "gptq":
                p_m, q_m, latent, fl = gptq_expert(w0, a_e, imp, a.qtype, a.damp, latent_dtype)
                floored += fl
            else:
                p_m, q_m, latent = p_rtn, q_rtn, w0
            terms = loss_terms(q_m - w0, a_e, g_e)
            for i in range(3):
                acc[a.method][i] += terms[i] * (w_e[e] if i < 2 else 1.0)
            p_f, q_f, full_f = p_m, q_m, terms[0]
            if a.ste:
                p_s, q_s, full_s, step = ste_refine(latent, w0, a_e, g_e, imp, a.qtype, a.ste, a.lr)
                ste_steps.append(step)
                if full_s < full_f:
                    ste_gain += (full_f - full_s) * w_e[e]
                    p_f, q_f, full_f = p_s, q_s, full_s
            terms = loss_terms(q_f - w0, a_e, g_e)
            for i in range(3):
                acc["final"][i] += terms[i] * (w_e[e] if i < 2 else 1.0)
            payloads.append(p_f.reshape(-1).cpu().numpy())
            if lat_out is not None:  # the latent of the FINAL payload: W_adj after GPTQ, or the STE iterate's start
                lat_out.append(latent.to(torch.float16).cpu())
        out_j = {k: {"kfac": 0.5 * v[0], "diag": 0.5 * v[1], "sq": v[2]} for k, v in acc.items()}
        out_j["experts"] = n_exp
        out_j["eigenvalues_floored"] = floored
        if refp is not None:
            out_j["rtn_bytes_match"] = f"{match}/{n_exp}"
        if a.ste:
            out_j["ste_best_step_mean"] = float(np.mean(ste_steps)) if ste_steps else 0.0
            out_j["ste_gain_kfac"] = 0.5 * ste_gain
        rep["per_layer"][j] = out_j
        for k in tot:
            for i in range(3):
                tot[k][i] += acc[k][i]
        if a.out:
            write_payload(a.out, j, np.concatenate(payloads))
        if lat_out is not None:
            Path(a.save_latents).mkdir(parents=True, exist_ok=True)
            lp = Path(a.save_latents) / f"layer_{j:02d}.pt"
            if lp.exists():
                raise SystemExit(f"KFACR_FAIL {lp} exists (never overwritten)")
            torch.save({"stem": STEM, "layer": j, "qtype": a.qtype, "method": a.method, "ref": a.ref,
                        "latent": torch.stack(lat_out)}, lp)  # [E, R, C] fp16, GGUF tensor order
        r, m, f = out_j["rtn"]["kfac"], out_j[a.method]["kfac"], out_j["final"]["kfac"]
        print(f"KFACR_LAYER {j} {time.time() - t0:.0f}s experts {n_exp} kfac rtn {r:.5f} {a.method} {m:.5f} "
              f"({m / max(r, 1e-30):.3f}x) final {f:.5f} ({f / max(r, 1e-30):.3f}x) sq rtn {out_j['rtn']['sq']:.1f} "
              f"final {out_j['final']['sq']:.1f} eig floored {floored}"
              + (f" rtn==ref {match}/{n_exp}" if refp is not None else "")
              + (f" ste best@{out_j['ste_best_step_mean']:.1f}" if a.ste else ""), flush=True)
        if refp is not None and match != n_exp:
            print(f"KFACR_ENCODER_MISMATCH layer {j}: rtn payload == --ref for {match}/{n_exp} experts", flush=True)
        del fac, w0_all
        torch.cuda.empty_cache()
    rep["total"] = {k: {"kfac": 0.5 * v[0], "diag": 0.5 * v[1], "sq": v[2]} for k, v in tot.items()}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    r, f = rep["total"]["rtn"]["kfac"], rep["total"]["final"]["kfac"]
    print(f"KFACR_DONE {len(layers)} layers {time.time() - t_all:.0f}s kfac rtn {r:.5f} final {f:.5f} "
          f"({f / max(r, 1e-30):.3f}x) out {a.out}", flush=True)


if __name__ == "__main__":
    main()
