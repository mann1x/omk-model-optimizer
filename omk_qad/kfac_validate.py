# SPDX-License-Identifier: Apache-2.0
"""Validate the K-FAC surrogate (bm-895 step 2): predict the KL of llama-quantize GGUF arms from their Δ alone.

For every arm (a GGUF quantized from the BF16 base) and every decoder layer, Δ = dequant(Q) − W0 is formed for the
routed experts' gate_up / down and the dense MLP's gate / up / down, and the K-FAC quadratic

    KL_pred = Σ_layers Σ_e (n_e / N_scored) · ½ · tr(Δ_e A_e Δ_eᵀ G_e)      (+ the dense terms with n_dense)

is evaluated with the factors written by :mod:`omk_qad.kfac` (A, G are per-token means over the n_e tokens routed
to expert e; the sampled-label Fisher G makes ½ΔᵀHΔ the second-order KL(teacher ‖ student) per scored token).
Two baselines come free: the diagonal Fisher Σ_ij Δ_ij² G_ii A_jj and the plain Σ Δ² (what a PTQ without curvature
minimises). Attention and embeddings carry no factors, so arms are comparable only within the same attention layout,
and the predicted scale can be checked only on arms whose attention is near-lossless (Q8_0).

Gate (plan_bm892_kfac_hrounding.md): the predicted ranking matches the measured klk within each attention group and
the scale is within ~2× on the Q8_0-attention arms. The measured klk is read from ``<arm>.json`` beside each GGUF.

The dequantisation (gguf-py, numpy) is the cost: ~13 s per arm and layer on one core, run threaded over the arms;
the K-FAC matmuls (~5 TFLOP per arm and layer) run on the GPU. The base GGUF must be the BF16 export of the HF model
the factors were collected on; with ``--hf`` layer 0 of both is compared bit for bit before anything else.

Usage: python -m omk_qad.kfac_validate --factors DIR --base BF16.gguf --arm A=A.gguf [--arm ...] --out validate.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

TENSORS = {  # GGUF tensor stem → (factor A key, factor G key); the routed ones are per expert
    "ffn_gate_up_exps": ("gu_A", "gu_G"),
    "ffn_down_exps": ("dn_A", "dn_G"),
    "ffn_gate": ("dense_mlp_in_A", "dense_gate_G"),
    "ffn_up": ("dense_mlp_in_A", "dense_up_G"),
    "ffn_down": ("dense_down_A", "dense_down_G"),
}
ROUTED = ("ffn_gate_up_exps", "ffn_down_exps")
ATTN = ("attn_q", "attn_k", "attn_v", "attn_output")


class Gguf:
    """One GGUF: tensors by name, dequantised to float32 numpy on demand."""

    def __init__(self, path: str):
        """Open ``path`` (header + memory-mapped data)."""
        import gguf

        self.path = path
        self.reader = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}

    def types(self, layer: int, stems) -> dict:
        """Quantisation type name of each ``blk.<layer>.<stem>.weight``."""
        return {s: self.t[f"blk.{layer}.{s}.weight"].tensor_type.name for s in stems}

    def dequant(self, layer: int, stem: str) -> np.ndarray:
        """float32 array in torch layout: [E, out, in] for routed experts, [out, in] for dense."""
        from gguf import quants

        t = self.t[f"blk.{layer}.{stem}.weight"]
        return quants.dequantize(t.data, t.tensor_type)

    def layer(self, layer: int) -> dict:
        """Every MLP tensor of one decoder layer, dequantised."""
        return {s: self.dequant(layer, s) for s in TENSORS}


def factors(path: Path, device) -> dict:
    """The fp32 factors of one layer file (scale × fp16) on ``device``, plus the counts."""
    f = torch.load(path, map_location="cpu", weights_only=False)
    out = {"n": f["n"].double(), "dense_n": float(f["dense_n"]), "meta": f["meta"]}
    for k in ("gu_A", "gu_G", "dn_A", "dn_G"):
        s = f[k + "_scale"].view(-1, *([1] * (f[k].dim() - 1)))
        out[k] = (f[k].to(device).float() * s.to(device))
    for k in ("mlp_in_A", "gate_G", "up_G", "down_A", "down_G"):
        out["dense_" + k] = f["dense_" + k].to(device).float() * float(f["dense_" + k + "_scale"])
    return out


def quad(d: torch.Tensor, a: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per matrix Δ=d[c] of the batch: tr(Δ A Δᵀ G) (G full [c,o,o] or diagonal [c,o]), Σ Δ² G_ii A_jj, Σ Δ²."""
    da = torch.bmm(d, a)  # [c, o, i]
    if g.dim() == 3:
        full = (da * torch.bmm(g, d)).sum((1, 2))
        gd = g.diagonal(dim1=-2, dim2=-1)
    else:
        full = (da * d * g[:, :, None]).sum((1, 2))
        gd = g
    ad = a.diagonal(dim1=-2, dim2=-1)
    sq = d.square()
    return full, (sq * gd[:, :, None] * ad[:, None, :]).sum((1, 2)), sq.sum((1, 2))


def layer_terms(delta: dict, fac: dict, n_scored: float, device, chunk: int = 16) -> dict:
    """The three predictions of one arm on one layer, per component (gu / dn / dense), as per-scored-token means."""
    out = {}
    w = fac["n"] / n_scored  # [E] weight of each expert's quadratic in the mean over scored tokens
    for stem in ROUTED:
        ka, kg = TENSORS[stem]
        d = torch.from_numpy(delta[stem])
        acc = torch.zeros(3, dtype=torch.float64)
        for c0 in range(0, d.shape[0], chunk):
            c1 = min(c0 + chunk, d.shape[0])
            full, dg, sq = (t.double().cpu() for t in quad(d[c0:c1].to(device), fac[ka][c0:c1], fac[kg][c0:c1]))
            acc += torch.stack([(full * w[c0:c1]).sum(), (dg * w[c0:c1]).sum(), sq.sum()])
        out[stem] = {"kfac": 0.5 * acc[0].item(), "diag": 0.5 * acc[1].item(), "sq": acc[2].item()}
    wd = fac["dense_n"] / n_scored
    for stem in ("ffn_gate", "ffn_up", "ffn_down"):
        ka, kg = TENSORS[stem]
        d = torch.from_numpy(delta[stem]).to(device)[None]
        full, dg, sq = (t.double().cpu().item() for t in quad(d, fac[ka][None], fac[kg][None]))
        out[stem] = {"kfac": 0.5 * full * wd, "diag": 0.5 * dg * wd, "sq": sq}
    # "sq" is the plain Σ Δ² of the layer (no token weighting): the curvature-free baseline
    return out


def identity_check(base: Gguf, hf_dir: str) -> None:
    """Layer 0 of the BF16 GGUF must equal the HF export the factors came from, bit for bit."""
    from safetensors import safe_open

    idx = json.loads((Path(hf_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    pairs = {"ffn_gate_up_exps": "model.language_model.layers.0.experts.gate_up_proj",
             "ffn_down_exps": "model.language_model.layers.0.experts.down_proj",
             "ffn_gate": "model.language_model.layers.0.mlp.gate_proj.weight",
             "ffn_down": "model.language_model.layers.0.mlp.down_proj.weight"}
    for stem, key in pairs.items():
        with safe_open(str(Path(hf_dir) / idx[key]), "pt") as f:
            hf = f.get_tensor(key).float()
        g = torch.from_numpy(base.dequant(0, stem))
        if hf.shape != g.shape or not torch.equal(hf, g):
            diff = (hf - g).abs().max().item() if hf.shape == g.shape else float("nan")
            raise SystemExit(f"KFACV_FAIL base GGUF {stem} != HF {key}: shapes {tuple(g.shape)} vs {tuple(hf.shape)}, "
                             f"max|diff| {diff}")
    print("KFACV_CHECK base GGUF layer 0 == HF export (gate_up, down, dense gate/down) bit for bit", flush=True)


def measured_of(gguf_path: str) -> float | None:
    """The klk from the ``<arm>.json`` the GGUF scorer wrote beside the file, if present."""
    p = Path(gguf_path).with_suffix(".json")
    if p.exists():
        return json.loads(p.read_text()).get("klk")
    return None


def summarize(res: dict) -> None:
    """Totals, the per-group ranking check and the scale check on Q8_0-attention arms."""
    arms = res["arms"]
    print(f"{'arm':5s} {'attention':28s} {'measured':>9s} {'pred_kfac':>10s} {'ratio':>6s} {'pred_diag':>10s} "
          f"{'sum_sq':>10s} {'gu':>8s} {'dn':>8s} {'dense':>8s}")
    for name, a in arms.items():
        t = a["total"]
        m = a["measured"]
        ratio = f"{t['kfac'] / m:6.2f}" if m else "   n/a"
        print(f"{name:5s} {a['attention']:28s} {m if m is not None else float('nan'):9.4f} {t['kfac']:10.4f} {ratio} "
              f"{t['diag']:10.4f} {t['sq']:10.1f} {t['gu']:8.4f} {t['dn']:8.4f} {t['dense']:8.4f}")
    groups: dict[str, list[str]] = {}
    for name, a in arms.items():
        groups.setdefault(a["attention"], []).append(name)
    verdicts = []
    for att, names in groups.items():
        names = [n for n in names if arms[n]["measured"] is not None]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                dm = arms[x]["measured"] - arms[y]["measured"]
                for key in ("kfac", "diag", "sq"):
                    dp = arms[x]["total"][key] - arms[y]["total"][key]
                    ok = (dm > 0) == (dp > 0)
                    verdicts.append((key, f"{x}>{y}" if dm > 0 else f"{x}<{y}", ok))
                    print(f"KFACV_RANK {key:4s} [{att}] {x} vs {y}: measured {dm:+.4f} predicted {dp:+.4f} "
                          f"{'OK' if ok else 'WRONG'}")
    for name, a in arms.items():
        if a["measured"] and all(v == "Q8_0" for v in a["attention_types"].values()):
            r = a["total"]["kfac"] / a["measured"]
            print(f"KFACV_SCALE {name}: predicted/measured {r:.2f} {'OK' if 0.5 <= r <= 2.0 else 'OUT'} "
                  f"(attention Q8_0, its own KL not predicted)")
    bad = [v for v in verdicts if v[0] == "kfac" and not v[2]]
    print(f"KFACV_RANKING kfac {len(verdicts) // 3 - len(bad)}/{len(verdicts) // 3} same-attention pairs ordered right"
          + (f"; wrong: {[v[1] for v in bad]}" if bad else ""), flush=True)


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", required=True, help="dir of layer_XX.pt from omk_qad.kfac")
    ap.add_argument("--base", required=True, help="BF16 GGUF of the model the factors were collected on (W0)")
    ap.add_argument("--hf", help="HF export dir: layer 0 must match --base bit for bit")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=GGUF")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=6, help="dequant threads (one per source per layer)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    base = Gguf(a.base)
    arms = {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        arms[name] = Gguf(path)
    if a.hf:
        identity_check(base, a.hf)
    fdir = Path(a.factors)
    files = sorted(fdir.glob("layer_*.pt"))
    layers = [int(p.stem.split("_")[1]) for p in files]
    if a.layers != "all":
        keep = sorted({j for part in a.layers.split(",")
                       for j in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})
        layers = [j for j in layers if j in keep]
    if not layers:
        raise SystemExit(f"KFACV_FAIL no layer files in {fdir}")
    meta0 = torch.load(fdir / f"layer_{layers[0]:02d}.pt", map_location="cpu", weights_only=False)["meta"]
    n_scored = float(meta0["scored"])
    res = {"factors": str(fdir), "base": a.base, "n_scored": n_scored, "layers": layers, "arms": {}}
    for name, g in arms.items():
        att = g.types(0, ATTN)
        res["arms"][name] = {"gguf": g.path, "measured": measured_of(g.path), "attention_types": att,
                             "attention": "/".join(att[s] for s in ATTN), "mlp_types": g.types(0, TENSORS.keys()),
                             "per_layer": {}, "total": {}}
    print(f"KFACV_START {len(layers)} layers x {len(arms)} arms, N_scored {int(n_scored)}, factors {fdir}", flush=True)
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for j in layers:
            t0 = time.time()
            fac = factors(fdir / f"layer_{j:02d}.pt", device)
            if int(fac["meta"]["scored"]) != int(n_scored):
                raise SystemExit(f"KFACV_FAIL layer {j} scored {fac['meta']['scored']} != {int(n_scored)}")
            futs = {name: pool.submit(g.layer, j) for name, g in arms.items()}
            w0 = base.layer(j)
            for name, fut in futs.items():
                q = fut.result()
                delta = {s: q[s] - w0[s] for s in TENSORS}
                for s in TENSORS:
                    if delta[s].shape != w0[s].shape:
                        raise SystemExit(f"KFACV_FAIL {name} layer {j} {s}: shape {delta[s].shape} vs {w0[s].shape}")
                terms = layer_terms(delta, fac, n_scored, device)
                res["arms"][name]["per_layer"][j] = terms
                del q, delta
            del fac, w0
            torch.cuda.empty_cache()
            line = " ".join(f"{n}={sum(t['kfac'] for t in res['arms'][n]['per_layer'][j].values()):.4f}" for n in arms)
            print(f"KFACV_LAYER {j} {time.time() - t0:.0f}s kfac {line}", flush=True)
    for name, r in res["arms"].items():
        tot = {"kfac": 0.0, "diag": 0.0, "sq": 0.0, "gu": 0.0, "dn": 0.0, "dense": 0.0}
        for terms in r["per_layer"].values():
            for s, t in terms.items():
                for k in ("kfac", "diag", "sq"):
                    tot[k] += t[k]
                tot["gu" if s == "ffn_gate_up_exps" else "dn" if s == "ffn_down_exps" else "dense"] += t["kfac"]
        r["total"] = tot
    Path(a.out).write_text(json.dumps(res, indent=1))
    summarize(res)
    print(f"KFACV_DONE {a.out} {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
