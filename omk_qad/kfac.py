# SPDX-License-Identifier: Apache-2.0
"""K-FAC curvature of a Gemma-4 MoE text model, collected in passes over cached windows (bm-895).

Around the BF16 weights W0 a distillation student equals its teacher, so the loss gradient is ~0 and, for any
quantization Δ = Q(W) − W0, ``L ≈ ½ Δᵀ H Δ``. Per linear layer H is approximated by Kronecker factors
``H ≈ G ⊗ A`` with ``A = E[x xᵀ]`` (layer input) and ``G = E[g gᵀ]`` (loss gradient at the layer output). The gradient
of ``½ΔᵀHΔ`` is then ``G · Δ · A`` per layer: any format can be rounded / refined against it with no forward pass.

G is the TRUE Fisher: at each scored position one token y is sampled from the model's own distribution and
``-log p(y)`` is back-propagated (summed, so each token's g is a per-sample gradient). At a matched optimum this is
the Gauss-Newton Hessian of the KL; the empirical gradient of the distillation loss itself would be ~0 here.
Scored positions are the trainer's: next-token role > 0 (anchor counted as teacher). A and G are accumulated over every
token of the window (context tokens feed the scored ones through attention); tokens after the last scored position
contribute g = 0, a uniform scale on G that does not change any rounding decision.

Attention sinks (window-start positions, delimiters ' ' '.' ',' and the turn / channel tokens): every later position
attends to them, so in ONE backward of the summed loss their gradient is the coherent sum Σ_t g_{p|t} over the later
losses, 50-4000x the median, and the K-FAC-expand term squares that sum — inflated ~1000x in the early layers (bm-895
probe, 2026-11-15: layer-0 prediction 15.0 vs measured KL 0.014 per token, layer 15 fine). ``--rademacher`` is the fix:
random-sign the per-position losses so E[g gᵀ] = Σ_t g_{p|t} g_{p|t}ᵀ (per-loss Fisher) at the same cost.
``--drop-resid K`` (positions whose residual norm exceeds K x the window median) is kept but did NOT identify the
sinks here: their residual norms are ordinary (x1-4), nothing was dropped. ``--rademacher`` did not change the early
layers either (22:25): the large gradients are the positions' own. What the probes then showed (22:40-23:00): the
per-layer KL has a Δ-independent floor (downstream route flips), and the early-layer excess is carried by 'void'
positions whose linear prediction is 20-13,700 nats while their measured KL is ~0; they have a gate_up-output gradient
norm of 28-1040, every legitimate position in every window of every layer stays below ~25. ``--drop-grow G`` applies
that ABSOLUTE cut to the rows entering G (median-relative cuts fail: layer 29's |g| is bimodal).

Collected per decoder layer:
  * routed experts, per expert e: ``gu_A[e]`` (hidden²), ``gu_G[e]`` ((2·moe_inter)²) for gate_up_proj[e];
    ``dn_A[e]`` (moe_inter²) and ``dn_G[e]`` (hidden², or its diagonal with ``--down-g diag``) for down_proj[e];
    ``n[e]`` = tokens routed to e (the normaliser of all four).
  * dense MLP (``--dense``): ``mlp_in_A`` (shared input of gate/up), ``gate_G``, ``up_G``, ``down_A``, ``down_G``.

The experts' forward is replaced per instance by an instrumented copy of transformers' eager loop
(``Gemma4TextExperts``; the class's ``@use_experts_implementation`` dispatch is bypassed). ``--check`` (always run
first) verifies, on one window, that the instrumented model reproduces the original logits and that every registered
gradient hook fired, before any pass starts.

Memory: fp32 accumulators ≈ 5.4 GB per layer (128 experts, hidden 2816, moe_inter 704) next to the bf16 model, so the
layers are collected in groups (``--layers-per-pass``), one pass over the same windows per group, model loaded once.
Each finished layer is written at once (``<out>/layer_XX.pt``, factors normalised by their count, stored as
``scale * fp16`` per matrix so tiny G values do not underflow); a rerun skips the layers already written.

Usage: python -m omk_qad.kfac --model DIR --cache CACHE --out DIR [--windows 256 --seq 4096 --layers-per-pass 5]
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

FACTOR_KEYS = ("gu_A", "gu_G", "dn_A", "dn_G")


class LayerAcc:
    """fp32 K-FAC accumulators of one decoder layer, on the model's device."""

    def __init__(self, experts, mlp, device, down_g: str, dense: bool, grow: float = 0.0):
        """Zeroed factors sized from ``experts`` (and the dense ``mlp`` with ``dense``).

        ``grow`` > 0: rows (position, expert) whose gate_up-OUTPUT gradient norm exceeds it are left out of G
        (gu_G and dn_G of that row; dense gate_G/up_G/down_G by the dense gate‖up gradient norm). Those are the
        'void' positions whose linearisation predicts an impossible loss change (bm-895, 2026-11-15). A and n are
        untouched (< 1 % of the rows).
        """
        e, h, i = experts.num_experts, experts.hidden_dim, experts.intermediate_dim
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)  # noqa: E731
        self.f = {"gu_A": z(e, h, h), "gu_G": z(e, 2 * i, 2 * i), "dn_A": z(e, i, i),
                  "dn_G": z(e, h, h) if down_g == "full" else z(e, h)}
        self.n = torch.zeros(e, dtype=torch.float64)
        self.down_g = down_g
        self.dense = None
        if dense and mlp is not None:
            hd, idd = mlp.gate_proj.in_features, mlp.gate_proj.out_features
            self.dense = {"mlp_in_A": z(hd, hd), "gate_G": z(idd, idd), "up_G": z(idd, idd), "down_A": z(idd, idd),
                          "down_G": z(hd, hd), "n": 0}
        self.pending = 0  # gradient hooks registered and not yet fired (checked after every backward)
        self.keep = None  # per window: bool [S] of the positions kept (None = all); set by the layer's pre-hook
        self.dropped = 0  # positions excluded so far (attention sinks)
        self.grow = grow
        self.rows_dropped = 0  # (position, expert) rows left out of G by the gradient-norm cut
        self.dense_rows_dropped = 0
        self._buf = {}  # grow > 0: dn_G rows per expert (fire before gu_G) and the dense trio, until the cut is known

    def add_a(self, key: str, e: int, x: torch.Tensor) -> None:
        """Add x xᵀ of expert ``e``'s input rows to A factor ``key``."""
        xf = x.detach().float()
        self.f[key][e].addmm_(xf.T, xf)

    def _acc_g(self, key: str, e: int, gf: torch.Tensor) -> None:
        if key == "dn_G" and self.down_g == "diag":
            self.f[key][e].add_(gf.square().sum(0))
        else:
            self.f[key][e].addmm_(gf.T, gf)

    def g_hook(self, key: str, e: int, keep: torch.Tensor | None = None):
        """Gradient hook adding g gᵀ (or its diagonal) of expert ``e``'s kept rows to G factor ``key``.

        With ``grow`` the down-output gradient (``dn_G``, fires first) is buffered and both rows are added when the
        gate_up-output gradient (``gu_G``) arrives and the rows above the cut are known.
        """
        self.pending += 1

        def hook(g: torch.Tensor) -> None:
            gf = g.detach().float()
            if keep is not None:
                gf = gf[keep]
            if self.grow > 0:
                if key == "dn_G":
                    self._buf[e] = gf
                    self.pending -= 1
                    return
                if e not in self._buf:
                    raise SystemExit(f"KFAC_FAIL expert {e}: gate_up gradient hook fired before the down hook")
                keep_row = gf.norm(dim=-1) <= self.grow
                self.rows_dropped += int((~keep_row).sum())
                self._acc_g("dn_G", e, self._buf.pop(e)[keep_row])
                gf = gf[keep_row]
            self._acc_g(key, e, gf)
            self.pending -= 1

        return hook

    def dense_hook(self, key: str, keep: torch.Tensor | None = None):
        """Gradient hook adding g gᵀ of the kept positions to dense-MLP factor ``key``.

        With ``grow`` the three dense gradients are buffered and added together once all have fired, leaving out the
        positions whose gate‖up output gradient norm exceeds the cut.
        """
        self.pending += 1

        def hook(g: torch.Tensor) -> None:
            gf = g.detach().reshape(-1, g.shape[-1]).float()
            if keep is not None:
                gf = gf[keep]
            self.pending -= 1
            if self.grow <= 0:
                self.dense[key].addmm_(gf.T, gf)
                return
            d = self._buf.setdefault("dense", {})
            d[key] = gf
            if len(d) < 3:
                return
            keep_row = torch.cat([d["gate_G"], d["up_G"]], dim=-1).norm(dim=-1) <= self.grow
            self.dense_rows_dropped += int((~keep_row).sum())
            for k, v in d.items():
                self.dense[k].addmm_(v[keep_row].T, v[keep_row])
            self._buf.pop("dense")

        return hook

    def bytes(self) -> int:
        """Size of the accumulators."""
        n = sum(t.numel() for t in self.f.values())
        if self.dense:
            n += sum(t.numel() for k, t in self.dense.items() if k != "n")
        return 4 * n


def _experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """The eager Gemma4TextExperts loop of transformers, plus K-FAC accumulation when ``self._kfac`` is set."""
    acc = getattr(self, "_kfac", None)
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        e = int(expert_idx[0])
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e])
        current_state = hidden_states[token_idx]
        h = F.linear(current_state, self.gate_up_proj[e])
        keep = acc.keep[token_idx] if acc is not None and acc.keep is not None else None
        if acc is not None:
            rows = current_state if keep is None else current_state[keep]
            acc.n[e] += rows.shape[0]
            acc.add_a("gu_A", e, rows)
            if h.requires_grad:
                h.register_hook(acc.g_hook("gu_G", e, keep))
        gate, up = h.chunk(2, dim=-1)
        a = self.act_fn(gate) * up
        y = F.linear(a, self.down_proj[e])
        if acc is not None:
            acc.add_a("dn_A", e, a if keep is None else a[keep])
            if y.requires_grad:
                y.register_hook(acc.g_hook("dn_G", e, keep))
        y = y * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, y.to(final_hidden_states.dtype))
    return final_hidden_states


def instrument(model, drop_resid: float = 0.0) -> list:
    """Replace every MoE layer's experts forward; hook the dense MLP linears. Returns the decoder layers.

    With ``drop_resid`` > 0 a pre-hook on each layer marks the positions whose residual-stream norm exceeds
    ``drop_resid`` x the window median as attention sinks; the accumulators skip them (``LayerAcc.keep``).
    """
    layers = list(model.model.layers)
    for layer in layers:
        if getattr(layer, "enable_moe_block", False):
            layer.experts.forward = types.MethodType(_experts_forward, layer.experts)
            layer.experts._kfac = None
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp._kfac = None

        if drop_resid > 0:
            def resid_hook(module, args, kwargs, _layer=layer):
                acc = getattr(getattr(_layer, "experts", None), "_kfac", None) or getattr(
                    getattr(_layer, "mlp", None), "_kfac", None)
                if acc is None:
                    return
                hs = args[0] if args else kwargs["hidden_states"]  # [1, S, H]
                rn = hs.detach().float().norm(dim=-1).reshape(-1)
                acc.keep = rn <= drop_resid * rn.median()
                acc.dropped += int((~acc.keep).sum())

            layer.register_forward_pre_hook(resid_hook, with_kwargs=True)
        if mlp is None:
            continue

        def lin_hook(name):
            def hook(module, args, output, _mlp=mlp, _name=name):
                acc = _mlp._kfac
                if acc is None or acc.dense is None:
                    return
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                keep = acc.keep
                if keep is not None:
                    x = x[keep]
                if _name == "gate_proj":  # gate and up share their input: one A
                    acc.dense["mlp_in_A"].addmm_(x.T, x)
                    acc.dense["n"] += x.shape[0]
                elif _name == "down_proj":
                    acc.dense["down_A"].addmm_(x.T, x)
                if output.requires_grad:
                    output.register_hook(acc.dense_hook({"gate_proj": "gate_G", "up_proj": "up_G",
                                                         "down_proj": "down_G"}[_name], keep))
            return hook

        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(mlp, name).register_forward_hook(lin_hook(name))
    return layers


def attach(layers, accs: dict) -> None:
    """Point each layer's experts / MLP at its accumulator (None = not collected in this pass)."""
    for li, layer in enumerate(layers):
        acc = accs.get(li)
        if getattr(layer, "enable_moe_block", False):
            layer.experts._kfac = acc
        if getattr(layer, "mlp", None) is not None:
            layer.mlp._kfac = acc


def head_logits(model, h: torch.Tensor) -> torch.Tensor:
    """Gemma4ForCausalLM's logits for hidden states ``h`` (lm_head + final logit softcapping), float32."""
    logits = model.lm_head(h).float()
    cap = model.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def fisher_backward(model, x: torch.Tensor, role: torch.Tensor, gen: torch.Generator, chunk: int = 512,
                    rademacher: bool = False) -> int:
    """One window: trunk forward, a sampled-label NLL at the scored positions, trunk backward. Returns #scored.

    With ``rademacher`` every scored position's NLL is multiplied by an independent random sign before the single
    backward: the gradient arriving at an earlier position p is then Σ_t ε_t g_{p|t} instead of Σ_t g_{p|t}, and its
    outer product has expectation Σ_t g_{p|t} g_{p|t}ᵀ (the per-loss Fisher the K-FAC term needs) rather than the
    square of a coherent sum (the attention-sink inflation). Same cost as the plain backward.
    """
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state  # [1, S, H]
    mask = (role[:, 1:] > 0).to(hidden.device)
    hs = hidden[:, :-1][mask]  # [n, H]
    n = hs.shape[0]
    if n == 0:
        return 0
    grad = torch.zeros_like(hs)
    hd = hs.detach()
    for s in range(0, n, chunk):  # the head's backward chunk by chunk, then ONE trunk backward
        hc = hd[s : s + chunk].clone().requires_grad_(True)
        logits = head_logits(model, hc)
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen)
            sgn = (torch.randint(0, 2, (y.shape[0],), generator=gen, device=y.device) * 2 - 1).float()
        nll_t = -torch.log_softmax(logits, dim=-1).gather(1, y).squeeze(1)
        nll = (nll_t * sgn).sum() if rademacher else nll_t.sum()
        (g,) = torch.autograd.grad(nll, hc)
        grad[s : s + chunk] = g.to(grad.dtype)
    torch.autograd.backward(hs, grad)
    return n


def _save_layer(acc: LayerAcc, path: Path, meta: dict) -> dict:
    """Normalise by counts, store each matrix as fp16 with an fp32 scale (mean |diag|, or mean for a diagonal)."""
    from .runctl import atomic_torch_save

    n = acc.n.clamp_min(1).to(torch.float32)
    out = {"n": acc.n.clone(), "meta": meta}
    for k, t in acc.f.items():
        t = t / n.to(t.device).view(-1, *([1] * (t.dim() - 1)))
        ref = t.diagonal(dim1=-2, dim2=-1) if t.dim() == 3 else t
        scale = ref.abs().mean(dim=-1).clamp_min(1e-30)  # [E]
        out[k] = (t / scale.view(-1, *([1] * (t.dim() - 1)))).half().cpu()
        out[k + "_scale"] = scale.cpu()
        if not torch.isfinite(out[k].float()).all():
            raise SystemExit(f"KFAC_FAIL {path.name} {k}: non-finite after fp16 scaling")
    if acc.dense:
        nd = max(acc.dense["n"], 1)
        for k, t in acc.dense.items():
            if k == "n":
                continue
            t = t / nd
            scale = t.diagonal().abs().mean().clamp_min(1e-30)
            out["dense_" + k] = (t / scale).half().cpu()
            out["dense_" + k + "_scale"] = scale.cpu()
        out["dense_n"] = acc.dense["n"]
    atomic_torch_save(out, path)
    return {"tokens_routed": int(acc.n.sum()), "experts_without_tokens": int((acc.n == 0).sum()),
            "positions_dropped": acc.dropped, "rows_dropped": acc.rows_dropped,
            "dense_rows_dropped": acc.dense_rows_dropped}


def _window(rows, w: int, seq: int, device):
    """Window ``w`` cropped to ``seq`` tokens: the span holding the most scored (role > 0) positions.

    Assistant-only roles can start thousands of tokens into a 16k row (r16k row 2564: first scored position 7985, so a
    start crop scored nothing and the self-check saw no gradient hook fire). Ties go to the earliest span.
    """
    b = rows.batch(w, 1)
    xa = b["x"]
    ra = b["role"] if b["role"] is not None else torch.ones_like(xa, dtype=torch.int8)
    s = 0
    if xa.shape[1] > seq:
        c = torch.cumsum((ra[0] > 0).to(torch.int64), 0)
        c = torch.cat([torch.zeros(1, dtype=c.dtype), c])
        s = int((c[seq:] - c[:-seq]).argmax())  # scored count of [s, s + seq) for every start s
    x = xa[:, s : s + seq].long().to(device)
    return x, ra[:, s : s + seq]


def reference_logits(model, rows, args, device):
    """Logits of the last 64 positions of window 0 from the UNTOUCHED model (call before :func:`instrument`).

    Goes through :func:`head_logits` like the instrumented side, so the check isolates the instrumentation (experts
    forward + hooks): the HF head softcaps in bf16, which alone moves logits near the cap by up to ~0.12.
    """
    x, _ = _window(rows, 0, min(args.seq, 1024), device)
    with torch.no_grad():
        return head_logits(model, model.model(input_ids=x, use_cache=False).last_hidden_state[0, -64:])


def self_check(model, layers, rows, args, device, ref: torch.Tensor) -> None:
    """Instrumented == original logits on one window; every gradient hook fires; factors finite, diagonals >= 0."""
    x, role = _window(rows, 0, min(args.seq, 1024), device)
    probe = sorted({0, len(layers) // 2, len(layers) - 1})
    mk = lambda: {li: LayerAcc(layers[li].experts, layers[li].mlp, device, args.down_g, args.dense,  # noqa: E731
                               args.drop_grow)
                  for li in probe}
    attach(layers, mk())
    with torch.no_grad():  # accumulators on, no gradient hooks (no graph)
        got = head_logits(model, model.model(input_ids=x, use_cache=False).last_hidden_state[0, -64:])
    err = (got - ref).abs().max().item()
    if err > 0.05:
        raise SystemExit(f"KFAC_FAIL check: instrumented logits differ from the original by {err:.4f}")
    accs = mk()
    attach(layers, accs)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    n = fisher_backward(model, x, role, gen)
    if n == 0:
        raise SystemExit("KFAC_FAIL check: the check window has no scored positions")
    for li, acc in accs.items():
        if acc.pending:
            raise SystemExit(f"KFAC_FAIL check: layer {li}: {acc.pending} gradient hooks did not fire")
        for k, t in acc.f.items():
            d = t.diagonal(dim1=-2, dim2=-1) if t.dim() == 3 else t
            if not torch.isfinite(t).all() or (d < -1e-6).any():
                raise SystemExit(f"KFAC_FAIL check: layer {li} {k} non-finite or negative diagonal")
        if acc.n.sum() == 0 or acc.f["gu_G"].abs().sum() == 0 or acc.f["dn_G"].abs().sum() == 0:
            raise SystemExit(f"KFAC_FAIL check: layer {li}: no tokens or a zero G")
        if acc.dense is not None and acc.dense["gate_G"].abs().sum() == 0:
            raise SystemExit(f"KFAC_FAIL check: layer {li}: dense MLP G is zero")
    attach(layers, {})
    print(f"KFAC_CHECK OK logits max|diff| {err:.2e} vs the untouched model, {n} scored positions, layers {probe}: "
          f"every gradient hook fired, factors finite", flush=True)


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True, help="omk_qad cache dir (cache.json + train_XXXX.pt shards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--windows", type=int, default=256, help="cache rows used (the first N of the --seed order)")
    ap.add_argument("--seq", type=int, default=4096, help="tokens per window (a cached row cropped to its start)")
    ap.add_argument("--seed", type=int, default=1, help="row order and label sampling")
    ap.add_argument("--layers", default="all", help="'all' or a comma list / a-b ranges of decoder layers")
    ap.add_argument("--layers-per-pass", type=int, default=5)
    ap.add_argument("--down-g", choices=["full", "diag"], default="diag")
    ap.add_argument("--dense", action="store_true", help="also the dense MLP's factors")
    ap.add_argument("--drop-resid", type=float, default=0.0,
                    help="exclude attention sinks: positions whose residual norm entering the layer exceeds this "
                         "multiple of the window median (0 = keep every position)")
    ap.add_argument("--drop-grow", type=float, default=0.0,
                    help="ABSOLUTE cut: leave the (position, expert) rows whose gate_up-output gradient norm exceeds "
                         "this out of G (the 'void' positions: prediction impossible, measured KL ~0); 0 off")
    ap.add_argument("--rademacher", action="store_true",
                    help="random-sign the per-position losses before the backward (per-loss Fisher; removes the "
                         "attention-sink inflation of G without excluding positions)")
    ap.add_argument("--anchor-as-teacher", action="store_true", default=True)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)

    from .ptq import load_text_causal_lm
    from .train import CacheRows

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.cache) / "cache.json").read_text())
    want = {k: getattr(a, k) for k in ("model", "cache", "windows", "seq", "seed", "down_g", "dense")}
    if a.drop_resid > 0:  # old factor dirs (no exclusion) keep matching their meta.json
        want["drop_resid"] = a.drop_resid
    if a.rademacher:
        want["rademacher"] = True
    if a.drop_grow > 0:
        want["drop_grow"] = a.drop_grow
    mpath = out / "meta.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old.get("args") != want:
            raise SystemExit(f"KFAC_REFUSED {out} holds factors of other settings: {old.get('args')} != {want}")
    rows = CacheRows(a.cache, meta["n_seq"], meta["shard"], "shuffle", a.seed, anchor_teacher=a.anchor_as_teacher)
    if a.windows > len(rows.rows):
        raise SystemExit(f"KFAC_FAIL --windows {a.windows} > {len(rows.rows)} cached rows")

    device = torch.device(a.device)
    model, _ = load_text_causal_lm(a.model, str(device), torch.bfloat16)
    # transformers dispatches Gemma4TextExperts.forward by config._experts_implementation (grouped_mm on CUDA by
    # default); the instrumented path is the eager loop, so the reference must run the eager loop too (the PTQ path
    # does the same), else the self-check compares two different bf16 kernels (max|diff| 17.4 on bs2, 2026-11-15).
    model.config._experts_implementation = "eager"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ref = reference_logits(model, rows, a, device)
    model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    layers = instrument(model, a.drop_resid)
    n_layers = len(layers)
    sel = list(range(n_layers)) if a.layers == "all" else sorted(
        {j for part in a.layers.split(",") for j in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
    if any(not getattr(layers[j], "enable_moe_block", False) for j in sel):
        raise SystemExit("KFAC_FAIL a selected layer has no MoE block")
    self_check(model, layers, rows, a, device, ref)
    if a.check_only:
        return
    torch.cuda.reset_peak_memory_stats(device)  # the pass's "mem" must not report the self-check's peak
    if not mpath.exists():
        mpath.write_text(json.dumps({"args": want, "cache_identity": meta.get("identity"),
                                     "rows": rows.rows[: a.windows],
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    todo = [j for j in sel if not (out / f"layer_{j:02d}.pt").exists()]
    groups = [todo[i : i + a.layers_per_pass] for i in range(0, len(todo), a.layers_per_pass)]
    print(f"KFAC_START {len(todo)} layers to collect ({len(sel) - len(todo)} already written) in {len(groups)} passes "
          f"x {a.windows} windows x {a.seq} tokens", flush=True)
    for gi, group in enumerate(groups):
        accs = {j: LayerAcc(layers[j].experts, layers[j].mlp, device, a.down_g, a.dense, a.drop_grow) for j in group}
        attach(layers, accs)
        gen = torch.Generator(device=device).manual_seed(a.seed)  # same labels in every pass
        t0, scored = time.time(), 0
        for w in range(a.windows):
            x, role = _window(rows, w, a.seq, device)
            k = fisher_backward(model, x, role, gen, rademacher=a.rademacher)
            if k == 0:  # nothing to score: no backward ran, the hooks registered in the forward never fire
                print(f"KFAC note: window {w} (row {rows.rows[w]}) has no scored positions, skipped", flush=True)
                for acc in accs.values():
                    acc.pending = 0
                continue
            scored += k
            for j, acc in accs.items():
                if acc.pending:
                    raise SystemExit(f"KFAC_FAIL window {w} layer {j}: {acc.pending} gradient hooks did not fire")
            if w % 16 == 15 or w == a.windows - 1:
                dt = time.time() - t0
                print(f"KFAC pass {gi + 1}/{len(groups)} layers {group} window {w + 1}/{a.windows} "
                      f"{dt / (w + 1):.2f}s/window scored {scored} "
                      f"mem {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB", flush=True)
        attach(layers, {})
        for j, acc in accs.items():
            info = _save_layer(acc, out / f"layer_{j:02d}.pt",
                               {"layer": j, "windows": a.windows, "seq": a.seq, "scored": scored, "seed": a.seed,
                                "drop_resid": a.drop_resid, "rademacher": a.rademacher,
                                "drop_grow": a.drop_grow})
            print(f"KFAC_LAYER {j} {info}", flush=True)
        del accs
        torch.cuda.empty_cache()
    print(f"KFAC_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
