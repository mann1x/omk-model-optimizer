# SPDX-License-Identifier: Apache-2.0
"""Probe the K-FAC surrogate on live windows (bm-895 step 2 diagnosis).

For ONE decoder layer and the Δ of one GGUF arm (its routed gate_up / down, dequantised, minus the BF16 base) the
probe measures, window by window:

  kl_measured   mean KL(base ‖ perturbed) per scored token with Δ written into that layer's experts (the truth);
  q_cross       ½ Σ_s (Σ_{e,m} g_{s,e,m}ᵀ Δ_{e,m} x_{s,e,m})² / n_scored — the exact per-position quadratic of the
                summed-loss gradient the collector back-propagates (what K-FAC-expand factorises), all cross terms;
  q_block       the same without the cross terms between experts and matrices (K-FAC's block diagonal);
  q_kfac_win    the K-FAC factorisation of THIS window's own x and g (isolates the factorisation error);
  q_kfac_saved  the prediction from the saved dataset factors, weighted with this window's expert loads
                (what omk_qad.kfac_validate computes).

Reading the ladder: kl ≈ q_cross ≈ q_block ≈ q_kfac_* means the surrogate holds; q_cross ≫ kl means the summed-gradient
Fisher is not the KL curvature at this Δ (expand approximation or the quadratic model); q_kfac ≫ q_block means the
Kronecker factorisation is at fault.

Usage: python -m omk_qad.kfac_probe --model DIR --cache CACHE --factors DIR --base BF16.gguf --arm A.gguf --layer 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .kfac import _window, head_logits
from .kfac_validate import Gguf


def capture_forward(self, hidden_states, top_k_index, top_k_weights):
    """Transformers' eager experts loop; per expert it records tokens, inputs and output gradients in ``self._cap``."""
    cap = getattr(self, "_cap", None)
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for idx in hit:
        e = int(idx[0])
        if e == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(mask[e])
        cs = hidden_states[token_idx]
        h = F.linear(cs, self.gate_up_proj[e])
        i = len(cap["tok"]) if cap is not None else -1
        if cap is not None:
            cap["tok"].append((e, token_idx.detach()))
            cap["x_gu"].append(cs.detach().float())
            if h.requires_grad:
                h.register_hook(lambda g, i=i: cap["g_gu"].__setitem__(i, g.detach().float()))
        gate, up = h.chunk(2, dim=-1)
        act = self.act_fn(gate) * up
        y = F.linear(act, self.down_proj[e])
        if cap is not None:
            cap["x_dn"].append(act.detach().float())
            if y.requires_grad:
                y.register_hook(lambda g, i=i: cap["g_dn"].__setitem__(i, g.detach().float()))
        y = y * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, y.to(final.dtype))
    return final


def scored_logp(model, x, role):
    """log-softmax of the head at the scored positions (fp32) and their count."""
    with torch.no_grad():
        hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
        mask = (role[:, 1:] > 0).to(hidden.device)
        hs = hidden[:, :-1][mask]
        out = [torch.log_softmax(head_logits(model, hs[s : s + 512]), dim=-1) for s in range(0, hs.shape[0], 512)]
    return torch.cat(out), hs.shape[0]


def fisher_backward_capture(model, x, role, gen, chunk=512, rademacher=False) -> int:
    """:func:`omk_qad.kfac.fisher_backward` (same sampled labels for the same seed): one summed backward."""
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
    mask = (role[:, 1:] > 0).to(hidden.device)
    hs = hidden[:, :-1][mask]
    n = hs.shape[0]
    grad = torch.zeros_like(hs)
    hd = hs.detach()
    for s in range(0, n, chunk):
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


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--factors", required=True)
    ap.add_argument("--base", required=True, help="BF16 GGUF (W0)")
    ap.add_argument("--arm", required=True, help="quantised GGUF whose Δ is probed")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply Δ (0.1: deep in the quadratic regime)")
    ap.add_argument("--clip", type=float, default=10.0,
                    help="robust variant: clip a position's gradient norm at this multiple of the window median; 0 off")
    ap.add_argument("--clip-q", type=float, default=0.0,
                    help="robust variant: cap a position's gradient norm at this quantile of the window's norms; 0 off")
    ap.add_argument("--drop-first", type=int, default=0,
                    help="robust variant: zero the gradient of the first N positions of the window")
    ap.add_argument("--drop-k", type=float, default=0.0,
                    help="robust variant: zero positions whose gradient norm exceeds k times the --drop-q quantile")
    ap.add_argument("--drop-q", type=float, default=0.9)
    ap.add_argument("--drop-grow", type=float, default=0.0,
                    help="ABSOLUTE per-row cut: zero the (position, expert) rows whose gate_up-output gradient norm "
                         "exceeds this (the collector's hook can apply it); 0 off")
    ap.add_argument("--drop-tok", default="",
                    help="structural: comma list of token ids whose positions are zeroed (delimiter / turn sinks)")
    ap.add_argument("--rademacher", action="store_true",
                    help="random-sign the per-position losses before the backward (per-loss Fisher)")
    ap.add_argument("--drop-resid", type=float, default=0.0,
                    help="structural sink exclusion: zero positions whose residual-stream norm entering the layer "
                         "exceeds k times the window median (massive activations); 0 off")
    a = ap.parse_args(argv)

    import json

    from .ptq import load_text_causal_lm
    from .train import CacheRows

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device)
    meta = json.loads((Path(a.cache) / "cache.json").read_text())
    rows = CacheRows(a.cache, meta["n_seq"], meta["shard"], "shuffle", a.seed, anchor_teacher=True)

    t0 = time.time()
    base, arm = Gguf(a.base), Gguf(a.arm)
    w0 = {s: torch.from_numpy(base.dequant(a.layer, s)) for s in ("ffn_gate_up_exps", "ffn_down_exps")}
    q = {s: torch.from_numpy(arm.dequant(a.layer, s)) for s in w0}
    d_gu = ((q["ffn_gate_up_exps"] - w0["ffn_gate_up_exps"]) * a.scale).to(device)  # [E, 2I, H] fp32
    d_dn = ((q["ffn_down_exps"] - w0["ffn_down_exps"]) * a.scale).to(device)  # [E, H, I]
    print(f"KFACP delta layer {a.layer}: rms gate_up {d_gu.pow(2).mean().sqrt():.5f} "
          f"down {d_dn.pow(2).mean().sqrt():.5f} ({time.time() - t0:.0f}s)", flush=True)

    f = torch.load(Path(a.factors) / f"layer_{a.layer:02d}.pt", map_location="cpu", weights_only=False)
    sa = {k: (f[k].float() * f[k + "_scale"].view(-1, *([1] * (f[k].dim() - 1)))).to(device)
          for k in ("gu_A", "gu_G", "dn_A", "dn_G")}
    # per-expert quadratics with the saved dataset factors (tr(Δ A Δᵀ G); dn_G diagonal)
    tr_gu = torch.zeros(d_gu.shape[0], dtype=torch.float64, device=device)
    tr_dn = torch.zeros_like(tr_gu)
    for c0 in range(0, d_gu.shape[0], 16):
        c1 = min(c0 + 16, d_gu.shape[0])
        dg = d_gu[c0:c1]
        tr_gu[c0:c1] = (torch.bmm(dg, sa["gu_A"][c0:c1]) * torch.bmm(sa["gu_G"][c0:c1], dg)).sum((1, 2)).double()
        dd = d_dn[c0:c1]
        tr_dn[c0:c1] = (torch.bmm(dd, sa["dn_A"][c0:c1]) * dd * sa["dn_G"][c0:c1][:, :, None]).sum((1, 2)).double()
    del sa
    n_e = f["n"].double().to(device)
    n_sc = f["meta"]["scored"]
    print(f"KFACP saved-factor quadratics ½ Σ_e n_e/N tr: gate_up {0.5 * (n_e * tr_gu).sum().item() / n_sc:.4f} "
          f"down {0.5 * (n_e * tr_dn).sum().item() / n_sc:.4f} (dataset mean, as kfac_validate)", flush=True)

    model, _ = load_text_causal_lm(a.model, str(device), torch.bfloat16)
    model.config._experts_implementation = "eager"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = model.model.layers[a.layer]
    experts = layer.experts
    import types

    experts.forward = types.MethodType(capture_forward, experts)
    experts._cap = None
    model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    orig_gu, orig_dn = experts.gate_up_proj.data.clone(), experts.down_proj.data.clone()
    pert_gu = (orig_gu.float() + d_gu).to(orig_gu.dtype)
    pert_dn = (orig_dn.float() + d_dn).to(orig_dn.dtype)
    print(f"KFACP model loaded ({time.time() - t0:.0f}s); perturbed weights: max|bf16 round| "
          f"{(pert_gu.float() - orig_gu.float() - d_gu).abs().max().item():.2e}", flush=True)

    gen = torch.Generator(device=device).manual_seed(a.seed)
    tot = {"n": 0, "kl": 0.0, "q_cross": 0.0, "q_block": 0.0, "q_block_clip": 0.0, "q_kfac_win": 0.0,
           "q_kfac_win_clip": 0.0, "q_kfac_saved": 0.0}
    print(f"{'w':>3s} {'n':>5s} {'kl_measured':>11s} {'q_cross':>9s} {'q_block':>9s} {'q_blk_clip':>9s} "
          f"{'q_kfac_win':>10s} {'q_kfw_clip':>10s} {'q_kfac_saved':>12s}")
    for w in range(a.windows):
        x, role = _window(rows, w, a.seq, device)
        # (1) measured KL
        experts._cap = None
        logp_b, n = scored_logp(model, x, role)
        if n == 0:
            print(f"{w:3d} window without scored positions, skipped")
            continue
        experts.gate_up_proj.data.copy_(pert_gu)
        experts.down_proj.data.copy_(pert_dn)
        logp_p, _ = scored_logp(model, x, role)
        experts.gate_up_proj.data.copy_(orig_gu)
        experts.down_proj.data.copy_(orig_dn)
        klv = (logp_b.exp() * (logp_b - logp_p)).sum(-1).double()  # per scored position
        kl = klv.sum().item() / n
        kl_pos = torch.zeros(x.shape[1], dtype=torch.float64, device=device)  # by input position
        kl_pos[(role[0, 1:] > 0).nonzero().squeeze(1)] = klv
        del logp_b, logp_p, klv
        # (2) the collector's gradients on this window (+ the residual-stream norm entering the layer, per position)
        cap = {"tok": [], "x_gu": [], "x_dn": [], "g_gu": {}, "g_dn": {}, "resid": None}
        hook = layer.register_forward_pre_hook(
            lambda m, args, kwargs, _c=cap: _c.__setitem__(
                "resid", (args[0] if args else kwargs["hidden_states"]).detach().float().norm(dim=-1)[0]),
            with_kwargs=True)
        experts._cap = cap
        n2 = fisher_backward_capture(model, x, role, gen, rademacher=a.rademacher)
        experts._cap = None
        hook.remove()
        assert n2 == n
        if len(cap["g_gu"]) != len(cap["tok"]):
            raise SystemExit(f"KFACP_FAIL window {w}: {len(cap['tok']) - len(cap['g_gu'])} gate_up hooks did not fire")
        # per-position gradient norm at the gate_up output (summed over the token's experts) → robust clip factor
        gn = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
        xn = torch.zeros_like(gn)
        for i, (e, tok) in enumerate(cap["tok"]):
            gn.index_add_(0, tok, cap["g_gu"][i].double().square().sum(-1))
            xn.index_add_(0, tok, cap["x_gu"][i].double().square().sum(-1))
        gnorm = gn.sqrt()
        gmed, xmed = gnorm[gn > 0].median().item(), xn.median().sqrt().item()
        clip = torch.ones_like(gnorm)
        live = gnorm[gn > 0].float()
        qs = torch.quantile(live, torch.tensor([0.5, 0.9, 0.99], device=live.device)).tolist()
        if a.clip_q > 0:  # cap at the q-quantile of the positions that carry gradient
            gcap = torch.quantile(live, a.clip_q).double().item()
            clip = torch.clamp(gcap / gnorm.clamp_min(1e-30), max=1.0)
        elif a.clip > 0:
            clip = torch.clamp(a.clip * gmed / gnorm.clamp_min(1e-30), max=1.0)
        if a.drop_k > 0:  # DROP (not cap) the sink-like positions: gradient norm > k times the q-quantile
            thr = a.drop_k * torch.quantile(live, a.drop_q).double().item()
            clip = torch.where(gnorm > thr, torch.zeros_like(clip), clip)
        if a.drop_first > 0:
            clip[: a.drop_first] = 0.0
        if a.drop_tok:  # delimiter / turn tokens by id (' ' 236743, '.' 236761, ',' 236764, <|turn> 105 ...)
            ids = torch.tensor([int(t) for t in a.drop_tok.split(",")], device=device)
            clip = torch.where(torch.isin(x[0], ids), torch.zeros_like(clip), clip)
        rn = cap["resid"].double()  # residual-stream norm per position at the layer input (massive activations)
        rmed = rn.median().item()
        if a.drop_resid > 0:  # structural sink exclusion: residual norm > k times the window median
            clip = torch.where(rn > a.drop_resid * rmed, torch.zeros_like(clip), clip)
        print(f"    |g| quantiles p50 {qs[0]:.2e} p90 {qs[1]:.2e} p99 {qs[2]:.2e} max {gnorm.max().item():.2e} "
              f"resid p50 {rmed:.1f} max {rn.max().item():.1f} (x{rn.max().item() / rmed:.0f}) "
              f"positions zeroed {int((clip == 0).sum())} capped {int(((clip < 1) & (clip > 0)).sum())}")
        s_pos = torch.zeros(x.shape[1], dtype=torch.float64, device=device)  # Σ over experts/matrices per position
        q_block = q_block_c = q_kw = q_kw_c = q_ks = 0.0
        rows_dropped, row_max = 0, 0.0
        for i, (e, tok) in enumerate(cap["tok"]):
            xg, gg, xd, gd = cap["x_gu"][i], cap["g_gu"][i], cap["x_dn"][i], cap["g_dn"][i]
            q_gu = ((xg @ d_gu[e].T) * gg).sum(-1).double()  # [n_e] g_tᵀ Δ x_t
            q_dn = ((xd @ d_dn[e].T) * gd).sum(-1).double()
            s_pos.index_add_(0, tok, q_gu + q_dn)
            q_block += (q_gu.square() + q_dn.square()).sum().item()
            c = clip[tok]
            if a.drop_grow > 0:  # absolute per-row cut on the gate_up-output gradient norm
                rn_row = gg.norm(dim=-1).double()
                row_max = max(row_max, rn_row.max().item())
                keep_row = rn_row <= a.drop_grow
                rows_dropped += int((~keep_row).sum())
                c = c * keep_row.to(c.dtype)
            q_block_c += ((q_gu * c).square() + (q_dn * c).square()).sum().item()
            ne = xg.shape[0]
            a_gu, a_dn = xg.T @ xg / ne, xd.T @ xd / ne
            for ggc, gdc, acc_key in ((gg, gd, "raw"), (gg * c[:, None].float(), gd * c[:, None].float(), "clip")):
                g_gu, g_dn = ggc.T @ ggc / ne, gdc.T @ gdc / ne
                v = ne * (((d_gu[e] @ a_gu) * (g_gu @ d_gu[e])).sum().item()
                          + ((d_dn[e] @ a_dn) * (g_dn @ d_dn[e])).sum().item())
                if acc_key == "raw":
                    q_kw += v
                else:
                    q_kw_c += v
            q_ks += ne * (tr_gu[e].item() + tr_dn[e].item())
        q_cross = s_pos.square().sum().item()
        sq = s_pos.square()
        if a.drop_grow > 0:
            print(f"    per-row |g| max {row_max:.2e}; rows dropped (> {a.drop_grow:g}) {rows_dropped}")
        # per position: the quadratic ½ s_p² (first-order loss change from p's experts, squared) vs the measured KL
        # at p. Quiet positions (prediction < 1e-4 nats) that still carry KL are the Δ-independent floor; positions
        # predicted above 20 nats are beyond anything a softcapped head can produce (the linearisation is void there).
        quiet = 0.5 * sq < 1e-4
        void = 0.5 * sq > 20.0
        kl_tot = max(kl_pos.sum().item(), 1e-30)
        qc = max(q_cross, 1e-30)
        print(f"    kl by position: quiet (pred<1e-4) {int(quiet.sum())} pos carry "
              f"{kl_pos[quiet].sum().item() / kl_tot:.2f} of the KL; predicted>20 nats {int(void.sum())} pos carry "
              f"{sq[void].sum().item() / qc:.2f} of q_cross and {kl_pos[void].sum().item() / kl_tot:.3f} of the KL "
              f"(max per-position KL {kl_pos.max().item():.3f})")
        for p in torch.topk(sq, 3).indices.tolist():  # the positions that carry the quadratic
            print(f"    pos {p:4d} tok {int(x[0, p]):6d} role {int(role[0, p])} share {sq[p].item() / qc:.3f} "
                  f"pred {0.5 * sq[p].item():.2e} kl_t {kl_pos[p].item():.2e} |g| {gnorm[p].item():.2e} "
                  f"(median {gmed:.2e}) |x| {xn[p].sqrt().item():.2e} (median {xmed:.2e}) "
                  f"resid x{rn[p].item() / rmed:.0f} clip {clip[p].item():.3f}")
        row = {"n": n, "kl": kl, "q_cross": 0.5 * q_cross / n, "q_block": 0.5 * q_block / n,
               "q_block_clip": 0.5 * q_block_c / n, "q_kfac_win": 0.5 * q_kw / n, "q_kfac_win_clip": 0.5 * q_kw_c / n,
               "q_kfac_saved": 0.5 * q_ks / n}
        for k, v in row.items():
            tot[k] += v if k != "n" else 0
        tot["n"] += n
        print(f"{w:3d} {n:5d} {row['kl']:11.5f} {row['q_cross']:9.4f} {row['q_block']:9.4f} {row['q_block_clip']:9.4f} "
              f"{row['q_kfac_win']:10.4f} {row['q_kfac_win_clip']:10.4f} {row['q_kfac_saved']:12.4f} "
              f"clipped {int((clip < 1).sum())}", flush=True)
        del cap, s_pos, kl_pos
        torch.cuda.empty_cache()
    m = {k: v / a.windows for k, v in tot.items() if k != "n"}
    kl0, kw0 = max(m["kl"], 1e-30), max(m["q_kfac_win"], 1e-30)  # scale 0 (control) gives exact zeros
    print(f"KFACP_MEAN scale {a.scale} layer {a.layer} clip {a.clip} clip_q {a.clip_q} drop_first {a.drop_first} "
          f"drop_k {a.drop_k} drop_q {a.drop_q} drop_resid {a.drop_resid} drop_tok [{a.drop_tok}] "
          f"rademacher {a.rademacher} drop_grow {a.drop_grow} "
          f"kl_measured {m['kl']:.5f} q_cross {m['q_cross']:.4f} q_block {m['q_block']:.4f} "
          f"q_block_clip {m['q_block_clip']:.4f} q_kfac_win {m['q_kfac_win']:.4f} "
          f"q_kfac_win_clip {m['q_kfac_win_clip']:.4f} q_kfac_saved {m['q_kfac_saved']:.4f} "
          f"ratios block/kl {m['q_block'] / kl0:.1f} block_clip/kl {m['q_block_clip'] / kl0:.2f} "
          f"kfac_win_clip/kl {m['q_kfac_win_clip'] / kl0:.2f} saved/win {m['q_kfac_saved'] / kw0:.1f}",
          flush=True)
    print(f"KFACP_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
