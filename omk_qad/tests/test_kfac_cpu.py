# SPDX-License-Identifier: Apache-2.0
"""omk_qad.kfac on a tiny random Gemma-4 MoE (CPU): the instrumented model is the original, and the factors are right.

* instrumented logits == the untouched model's logits;
* every expert's token count == the router's top-k assignment; gu_A[e] == Σ x xᵀ recomputed by independent hooks;
* Σ_t g_t x_tᵀ from the hooked g and routed x == autograd's dL/d gate_up_proj[e], and gu_G[e] == Σ g gᵀ;
* every registered gradient hook fires; fp16 + scale storage round-trips.
"""
# ruff: noqa: D101, D103, N802, N806

import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Gemma4ForCausalLM  # noqa: E402
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig  # noqa: E402

from omk_qad import kfac  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(vocab_size=128, hidden_size=64, intermediate_size=48, num_hidden_layers=3,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
                           num_global_key_value_heads=1, layer_types=["sliding_attention", "sliding_attention",
                                                                      "full_attention"],
                           sliding_window=16, enable_moe_block=True, num_experts=4, top_k_experts=2,
                           moe_intermediate_size=16, hidden_size_per_layer_input=0, final_logit_softcapping=30.0,
                           max_position_embeddings=128)
    m = Gemma4ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.08)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Args(types.SimpleNamespace):
    pass


def _prep(m):
    m.model.embed_tokens.register_forward_hook(lambda mod, i, o: o.requires_grad_(True))
    return kfac.instrument(m)


def test_instrumented_logits_equal_original():
    m = tiny()
    x = torch.randint(0, 128, (1, 24))
    with torch.no_grad():
        ref = m(input_ids=x, use_cache=False).logits.float()
    layers = _prep(m)
    accs = {i: kfac.LayerAcc(layers[i].experts, layers[i].mlp, "cpu", "full", True) for i in range(3)}
    kfac.attach(layers, accs)
    with torch.no_grad():
        got = kfac.head_logits(m, m.model(input_ids=x, use_cache=False).last_hidden_state)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_counts_A_and_hooks():
    m = tiny(1)
    x = torch.randint(0, 128, (1, 24))
    role = torch.ones(1, 24, dtype=torch.int8)
    seen = {}

    def spy(layer_i):  # independent record of the experts' inputs and routing
        def hook(mod, args, out):
            seen[layer_i] = (args[0].detach().clone(), args[1].clone())
        return hook

    layers = _prep(m)
    for i, layer in enumerate(layers):
        layer.experts.register_forward_hook(spy(i))
    accs = {i: kfac.LayerAcc(layers[i].experts, layers[i].mlp, "cpu", "full", True) for i in range(3)}
    kfac.attach(layers, accs)
    n = kfac.fisher_backward(m, x, role, torch.Generator().manual_seed(0))
    assert n == 23
    for i, acc in accs.items():
        assert acc.pending == 0
        hs, topk = seen[i]
        for e in range(4):
            tok = (topk == e).any(dim=-1)
            assert int(acc.n[e]) == int(tok.sum())
            xe = hs[tok].float()
            assert torch.allclose(acc.f["gu_A"][e], xe.T @ xe, atol=1e-5)
            G = acc.f["gu_G"][e]
            assert torch.allclose(G, G.T, atol=1e-6)
            if tok.any():
                assert G.diagonal().sum() > 0
        assert acc.dense["mlp_in_A"].diagonal().sum() > 0 and acc.dense["gate_G"].diagonal().sum() > 0


def test_G_matches_autograd_gradient():
    """For each expert, Σ_t g_t x_tᵀ (from the hooked g and the routed x) == autograd's dL/d gate_up_proj[e]."""
    m = tiny(2)
    x = torch.randint(0, 128, (1, 20))
    role = torch.ones(1, 20, dtype=torch.int8)
    layers = _prep(m)
    ex = layers[1].experts
    ex.gate_up_proj.requires_grad_(True)
    rec = {}
    orig = kfac.LayerAcc.g_hook

    def g_hook(self, key, e, keep=None):  # also keep the raw g of gate_up per expert
        inner = orig(self, key, e, keep)

        def hook(g):
            if key == "gu_G":
                rec.setdefault(e, []).append(g.detach().clone())
            inner(g)
        return hook

    xs = {}

    def add_a(self, key, e, xin):
        if key == "gu_A":
            xs.setdefault(e, []).append(xin.detach().clone())
        kfac.LayerAcc._add_a_orig(self, key, e, xin)

    kfac.LayerAcc._add_a_orig = kfac.LayerAcc.add_a
    try:
        kfac.LayerAcc.g_hook = g_hook
        kfac.LayerAcc.add_a = add_a
        acc = kfac.LayerAcc(ex, layers[1].mlp, "cpu", "full", False)
        kfac.attach(layers, {1: acc})
        kfac.fisher_backward(m, x, role, torch.Generator().manual_seed(3))
    finally:
        kfac.LayerAcc.g_hook = orig
        kfac.LayerAcc.add_a = kfac.LayerAcc._add_a_orig
    grad = ex.gate_up_proj.grad
    for e in rec:
        g, xe = torch.cat(rec[e]).float(), torch.cat(xs[e]).float()
        assert torch.allclose(g.T @ xe, grad[e].float(), atol=1e-5), (g.T @ xe - grad[e]).abs().max()
        assert torch.allclose(acc.f["gu_G"][e], g.T @ g, atol=1e-6)


def test_keep_mask_excludes_positions():
    """``LayerAcc.keep`` (the --drop-resid sink mask) removes the masked positions from n, A and G of every expert."""
    m = tiny(2)
    x = torch.randint(0, 128, (1, 20))
    role = torch.ones(1, 20, dtype=torch.int8)
    layers = _prep(m)
    ex = layers[1].experts

    def run(keep):
        acc = kfac.LayerAcc(ex, layers[1].mlp, "cpu", "full", True)
        acc.keep = keep
        kfac.attach(layers, {1: acc})
        kfac.fisher_backward(m, x, role, torch.Generator().manual_seed(3))
        kfac.attach(layers, {})
        return acc

    full = run(None)
    keep = torch.ones(20, dtype=torch.bool)
    keep[0] = False  # drop the first position (the sink) everywhere
    part = run(keep)
    dropped_routes = int(full.n.sum() - part.n.sum())
    assert 1 <= dropped_routes <= ex.num_experts, dropped_routes  # one token, routed to top-k experts
    assert part.dense["n"] == full.dense["n"] - 1
    # excluding rows can only lower the Gram traces, and must change them for the experts that saw the token
    for k in ("gu_A", "gu_G", "dn_A"):
        tf = full.f[k].diagonal(dim1=-2, dim2=-1).sum(-1)
        tp = part.f[k].diagonal(dim1=-2, dim2=-1).sum(-1)
        assert (tp <= tf + 1e-6).all()
        assert (tp < tf - 1e-9).sum() == dropped_routes, (k, (tp < tf - 1e-9).sum(), dropped_routes)
    assert part.dense["mlp_in_A"].trace() < full.dense["mlp_in_A"].trace()


def test_rademacher_signs_only_flip_per_position_gradients():
    """``rademacher=True`` multiplies each scored position's loss by an independent ±1.

    At the LAST layer's output position t feeds only its own loss (norm + head, no attention after), so its gradient
    must be exactly ±(the plain one), with both signs occurring; the sampled labels are the same for the same seed.
    """
    m = tiny(4)
    x = torch.randint(0, 128, (1, 20))
    role = torch.ones(1, 20, dtype=torch.int8)
    layers = _prep(m)

    def run(flag):
        rec = []

        def fwd(mod, i, o):
            (o if torch.is_tensor(o) else o[0]).register_hook(lambda g: rec.append(g.detach().clone()))

        h = layers[2].register_forward_hook(fwd)
        n = kfac.fisher_backward(m, x, role, torch.Generator().manual_seed(5), rademacher=flag)
        h.remove()
        assert n == 19 and len(rec) == 1
        return rec[0][0]  # [S, H] gradient of the loss w.r.t. the last layer's output

    plain, rad = run(False), run(True)
    assert torch.allclose(rad.abs(), plain.abs(), atol=1e-6)
    sign = (rad[:-1] * plain[:-1]).sum(-1).sign()  # per scored position: +1 kept, -1 flipped
    assert (sign > 0).any() and (sign < 0).any()


def test_drop_grow_cuts_rows_from_g_only():
    """``grow``: rows above the gate_up-output gradient-norm cut leave gu_G AND dn_G (and the dense trio), never A or n.

    An infinite cut reproduces the plain factors exactly, a tiny cut empties G.
    """
    m = tiny(2)
    x = torch.randint(0, 128, (1, 20))
    role = torch.ones(1, 20, dtype=torch.int8)
    layers = _prep(m)
    ex = layers[1].experts

    def run(grow):
        acc = kfac.LayerAcc(ex, layers[1].mlp, "cpu", "full", True, grow)
        kfac.attach(layers, {1: acc})
        kfac.fisher_backward(m, x, role, torch.Generator().manual_seed(3))
        kfac.attach(layers, {})
        assert acc.pending == 0 and not acc._buf
        return acc

    # tiny-model gate_up gradient row norms: median 1.6e-3, max 1.6e-2 → 0.003 cuts some rows, 1e-12 all nonzero ones
    plain, same, cut, none = run(0.0), run(1e9), run(0.003), run(1e-12)
    for k in ("gu_G", "dn_G"):
        assert torch.allclose(same.f[k], plain.f[k], atol=1e-7)
        assert (cut.f[k].abs().sum(dim=(-2, -1)) <= plain.f[k].abs().sum(dim=(-2, -1)) + 1e-7).all()
        assert none.f[k].abs().sum() == 0
    for k in ("gu_A", "dn_A"):
        assert torch.allclose(cut.f[k], plain.f[k]) and torch.allclose(none.f[k], plain.f[k])
    assert torch.equal(cut.n, plain.n) and torch.equal(none.n, plain.n)
    # rows with an exactly-zero gradient (the position after the last scored one) are never above any cut
    assert same.rows_dropped == 0 and 0 < cut.rows_dropped < none.rows_dropped <= int(plain.n.sum())
    assert torch.allclose(same.dense["gate_G"], plain.dense["gate_G"], atol=1e-7)
    assert none.dense["gate_G"].abs().sum() == 0 and none.dense["down_G"].abs().sum() == 0
    assert torch.allclose(none.dense["mlp_in_A"], plain.dense["mlp_in_A"]) and none.dense["n"] == plain.dense["n"]
    assert 0 < cut.dense_rows_dropped <= none.dense_rows_dropped <= 20


def test_save_roundtrip(tmp_path):
    m = tiny(3)
    layers = _prep(m)
    acc = kfac.LayerAcc(layers[0].experts, layers[0].mlp, "cpu", "diag", True)
    kfac.attach(layers, {0: acc})
    kfac.fisher_backward(m, torch.randint(0, 128, (1, 24)), torch.ones(1, 24, dtype=torch.int8),
                         torch.Generator().manual_seed(1))
    kfac.attach(layers, {})
    p = tmp_path / "layer_00.pt"
    kfac._save_layer(acc, p, {"layer": 0})
    d = torch.load(p)
    n = acc.n.clamp_min(1).float()
    for k in kfac.FACTOR_KEYS:
        want = acc.f[k] / n.view(-1, *([1] * (acc.f[k].dim() - 1)))
        got = d[k].float() * d[k + "_scale"].view(-1, *([1] * (acc.f[k].dim() - 1)))
        assert torch.allclose(got, want, rtol=2e-3, atol=1e-7 * want.abs().max())
    assert d["dn_G"].dim() == 2  # diag storage
