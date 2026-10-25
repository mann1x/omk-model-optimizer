# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests for omk_qad.losses (test names are their docs)."""

import math

import pytest
import torch
import torch.nn.functional as F

from omk_qad.losses import (
    JaccardDistillationLoss,
    jaccard_full,
    jaccard_topk_rest,
    kl_topk_rest,
    masked_mean,
    rkl_topk_rest,
)


def _logits_of(probs):
    return torch.tensor(probs, dtype=torch.float64).log()


def test_hand_computed_case():
    # p_s = [0.5, 0.3, 0.2], p_t = [0.2, 0.3, 0.5]: sum min = 0.2+0.3+0.2 = 0.7, sum max = 0.5+0.3+0.5 = 1.3
    loss = jaccard_full(_logits_of([0.5, 0.3, 0.2]), _logits_of([0.2, 0.3, 0.5]))
    assert loss.item() == pytest.approx(1 - 0.7 / 1.3, abs=1e-6)


def test_identity_and_disjoint():
    p = _logits_of([0.1, 0.2, 0.7])
    assert jaccard_full(p, p).item() == pytest.approx(0.0, abs=1e-6)
    s = torch.tensor([0.0, -1e4, -1e4])
    t = torch.tensor([-1e4, 0.0, -1e4])
    assert jaccard_full(s, t).item() == pytest.approx(1.0, abs=1e-6)


def test_equals_two_tv_over_one_plus_tv():
    torch.manual_seed(0)
    s, t = torch.randn(64, 50), torch.randn(64, 50)
    tv = 0.5 * (F.softmax(s, -1) - F.softmax(t, -1)).abs().sum(-1)
    torch.testing.assert_close(jaccard_full(s, t), 2 * tv / (1 + tv), atol=1e-5, rtol=0)
    # and the explicit two-sum definition
    ps, pt = F.softmax(s, -1), F.softmax(t, -1)
    explicit = 1 - torch.minimum(ps, pt).sum(-1) / torch.maximum(ps, pt).sum(-1)
    torch.testing.assert_close(jaccard_full(s, t), explicit, atol=1e-5, rtol=0)


@pytest.mark.parametrize("k", [1, 4, 16, 50])
def test_topk_rest_is_a_lower_bound_and_exact_at_full_k(k):
    torch.manual_seed(1)
    s, t = torch.randn(32, 50) * 3, torch.randn(32, 50) * 3
    logp_t = F.log_softmax(t, -1)
    vals, ids = logp_t.topk(k, dim=-1)
    approx = jaccard_topk_rest(s, ids, vals)
    full = jaccard_full(s, t)
    assert torch.all(approx <= full + 1e-6)
    if k == 50:
        torch.testing.assert_close(approx, full, atol=1e-5, rtol=0)


def test_kl_topk_rest_matches_full_kl_at_full_k():
    torch.manual_seed(2)
    s, t = torch.randn(16, 40), torch.randn(16, 40)
    logp_t = F.log_softmax(t, -1)
    vals, ids = logp_t.topk(40, dim=-1)
    full = F.kl_div(F.log_softmax(s, -1), logp_t, log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(kl_topk_rest(s, ids, vals), full, atol=1e-5, rtol=0)


def test_gradient_flows_to_student_only():
    torch.manual_seed(3)
    s = torch.randn(8, 30, requires_grad=True)
    t = torch.randn(8, 30, requires_grad=True)
    jaccard_full(s, t).mean().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
    assert t.grad is None
    s2 = torch.randn(8, 30, requires_grad=True)
    vals, ids = F.log_softmax(torch.randn(8, 30), -1).topk(8, -1)
    jaccard_topk_rest(s2, ids, vals).mean().backward()
    assert s2.grad is not None and torch.isfinite(s2.grad).all() and s2.grad.abs().sum() > 0


def test_descent_reduces_loss():
    torch.manual_seed(4)
    t = torch.randn(4, 20) * 2
    s = torch.zeros(4, 20, requires_grad=True)
    opt = torch.optim.SGD([s], lr=5.0)
    first = jaccard_full(s, t).mean().item()
    for _ in range(200):
        opt.zero_grad()
        jaccard_full(s, t).mean().backward()
        opt.step()
    assert jaccard_full(s, t).mean().item() < 0.25 * first


def test_masked_mean_and_module():
    per_token = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    mask = torch.tensor([[1, 0], [1, 1]])
    assert masked_mean(per_token, mask).item() == pytest.approx((1 + 5 + 7) / 3)
    assert masked_mean(per_token, None).item() == pytest.approx(4.0)
    s, t = torch.randn(2, 3, 11), torch.randn(2, 3, 11)
    assert JaccardDistillationLoss(reduction="none")(s, t).shape == (2, 3)
    assert math.isfinite(JaccardDistillationLoss()(s, t).item())
    with pytest.raises(ValueError):
        JaccardDistillationLoss(reduction="sum")


def _kl_eps_form(s, ids, vals, eps=1e-12):  # the pre-bug-825 probability-space form, as a reference
    p_s = F.softmax(s.float(), -1)
    p_s_k, p_t_k = p_s.gather(-1, ids), vals.float().exp()
    p_s_rest, p_t_rest = (1 - p_s_k.sum(-1)).clamp_min(0), (1 - p_t_k.sum(-1)).clamp_min(0)
    return (p_t_k * ((p_t_k + eps).log() - (p_s_k + eps).log())).sum(-1) + \
        p_t_rest * ((p_t_rest + eps).log() - (p_s_rest + eps).log())


def test_kl_topk_rest_agrees_with_the_eps_form_off_the_floor():
    torch.manual_seed(5)
    s, t = torch.randn(64, 200) * 2, torch.randn(64, 200) * 2
    vals, ids = F.log_softmax(t, -1).topk(16, dim=-1)
    torch.testing.assert_close(kl_topk_rest(s, ids, vals), _kl_eps_form(s, ids, vals), atol=1e-5, rtol=1e-5)


def test_kl_topk_rest_keeps_its_gradient_when_the_student_misses_every_topk_id():
    # bug-825: a student with ~e-80 mass on the teacher's K ids sat on the eps form's -log(1e-12) plateau
    torch.manual_seed(6)
    t = torch.randn(8, 100) * 2
    vals, ids = F.log_softmax(t, -1).topk(16, dim=-1)
    s0 = torch.zeros(8, 100).scatter(-1, ids, -80.0)
    old = s0.clone().requires_grad_()
    _kl_eps_form(old, ids, vals).sum().backward()
    new = s0.clone().requires_grad_()
    loss = kl_topk_rest(new, ids, vals)
    loss.sum().backward()
    assert _kl_eps_form(s0, ids, vals).max() < -math.log(1e-12)  # the old loss was capped ...
    assert loss.min() > -2 * math.log(1e-12)  # ... the true KL is not
    assert old.grad.gather(-1, ids).abs().sum(-1).max() < 1e-6  # ... and the old gradient was gone
    g = new.grad.gather(-1, ids).abs().sum(-1)
    torch.testing.assert_close(g, vals.exp().sum(-1), atol=1e-5, rtol=1e-5)  # d/dz_i = p_s_i - p_t_i ~ -p_t_i
    ends = {}
    for name, fn in (("log", kl_topk_rest), ("eps", _kl_eps_form)):  # same optimizer, both forms
        s = s0.clone().requires_grad_()
        opt = torch.optim.Adam([s], lr=1.0)
        for _ in range(200):
            opt.zero_grad()
            fn(s, ids, vals).mean().backward()
            opt.step()
        ends[name] = kl_topk_rest(s.detach(), ids, vals).mean().item()
    assert ends["log"] < 1.0 and ends["eps"] > 50.0, ends  # only the log-space form leaves the dead zone


def test_kl_topk_rest_full_k_backward_is_finite():
    torch.manual_seed(7)
    s = torch.randn(4, 20, requires_grad=True)
    vals, ids = F.log_softmax(torch.randn(4, 20), -1).topk(20, dim=-1)
    kl_topk_rest(s, ids, vals).sum().backward()
    assert torch.isfinite(s.grad).all()


def test_rkl_topk_rest_matches_full_reverse_kl_at_full_k():
    torch.manual_seed(8)
    s, t = torch.randn(16, 40), torch.randn(16, 40)
    lp_s, lp_t = F.log_softmax(s, -1), F.log_softmax(t, -1)
    vals, ids = lp_t.topk(40, dim=-1)
    full = (lp_s.exp() * (lp_s - lp_t)).sum(-1)  # KL(s || t)
    torch.testing.assert_close(rkl_topk_rest(s, ids, vals), full, atol=1e-5, rtol=0)


@pytest.mark.parametrize("k", [1, 8, 39])
def test_rkl_topk_rest_is_the_reverse_kl_of_the_collapsed_bins(k):
    torch.manual_seed(9)
    s, t = torch.randn(16, 40) * 2, torch.randn(16, 40) * 2
    lp_s, lp_t = F.log_softmax(s, -1), F.log_softmax(t, -1)
    vals, ids = lp_t.topk(k, dim=-1)
    q_k, p_k = lp_s.gather(-1, ids).exp(), vals.exp()
    q = torch.cat([q_k, 1 - q_k.sum(-1, keepdim=True)], -1)
    p = torch.cat([p_k, 1 - p_k.sum(-1, keepdim=True)], -1)
    want = (q * (q.log() - p.log())).sum(-1)
    got = rkl_topk_rest(s, ids, vals)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
    assert torch.all(got >= -1e-6)
    # a different direction from the forward KL on the same bins
    assert not torch.allclose(got, kl_topk_rest(s, ids, vals), atol=1e-3)


def test_rkl_topk_rest_is_zero_on_the_teacher_itself():
    torch.manual_seed(10)
    t = torch.randn(8, 50) * 3
    vals, ids = F.log_softmax(t, -1).topk(16, dim=-1)
    assert rkl_topk_rest(t, ids, vals).abs().max().item() < 1e-5


def test_rkl_topk_rest_stays_finite_when_the_cached_k_mass_reaches_one():
    """fp16-cached teacher log-probs can sum to >= 1 over the K: the floored, renormalised rest bin keeps it finite."""
    torch.manual_seed(11)
    t = torch.full((4, 30), -30.0)
    t[:, :4] = torch.randn(4, 4)
    vals, ids = F.log_softmax(t, -1).topk(4, dim=-1)
    vals = (vals + 1e-3).half()  # K mass now slightly over 1: rest = 1 - sum < 0 before the floor
    s = torch.randn(4, 30, requires_grad=True)  # a student with real mass outside the K
    loss = rkl_topk_rest(s, ids, vals)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and (loss >= -1e-5).all() and torch.isfinite(s.grad).all()


def test_rkl_topk_rest_keeps_its_gradient_when_the_student_misses_every_topk_id():
    s = torch.zeros(2, 64)
    s[:, 32:] = 30.0  # all student mass outside the teacher's K
    s.requires_grad_()
    vals, ids = F.log_softmax(torch.randn(2, 64), -1)[:, :32].topk(8, dim=-1)
    rkl_topk_rest(s, ids, vals).sum().backward()
    g = s.grad[:, :32].gather(-1, ids)
    assert torch.isfinite(s.grad).all() and (g < 0).all()  # descent raises the student on the teacher's ids
