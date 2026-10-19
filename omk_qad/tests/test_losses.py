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
