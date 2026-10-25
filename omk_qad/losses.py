# SPDX-License-Identifier: Apache-2.0
"""Distillation losses for 2-bit QAD (Nemotron 3.5 Bolt report, section 5.2).

The Bolt recipe minimises a weighted Jaccard distance between the quantized student's and the frozen
teacher's next-token distributions, at temperature 1, averaged over loss-mask tokens, with no CE term:

    L = 1 - sum_v min(p_s(v), p_t(v)) / sum_v max(p_s(v), p_t(v))

For two probability vectors min + max = p_s + p_t, so sum max = 2 - sum min. With m = sum min = 1 - TV
(TV = total-variation distance) the loss is 1 - m / (2 - m) = 2 TV / (1 + TV): a monotone transform of TV.
That is how it is computed here; it needs one reduction over the vocabulary instead of two.

Two forms:
  * full vocabulary, from student and teacher logits (online teacher);
  * top-K + rest, from the teacher's cached top-K log-probs (offline teacher, as bonsai27_qat.py caches
    them). The tail outside the K ids collapses into one "rest" bin on both sides. Collapsing bins can only
    raise sum min (min(a+b, c+d) >= min(a, c) + min(b, d)), so this is a LOWER bound on the full loss.

The KL forms of both are kept as the control arm; the top-K + rest form also has the reverse KL
(student || teacher), the on-policy distillation objective of the bf16 recovery (bm-889).
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = [
    "JaccardDistillationLoss",
    "jaccard_from_overlap",
    "jaccard_full",
    "jaccard_topk_rest",
    "kl_topk_rest",
    "masked_mean",
    "rkl_topk_rest",
]


def jaccard_from_overlap(overlap: torch.Tensor) -> torch.Tensor:
    """Jaccard distance from m = sum_v min(p_s, p_t), using sum_v max = 2 - m."""
    overlap = overlap.clamp(0.0, 1.0)
    return 1.0 - overlap / (2.0 - overlap)


def masked_mean(per_token: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the tokens selected by ``mask`` (all tokens when None)."""
    if mask is None:
        return per_token.mean()
    mask = mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


def jaccard_full(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token Jaccard distance over the full vocabulary. Shapes [..., V] -> [...]."""
    p_s = F.softmax(logits_s.float() / temperature, dim=-1)
    p_t = F.softmax(logits_t.float() / temperature, dim=-1).detach()
    return jaccard_from_overlap(torch.minimum(p_s, p_t).sum(-1))


def _topk_rest_probs(
    logits_s: torch.Tensor, topk_ids: torch.Tensor, topk_logprobs_t: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logp_s = F.log_softmax(logits_s.float() / temperature, dim=-1)
    p_s_k = logp_s.gather(-1, topk_ids.long()).exp()
    p_t_k = topk_logprobs_t.float().exp().detach()
    p_s_rest = (1.0 - p_s_k.sum(-1)).clamp_min(0.0)
    p_t_rest = (1.0 - p_t_k.sum(-1)).clamp_min(0.0)
    return p_s_k, p_t_k, p_s_rest, p_t_rest


def jaccard_topk_rest(
    logits_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logprobs_t: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token Jaccard distance over the teacher's K cached ids plus one rest bin.

    ``logits_s`` is [..., V]; ``topk_ids`` and ``topk_logprobs_t`` are [..., K] (teacher log-probs at
    temperature 1, as cached). Lower bound on :func:`jaccard_full` (see the module docstring).
    """
    p_s_k, p_t_k, p_s_rest, p_t_rest = _topk_rest_probs(logits_s, topk_ids, topk_logprobs_t, temperature)
    overlap = torch.minimum(p_s_k, p_t_k).sum(-1) + torch.minimum(p_s_rest, p_t_rest)
    return jaccard_from_overlap(overlap)


def kl_topk_rest(
    logits_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logprobs_t: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token KL(teacher || student) over K ids plus one rest bin (the bonsai27_qat.py control loss).

    Computed in log space: the student's K log-probs are logits minus logsumexp and its rest log-mass is the
    logsumexp over the ids outside the K, so d KL / d logit = p_s - p_t never vanishes. The earlier
    ``log(p + 1e-12)`` form saturated at -log(1e-12) once the student's mass on the teacher's K ids fell
    under 1e-12: a flat loss with a ~p / eps gradient that training could not leave (M6 tern, bug-825).
    """
    z = logits_s.float() / temperature
    lse = torch.logsumexp(z, dim=-1)
    ids = topk_ids.long()
    lp_s_k = z.gather(-1, ids) - lse[..., None]
    lp_t_k = topk_logprobs_t.float().detach()
    p_t_k = lp_t_k.exp()
    kl_k = (p_t_k * (lp_t_k - lp_s_k)).sum(-1)
    if ids.shape[-1] >= z.shape[-1]:  # the K distinct ids are the whole vocabulary: no rest bin on either side
        return kl_k
    lp_s_rest = torch.logsumexp(z.scatter(-1, ids, float("-inf")), dim=-1) - lse
    p_t_rest = (1.0 - p_t_k.sum(-1)).clamp_min(0.0)
    return kl_k + torch.xlogy(p_t_rest, p_t_rest) - p_t_rest * lp_s_rest


def rkl_topk_rest(
    logits_s: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logprobs_t: torch.Tensor,
    temperature: float = 1.0,
    rest_floor: float = 1e-6,
) -> torch.Tensor:
    """Per-token reverse KL(student || teacher) over K ids plus one rest bin (mode-seeking, GKD-style).

    Log space as :func:`kl_topk_rest`, so the gradient never saturates. The teacher's bins are its K cached
    log-probs plus a rest bin of ``max(1 - sum p_t, rest_floor)``, renormalised: fp16-cached log-probs can put
    the K mass at or over 1, and a zero rest bin would make any student mass outside the K infinitely costly
    (log 0). Renormalising keeps both sides proper distributions, so the loss is >= 0.
    """
    z = logits_s.float() / temperature
    lse = torch.logsumexp(z, dim=-1)
    ids = topk_ids.long()
    lp_s_k = z.gather(-1, ids) - lse[..., None]
    lp_t_k = topk_logprobs_t.float().detach()
    if ids.shape[-1] >= z.shape[-1]:  # the K ids are the whole vocabulary: no rest bin on either side
        lp_t_k = lp_t_k - torch.logsumexp(lp_t_k, dim=-1, keepdim=True)
        return (lp_s_k.exp() * (lp_s_k - lp_t_k)).sum(-1)
    lp_s_rest = torch.logsumexp(z.scatter(-1, ids, float("-inf")), dim=-1) - lse
    lp_t_rest = (1.0 - lp_t_k.exp().sum(-1)).clamp_min(rest_floor).log()
    lp_t = torch.cat([lp_t_k, lp_t_rest[..., None]], -1)
    lp_t = lp_t - torch.logsumexp(lp_t, dim=-1, keepdim=True)
    lp_s = torch.cat([lp_s_k, lp_s_rest[..., None]], -1)
    return (lp_s.exp() * (lp_s - lp_t)).sum(-1)


class JaccardDistillationLoss(_Loss):
    """Jaccard distance on output logits, drop-in for ``modelopt.torch.distill.LogitsDistillationLoss``.

    ``reduction="none"`` returns the per-token loss so the caller can apply its loss mask; ``"mean"``
    averages over all tokens. No temperature-squared rescaling: the Bolt recipe runs at temperature 1.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean"):
        """Set the softmax temperature and the reduction (``"mean"`` or ``"none"``)."""
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self._temperature = temperature
        self._reduction = reduction

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        """Loss of student logits ``logits_s`` against teacher logits ``logits_t`` (class dim last)."""
        per_token = jaccard_full(logits_s, logits_t, self._temperature)
        return per_token if self._reduction == "none" else per_token.mean()
