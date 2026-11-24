# SPDX-License-Identifier: Apache-2.0
"""Router telemetry, load-balance loss and routing-collapse gate for MoE students (bm-898 sft11r, 2026-11-20).

Owner: "this time we target fully the router too" -- once ``router.scale`` and ``router.per_expert_scale`` are
trainable, the cheapest way down the loss is to turn a few experts up and starve the rest. This module watches that.

``RouterProbe`` hooks every router module of the student (a module whose forward returns
``(router_probabilities [T, E], top_k_weights [T, K], top_k_index [T, K])``, the Gemma-4 ``Gemma4TextRouter``
contract) and runs in one of three modes:

- ``off``: hooks do nothing (the default between phases).
- ``train``: each forward adds one Switch-Transformer load-balance term per layer, ``E * sum_e f_e * P_e`` with
  ``f_e`` the fraction of top-k assignments routed to expert e (no gradient) and ``P_e`` the mean router probability
  of e (gradient flows into proj / scale); ``pop_lb()`` returns their mean over layers, a scalar with a graph.
- ``eval``: each forward accumulates, per layer, the assignment counts per expert, the summed routing entropy, the
  token count and the top-1 expert of every token; ``end_eval()`` turns that into ``RouterStats``.

``RouterStats`` per layer: ``load`` (share of assignments per expert), ``top_share`` (share taken by the ``top_n``
most-loaded experts, ``top_n = max(1, round(E / 12))`` = 8 of 98 on the 20B), ``entropy`` (mean routing entropy in
nats), ``flip`` (share of tokens whose top-1 expert differs from the reference, usually the init model). The eval rows
are fixed, so the reference top-1 is stored once (``init``) and the flip rate is exact.

``gate(stats, init, spec)`` with ``spec = (top_ratio, entropy_ratio, top_abs)``: trips when any layer's ``top_share``
exceeds ``top_ratio x`` its init value, or its entropy falls below ``entropy_ratio x`` init, or ``top_share``
exceeds ``top_abs`` while the init was below it. Returns the list of reasons (empty = pass).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


def router_modules(model: nn.Module) -> dict[int, nn.Module]:
    """{layer index: router module} of a Gemma-4-shaped MoE model; refuses when there is none."""
    out = {}
    for n, m in model.named_modules():
        if n.endswith(".router") and ".layers." in n:
            out[int(n.split(".layers.")[1].split(".")[0])] = m
    if not out:
        raise SystemExit("router telemetry: the model has no `layers.N.router` module")
    return out


def top_n_of(n_experts: int) -> int:
    """How many most-loaded experts the concentration metric sums: 8 of 98 (20B), 11 of 128, 1 of 8 (tiny)."""
    return max(1, round(n_experts / 12))


@dataclass
class LayerStats:
    """One layer's routing statistics over the eval rows (see the module docstring)."""

    load: list[float]
    top_share: float
    entropy: float
    flip: float | None  # None when there is no reference yet
    tokens: int


@dataclass
class RouterStats:
    """Per-layer ``LayerStats`` plus the ``top_n`` the concentration metric used."""

    layers: dict[int, LayerStats]
    top_n: int

    def summary(self) -> dict:
        """The numbers that go to the log / W&B: worst layer first, then per layer."""
        ls = self.layers
        out = {"top_n": self.top_n,
               "top_share_max": max(s.top_share for s in ls.values()),
               "top_share_mean": sum(s.top_share for s in ls.values()) / len(ls),
               "entropy_min": min(s.entropy for s in ls.values()),
               "entropy_mean": sum(s.entropy for s in ls.values()) / len(ls)}
        flips = [s.flip for s in ls.values() if s.flip is not None]
        if flips:
            out["flip_max"] = max(flips)
            out["flip_mean"] = sum(flips) / len(flips)
        out["layers"] = {str(i): {"top_share": s.top_share, "entropy": s.entropy,
                                  **({"flip": s.flip} if s.flip is not None else {}),
                                  "load_max": max(s.load), "load_min": min(s.load)} for i, s in sorted(ls.items())}
        return out


@dataclass
class _Acc:
    counts: torch.Tensor | None = None  # [E] assignments per expert
    ent: float = 0.0  # summed entropy over tokens
    tokens: int = 0
    top1: list = field(default_factory=list)  # per-forward [T] int16 CPU tensors


class RouterProbe:
    """Forward hooks on every router of ``model``; see the module docstring for the modes."""

    def __init__(self, model: nn.Module):
        """Hook every ``layers.N.router`` of ``model``; the hooks stay inert until a mode is set."""
        self.routers = router_modules(model)
        self.mode = "off"
        self.n_experts: int | None = None
        self._seen: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # train window: (probs, idx) per layer
        self._acc: dict[int, _Acc] = {}
        self.init_top1: dict[int, torch.Tensor] | None = None  # reference top-1 per layer (eval rows, in order)
        self._handles = [m.register_forward_hook(self._hook(i)) for i, m in self.routers.items()]

    def remove(self) -> None:
        """Detach the hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    # ------------------------------------------------------------------ hook
    def _hook(self, layer: int):
        def fn(_module, _inputs, output):
            if self.mode == "off":
                return
            if not (isinstance(output, tuple) and len(output) == 3):
                raise SystemExit(f"router telemetry: layer {layer} router returned {type(output).__name__}, expected "
                                 "(probabilities, top_k_weights, top_k_index)")
            probs, _w, idx = output
            if probs.ndim != 2 or idx.ndim != 2:
                raise SystemExit(f"router telemetry: layer {layer} router output shapes {tuple(probs.shape)} / "
                                 f"{tuple(idx.shape)}, expected [T, E] / [T, K]")
            e = probs.shape[1]
            if self.n_experts is None:
                self.n_experts = e
            elif self.n_experts != e:
                raise SystemExit(f"router telemetry: layer {layer} has {e} experts, layer 0 has {self.n_experts}")
            if self.mode == "train":
                # References only -- NO autograd op in here. The router runs inside a gradient-checkpoint frame, and
                # non-reentrant checkpointing requires the backward's recompute pass (hook inert, mode "off") to save
                # exactly the tensors the original forward saved; the balance arithmetic happens in pop_lb(), outside.
                self._seen.setdefault(layer, []).append((probs, idx))
            else:  # eval
                with torch.no_grad():
                    acc = self._acc.setdefault(layer, _Acc())
                    c = torch.bincount(idx.reshape(-1), minlength=e)
                    acc.counts = c if acc.counts is None else acc.counts + c
                    pf = probs.float().clamp_min(1e-12)
                    acc.ent += float(-(pf * pf.log()).sum(1).sum())
                    acc.tokens += probs.shape[0]
                    acc.top1.append(probs.argmax(1).to(torch.int16).cpu())
        return fn

    # ------------------------------------------------------------------ train mode
    def begin_train(self) -> None:
        """Collect load-balance terms on the next forwards."""
        self._seen = {}
        self.mode = "train"

    def pop_lb(self) -> torch.Tensor | None:
        """Mean load-balance term over the layers seen since ``begin_train`` (None when no router ran).

        Popping ENDS the collection window (mode -> ``off``). It has to: under gradient checkpointing the backward
        re-runs every layer's forward, the hook fires again, and a term collected from that recompute pass would hold
        the recomputed activations of the whole layer alive until the next ``begin_train`` -- on the 20B student that
        was +30 GiB and a CUDA OOM at step 1 (2026-11-20). Call ``begin_train`` again before the next forward.
        """
        self.mode = "off"
        seen, self._seen = self._seen, {}
        if not seen:
            return None
        terms = []
        for pairs in seen.values():  # one term per layer; a layer seen twice in the window sums
            t = None
            for probs, idx in pairs:
                e = probs.shape[1]
                f = torch.bincount(idx.reshape(-1), minlength=e).float() / max(1, idx.numel())  # no grad
                p = probs.float().mean(0)  # [E], with grad into proj / scale
                term = e * (f * p).sum()
                t = term if t is None else t + term
            terms.append(t)
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------ eval mode
    def begin_eval(self) -> None:
        """Accumulate routing statistics on the next forwards."""
        self._acc = {}
        self.mode = "eval"

    def end_eval(self, set_reference: bool = False) -> RouterStats:
        """Stats of the forwards since ``begin_eval``; with ``set_reference`` their top-1 becomes the flip reference."""
        self.mode = "off"
        if not self._acc:
            raise SystemExit("router telemetry: no router forward ran during the eval")
        e = self.n_experts or 0
        top_n = top_n_of(e)
        layers = {}
        top1 = {}
        for i, acc in sorted(self._acc.items()):
            assert acc.counts is not None
            load = (acc.counts.float() / max(1, int(acc.counts.sum()))).tolist()
            share = float(sum(sorted(load, reverse=True)[:top_n]))
            t1 = torch.cat(acc.top1)
            top1[i] = t1
            flip = None
            if self.init_top1 is not None and i in self.init_top1:
                ref = self.init_top1[i]
                if t1.numel() < ref.numel():
                    raise SystemExit(f"router telemetry: layer {i} saw {t1.numel()} eval tokens, the reference has "
                                     f"{ref.numel()} (eval rows were removed or changed)")
                if t1.numel() > ref.numel() and i == min(self._acc):
                    # bm-898 2026-11-24 (bug-996): eval files APPENDED on resume (sft_eval is a runtime
                    # arg). Eval rows are fixed-length and evaluated in file order, so the first
                    # ref.numel() tokens are the reference rows: the flip is measured on them only.
                    print(f"router telemetry: {t1.numel() - ref.numel()} eval tokens beyond the {ref.numel()} "
                          "reference tokens (eval rows added on resume); flip measured on the reference rows",
                          flush=True)
                flip = float((ref != t1[: ref.numel()]).float().mean())
            layers[i] = LayerStats(load=load, top_share=share, entropy=acc.ent / max(1, acc.tokens), flip=flip,
                                   tokens=acc.tokens)
        if set_reference or self.init_top1 is None:
            self.init_top1 = top1
            if set_reference:
                for s in layers.values():
                    s.flip = 0.0
        self._acc = {}
        return RouterStats(layers=layers, top_n=top_n)

    # ------------------------------------------------------------------ reference persistence
    def reference_state(self) -> dict:
        """The flip reference (per-layer top-1 of the eval rows), for ``torch.save``."""
        return {"top1": {str(i): t for i, t in (self.init_top1 or {}).items()}, "n_experts": self.n_experts}

    def load_reference(self, state: dict) -> None:
        """Restore a saved ``reference_state`` (a resumed run)."""
        self.init_top1 = {int(i): t for i, t in state["top1"].items()}
        self.n_experts = state.get("n_experts", self.n_experts)


def parse_gate(spec: str) -> tuple[float, float, float] | None:
    """``--router-gate top_ratio,entropy_ratio,top_abs`` -> tuple; "0" / "" / "off" -> None (gate off)."""
    if not spec or spec.strip() in ("0", "off", "none"):
        return None
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3 or parts[0] <= 1.0 or not 0.0 < parts[1] < 1.0 or not 0.0 < parts[2] <= 1.0:
        raise SystemExit(f"--router-gate {spec!r}: expected top_ratio>1,entropy_ratio in (0,1),top_abs in (0,1]")
    return parts[0], parts[1], parts[2]


def gate(stats: RouterStats, init: RouterStats, spec: tuple[float, float, float]) -> list[str]:
    """Reasons the gate trips (empty = pass); see the module docstring."""
    top_ratio, ent_ratio, top_abs = spec
    why = []
    for i, s in sorted(stats.layers.items()):
        s0 = init.layers.get(i)
        if s0 is None:
            why.append(f"layer {i}: no init reference")
            continue
        if s.top_share > top_ratio * s0.top_share:
            why.append(f"layer {i}: top-{stats.top_n} share {s.top_share:.3f} > {top_ratio} x init {s0.top_share:.3f}")
        if s.entropy < ent_ratio * s0.entropy:
            why.append(f"layer {i}: entropy {s.entropy:.3f} < {ent_ratio} x init {s0.entropy:.3f}")
        if s0.top_share < top_abs < s.top_share:
            why.append(f"layer {i}: top-{stats.top_n} share {s.top_share:.3f} > {top_abs} (init {s0.top_share:.3f})")
    return why


def fp_groups(named: list[tuple[str, torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    """Group the trained 1-D parameters for update/drift telemetry (router, FFN-norm and attention-norm scales)."""
    out: dict[str, list[torch.Tensor]] = {}
    for n, p in named:
        if n.endswith("router.scale"):
            g = "router_scale"
        elif n.endswith("router.per_expert_scale"):
            g = "per_expert_scale"
        elif "feedforward_layernorm" in n:
            g = "ffn_norm"
        elif n.endswith(("q_norm.weight", "k_norm.weight")):
            g = "qk_norm"
        elif n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
            g = "attn_norm"
        else:
            g = "other"
        out.setdefault(g, []).append(p)
    return out


class FpDrift:
    """Update ratio and drift-from-init of the fp (1-D) parameter groups, per optimizer step."""

    def __init__(self, groups: dict[str, list[torch.Tensor]]):
        """``groups`` as ``fp_groups`` returns them; the current values become the drift reference."""
        self.groups = groups
        self.init = {g: torch.cat([p.detach().flatten().float() for p in ps]).clone() for g, ps in groups.items()}
        self._before: dict[str, torch.Tensor] = {}

    def _flat(self, g: str) -> torch.Tensor:
        return torch.cat([p.detach().flatten().float() for p in self.groups[g]])

    def before(self) -> None:
        """Snapshot the groups before an optimizer step."""
        self._before = {g: self._flat(g).clone() for g in self.groups}

    def after(self) -> dict[str, float]:
        """Per group after the step: ``fp/<g>/update_ratio`` vs the snapshot, ``drift`` and ``max_abs_dev`` vs init."""
        out = {}
        for g in self.groups:
            now = self._flat(g)
            b, i0 = self._before.get(g), self.init[g]
            if b is not None:
                out[f"fp/{g}/update_ratio"] = float((now - b).norm() / b.norm().clamp_min(1e-30))
            out[f"fp/{g}/drift"] = float((now - i0).abs().mean() / i0.abs().mean().clamp_min(1e-30))
            out[f"fp/{g}/max_abs_dev"] = float((now - i0).abs().max())
        return out


def entropy_uniform(n_experts: int) -> float:
    """Entropy of the uniform routing distribution (the ceiling the per-layer entropy is read against)."""
    return math.log(n_experts)
