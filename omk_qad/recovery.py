# SPDX-License-Identifier: Apache-2.0
"""bf16 recovery mode: train a model's own bf16 weights (no quantizer) against a different teacher.

The student's trained tensors are latents with no quantized segment: served as themselves, updated inside
backward by the same momentum-free Adafactor as QAD, written back to bf16 with stochastic rounding (no fp32
master copy: a 26B model fits one 96 GB GPU). Everything outside the training scope is frozen bf16.

Scopes (``--train-scope``):

  gemma4   the reviewed Gemma-4 MoE scope (omnimergekit gepo_brevity.lora_scope_for): every attention
           projection ``self_attn.{q,k,v,o}_proj`` and the per-layer dense MLP ``mlp.{gate,up,down}_proj``.
           The count is DERIVED from the loaded model (layers x 6 + the layers that carry a v_proj: Gemma-4's
           full-attention layers share K and V and have none), and nothing outside the regex may match:
           ``router.proj`` (opt-in, ``--train-router``), the routed experts, the embedding / head and the norms
           are not trained by accident.
  classes:<a,b,...>  census classes (attention, dense_mlp, router, routed_experts, shared_experts, ...).

Export writes the source checkpoint back with only the trained tensors replaced (every other tensor, the
vision tower included, byte for byte), under the source's own key names and shard layout.
"""

import json
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

GEMMA4_SCOPE_RE = r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight"
GEMMA4_ROUTER_RE = r"model\.layers\.\d+\.router\.proj\.weight"


def softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    """Final-logit soft capping (Gemma-4: cap * tanh(logits / cap)); identity when ``cap`` is falsy."""
    return torch.tanh(logits / cap) * cap if cap else logits


def softcap_of(model: nn.Module) -> float | None:
    """The model's final-logit soft cap (None when it has none)."""
    return getattr(model.config, "final_logit_softcapping", None) or None


def scope_params(model: nn.Module, scope: str, train_router: bool = False) -> list[str]:
    """Names of the parameters ``scope`` trains; refuses when the model does not match the scope exactly."""
    names = [n for n, _ in model.named_parameters()]
    if scope == "gemma4":
        hit = [n for n in names if re.fullmatch(GEMMA4_SCOPE_RE, n)]
        if not hit:
            raise SystemExit("REFUSE: --train-scope gemma4 matches no parameter of this model")
        layers = {int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in hit}
        n_v = sum(n.endswith("self_attn.v_proj.weight") for n in hit)
        n_layers = getattr(model.config, "num_hidden_layers", len(layers))
        if len(layers) != n_layers or len(hit) != len(layers) * 6 + n_v:
            raise SystemExit(f"REFUSE: gemma4 scope matched {len(hit)} tensors over {len(layers)}/{n_layers} layers, "
                             f"expected {len(layers) * 6 + n_v} (layers x 6 + {n_v} v_proj)")
        if train_router:
            router = [n for n in names if re.fullmatch(GEMMA4_ROUTER_RE, n)]
            if len(router) != n_layers:
                raise SystemExit(f"REFUSE: --train-router matched {len(router)} router.proj, expected {n_layers}")
            hit += router
        bad = [n for n in hit if ".experts." in n or "embed" in n or "lm_head" in n or "vision" in n]
        if bad:
            raise SystemExit(f"REFUSE: gemma4 scope reached excluded tensors {bad[:4]}")
        return hit
    if scope.startswith("classes:"):
        from .census import classify_model

        want = {c for c in scope.split(":", 1)[1].split(",") if c}
        mods = {e.module for e in classify_model(model).entries if e.cls in want}
        if "vectors" in want:
            mods.add("<1-D vectors>")
        hit = [n for n, p in model.named_parameters()
               if n.rsplit(".", 1)[0] in mods or n in mods or ("<1-D vectors>" in mods and p.ndim < 2)]
        if not hit:
            raise SystemExit(f"REFUSE: classes {sorted(want)} match no parameter")
        return hit
    raise SystemExit(f"unknown --train-scope {scope!r} (gemma4 | classes:<a,b,...>)")


class FrozenHead:
    """The output head when it is not trained (e.g. tied to a frozen embedding): the latent interface, inert."""

    frozen = True  # train_step computes no weight gradient for it

    def __init__(self, weight: torch.Tensor, ctl):
        """``weight`` is the head's parameter; ``ctl`` gives the compute dtype."""
        self.weight, self.ctl = weight, ctl

    def served(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The weight, detached: a gradient on it is dropped, never accumulated on the frozen parameter."""
        return self.weight.detach().to(dtype or self.ctl.compute_dtype)

    def add_use(self) -> None:
        """No pending gradient is ever delivered."""

    def on_grad(self, g: torch.Tensor) -> None:
        """Discarded."""
        del g

    def flush(self) -> None:
        """Nothing pending."""


def build_student_bf16(a, load_fn, say):
    """Student for ``--mode bf16``: (model, latents, fp_params, ctl, head) with only ``--train-scope`` trainable."""
    from .fakequant import Control, Latent, install

    cuda = a.device != "cpu"
    student, _ = load_fn(a.model, a.device, torch.bfloat16 if cuda else torch.float32)
    trained = set(scope_params(student, a.train_scope, getattr(a, "train_router", False)))
    ctl = Control(compute_dtype=torch.bfloat16 if cuda else torch.float32, beta2=a.beta2, clip=a.clip,
                  dummy=torch.zeros((), requires_grad=True, device=a.device))
    latents, fp_params = {}, []
    for n, p in student.named_parameters():
        if n not in trained:
            p.requires_grad_(False)
        elif p.ndim >= 2:
            latents[n] = Latent(n, p, [], ctl, None, torch.bfloat16)
        else:
            p.data = p.data.float()
            p.requires_grad_(True)
            fp_params.append(p)
    head_param = student.get_output_embeddings().weight
    head_name = next(n for n, p in student.named_parameters() if p is head_param)
    head = latents.get(head_name) or FrozenHead(head_param, ctl)
    install(student, latents)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    a.softcap = softcap_of(student)
    n_frozen = sum(p.numel() for n, p in student.named_parameters() if n not in trained)
    say(f"student (bf16 mode, scope {a.train_scope}{' +router' if getattr(a, 'train_router', False) else ''}): "
        f"{len(latents)} latents {sum(lat.param.numel() for lat in latents.values()) / 1e9:.3f}B bf16, "
        f"fp params {sum(p.numel() for p in fp_params) / 1e6:.2f}M, frozen {n_frozen / 1e9:.3f}B, "
        f"head {head_name} ({'trained' if head_name in latents else 'frozen'}), softcap {a.softcap}")
    return student, latents, fp_params, ctl, head


# ===================================================================================================== export
def checkpoint_keys(model_dir: Path) -> dict[str, str]:
    """{checkpoint tensor key: shard file} of a safetensors checkpoint (sharded or single)."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return dict(json.loads(idx.read_text())["weight_map"])
    from safetensors import safe_open

    with safe_open(model_dir / "model.safetensors", "pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")


def key_for(name: str, keys: dict[str, str]) -> str:
    """The checkpoint key of a text-model parameter (a multimodal checkpoint nests it under language_model)."""
    for k in (name, name.replace("model.", "model.language_model.", 1)):
        if k in keys:
            return k
    raise SystemExit(f"export: {name} has no checkpoint key in the source (tried model. and model.language_model.)")


def export_bf16(src: Path, ckpt: str, out: Path, say) -> dict:
    """Source checkpoint with the trained tensors of ``ckpt`` written in; every other tensor byte-identical."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    t0 = time.time()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    trained = {**ck["latents"], **ck["fp"]}
    keys = checkpoint_keys(src)
    mapped = {key_for(n, keys): t for n, t in trained.items()}
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"export: {out} is not empty (an export is never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():  # configs, tokenizer, chat template, processor files: copied as they are
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    replaced, shards = [], sorted(set(keys.values()))
    for shard in shards:
        with safe_open(src / shard, "pt") as f:
            meta = f.metadata()
            tensors = {}
            for k in f.keys():  # noqa: SIM118 (safe_open handle, not a dict)
                t = f.get_tensor(k)
                if k in mapped:
                    new = mapped[k].to(t.dtype)
                    if new.shape != t.shape:
                        raise SystemExit(f"export: {k} shape {tuple(new.shape)} != source {tuple(t.shape)}")
                    tensors[k] = new.contiguous()
                    replaced.append(k)
                else:
                    tensors[k] = t
        save_file(tensors, out / shard, metadata=meta)
        del tensors
    if len(replaced) != len(mapped):
        raise SystemExit(f"export: wrote {len(replaced)} of {len(mapped)} trained tensors")
    unchanged = verify_export(src, out, set(replaced))
    rep = {"source": str(src), "checkpoint": ckpt, "step": ck["step"], "replaced": len(replaced),
           "unchanged_verified": unchanged, "shards": shards, "seconds": round(time.time() - t0, 1),
           "replaced_keys": sorted(replaced)}
    (out / "omk_bf16_export.json").write_text(json.dumps(rep, indent=1))
    say(f"EXPORT_BF16 {out}: {len(replaced)} trained tensors written, {unchanged} others verified identical "
        f"({rep['seconds']}s)")
    return rep


def verify_export(src: Path, out: Path, replaced: set[str]) -> int:
    """Every untouched tensor of ``out`` equals the source bit for bit; returns how many were checked."""
    from safetensors import safe_open

    keys, n = checkpoint_keys(src), 0
    for shard in sorted(set(keys.values())):
        with safe_open(src / shard, "pt") as a, safe_open(out / shard, "pt") as b:
            if set(a.keys()) != set(b.keys()):
                raise SystemExit(f"export verify: {shard} key sets differ")
            for k in a.keys():  # noqa: SIM118 (safe_open handle, not a dict)
                if k in replaced:
                    continue
                if not torch.equal(a.get_tensor(k).view(torch.uint8), b.get_tensor(k).view(torch.uint8)):
                    raise SystemExit(f"export verify: untouched tensor {k} changed")
                n += 1
    return n
