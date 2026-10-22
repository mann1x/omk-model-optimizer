# SPDX-License-Identifier: Apache-2.0
"""MTP (nextn) head of Qwen3.5/3.6-style checkpoints: build, load, run, train -- as llama.cpp runs it.

transformers' ForCausalLM has no MTP module, so the ``mtp.*`` tensors of the checkpoint are dropped on load and a
PTQ/QAD export would ship without a draft head. Here the head is rebuilt from the model's OWN full-attention
decoder-layer class and attached as ``model.mtp`` with the checkpoint's exact parameter names (``mtp.fc``,
``mtp.pre_fc_norm_embedding``, ``mtp.pre_fc_norm_hidden``, ``mtp.layers.0.*``, ``mtp.norm``), so the layout
probe, the trainer and llama.cpp's converter (``mtp.* -> blk.<n_layer>.nextn.* / blk.<n_layer>.*``) all see it.

Forward = llama.cpp ``llama_model_qwen35*::graph_mtp`` (src/models/qwen35.cpp):
    h_t      = trunk output after the final norm (``h_nextn``, = HF ``last_hidden_state``)
    z_t      = eh_proj( concat( enorm(embed(x_{t+1})), hnorm(h_t) ) )
    z        = one full-attention decoder layer over z (causal)
    logits_t = output( shared_head_norm(z_t) )            # predicts x_{t+2}
so its distillation target at t is the teacher's next-token distribution at position t+1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def mtp_layers(config) -> int:
    """Number of MTP layers the (text) config declares."""
    return int(getattr(config, "mtp_num_hidden_layers", 0) or 0)


class MTPHead(nn.Module):
    """``mtp.*`` of a Qwen3.5/3.6-style checkpoint, built from the trunk's own module classes."""

    def __init__(self, model: nn.Module):
        """Build (uninitialised) from ``model`` (the text causal LM the head belongs to)."""
        super().__init__()
        cfg = model.config
        body = model.get_decoder()
        if mtp_layers(cfg) != 1:
            raise ValueError(f"only a single MTP layer is supported (llama.cpp too); config has {mtp_layers(cfg)}")
        full = [i for i, t in enumerate(getattr(cfg, "layer_types", [])) if t == "full_attention"]
        if not full:
            raise ValueError("MTP layer is a full-attention layer; the config declares none")
        layer_cls, norm_cls = type(body.layers[full[0]]), type(body.norm)
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = norm_cls(h, eps=eps)
        self.pre_fc_norm_hidden = norm_cls(h, eps=eps)
        self.layers = nn.ModuleList([layer_cls(cfg, full[0])])
        self.norm = norm_cls(h, eps=eps)
        self._cfg = cfg

    def forward(self, emb_next: torch.Tensor, hidden: torch.Tensor, rotary: nn.Module) -> torch.Tensor:
        """``emb_next`` = embed(x_{t+1}) and ``hidden`` = h_t, both [B, T, H]; returns normed z [B, T, H]."""
        from transformers.masking_utils import create_causal_mask

        z = self.fc(torch.cat([self.pre_fc_norm_embedding(emb_next), self.pre_fc_norm_hidden(hidden)], dim=-1))
        b, t = z.shape[:2]
        pos = (torch.arange(t, device=z.device) + 1).view(1, -1).expand(b, -1)  # position of x_{t+1}
        rope_pos = pos[None].expand(3, b, t)  # as Qwen3_5*TextModel: [3, B, T] ids to the rotary, text ids to attn
        mask = create_causal_mask(config=self._cfg, inputs_embeds=z, attention_mask=None, past_key_values=None,
                                  position_ids=pos)
        z = self.layers[0](z, position_embeddings=rotary(z, rope_pos), attention_mask=mask, position_ids=pos)
        return self.norm(z)


def attach(model: nn.Module) -> MTPHead:
    """Create ``model.mtp`` (same device/dtype as the trunk's final norm) and return it."""
    ref = model.get_decoder().norm.weight
    head = MTPHead(model).to(device=ref.device, dtype=ref.dtype)
    if ref.device.type == "cpu":  # same rule as the trunk (ptq.use_torch_kernels): no CUDA-only kernel on the CPU
        from .ptq import use_torch_kernels

        use_torch_kernels(head)
    model.mtp = head
    return head


def checkpoint_mtp_keys(model_dir: Path) -> dict[str, Path]:
    """``mtp.*`` tensor name -> safetensors file, over every ``*.safetensors`` file of ``model_dir``."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    idx = model_dir / "model.safetensors.index.json"
    out: dict[str, Path] = {}
    if idx.exists():
        for k, f in json.loads(idx.read_text())["weight_map"].items():
            if k.startswith("mtp."):
                out[k] = model_dir / f
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():  # noqa: SIM118 - safe_open is not a dict
                if k.startswith("mtp."):
                    out.setdefault(k, f)
    return out


def load_weights(model: nn.Module, model_dir: Path) -> int:
    """Copy the checkpoint's ``mtp.*`` tensors into ``model.mtp``; every MTP parameter must be found."""
    from safetensors import safe_open

    keys = checkpoint_mtp_keys(model_dir)
    params = {f"mtp.{n}": p for n, p in model.mtp.named_parameters()}
    missing = sorted(set(params) - set(keys))
    extra = sorted(set(keys) - set(params))
    if missing or extra:
        raise SystemExit(f"MTP tensors do not match the checkpoint: missing {missing[:5]}, unexpected {extra[:5]}")
    by_file: dict[Path, list[str]] = {}
    for k, f in keys.items():
        by_file.setdefault(f, []).append(k)
    with torch.no_grad():
        for f, ks in by_file.items():
            with safe_open(str(f), "pt") as sf:
                for k in ks:
                    t = sf.get_tensor(k)
                    if tuple(t.shape) != tuple(params[k].shape):
                        raise SystemExit(f"{k}: checkpoint {tuple(t.shape)} != module {tuple(params[k].shape)}")
                    params[k].copy_(t.to(params[k].dtype))
    return len(params)


def save_random_weights(model: nn.Module, out_dir: Path, std: float = 0.05, seed: int = 1) -> Path:
    """Tiny-model helper: random ``mtp.*`` weights (same init rule as omk_qad.tiny) in ``model-mtp.safetensors``."""
    from safetensors.torch import save_file

    head = attach(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in head.parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * std)
    sd = {f"mtp.{n}": p.detach().contiguous() for n, p in head.named_parameters()}
    path = Path(out_dir) / "model-mtp.safetensors"
    save_file(sd, str(path))
    del model.mtp
    return path
