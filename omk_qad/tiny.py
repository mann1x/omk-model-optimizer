# SPDX-License-Identifier: Apache-2.0
"""Tiny random models with the exact architecture of the real QAD targets, for CPU tests.

The config is the reference model's own text config (architecture, rope, attention/linear-attention head
layout, tokenizer, vocab) with only the sizes shrunk, so module names, fused-expert layout and the
llama.cpp converter path are the ones the real model takes. Every IQ2 target input dim stays a multiple
of 256 (the GGML super-block).

  python -m omk_qad.tiny --kind moe   --out DIR   # Qwen3.6-27B-A3B-Coder shape (qwen3_5_moe_text)
  python -m omk_qad.tiny --kind dense --out DIR   # Qwen3.6-27B shape (qwen3_5_text)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REFERENCE = {
    "moe": "ManniX-ITA/Qwen3.6-27B-A3B-Coder",
    "dense": "Qwen/Qwen3.6-27B",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")

SHRINK = {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "max_position_embeddings": 4096,
}
SHRINK_MOE = {
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 256,
}
SHRINK_DENSE = {"intermediate_size": 512}


def tiny_text_config(kind: str) -> dict:
    """The reference text config with sizes shrunk (no weights downloaded)."""
    cfg = json.loads(Path(hf_hub_download(REFERENCE[kind], "config.json")).read_text())
    text = dict(cfg.get("text_config", cfg))
    text.update(SHRINK)
    text.update(SHRINK_MOE if kind == "moe" else SHRINK_DENSE)
    text.pop("mtp_num_hidden_layers", None)
    text.pop("mtp_use_dedicated_embeddings", None)
    return text


def make_tiny(kind: str, out: Path, seed: int = 0, std: float = 0.05) -> Path:
    """Build, randomise and save a tiny model of ``kind`` ("moe" | "dense") with the reference tokenizer."""
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )

    text = tiny_text_config(kind)
    config_cls, model_cls = (
        (Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM) if kind == "moe" else (Qwen3_5TextConfig, Qwen3_5ForCausalLM)
    )
    for key in ("model_type", "architectures", "transformers_version", "dtype", "torch_dtype"):
        text.pop(key, None)
    config = config_cls(**text)
    config.architectures = [model_cls.__name__]
    torch.manual_seed(seed)
    model = model_cls(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.ndim >= 2:  # 1-D tensors (norms, A_log, dt_bias) keep the model's own init
                param.normal_(0.0, std)
    model = model.to(torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    for fname in TOKENIZER_FILES:
        shutil.copy(hf_hub_download(REFERENCE[kind], fname), out / fname)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=sorted(REFERENCE), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_tiny(a.kind, a.out, a.seed))


if __name__ == "__main__":
    main()
