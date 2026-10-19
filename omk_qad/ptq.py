# SPDX-License-Identifier: Apache-2.0
"""PTQ of any HF causal LM to the census policy, exported as a ModelOpt GGML-packed HF checkpoint.

This is (a) the PTQ floor every QAD run must beat and (b) the producer of the packed checkpoint that
omk_qad.gguf_export turns into a GGUF. Text-only: multimodal checkpoints are loaded as their text causal LM.

  python -m omk_qad.ptq --model DIR --out DIR [--policy routed_experts=iq2_xxs,...] [--algorithm max|imatrix]
                        [--calib-text FILE] [--calib-seqs 16] [--seq 512] [--device cpu|cuda:N]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .census import build_quant_cfg, run_census


def load_text_causal_lm(model_dir: str, device: str, dtype: torch.dtype = torch.bfloat16):
    """The text causal LM of ``model_dir`` (vision tower / MTP head not loaded) and its tokenizer."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_dir).get_text_config()
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, dtype=dtype, device_map={"": device})
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_dir)


def calib_batches(tokenizer, path: str, n_seqs: int, seq: int, seed: int = 0) -> list[torch.Tensor]:
    """``n_seqs`` token windows of length ``seq`` from a text file, or random tokens when no file is given."""
    if path:
        ids = tokenizer(Path(path).read_text(), return_tensors="pt").input_ids[0]
        if ids.numel() < seq:
            raise ValueError(f"calibration text has {ids.numel()} tokens, fewer than one window of {seq}")
        step = max(1, (ids.numel() - seq) // max(1, n_seqs - 1))
        return [ids[i * step : i * step + seq].unsqueeze(0) for i in range(n_seqs)]
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, tokenizer.vocab_size, (1, seq), generator=g) for _ in range(n_seqs)]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", default="", help="comma list class=format overriding the default policy")
    ap.add_argument("--algorithm", default="max", help="ModelOpt calibration algorithm (max | imatrix | ...)")
    ap.add_argument("--calib-text", default="")
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    t0 = time.time()
    model, tok = load_text_causal_lm(a.model, a.device)
    policy = dict(kv.split("=", 1) for kv in a.policy.split(",") if kv)
    census = run_census(model, policy)
    print(census.table(), flush=True)
    if "MISMATCH" in census.table():
        raise SystemExit("census does not cover every parameter; refusing to quantize")
    cfg = build_quant_cfg(census, algorithm=a.algorithm)
    batches = calib_batches(tok, a.calib_text, a.calib_seqs, a.seq)

    def forward_loop(m):
        with torch.no_grad():
            for b in batches:
                m(input_ids=b.to(a.device))

    mtq.quantize(model, cfg, forward_loop)
    a.out.mkdir(parents=True, exist_ok=True)
    export_hf_checkpoint(model, export_dir=a.out)
    tok.save_pretrained(a.out)
    for extra in ("chat_template.jinja",):
        src = Path(a.model) / extra
        if src.exists():
            (a.out / extra).write_bytes(src.read_bytes())
    (a.out / "omk_census.json").write_text(json.dumps(census.to_json(), indent=1))
    (a.out / "omk_ptq.json").write_text(
        json.dumps({"model": a.model, "policy": policy, "algorithm": a.algorithm, "calib_text": a.calib_text,
                    "calib_seqs": a.calib_seqs, "seq": a.seq, "seconds": round(time.time() - t0, 1)}, indent=1)
    )
    print(f"exported {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
