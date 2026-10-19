# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the QAD trainer: layout plan, libggml codec, exact gradients, descent, train -> GGUF.

Artifacts go to $OMK_QAD_WORK (persistent disk; /tmp is tmpfs on solidpc). Needs the HF cache (or network)
for the reference configs/tokenizers and a built llama.cpp (libggml-base.so, llama-quantize, llama-perplexity).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
WORK = Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK)) / "qad"
LLAMA_CPP = Path(os.environ.get("OMK_LLAMA_CPP", "/opt/llama.cpp"))
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _run(*args: str) -> str:
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=CPU_ENV, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def _args(model: Path, **kw) -> argparse.Namespace:
    base = {"device": "cpu", "model": str(model), "beta2": 0.99, "clip": 1.0, "ggml_refresh": 1, "latent_dtype": "fp32",
            "export_only": "embedding", "latent_min": 65536, "loss": "jaccard", "ce_weight": 0.0, "loss_chunk": 32}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def setup():
    """Sharp tiny MoE teacher (lm_head x10: a peaked target), its IQ2_M layout, a small teacher cache."""
    from transformers import AutoModelForCausalLM

    from omk_qad.layout import build_layout
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("moe", "dense"):
        tiny = WORK / f"tiny_{kind}"
        if not (tiny / "config.json").exists():
            make_tiny(kind, tiny)
        sharp = WORK / f"tiny_{kind}_sharp"
        if not (sharp / "config.json").exists():
            m = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.bfloat16)  # bf16 as the real checkpoints
            with torch.no_grad():
                m.lm_head.weight.mul_(10)
            m.save_pretrained(sharp)
            for f in tiny.iterdir():
                if f.suffix in (".json", ".txt", ".jinja") and f.name not in ("config.json", "generation_config.json"):
                    (sharp / f.name).write_bytes(f.read_bytes())
        lay = WORK / f"layout_{kind}.pt"
        if not lay.exists():
            torch.save(build_layout(sharp, LLAMA_CPP, "IQ2_M", work=WORK), lay)
        out[kind] = (sharp, lay)
    text = WORK / "text.txt"
    if not text.exists():
        text.write_text((LLAMA_CPP / "README.md").read_text()[:60000])
    cache = WORK / "cache_moe"
    if not (cache / "cache.json").exists():
        _run("omk_qad.train", "--phase", "cache", "--model", str(out["moe"][0]), "--cache", str(cache),
             "--device", "cpu", "--data", str(text), "--eval-data", str(text), "--n-seq", "8", "--n-eval", "2",
             "--seq", "64", "--shard", "4")
    out["cache"] = cache
    out["text"] = text
    return out


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_layout_plan(setup, kind):
    from omk_qad.layout import summary

    layout = torch.load(setup[kind][1], weights_only=False)
    s = summary(layout)
    assert s["unsupported_quantized"] == []
    two_bit = "routed_experts" if kind == "moe" else "dense_mlp"
    assert set(s["per_class"][two_bit]) == {"IQ2_XXS"}
    assert s["per_class"]["embedding"] == {"IQ3_S": 1}  # llama.cpp IQ2_M rule for token_embd
    assert s["per_class"]["lm_head"] == {"Q5_K": 1}  # llama.cpp IQ2_M rule for output
    # the converter's V-head reorder of ssm_out is found by the probe, never assumed
    assert {g.split(".", 2)[2] for g in s["permuted_columns"] if g.endswith(".weight")} == {"ssm_out.weight"}
    for g, t in layout["targets"].items():
        if t["type"] not in ("F32", "F16", "BF16"):
            assert t["mapping"] == "direct", g


def test_ggml_codec_matches_gguf_py():
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    from omk_qad.ggml_lib import GGMLLib

    lib = GGMLLib(LLAMA_CPP)
    x = (np.random.default_rng(0).standard_normal((64, 512)) * 0.02).astype(np.float32)
    for name in ("Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ3_S", "Q2_K", "IQ4_XS"):
        qt = gguf.GGMLQuantizationType[name]
        q = lib.quantize(int(qt), x)
        assert np.array_equal(lib.quantize(int(qt), x), q), name  # deterministic
        d = lib.dequantize(int(qt), q, 512)
        assert np.array_equal(d, gguf.quants.dequantize(q, qt).reshape(d.shape)), name


def test_gradients_match_autograd(setup):
    """Hook path (served weights, use counting, checkpoint recompute, chunked head) == plain autograd."""
    from omk_qad import train as T
    from omk_qad.ptq import load_text_causal_lm

    model, lay = setup["moe"]
    a = _args(model, ce_weight=0.3, loss_chunk=24)
    student, lats, fp, ctl, head = T.build_student(a, torch.load(lay, weights_only=False))
    sh = torch.load(setup["cache"] / "train_0000.pt")
    x, idx, val = sh["x"][:2].long(), sh["idx"][:2], sh["val"][:2]
    cap = {}
    for n, lat in lats.items():
        lat.update = lambda g, n=n: cap.__setitem__(n, g.clone())
    served = {n: lat.served().clone() for n, lat in lats.items()}
    T.train_step(student, lats, lats[head], fp, None, x, idx, val, a)
    assert set(cap) == set(lats) and ctl.stats["undelivered"] == 0
    named = dict(student.named_parameters())
    fp_names = [n for n, p in named.items() if p.requires_grad]
    ref, _ = load_text_causal_lm(a.model, "cpu", torch.float32)
    leaves = {n: t.clone().requires_grad_() for n, t in served.items()}
    fpl = {n: named[n].detach().clone().requires_grad_() for n in fp_names}
    lg = torch.func.functional_call(ref, {**leaves, **fpl}, (x,)).logits[:, :-1].float()
    j, _, ce, _ = T.token_terms(lg.reshape(-1, lg.shape[-1]), idx.reshape(-1, idx.shape[-1]),
                               val.reshape(-1, val.shape[-1]), x[:, 1:].reshape(-1))
    ((j.sum() + a.ce_weight * ce.sum()) / j.numel()).backward()
    for n, g in cap.items():
        r = leaves[n].grad
        assert r.norm() > 0, n
        assert (g - r).norm() / r.norm() < 1e-4, n
    for n in fp_names:
        r = fpl[n].grad
        assert torch.allclose(named[n].grad, r, rtol=1e-4, atol=1e-8 * float(r.abs().max())), n


def test_descends_on_one_batch(setup):
    """The quantized student's loss on a fixed batch goes down (optimizer + re-encode loop works).

    KL, not Jaccard: on this random tiny model the 2-bit student starts almost disjoint from the teacher, and
    Jaccard (= 2TV/(1+TV)) has zero gradient once the supports are disjoint (seen: loss frozen at 0.9953).
    """
    from omk_qad import train as T

    model, lay = setup["moe"]
    a = _args(model, export_only="embedding,lm_head", loss_chunk=64, loss="kl")
    student, lats, fp, ctl, head = T.build_student(a, torch.load(lay, weights_only=False))
    opt = torch.optim.AdamW(fp, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    sh = torch.load(setup["cache"] / "train_0000.pt")
    x, idx, val = sh["x"][:1].long(), sh["idx"][:1], sh["val"][:1]
    ctl.lr = 3e-3
    hist = []
    for s in range(1, 16):
        ctl.step = s
        enc = ctl.stats["encodes"]
        hist.append(T.train_step(student, lats, lats[head], fp, opt, x, idx, val, a))
        assert ctl.stats["encodes"] > enc or s == 1  # updated latents are re-encoded every step
    assert np.mean(hist[-3:]) < 0.9 * hist[0], hist


def test_train_export_verify_load(setup):
    """Lossless gate + 2 steps, then GGUF at the plan; the GGUF holds exactly the blocks training served."""
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    from omk_qad import train as T

    model, lay = setup["moe"]
    run, out = WORK / "run_moe", WORK / "qad_moe.gguf"
    _run("omk_qad.train", "--phase", "train", "--model", str(model), "--layout", str(lay),
         "--cache", str(setup["cache"]), "--out", str(run), "--device", "cpu", "--steps", "2", "--bs", "2",
         "--lr", "1e-3", "--warmup", "1", "--eval-every", "2", "--save-every", "2", "--latent-dtype", "fp16",
         "--latent-min", "65536", "--loss-chunk", "32")
    log = (run / "log.txt").read_text()
    assert "LOSSLESS_GATE" in log and "QAD_DONE" in log
    res = json.loads((run / "summary.json").read_text())["results"]
    assert [r["tag"] for r in res] == ["init", "step2"]
    _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device", "cpu",
         "--load", str(run / "state.pt"), "--gguf", str(out), "--latent-dtype", "fp16", "--latent-min", "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    assert rep["type_counts"]["IQ2_XXS"] == 4 * 6  # 4 layers x (gate/up/down exps + shexp)

    # train == ship: the trained latents' served payloads are byte-identical to the GGUF tensors
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp16"), torch.load(lay, weights_only=False))
    T.load_state(str(run / "state.pt"), lats, student, None)
    tensors = {t.name: t for t in gguf.GGUFReader(out).tensors}
    checked = 0
    for lat in lats.values():
        for g, chunks in lat.payloads().items():
            want = np.concatenate([_flat(c) for c in chunks])
            assert np.array_equal(np.asarray(tensors[g].data).reshape(-1), want), g
            checked += 1
    assert checked >= 4 * 6

    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(out), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]


def _flat(payload) -> np.ndarray:
    arr = payload.detach().cpu().numpy() if isinstance(payload, torch.Tensor) else payload
    return np.ascontiguousarray(arr, dtype=np.uint8).reshape(-1)


def test_dense_ptq_floor_export(setup):
    """Dense arch: export without training (the PTQ floor at the plan) verifies and loads."""
    model, lay = setup["dense"]
    out = WORK / "ptq_dense.gguf"
    _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device", "cpu",
         "--gguf", str(out), "--latent-min", "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == []
    assert rep["type_counts"]["IQ2_XXS"] == 4 * 3
