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
TRAIN = ("--phase", "train", "--device", "cpu", "--bs", "2", "--lr", "1e-3", "--warmup", "1", "--latent-dtype", "fp16",
         "--latent-min", "65536", "--loss-chunk", "32")


def _run(*args: str) -> str:
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=CPU_ENV, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def _args(model: Path, **kw) -> argparse.Namespace:
    base = {"device": "cpu", "model": str(model), "beta2": 0.99, "clip": 1.0, "ggml_refresh": 1, "latent_dtype": "fp32",
            "export_only": "embedding", "latent_min": 65536, "loss": "jaccard", "ce_weight": 0.0, "loss_chunk": 32,
            "mtp": "train", "mtp_weight": 0.3, "imatrix": "none"}
    base.update(kw)
    return argparse.Namespace(**base)


def _sharp(tiny: Path, sharp: Path) -> None:
    """lm_head x10 copy of ``tiny`` (a peaked teacher target); the MTP file is carried over (HF drops mtp.*)."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.bfloat16)  # bf16 as the real checkpoints
    with torch.no_grad():
        m.lm_head.weight.mul_(10)
    m.save_pretrained(sharp)
    for f in tiny.iterdir():
        if (f.suffix in (".json", ".txt", ".jinja") and f.name not in ("config.json", "generation_config.json")) or \
                f.name == "model-mtp.safetensors":
            (sharp / f.name).write_bytes(f.read_bytes())


def _cache(model: Path, layout: Path, cache: Path, text: Path) -> Path:
    """Small teacher cache + imatrix (resumes an older cache that has no imatrix yet)."""
    from omk_qad.imatrix import folded_candidates

    _, block = folded_candidates(torch.load(layout, weights_only=False))
    fresh = all((cache / f).exists() for f in ("cache.json", "imatrix.pt", "imatrix.folded.pt")) and {
        e["block"] for e in torch.load(cache / "imatrix.folded.pt", weights_only=False)["entries"].values()} == {block}
    if not fresh:  # the cache phase backfills (a stale-block folded statistic is set aside, never deleted)
        _run("omk_qad.train", "--phase", "cache", "--model", str(model), "--cache", str(cache), "--layout", str(layout),
             "--device", "cpu", "--data", str(text), "--eval-data", str(text), "--n-seq", "8", "--n-eval", "2",
             "--seq", "64", "--shard", "4", "--llama-cpp", str(LLAMA_CPP))
    return cache


@pytest.fixture(scope="module")
def setup():
    """Sharp tiny MoE / dense / MoE+MTP / tied dense+MTP teachers, IQ2_M-rest layouts, teacher caches + imatrix."""
    from omk_qad.layout import build_layout
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, kind, mtp in (("moe", "moe", False), ("dense", "dense", False), ("moe_mtp", "moe", True),
                            ("dense_tied", "dense_tied", True)):  # Qwen3.5-4B shape: tied embeddings + MTP
        tiny = WORK / f"tiny_{name}"
        if not (tiny / "config.json").exists():
            make_tiny(kind, tiny, mtp=mtp)
        sharp = WORK / f"tiny_{name}_sharp"
        if not (sharp / "config.json").exists():
            _sharp(tiny, sharp)
        lay = WORK / f"layout2_{name}.pt"  # v2: carries rest/mtp (2026-10-19)
        if not lay.exists():
            torch.save(build_layout(sharp, LLAMA_CPP, "IQ2_M", work=WORK / f"lay_{name}"), lay)
        out[name] = (sharp, lay)
    text = WORK / "text.txt"
    if not text.exists():
        text.write_text((LLAMA_CPP / "README.md").read_text()[:60000])
    out["cache"] = _cache(*out["moe"], WORK / "cache_moe", text)
    out["cache_mtp"] = _cache(*out["moe_mtp"], WORK / "cache_moe_mtp", text)
    out["cache_tied"] = _cache(*out["dense_tied"], WORK / "cache_dense_tied", text)
    out["text"] = text
    return out


@pytest.mark.parametrize("kind", ["moe", "dense", "dense_tied"])
def test_layout_plan(setup, kind):
    from omk_qad.layout import summary

    layout = torch.load(setup[kind][1], weights_only=False)
    s = summary(layout)
    assert s["unsupported_quantized"] == []
    two_bit = "routed_experts" if kind == "moe" else "dense_mlp"
    assert set(s["per_class"][two_bit]) == {"IQ2_XXS"}
    if kind == "dense_tied":  # one tensor: classed as the head, shipped as token_embd with the output's rule
        assert "embedding" not in s["per_class"] and s["per_class"]["lm_head"] == {"Q5_K": 1}
        assert layout["targets"]["token_embd.weight"]["cls"] == "lm_head" and "output.weight" not in layout["targets"]
    else:
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


@pytest.mark.parametrize("kind", ["moe", "dense_tied"])
def test_gradients_match_autograd(setup, kind):
    """Hook path (served weights, use counting, checkpoint recompute, chunked head) == plain autograd.

    Tied: the one tensor is used twice per step (lookup in the body, head on the final hidden state); its update
    must receive the sum of both. The MTP head is frozen here (the reference model has none).
    """
    from omk_qad import train as T
    from omk_qad.ptq import load_text_causal_lm

    model, lay = setup[kind]
    a = _args(model, ce_weight=0.3, loss_chunk=24, mtp="freeze")
    student, lats, fp, ctl, head = T.build_student(a, torch.load(lay, weights_only=False))
    if kind == "dense_tied":
        assert head == "model.embed_tokens.weight" and lats[head].segments  # trained, fake-quantized
    sh = torch.load(setup["cache" if kind == "moe" else "cache_tied"] / "train_0000.pt")
    x, idx, val = sh["x"][:2].long(), sh["idx"][:2], sh["val"][:2]
    cap = {}
    for n, lat in lats.items():
        lat.update = lambda g, n=n: cap.__setitem__(n, g.clone())
    served = {n: lat.served().clone() for n, lat in lats.items()}
    T.train_step(student, lats, lats[head], fp, None, x, idx, val, a)
    trunk = {n for n in lats if not n.startswith("mtp.")}
    assert set(cap) == trunk and ctl.stats["undelivered"] == 0
    named = dict(student.named_parameters())
    fp_names = [n for n, p in named.items() if p.requires_grad and not n.startswith("mtp.")]
    ref, _ = load_text_causal_lm(a.model, "cpu", torch.float32)
    leaves = {n: t.clone().requires_grad_() for n, t in served.items() if n in trunk}
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
    import shutil

    run, out = WORK / "run_moe", WORK / "qad_moe.gguf"
    if run.exists():
        shutil.rmtree(run)  # this test's own scratch: a finished stage is immutable, so a rerun starts clean
    _run("omk_qad.train", *TRAIN, "--model", str(model), "--layout", str(lay), "--cache", str(setup["cache"]),
         "--out", str(run), "--steps", "2", "--eval-every", "2", "--save-every", "2")
    s1 = run / "stage1"
    log = (s1 / "log.txt").read_text()
    assert "LOSSLESS_GATE" in log and "STAGE_DONE stage 1" in log and "QAD_DONE" in log
    res = json.loads((s1 / "summary.json").read_text())["results"]
    assert [r["tag"] for r in res] == ["init", "step2"]
    imat = str(setup["cache"] / "imatrix.pt")
    assert torch.load(s1 / "final.pt", weights_only=False)["imatrix"] == imat  # recorded: export must reuse it
    _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device", "cpu",
         "--load", str(s1 / "final.pt"), "--gguf", str(out), "--latent-dtype", "fp16", "--latent-min", "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    assert rep["type_counts"]["IQ2_XXS"] == 4 * 6  # 4 layers x (gate/up/down exps + shexp)
    assert rep["with_imatrix"] > 0 and out.with_suffix(".imatrix.gguf").exists()

    # train == ship: the trained latents' served payloads (same imatrix) are byte-identical to the GGUF tensors
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp16", imatrix=imat),
                                             torch.load(lay, weights_only=False))
    T.load_state(str(s1 / "final.pt"), lats, student, None)
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


def test_layout_presets_overrides_and_refusals(setup):
    """Every rest preset keeps the fixed classes at IQ2_XXS; overrides apply; a pure llama.cpp mix is refused."""
    from omk_qad.layout import plan_layout, probe_model, summary

    model, _ = setup["moe"]
    probe = probe_model(model, LLAMA_CPP, WORK / "lay_presets")
    sizes = {}
    for rest in ("IQ2_M", "IQ3_S", "Q3_K_S", "bolt"):
        s = summary(plan_layout(probe, rest))
        assert set(s["per_class"]["routed_experts"]) == {"IQ2_XXS"}, rest
        assert set(s["per_class"]["shared_experts"]) == {"IQ2_XXS"}, rest
        sizes[rest] = s["sizes_mib"]["quant"]
    assert set(summary(plan_layout(probe, "bolt"))["per_class"]["attention"]) == {"Q8_0"}
    assert sizes["IQ2_M"] < sizes["bolt"]

    lay = plan_layout(probe, "IQ2_M", overrides={"attention": "Q6_K"}, regex={r"output\.weight": "Q8_0"})
    s = summary(lay)
    assert set(s["per_class"]["attention"]) == {"Q6_K"} and s["per_class"]["lm_head"] == {"Q8_0": 1}
    with pytest.raises(SystemExit, match=r"pure llama\.cpp mix"):
        plan_layout(probe, "IQ2_M", fixed={})
    with pytest.raises(SystemExit, match="both fixed and overridden"):
        plan_layout(probe, "IQ2_M", overrides={"routed_experts": "Q4_K"})


def test_mtp_layout_imatrix_and_export(setup):
    """MTP head: nextn tensors planned + trained-mapped; imatrix covers trunk, experts and MTP; GGUF ships it."""
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    model, lay_path = setup["moe_mtp"]
    layout = torch.load(lay_path, weights_only=False)
    assert layout["mtp"] and layout["mtp_type"] == "Q4_K"
    n_layer = 4
    nextn = {g: t for g, t in layout["targets"].items() if g.startswith(f"blk.{n_layer}.")}
    assert {f"blk.{n_layer}.nextn.{k}.weight" for k in ("eh_proj", "enorm", "hnorm", "shared_head_norm")} <= set(nextn)
    assert sum(t["type"] == "Q4_K" for t in nextn.values()) >= 10  # eh_proj + attention + experts + shared expert
    for g, t in nextn.items():
        if t["type"] not in ("F32", "F16", "BF16"):  # quantized: proven mtp.* rows/cols (norms/router stay float)
            assert t["cls"] == "mtp" and t["mapping"] == "direct" and t["type"] == "Q4_K", g

    # imatrix: every quantized tensor's source has data; routed-expert rows = top_k x the tokens the layer saw
    rec = torch.load(setup["cache_mtp"] / "imatrix.pt", weights_only=False)
    e = rec["entries"]
    for g, t in layout["targets"].items():
        if t["type"] not in ("F32", "F16", "BF16") and t["cls"] != "embedding":
            assert t["src"] in e, g
    top_k = 2
    for pre in [f"model.layers.{i}." for i in range(n_layer)] + ["mtp.layers.0."]:
        tokens = int(e[pre + "mlp.shared_expert.gate_proj.weight"]["counts"].sum())
        assert tokens > 0, pre
        for w in ("gate_up_proj", "down_proj"):
            c = e[pre + f"mlp.experts.{w}"]["counts"]
            assert int(c.sum()) == top_k * tokens and c.shape[0] == 8, (pre, w)
    imat_gguf = setup["cache_mtp"] / "imatrix.gguf"
    names = {t.name for t in gguf.GGUFReader(imat_gguf).tensors}
    assert f"blk.{n_layer}.ffn_down_exps.weight.in_sum2" in names
    assert f"blk.{n_layer}.nextn.eh_proj.weight.counts" in names

    out = WORK / "ptq_moe_mtp.gguf"
    _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay_path), "--device", "cpu",
         "--gguf", str(out), "--latent-min", "65536", "--imatrix", str(setup["cache_mtp"] / "imatrix.pt"))
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    quantized = {n for n, t in rep["tensors"].items() if t["how"] != "float"}
    weighted = {n for n in quantized if rep["tensors"][n]["imatrix"]}
    assert weighted == quantized - {"token_embd.weight"}  # a lookup table has no input activations
    r = gguf.GGUFReader(out)
    kv = {f.name: f.contents() for f in r.fields.values()}
    assert kv["qwen35moe.nextn_predict_layers"] == 1 and kv["qwen35moe.block_count"] == n_layer + 1
    assert out.with_suffix(".imatrix.gguf").exists() and out.with_suffix(".imatrix.pt").exists()

    # llama.cpp consumes both artifacts: the model loads, and our imatrix.gguf drives an IQ2_XXS requantize
    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(out), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]
    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-quantize"), "--allow-requantize", "--imatrix",
         str(out.with_suffix(".imatrix.gguf")), str(out), str(WORK / "requant_iq2xxs.gguf"), "IQ2_XXS", "8"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]


def _train(model: Path, lay: Path, cache: Path, out: Path, *extra: str, check: bool = True):
    cmd = [sys.executable, "-m", "omk_qad.train", *TRAIN, "--model", str(model), "--layout", str(lay),
           "--cache", str(cache), "--out", str(out), "--steps", "3", "--eval-every", "3", "--save-every", "1", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=CPU_ENV, check=False)
    if check:
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    return proc


def test_pause_resume_is_bitwise_and_stage2_branches(setup):
    """PAUSE mid-run + relaunch == straight run (bitwise); finished stages are immutable; stage-2 branches."""
    import hashlib
    import shutil

    model, lay = setup["moe"]
    a, b = WORK / "run_straight", WORK / "run_paused"
    for d in (a, b):
        if d.exists():
            shutil.rmtree(d)  # test scratch of this test only
    _train(model, lay, setup["cache"], a)
    (b / "stage1").mkdir(parents=True)
    (b / "stage1" / "PAUSE").touch()
    p = _train(model, lay, setup["cache"], b)
    assert "PAUSED at step 1" in p.stdout + p.stderr and not (b / "stage1" / "final.pt").exists()
    assert not (b / "stage1" / "PAUSE").exists()  # consumed: the relaunch is not paused again
    p = _train(model, lay, setup["cache"], b)
    assert "RESUME from step 1" in p.stdout + p.stderr
    fa = torch.load(a / "stage1" / "final.pt", weights_only=False)
    fb = torch.load(b / "stage1" / "final.pt", weights_only=False)
    assert fa["step"] == fb["step"] == 3
    for k in ("latents", "fp"):
        assert fa[k].keys() == fb[k].keys()
        assert all(torch.equal(fa[k][n], fb[k][n]) for n in fa[k]), k

    p = _train(model, lay, setup["cache"], a, check=False)
    assert p.returncode != 0 and "immutable" in p.stdout + p.stderr

    def sha():
        return hashlib.sha256((a / "stage1" / "final.pt").read_bytes()).hexdigest()

    before = sha()
    _train(model, lay, setup["cache"], a, "--stage", "2", "--branch", "kl", "--loss", "kl")
    meta = json.loads((a / "stage2" / "kl" / "final.json").read_text())
    assert meta["stage"] == 2 and meta["init_from"] == str(a / "stage1" / "final.pt")
    assert sha() == before
    p = _train(model, lay, setup["cache"], a, "--stage", "2", check=False)
    assert p.returncode != 0 and "--branch" in p.stdout + p.stderr


def test_refresh_window_defers_encodes_and_resumes_bitwise(setup):
    """--refresh / --ggml-refresh N: fewer encodes than every update; PAUSE + relaunch == straight run (bitwise)."""
    import shutil

    model, lay = setup["moe"]
    every, win, paused = WORK / "run_refresh1", WORK / "run_refresh2", WORK / "run_refresh2_paused"
    for d in (every, win, paused):
        if d.exists():
            shutil.rmtree(d)  # test scratch of this test only
    w2 = ("--refresh", "2", "--ggml-refresh", "2")
    _train(model, lay, setup["cache"], every)
    _train(model, lay, setup["cache"], win, *w2)
    (paused / "stage1").mkdir(parents=True)
    (paused / "stage1" / "PAUSE").touch()
    p = _train(model, lay, setup["cache"], paused, *w2)
    assert "PAUSED at step 1" in p.stdout + p.stderr
    p = _train(model, lay, setup["cache"], paused, *w2)
    assert "RESUME from step 1" in p.stdout + p.stderr
    fe, fw, fp = (torch.load(d / "stage1" / "final.pt", weights_only=False) for d in (every, win, paused))
    assert fw["stats"]["encodes"] < fe["stats"]["encodes"], (fw["stats"], fe["stats"])
    assert fw["payloads"] and not fe["payloads"]  # carried only when a window can serve a stale payload
    for k in ("latents", "fp"):
        assert all(torch.equal(fw[k][n], fp[k][n]) for n in fw[k]), k


def test_touch_drops_cached_payloads():
    """A write outside the optimizer is never served from a refresh window's cache."""
    from omk_qad import fakequant as fq

    ctl = fq.Control(refresh=100, ggml_refresh=100)
    lat = fq.Latent("w", torch.nn.Parameter(torch.randn(8, 16)), [], ctl, None, torch.float32)
    lat._cache = {0: (lat.version, 0, torch.zeros(4, dtype=torch.uint8))}
    state = lat.cache_state()
    lat.touch()
    assert lat._cache == {}
    lat.load_cache(state)
    assert lat._cache[0][0] == lat.version and lat._cache[0][1] == 0


def test_tied_train_export_verify_load(setup):
    """Tied embeddings + MTP (Qwen3.5-4B shape) train, export and load.

    The head trains fake-quantized with its imatrix and ships as token_embd only; the GGUF holds exactly what
    training served; llama.cpp loads it.
    """
    import shutil

    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    from omk_qad import train as T

    model, lay = setup["dense_tied"]
    cache = setup["cache_tied"]
    imat = torch.load(cache / "imatrix.pt", weights_only=False)
    e = imat["entries"]["model.embed_tokens.weight"]  # the head's input (final hidden state), named by identity
    assert int(e["counts"].sum()) == imat["chunk_count"] * (imat["chunk_size"] - 1)

    run, out = WORK / "run_tied", WORK / "qad_tied.gguf"
    if run.exists():
        shutil.rmtree(run)  # this test's own scratch
    _run("omk_qad.train", *TRAIN, "--model", str(model), "--layout", str(lay), "--cache", str(cache),
         "--out", str(run), "--steps", "2", "--eval-every", "2", "--save-every", "2")
    final = run / "stage1" / "final.pt"
    _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(lay), "--device", "cpu",
         "--load", str(final), "--gguf", str(out), "--latent-dtype", "fp16", "--latent-min", "65536")
    rep = json.loads(out.with_suffix(".export.json").read_text())
    assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
    quantized = {n for n, t in rep["tensors"].items() if t["how"] != "float"}
    assert "token_embd.weight" in quantized and "output.weight" not in rep["tensors"]
    assert {n for n in quantized if rep["tensors"][n]["imatrix"]} == quantized  # tied head included

    student, lats, _, _, head = T.build_student(_args(model, latent_dtype="fp16", imatrix=str(cache / "imatrix.pt")),
                                                torch.load(lay, weights_only=False))
    T.load_state(str(final), lats, student, None)
    tensors = {t.name: t for t in gguf.GGUFReader(out).tensors}
    for lat in lats.values():
        for g, chunks in lat.payloads().items():
            want = np.concatenate([_flat(c) for c in chunks])
            assert np.array_equal(np.asarray(tensors[g].data).reshape(-1), want), g
    assert lats[head].payloads().keys() == {"token_embd.weight"}

    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(out), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]


# --- ternary student (plan M2): PQ2_0 latents in the Hadamard-folded basis ------------------------------------------
TERNARY_CLASSES = ("attention", "linear_attn", "dense_mlp", "lm_head")  # bonsai4b's set (tied embedding = head)


def _ternary_layout(lay_path: Path, out: Path, qtype: str = "PQ2_0") -> Path:
    """The IQ2_M-rest layout with TERNARY_CLASSES retyped to a ternary type (one model-wide Hadamard block)."""
    from omk_qad.layout import retype

    torch.save(retype(torch.load(lay_path, weights_only=False), dict.fromkeys(TERNARY_CLASSES, qtype)), out)
    return out


def test_hadamard_basis():
    from omk_qad.ternary import HadamardBasis, hadamard_block, model_block

    assert (hadamard_block(2560), hadamard_block(9216), hadamard_block(256)) == (512, 1024, 256)
    assert model_block([2560, 9216]) == 512 and model_block([5120, 17408]) == 1024  # Qwen3.5-4B / 27B: one block
    with pytest.raises(ValueError):
        model_block([2560, 2688])
    for width in (256, 512, 2560):
        b = HadamardBasis(width)
        w = torch.randn(5, width, generator=torch.Generator().manual_seed(width))
        assert torch.allclose(b.unfold(b.fold(w)), w, atol=1e-5)
        assert torch.allclose(b.fold(w).norm(dim=-1), w.norm(dim=-1), rtol=1e-5)  # orthogonal


def test_ternary_student_serves_ternary_with_exact_gradients(setup):
    """Served = unfold(ternary); bypass = the original weights; the folded-latent gradient == fold(autograd)."""
    from omk_qad import train as T
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.ternary import decode, unpack

    model, lay = setup["dense_tied"]
    tl = _ternary_layout(lay, WORK / "layout_tied_ternary.pt")
    a = _args(model, ce_weight=0.3, loss_chunk=24, mtp="freeze", latent_dtype="fp32")
    student, lats, fp, ctl, head = T.build_student(a, torch.load(tl, weights_only=False))
    tern = {n for n, lat in lats.items() if lat.basis is not None}
    assert head in tern and any(".mlp." in n for n in tern) and any(".linear_attn." in n for n in tern)
    ref, _ = load_text_causal_lm(a.model, "cpu", torch.float32)
    ref_w = dict(ref.named_parameters())

    for n in tern:
        lat = lats[n]
        ctl.bypass = True
        assert torch.allclose(lat.served(), ref_w[n].detach(), atol=1e-5), n  # fold is lossless
        ctl.bypass = False
        served = lat.served().reshape(lat.rows, lat.cols)
        lat.payloads()
        q = torch.zeros(lat.rows, lat.cols)  # folded ternary values placed at their latent rows (GGUF may reorder)
        nnz, k = [], 0
        for seg in lat.segments:
            for rows, _, _ in seg.chunks:
                t, d = unpack("PQ2_0", lat._cache[k][2], lat.cols)
                q[rows] = decode(t, d)
                nnz.append(t.reshape(t.shape[0], -1, 128).ne(0).sum(-1).reshape(-1))
                k += 1
        nnz = torch.cat(nnz)
        # PrismML's top-86 in the folded basis; bf16 source weights can tie at the threshold (ties are kept)
        assert bool((nnz >= 86).all()) and float((nnz == 86).float().mean()) > 0.9, n
        assert torch.allclose(lat._to_latent(served), q, atol=1e-5 * float(q.abs().max())), n

    sh = torch.load(setup["cache_tied"] / "train_0000.pt")
    x, idx, val = sh["x"][:2].long(), sh["idx"][:2], sh["val"][:2]
    cap = {}
    for n, lat in lats.items():
        lat.update = lambda g, n=n: cap.__setitem__(n, g.clone())
    trunk = {n for n in lats if not n.startswith("mtp.")}
    served = {n: lats[n].served().clone() for n in trunk}
    T.train_step(student, lats, lats[head], fp, None, x, idx, val, a)
    assert set(cap) == trunk and ctl.stats["undelivered"] == 0
    named = dict(student.named_parameters())
    fp_names = [n for n, p in named.items() if p.requires_grad and not n.startswith("mtp.")]
    leaves = {n: t.clone().requires_grad_() for n, t in served.items()}
    fpl = {n: named[n].detach().clone().requires_grad_() for n in fp_names}
    lg = torch.func.functional_call(ref, {**leaves, **fpl}, (x,)).logits[:, :-1].float()
    j, _, ce, _ = T.token_terms(lg.reshape(-1, lg.shape[-1]), idx.reshape(-1, idx.shape[-1]),
                               val.reshape(-1, val.shape[-1]), x[:, 1:].reshape(-1))
    ((j.sum() + a.ce_weight * ce.sum()) / j.numel()).backward()
    for n, g in cap.items():
        lat = lats[n]
        r = leaves[n].grad
        if lat.basis is not None:  # the update sees the folded gradient
            r = lat._to_latent(r.reshape(lat.rows, lat.cols)).reshape(lat.shape)
        assert r.norm() > 0, n
        assert (g - r).norm() / r.norm() < 1e-4, n


def test_ternary_student_descends_on_one_batch(setup):
    """Started away from the teacher (seeded noise on the folded latents), the ternary student descends.

    The sharp tied teacher is saturated on these rows (top-1 p >= 0.9998), so the unperturbed ternary student already
    sits at KL ~6e-5, the IQ2 student's floor: there is nothing to descend from (bug-815).
    """
    from omk_qad import train as T

    model, lay = setup["dense_tied"]
    tl = _ternary_layout(lay, WORK / "layout_tied_ternary.pt")
    a = _args(model, loss_chunk=64, loss="kl", mtp="freeze")
    student, lats, fp, ctl, head = T.build_student(a, torch.load(tl, weights_only=False))
    g = torch.Generator().manual_seed(0)
    for lat in lats.values():
        if lat.basis is not None:
            p = lat.param.data
            p.add_(p.std() * torch.randn(p.shape, generator=g))
            lat.touch()
    opt = torch.optim.AdamW(fp, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    sh = torch.load(setup["cache_tied"] / "train_0000.pt")
    x, idx, val = sh["x"][:1].long(), sh["idx"][:1], sh["val"][:1]
    ctl.lr = 3e-4
    hist = []
    for s in range(1, 16):
        ctl.step = s
        hist.append(T.train_step(student, lats, lats[head], fp, opt, x, idx, val, a))
    assert hist[0] > 0.5, hist  # the start really is away from the teacher (a zero needs a nonzero floor)
    assert np.mean(hist[-3:]) < 0.1 * hist[0], hist


def test_ternary_layout_plan_and_refusals(setup):
    """--fixed / --override ternary: planned through an IQ2_XXS placeholder, sized at 34 B / 128, one block.

    Weights the runtime does not fold and regex-set ternary types are refused.
    """
    from omk_qad.layout import plan_layout, probe_model, summary

    model, _ = setup["dense_tied"]
    probe = probe_model(model, LLAMA_CPP, WORK / "lay_ternary")
    iq2, tern = plan_layout(probe, "IQ2_M"), plan_layout(probe, "IQ2_M", fixed={"dense_mlp": "PQ2_0"})
    s = summary(tern)
    assert set(s["per_class"]["dense_mlp"]) == {"PQ2_0"} and s["hadamard_block"] == 256
    assert summary(iq2)["hadamard_block"] is None
    n = sum(int(np.prod(t["shape"])) for t in tern["targets"].values() if t["type"] == "PQ2_0")
    want = iq2["sizes_mib"]["quant"] + n * (34 / 128 - 66 / 256) / 2**20  # same rest plan, other bytes per weight
    assert abs(tern["sizes_mib"]["quant"] - want) < 0.02
    with pytest.raises(SystemExit, match="does not fold"):
        plan_layout(probe, "IQ2_M", overrides={"small_proj": "PQ2_0"})
    with pytest.raises(SystemExit, match="not by --override-regex"):
        plan_layout(probe, "IQ2_M", regex={r".*ffn_up.*": "PQ2_0"})


def test_ternary_train_export_ships_the_trained_trits(setup):
    """Ternary train -> GGUF: types 142 / 143, prism.hadamard KV, bytes == training payloads, both formats agree.

    The unfold control (omk_qad.ternary_gate) == the student's served weights, so an engine that matches its
    perplexity on the ternary file serves what training served; stock llama.cpp loads the control. The flip
    control only negates the signs.
    """
    import shutil

    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    import gguf

    from omk_qad import train as T
    from omk_qad.ternary import decode, register_gguf_types, seeded_signs, unpack
    from omk_qad.ternary_gate import build

    register_gguf_types(gguf)
    model, lay = setup["dense_tied"]
    tl = _ternary_layout(lay, WORK / "layout_tied_ternary.pt")
    tl1 = _ternary_layout(lay, WORK / "layout_tied_ptq1.pt", "PTQ1_0")
    run = WORK / "run_tied_ternary"
    if run.exists():
        shutil.rmtree(run)  # this test's own scratch
    out = _run("omk_qad.train", *TRAIN, "--model", str(model), "--layout", str(tl), "--cache",
               str(setup["cache_tied"]), "--out", str(run), "--steps", "2", "--eval-every", "2", "--save-every", "2",
               "--lr", "6e-5", "--loss", "kl")
    log = (run / "stage1" / "log.txt").read_text()
    assert "LOSSLESS_GATE" in log and "STAGE_DONE" in log and "ternary" in out + log
    final = run / "stage1" / "final.pt"
    files = {}
    for fmt, layout in (("PQ2_0", tl), ("PTQ1_0", tl1)):
        files[fmt] = WORK / f"tied_{fmt.lower()}.gguf"
        _run("omk_qad.train", "--phase", "export", "--model", str(model), "--layout", str(layout), "--device", "cpu",
             "--load", str(final), "--gguf", str(files[fmt]), "--latent-dtype", "fp16", "--latent-min", "65536")
        rep = json.loads(files[fmt].with_suffix(".export.json").read_text())
        assert rep["verify_failures"] == [] and rep["type_mismatches"] == [] and rep["export_only"] == []
        assert rep["type_counts"][fmt] > 0

    reader = gguf.GGUFReader(files["PQ2_0"])
    kv = {k[len("prism.hadamard."):]: f.contents() for k, f in reader.fields.items() if k.startswith("prism.")}
    tern = sorted(t.name for t in reader.tensors if t.tensor_type == 142)
    assert kv["version"] == 2 and kv["tied_output"] and kv["block_size"] == 256 and kv["sign_mode"] == "explicit"
    assert kv["inverse_weight_names"] == ["token_embd.weight"] and "output.weight" not in tern
    assert sorted(kv["weight_names"] + kv["inverse_weight_names"]) == tern
    assert kv["sign_values"] == [int(v) for w in kv["sign_widths"] for v in seeded_signs(w).tolist()]

    a = _args(model, latent_dtype="fp16", loss="kl", imatrix=str(setup["cache_tied"] / "imatrix.pt"))
    student, lats, _, _, _ = T.build_student(a, torch.load(tl, weights_only=False))
    T.load_state(str(final), lats, student, None, weights_only=True)
    tensors = {t.name: t for t in reader.tensors}
    ptq1 = {t.name: t for t in gguf.GGUFReader(files["PTQ1_0"]).tensors}
    by_gguf = {}
    for n, lat in lats.items():
        if lat.basis is not None:
            for seg in lat.segments:
                by_gguf.setdefault(seg.gguf, []).append((n, seg))
    assert sorted(by_gguf) == tern
    for g, segs in by_gguf.items():
        assert len({n for n, _ in segs}) == 1, g
        lat = lats[segs[0][0]]
        want = np.concatenate([_flat(c) for c in lat.payloads()[g]])  # encode cache: PQ2_0 bytes per chunk
        assert np.array_equal(np.asarray(tensors[g].data).reshape(-1), want), g
        n_in = int(tensors[g].shape[0])
        t2, d2 = unpack("PQ2_0", np.asarray(tensors[g].data), n_in)
        t1, d1 = unpack("PTQ1_0", np.asarray(ptq1[g].data), n_in)
        assert torch.equal(t1, t2) and torch.equal(d1, d2), g  # train once, ship either packing
        assert torch.equal(decode(t2, d2), decode(t1, d1)), g

    unfold, flip = WORK / "tied_pq2_0.unfold.gguf", WORK / "tied_pq2_0.flip.gguf"
    assert build("unfold", files["PQ2_0"], unfold, LLAMA_CPP)["unfolded"] == len(tern)
    build("flip", files["PQ2_0"], flip, LLAMA_CPP)
    un = {t.name: t for t in gguf.GGUFReader(unfold).tensors}
    assert not any(k.startswith("prism.") for k in gguf.GGUFReader(unfold).fields)
    for g, segs in by_gguf.items():
        lat = lats[segs[0][0]]
        served = lat.served().reshape(lat.rows, lat.cols).float()
        if lat.colperm is not None:
            served = served[:, lat.colperm]
        want = torch.cat([served[rows] for _, seg in segs for rows, _, _ in seg.chunks]).numpy()
        got = np.asarray(un[g].data).reshape(want.shape)
        assert np.allclose(got, want, rtol=0, atol=1e-6 * float(np.abs(want).max())), g
    fl = gguf.GGUFReader(flip)
    assert [-v for v in kv["sign_values"]] == fl.fields["prism.hadamard.sign_values"].contents()
    for t in fl.tensors:
        assert np.array_equal(np.asarray(t.data), np.asarray(tensors[t.name].data)), t.name

    proc = subprocess.run(
        [str(LLAMA_CPP / "build/bin/llama-perplexity"), "-m", str(unfold), "-f", str(setup["text"]), "-c", "128",
         "--chunks", "2", "-t", "8", "-ngl", "0"],
        capture_output=True, text=True, env=CPU_ENV, check=False,
    )
    assert proc.returncode == 0 and "Final estimate" in proc.stdout + proc.stderr, proc.stderr[-2000:]


def test_ternary_stage2_branch_from_iq2_stage1(setup):
    """A model-basis stage-1 checkpoint seeds a ternary stage-2 branch by folding; a basis mix is refused."""
    import shutil

    from omk_qad import train as T

    model, lay = setup["dense_tied"]
    tl = _ternary_layout(lay, WORK / "layout_tied_ternary.pt")
    run = WORK / "run_iq2_to_ternary"
    if run.exists():
        shutil.rmtree(run)  # this test's own scratch
    common = ("--model", str(model), "--cache", str(setup["cache_tied"]), "--out", str(run), "--eval-every", "2",
              "--save-every", "2", "--steps", "2")
    _run("omk_qad.train", *TRAIN, *common, "--layout", str(lay))
    _run("omk_qad.train", *TRAIN, *common, "--layout", str(tl), "--stage", "2", "--branch", "ternary", "--loss", "kl")
    assert (run / "stage2" / "ternary" / "final.pt").exists()

    # the conversion itself: stage-1 latents folded into the ternary student == the stage-1 weights
    s1 = torch.load(run / "stage1" / "final.pt", weights_only=False)
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp32"), torch.load(tl, weights_only=False))
    T.load_state(str(run / "stage1" / "final.pt"), lats, student, None, weights_only=True)
    for n, lat in lats.items():
        if lat.basis is not None:
            assert torch.allclose(lat.model_weight(), s1["latents"][n].float(), atol=1e-5), n

    # a ternary checkpoint cannot be resumed / loaded into a model-basis (IQ2) student
    student, lats, _, _, _ = T.build_student(_args(model, latent_dtype="fp16"), torch.load(lay, weights_only=False))
    with pytest.raises(SystemExit, match="stored folded"):
        T.load_state(str(run / "stage2" / "ternary" / "final.pt"), lats, student, None, weights_only=True)


# --- folded-basis imatrix (plan M3) -------------------------------------------------------------------------------
def test_folded_imatrix_is_the_folded_statistic(setup):
    """Collector's folded sums == sum((H s x[:, perm])^2) of the hooked inputs; trace preserved; not a reorder."""
    from omk_qad.imatrix import Collector, folded_candidates
    from omk_qad.ptq import load_text_causal_lm

    model, lay = setup["dense_tied"]
    layout = torch.load(lay, weights_only=False)
    cand, block = folded_candidates(layout)
    assert block == 256  # tiny widths 256 / 512: the one model-wide block
    out_proj = next(n for n in cand if n.endswith("linear_attn.out_proj.weight"))
    assert cand[out_proj] is not None  # ssm_out: the converter's V-head column reorder is carried
    m, _ = load_text_causal_lm(str(model), "cpu", torch.float32)
    m.requires_grad_(False)
    seen = {}
    mod = m.get_submodule(out_proj.rsplit(".", 1)[0])
    h = mod.register_forward_pre_hook(lambda _m, args: seen.setdefault("x", []).append(args[0].detach().clone()))
    col = Collector(m, folded=cand, block=block)
    x = torch.load(setup["cache_tied"] / "train_0000.pt")["x"][:2].long()
    with torch.no_grad():
        m(input_ids=x)
    col.close()
    h.remove()
    st, fst = col.state(), col.folded_state()
    xin = torch.cat(seen["x"]).reshape(-1, mod.in_features).float()
    want = col.basis(xin.shape[-1]).fold(xin[:, cand[out_proj]]).double().square().sum(0)
    assert torch.allclose(fst[out_proj]["sums"][0], want, rtol=1e-5)
    for n in fst:  # orthogonal fold: the total energy per expert is the same in both bases
        assert torch.allclose(fst[n]["sums"].sum(-1), st[n]["sums"].sum(-1), rtol=1e-5), n
    model_diag = st[out_proj]["sums"][0][cand[out_proj]]
    assert not torch.allclose(fst[out_proj]["sums"][0].sort().values, model_diag.sort().values, rtol=1e-2)


def test_ternary_student_uses_the_folded_imatrix(setup):
    from omk_qad import train as T

    model, lay = setup["dense_tied"]
    tl = _ternary_layout(lay, WORK / "layout_tied_ternary.pt")
    cache = setup["cache_tied"]
    rec = torch.load(cache / "imatrix.folded.pt", weights_only=False)
    assert rec["basis"] == "seeded" and rec["entries"]
    layout = torch.load(tl, weights_only=False)
    _, plain, _, _, _ = T.build_student(_args(model, ternary_method="opt"), layout)
    _, weighted, _, _, _ = T.build_student(_args(model, ternary_method="opt", imatrix=str(cache / "imatrix.pt")),
                                           layout)
    tern = [n for n, lat in weighted.items() if lat.basis is not None]
    assert tern and all(weighted[n].msq is not None for n in tern)
    assert all(weighted[n].msq.shape[-1] == weighted[n].cols for n in tern)
    changed = sum(not torch.equal(weighted[n].served(), plain[n].served()) for n in tern)
    assert changed == len(tern)  # importance moves the weighted-L2 optimum of every ternary weight


# --- separate students over one shared teacher (plan M5) -----------------------------------------------------------
def _multi_spec(model: Path, out: Path, text: Path, evals: str) -> dict:
    common = {"steps": 2, "bs": 2, "warmup": 1, "eval_every": 2, "save_every": 2, "latent_dtype": "fp16",
              "latent_min": 65536, "loss_chunk": 32}
    return {"model": str(model), "out": str(out), "llama_cpp": str(LLAMA_CPP), "layout": {"rest": "IQ2_M"},
            "cache": {"data": str(text), "eval_data": evals, "n_seq": 8, "n_eval": 1, "seq": 64, "shard": 4},
            "preflight_steps": 2, "references": {"elsewhere": {"note": "another metric"}},
            "students": [
                {"name": "iq2", "gpu": None, "layout": {"fixed": {"dense_mlp": "iq2_xxs"}},
                 "train": {**common, "lr": 1e-3, "loss": "jaccard"}},
                {"name": "tern", "gpu": None,
                 "layout": {"fixed": {"dense_mlp": "pq2_0"},
                            "override": {"attention": "pq2_0", "linear_attn": "pq2_0", "lm_head": "pq2_0"}},
                 "train": {**common, "lr": 6e-5, "loss": "kl"}},
            ]}


def test_multi_students_share_only_the_teacher(setup):
    """An IQ2 and a ternary student from one spec share only the teacher.

    One probe, one cache + imatrix; two concurrent processes with their own hyper-parameters, layouts, checkpoints
    and GGUFs; one report over the same eval rows.
    """
    import shutil

    from omk_qad import multi

    model, _ = setup["dense_tied"]
    out = WORK / "multi"
    if out.exists():
        shutil.rmtree(out)  # this test's own scratch
    out.mkdir(parents=True)
    words = setup["text"].read_text().split()
    evals = []
    for k, part in enumerate((words[: len(words) // 2], words[len(words) // 2 :])):  # two eval families
        evals.append(out / f"eval_{'ab'[k]}.txt")
        evals[-1].write_text(" ".join(part))
    spec = _multi_spec(model, out, setup["text"], ",".join(map(str, evals)))
    sp = out / "spec.json"
    sp.write_text(json.dumps(spec))
    _run("omk_qad.multi", "all", "--spec", str(sp))

    assert (out / "cache" / "cache.log").read_text().count("CACHE_DONE") == 1
    assert (out / "cache" / "imatrix.folded.pt").exists()
    sums = {n: json.loads((out / n / "stage1" / "summary.json").read_text()) for n in ("iq2", "tern")}
    assert {sm["origin"]["cache"] for sm in sums.values()} == {str(out / "cache")}  # the same teacher rows
    assert (sums["iq2"]["args"]["lr"], sums["iq2"]["args"]["loss"]) == (1e-3, "jaccard")  # each its own recipe
    assert (sums["tern"]["args"]["lr"], sums["tern"]["args"]["loss"]) == (6e-5, "kl")
    assert all((out / n / "stage1" / "final.pt").exists() for n in ("iq2", "tern"))

    st = json.loads((out / "multi.json").read_text())["train"]
    assert st["iq2"]["pid"] != st["tern"]["pid"]
    assert st["iq2"]["started"] < st["tern"]["ended"] and st["tern"]["started"] < st["iq2"]["ended"]  # concurrent

    lay = {n: torch.load(out / n / "layout.pt", weights_only=False) for n in ("iq2", "tern")}
    assert lay["iq2"].get("hadamard_block") is None and lay["tern"]["hadamard_block"] == 256
    rep = {n: json.loads((out / n / f"{n}.export.json").read_text()) for n in ("iq2", "tern")}
    assert "IQ2_XXS" in rep["iq2"]["type_counts"] and "PQ2_0" not in rep["iq2"]["type_counts"]
    assert rep["tern"]["type_counts"]["PQ2_0"] > 0
    assert all(r["verify_failures"] == [] and r["type_mismatches"] == [] for r in rep.values())
    report = json.loads((out / "report.json").read_text())
    assert report["same_eval_rows"] and [r["student"] for r in report["rows"]] == ["iq2", "tern"]
    assert all(r["status"] == "final" and r["klk"] is not None for r in report["rows"])
    for r in report["rows"]:  # one row per family, equal token counts: the overall KL is their mean
        assert r["klk"] == pytest.approx((r["klk:eval_a"] + r["klk:eval_b"]) / 2, rel=1e-6)
    md = (out / "report.md").read_text()
    assert "klk:eval_a" in md and "elsewhere" in md and "elsewhere" not in md.split("References")[0]

    again = _run("omk_qad.multi", "train", "--spec", str(sp))  # finished students are reported, not relaunched
    assert again.count("not relaunched") == 2

    pre = _run("omk_qad.multi", "preflight", "--spec", str(sp), "--students", "tern")
    pf = json.loads((out / "multi.json").read_text())["preflight"]["tern"]
    assert pf["reserved_gib"] == 0.0 and "no GPU is shared" in pre
    assert not (out / "tern" / "stage1" / "last.pt").exists() or (out / "tern" / "stage1" / "final.pt").exists()
    assert (out / "_preflight" / "tern" / "stage1" / "final.pt").exists()  # its own scratch run, not the student's

    changed = json.loads(sp.read_text())
    changed["students"][0]["layout"] = {"fixed": {"dense_mlp": "iq2_xs"}}
    with pytest.raises(SystemExit, match="differs from its existing"):
        multi.phase_plan(changed, changed["students"][:1])

    other = WORK / "multi_block"  # a ternary student folding with another block than the shared folded imatrix
    if other.exists():
        shutil.rmtree(other)
    s2 = {**spec, "out": str(other)}
    for n in ("iq2", "tern"):
        (other / n).mkdir(parents=True)
        lay2 = dict(lay[n])
        if n == "tern":
            lay2["hadamard_block"] = 512
        torch.save(lay2, other / n / "layout.pt")
    with pytest.raises(SystemExit, match="folds with block 512"):
        multi.check_blocks(s2)

    gpu = {**spec, "students": [{**s, "gpu": 0} for s in spec["students"]]}  # sharing a GPU needs a measured peak
    with pytest.raises(SystemExit, match="run 'preflight' first"):
        multi.fit(gpu, gpu["students"], {})


def test_big_matrix_update_is_the_whole_matrix_step(monkeypatch):
    """A matrix above CHUNK_ELEMS updates in row chunks (bounded memory) and lands where the whole-matrix step does."""
    from omk_qad import fakequant as fq

    torch.manual_seed(0)
    w = torch.randn(300, 64)
    grads = [torch.randn(300, 64) for _ in range(3)]
    grads[1][17:40] = 0  # rows without gradient keep their statistics apart from the decay

    def run(chunk: int) -> fq.Latent:
        monkeypatch.setattr(fq, "CHUNK_ELEMS", chunk)
        ctl = fq.Control(lr=1e-2, beta2=0.99)
        lat = fq.Latent("w", torch.nn.Parameter(w.clone()), [], ctl, None, torch.float32)
        for g in grads:
            lat.update(g.clone())
        return lat

    whole, big = run(10**9), run(1000)  # 1000 elements -> 15 rows per chunk, 20 chunks
    assert big.ctl.stats["updates"] == whole.ctl.stats["updates"] == 3
    for x, y in ((big.param.data, whole.param.data), (big.r, whole.r), (big.c, whole.c), (big.t, whole.t)):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
    assert not torch.equal(big.param.data, w)


def _train_spiked(monkeypatch, model, lay, cache, out, spike_at, *extra):
    """omk_qad.train in-process with the loss of step ``spike_at`` reported x100 (the updates stay real)."""
    from omk_qad import train as T

    real, calls = T.train_step, [0]

    def spiked(*a, **k):
        calls[0] += 1
        loss = real(*a, **k)
        return loss * 100 if calls[0] == spike_at else loss

    monkeypatch.setattr(T, "train_step", spiked)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    T.main([*TRAIN, "--model", str(model), "--layout", str(lay), "--cache", str(cache), "--out", str(out),
            "--steps", "3", "--eval-every", "3", "--save-every", "1", *extra])


def test_guard_rolls_back_to_the_last_healthy_state_and_cuts_the_lr(setup, monkeypatch):
    """bug-825: a diverging loss restores the snapshot bitwise, halves the LR scale and keeps it across a save."""
    import shutil

    model, lay = setup["moe"]
    ref, run = WORK / "run_guard_ref", WORK / "run_guard"
    for d in (ref, run):
        if d.exists():
            shutil.rmtree(d)  # test scratch of this test only
    _train(model, lay, setup["cache"], ref, "--stop-at", "1")
    guard = ("--guard-ratio", "2", "--guard-patience", "1", "--guard-every", "1", "--stop-at", "2")
    _train_spiked(monkeypatch, model, lay, setup["cache"], run, 2, *guard)
    log = (run / "stage1" / "log.txt").read_text()
    assert "GUARD_ROLLBACK step 2 -> state of step 1, LR scale 0.5" in log, log[-2000:]
    a = torch.load(ref / "stage1" / "last.pt", weights_only=False)
    b = torch.load(run / "stage1" / "last.pt", weights_only=False)
    assert a["step"] == 1 and b["step"] == 2 and b["guard"]["scale"] == 0.5 and b["guard"]["rollbacks"] == 1
    for k in ("latents", "fp"):
        assert all(torch.equal(a[k][n], b[k][n]) for n in a[k]), k  # step 2's update was undone exactly
    for n, (r, c, t) in a["adafactor"].items():
        r2, c2, t2 = b["adafactor"][n]
        assert torch.equal(r, r2) and torch.equal(c, c2) and torch.equal(t, t2), n


def test_guard_fails_loud_past_its_rollback_budget(setup, monkeypatch):
    import shutil

    model, lay = setup["moe"]
    run = WORK / "run_guard_fail"
    if run.exists():
        shutil.rmtree(run)  # test scratch of this test only
    with pytest.raises(SystemExit, match="GUARD_FAIL step 2"):
        _train_spiked(monkeypatch, model, lay, setup["cache"], run, 2, "--guard-ratio", "2", "--guard-patience", "1",
                      "--guard-every", "1", "--guard-max", "0")
