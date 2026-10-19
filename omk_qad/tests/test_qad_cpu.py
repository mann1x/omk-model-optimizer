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
    if not (cache / "cache.json").exists() or not (cache / "imatrix.pt").exists():
        _run("omk_qad.train", "--phase", "cache", "--model", str(model), "--cache", str(cache), "--layout", str(layout),
             "--device", "cpu", "--data", str(text), "--eval-data", str(text), "--n-seq", "8", "--n-eval", "2",
             "--seq", "64", "--shard", "4", "--llama-cpp", str(LLAMA_CPP))
    return cache


@pytest.fixture(scope="module")
def setup():
    """Sharp tiny MoE / dense / MoE+MTP teachers, their IQ2_M-rest layouts, small teacher caches with imatrix."""
    from omk_qad.layout import build_layout
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, kind, mtp in (("moe", "moe", False), ("dense", "dense", False), ("moe_mtp", "moe", True)):
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
