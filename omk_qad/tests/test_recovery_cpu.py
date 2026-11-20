# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of bf16 recovery mode (omk_qad.recovery) on a tiny Gemma-4 26B-A4B-shaped model.

Artifacts go to $OMK_QAD_WORK (persistent disk; /tmp is tmpfs on solidpc).
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
WORK = Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK)) / "recovery"
LLAMA_CPP = Path(os.environ.get("OMK_LLAMA_CPP", "/opt/llama.cpp"))
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"}
TRAIN = ("--phase", "train", "--mode", "bf16", "--device", "cpu", "--bs", "2", "--lr", "1e-3", "--warmup", "1",
         "--loss-chunk", "32", "--loss", "kl")
N_LAYERS, N_FULL = 6, 1  # tiny gemma4_moe: 5 sliding + 1 full-attention layer (no v_proj)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=CPU_ENV, check=False)
    if check:
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    return proc


@pytest.fixture(scope="module")
def g4():
    """Tiny Gemma-4 student (seed 0), a different-weights teacher of the same tokenizer (seed 1), text, caches."""
    from omk_qad.tiny import make_tiny

    WORK.mkdir(parents=True, exist_ok=True)
    student, teacher = WORK / "tiny_g4_s0", WORK / "tiny_g4_s1"
    for d, seed in ((student, 0), (teacher, 1)):
        if not (d / "config.json").exists():
            make_tiny("gemma4_moe", d, seed=seed)
    text = WORK / "text.txt"
    if not text.exists():
        text.write_text((LLAMA_CPP / "README.md").read_text()[:60000])
    caches = {}
    for name, model in (("teacher", teacher), ("self", student)):
        c = WORK / f"cache_{name}"
        if not (c / "cache.json").exists():
            _run("omk_qad.train", "--phase", "cache", "--model", str(model), "--cache", str(c), "--device", "cpu",
                 "--data", str(text), "--eval-data", str(text), "--n-seq", "16", "--n-eval", "2", "--seq", "64",
                 "--shard", "8", "--imatrix-seqs", "0")
        caches[f"cache_{name}"] = c
    return {"student": student, "teacher": teacher, "text": text, **caches}


def test_gemma4_census_router_and_scope(g4):
    from omk_qad.census import classify_model
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import scope_params

    m, _ = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    cls = {e.module: e.cls for e in classify_model(m).entries}
    assert {cls[f"model.layers.{i}.router.proj"] for i in range(N_LAYERS)} == {"router"}
    assert cls["model.layers.0.experts"] == "routed_experts"
    hit = scope_params(m, "gemma4")
    n_v = N_LAYERS - N_FULL
    assert len(hit) == N_LAYERS * 6 + n_v
    assert "model.layers.5.self_attn.v_proj.weight" not in dict(m.named_parameters())  # k = v on full layers
    assert not [n for n in hit if "router" in n or "experts" in n or "embed" in n or "norm" in n]
    with_router = scope_params(m, "gemma4", train_router=True)
    assert len(with_router) == len(hit) + N_LAYERS
    assert all(n.endswith("router.proj.weight") for n in set(with_router) - set(hit))


def test_gemma4_scope_refuses_another_architecture():
    from transformers import AutoModelForCausalLM

    from omk_qad.recovery import scope_params
    from omk_qad.tiny import make_tiny

    qwen = WORK / "tiny_qwen_dense"
    if not (qwen / "config.json").exists():
        make_tiny("dense", qwen)
    m = AutoModelForCausalLM.from_pretrained(qwen, dtype=torch.float32)
    with pytest.raises(SystemExit, match="REFUSE"):
        scope_params(m, "gemma4")


def test_softcapped_head_is_the_models_own_logits(g4):
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import softcap, softcap_of

    m, tok = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    assert softcap_of(m) == 30.0
    x = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids
    with torch.no_grad():
        ref = m(input_ids=x).logits
        h = m.get_decoder()(input_ids=x).last_hidden_state
        ours = softcap(F.linear(h, m.get_output_embeddings().weight), softcap_of(m))
        raw = F.linear(h, m.get_output_embeddings().weight)
    torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)
    assert raw.abs().max() > 30.0 or not torch.allclose(raw, ref, atol=1e-4)  # the uncapped head is not the model's


def test_fp16_storage_cuts_the_rounding_noise_of_a_sub_ulp_step():
    from omk_qad.fakequant import sr_cast

    torch.manual_seed(0)
    start, step, n = 1.0 + 2.0 ** -3, 1e-4, 100
    out = {}
    for dtype in (torch.bfloat16, torch.float16):
        w = torch.full((400_000,), start)
        assert torch.equal(w, w.to(dtype).float())
        for _ in range(n):
            w = sr_cast(w - step, dtype).float()
        out[dtype] = w - start
    for moved in out.values():  # both unbiased: the rounding keeps the intended movement on average
        assert abs(moved.mean().item() + n * step) < 2e-4
    noise = out[torch.float16].std() / out[torch.bfloat16].std()
    assert 0.3 < noise < 0.4  # ulp 8x finer, noise sqrt(8)x lower


def test_bf16_mode_stores_fp16_serves_bf16_and_refuses_a_storage_change(g4):
    out = WORK / "fp16_storage"
    shutil.rmtree(out, ignore_errors=True)
    args = ("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
            "--save-every", "1", "--steps", "4", "--stop-at", "2", "--out", str(out))
    log = _run(*args).stdout
    assert "stored fp16" in log
    last = out / "stage1" / "last.pt"
    latents = torch.load(last, map_location="cpu", weights_only=False)["latents"]
    assert {t.dtype for t in latents.values()} == {torch.float16}
    bad = _run(*args, "--bf16-latent", "bf16", check=False)
    assert bad.returncode != 0 and "bf16_latent" in bad.stderr
    bad = _run(*args, "--bf16-latent", "bf16", "--allow-change", "bf16_latent", check=False)  # no identity
    assert bad.returncode != 0 and "latents stored as torch.float16" in bad.stdout + bad.stderr


def _ckpt(out: Path) -> dict:
    return torch.load(out / "stage1" / "final.pt", map_location="cpu", weights_only=False)


def test_bf16_mode_trains_only_the_scope_and_descends(g4):
    out = WORK / "run_teacher"
    # this test's own scratch: a finished run from an earlier session is refused ("is finished and immutable") and
    # tolerating that made the test read a 2026-10-23 checkpoint with bf16 latents (bug-977, 2026-11-19)
    shutil.rmtree(out, ignore_errors=True)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--steps", "8", "--eval-every", "8", "--save-every", "4", check=False)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    assert "LOSSLESS_GATE skipped" in log
    assert "scope gemma4" in log and "head model.embed_tokens.weight (frozen)" in log
    ck = _ckpt(out)
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import scope_params

    m, _ = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    assert set(ck["latents"]) == set(scope_params(m, "gemma4"))
    assert ck["fp"] == {}  # norms are outside the gemma4 scope: nothing else is written
    assert all(t.dtype == torch.float16 for t in ck["latents"].values())
    res = ck["results"]
    assert res[0]["tag"] == "init" and res[-1]["klk"] < res[0]["klk"], [r["klk"] for r in res]
    moved = [n for n, t in ck["latents"].items() if not torch.equal(t.float(), dict(m.named_parameters())[n])]
    assert len(moved) == len(ck["latents"])


@pytest.mark.parametrize(("name", "extra"), [
    ("moe_scope", ("--train-scope", "classes:attention,dense_mlp,router,routed_experts", "--steps", "8",
                   "--eval-every", "8")),
    ("accum_momentum", ("--accum", "2", "--beta1", "0.9", "--steps", "4", "--eval-every", "4")),
])
def test_bf16_trial_configs_train_and_descend(g4, name, extra):
    """bm-889 trials: the MoE scope (router + routed experts too) and --accum/--beta1 train every latent and descend."""
    import shutil

    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import scope_params

    out = WORK / f"run_trial_{name}"
    if out.exists():
        shutil.rmtree(out)  # test scratch of this test only
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--save-every", "100", *extra)
    ck = _ckpt(out)
    m, _ = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    scope = extra[1] if extra[0] == "--train-scope" else "gemma4"
    assert set(ck["latents"]) == set(scope_params(m, scope))
    log = proc.stdout + proc.stderr
    tel = re.findall(r"rows \[([\d, ]+)\] tokens (\d+) grad_norm ([\d.e+-]+) update_ratio ([\d.e+-]+)", log)
    assert tel, log[-2000:]  # every logged step carries its rows, scored tokens, gradient norm and update ratio
    assert all(int(t) > 0 and float(g) > 0 and float(u) > 0 for _, t, g, u in tel), tel
    if name == "moe_scope":
        assert any(".router." in n for n in ck["latents"]) and any(".experts." in n for n in ck["latents"])
    else:
        assert "OPTIM accum 2 x bs 2 = 4 rows/update, beta1 0.9" in proc.stdout + proc.stderr
        assert all(len(st) == 4 for st in ck["adafactor"].values())
    base = dict(m.named_parameters())
    assert all(not torch.equal(t, base[n].bfloat16()) for n, t in ck["latents"].items())
    res = ck["results"]
    assert res[-1]["klk"] < res[0]["klk"], [r["klk"] for r in res]


def test_bf16_lossless_gate_when_the_teacher_is_the_student(g4):
    out = WORK / "run_self"
    shutil.rmtree(out, ignore_errors=True)  # own scratch; a leftover run is RUN_REFUSED (bug-977)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_self"]),
                "--out", str(out), "--steps", "2", "--eval-every", "2", "--stop-at", "1", check=False)
    log = proc.stdout + proc.stderr
    assert "LOSSLESS_GATE klk=" in log, log[-3000:]


def test_bf16_reverse_kl_trains_and_is_evaluated(g4, tmp_path):
    out = tmp_path / "run_rkl"
    tr = [*TRAIN[: TRAIN.index("--loss")], "--loss", "rkl", *TRAIN[TRAIN.index("--loss") + 2 :]]
    _run("omk_qad.train", *tr, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
         "--out", str(out), "--steps", "8", "--eval-every", "8", "--save-every", "4")
    res = _ckpt(out)["results"]
    assert all("rkl" in r for r in res)
    assert res[0]["tag"] == "init" and res[-1]["rkl"] < res[0]["rkl"], [r["rkl"] for r in res]
    assert json.loads((out / "stage1" / "summary.json").read_text())["args"]["loss"] == "rkl"


def test_bf16_gguf_teacher_cache_skips_the_lossless_gate(g4, tmp_path):
    """A --teacher-gguf cache names the student's dir as --model (tokenizer only): the gate must not run on it."""
    import shutil

    c = tmp_path / "cache_gguf"
    shutil.copytree(g4["cache_self"], c)
    meta = json.loads((c / "cache.json").read_text())
    meta["args"]["teacher_gguf"] = "/models/another-teacher.gguf"
    (c / "cache.json").write_text(json.dumps(meta))
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(c),
                "--out", str(tmp_path / "run"), "--steps", "1", "--eval-every", "1")
    log = proc.stdout + proc.stderr
    assert "LOSSLESS_GATE skipped: cache teacher /models/another-teacher.gguf" in log, log[-3000:]


def test_bf16_export_replaces_only_trained_tensors(g4):
    from safetensors import safe_open

    out = WORK / "run_teacher"
    if not (out / "stage1" / "final.pt").exists():
        pytest.skip("needs test_bf16_mode_trains_only_the_scope_and_descends")
    hf = WORK / "export_hf"
    import shutil

    for d in (hf, hf.with_name(hf.name + ".partial")):
        if d.exists():
            shutil.rmtree(d)  # test scratch of this test only
    (hf.with_name(hf.name + ".partial")).mkdir()  # what a killed export leaves: removed, then rebuilt
    (hf.with_name(hf.name + ".partial") / "model.safetensors").write_bytes(b"torn")
    p = _run("omk_qad.train", "--phase", "export", "--mode", "bf16", "--device", "cpu", "--model",
             str(g4["student"]), "--load", str(out / "stage1" / "final.pt"), "--out-hf", str(hf))
    assert "never finished" in p.stdout and not hf.with_name(hf.name + ".partial").exists()
    rep = json.loads((hf / "omk_bf16_export.json").read_text())
    ck = _ckpt(out)
    assert rep["replaced"] == len(ck["latents"]) and rep["unchanged_verified"] > 0
    for f in ("config.json", "tokenizer.json", "chat_template.jinja"):
        assert (hf / f).read_bytes() == (g4["student"] / f).read_bytes()
    with safe_open(hf / "model.safetensors", "pt") as f:
        for n, t in ck["latents"].items():
            w = f.get_tensor(n)
            assert torch.equal(w, t.to(w.dtype)), n  # the same cast as the forward: the export is what was evaluated
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)  # loads as a normal HF checkpoint
    assert m.config.final_logit_softcapping == 30.0
    proc = _run("omk_qad.train", "--phase", "export", "--mode", "bf16", "--device", "cpu", "--model",
                str(g4["student"]), "--load", str(out / "stage1" / "final.pt"), "--out-hf", str(hf), check=False)
    assert proc.returncode != 0 and "never overwritten" in proc.stdout + proc.stderr


# ===================================================================================================== roles
def test_role_loss_terms():
    import argparse

    from omk_qad.train import role_loss

    torch.manual_seed(0)
    v, n = 32, 6
    lg = torch.randn(n, v)
    lp_s = F.log_softmax(lg, -1)
    t_val, t_idx = F.log_softmax(torch.randn(n, v), -1).topk(v, -1)  # K = whole vocab: KL is exact
    a_val, a_idx = lp_s.topk(v, -1)  # the anchor IS the student: zero anchor loss
    x_next = torch.randint(0, v, (n,))
    ns = argparse.Namespace(loss="kl", anchor_weight=1.0, ul_weight=1.0)
    r = torch.tensor([0, 1, 2, 3, 0, 2, 4])
    lg, x_next = torch.cat([lg, lg[3:4]]), torch.cat([x_next, x_next[3:4]])  # position 6 = position 3 as lead-in
    t_idx, t_val, a_idx, a_val = (torch.cat([t, t[3:4]]) for t in (t_idx, t_val, a_idx, a_val))
    lp_s = F.log_softmax(lg, -1)
    per = role_loss(lg, x_next, r, t_idx, t_val, a_idx, a_val, ns)
    assert per[0] == 0 and per[4] == 0  # ignored positions
    assert per[2].abs() < 1e-5 and per[5].abs() < 1e-5  # anchor == student
    kl_t = (t_val.exp() * (t_val - lp_s.gather(-1, t_idx))).sum(-1)
    ul = -torch.log(1 - lp_s.exp().gather(-1, x_next[:, None]).squeeze(-1))
    torch.testing.assert_close(per[1], kl_t[1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(per[3], kl_t[3] + ul[3], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(per[6], kl_t[3], rtol=1e-4, atol=1e-5)  # lead-in: teacher KL, no unlikelihood
    with pytest.raises(SystemExit, match="anchor"):
        role_loss(lg, x_next, r, t_idx, t_val, None, None, ns)


@pytest.fixture(scope="module")
def roles_cache(g4):
    """Role windows from synthetic agentic conversations -> teacher cache (seed-1 model) + anchor pass (student)."""
    from omk_qad.tests.test_rollouts_cpu import CONV, TOOLS

    rows = []
    for k in range(6):  # small variations so the windows differ
        conv = json.loads(json.dumps(CONV))
        conv[1]["content"] = f"Fix game{k}.js"
        rows.append({"id": f"c{k}", "messages": conv, "tools": TOOLS})
    src = WORK / "conv.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rfile = WORK / "conv.roles.jsonl"
    _run("omk_qad.rollouts", "--model", str(g4["student"]), "--in", str(src), "--out", str(rfile), "--seq", "64",
         "--stride", "32", "--history-reasoning", "keep")
    cache = WORK / "cache_roles"
    if not (cache / "cache.json").exists():
        _run("omk_qad.train", "--phase", "cache", "--model", str(g4["teacher"]), "--cache", str(cache), "--device",
             "cpu", "--data", f"{rfile}=3,{g4['text']}=1", "--eval-data", str(g4["text"]), "--n-seq", "16",
             "--n-eval", "2", "--seq", "64", "--shard", "8", "--imatrix-seqs", "0")
        _run("omk_qad.train", "--phase", "cache", "--anchor", "--model", str(g4["student"]), "--cache", str(cache),
             "--device", "cpu", "--cache-bs", "4")
    return cache


def test_role_cache_carries_roles_and_anchor_targets(g4, roles_cache):
    from omk_qad.rollouts import ANCHOR

    meta = json.loads((roles_cache / "cache.json").read_text())
    assert meta["anchor_model"] == str(g4["student"])
    seen_anchor = False
    for s in range(2):
        sh = torch.load(roles_cache / f"train_{s:04d}.pt")
        assert sh["role"].shape == sh["x"].shape and sh["role"].dtype == torch.int8
        has = bool((sh["role"] == ANCHOR).any())
        assert ("aidx" in sh) == has
        if has:
            seen_anchor = True
            assert sh["aidx"].shape == sh["idx"].shape
    assert seen_anchor
    proc = _run("omk_qad.train", "--phase", "cache", "--anchor", "--model", str(g4["teacher"]), "--cache",
                str(roles_cache), "--device", "cpu", check=False)
    assert proc.returncode != 0 and "already holds anchor targets" in proc.stdout + proc.stderr


def test_bf16_trains_on_roles(g4, roles_cache):
    out = WORK / "run_roles"
    shutil.rmtree(out, ignore_errors=True)  # own scratch (bug-977)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(roles_cache),
                "--out", str(out), "--steps", "4", "--eval-every", "4", "--ul-weight", "0.5", check=False)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    ck = _ckpt(out)
    assert ck["step"] == 4 and all(torch.isfinite(t.float()).all() for t in ck["latents"].values())
    losses = [float(ln.split("loss=")[1].split()[0]) for ln in (out / "stage1" / "log.txt").read_text().splitlines()
              if ln.startswith("step ") and "loss=" in ln]
    assert losses and all(0 < q < 50 for q in losses), losses


def _snapshot(d: Path) -> dict:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir() if p.is_file()}


@pytest.fixture(scope="module")
def held_set(g4, roles_cache):
    """Role windows of conversations the cache never saw -> evalset_held.pt beside cache_roles."""
    from omk_qad.tests.test_rollouts_cpu import CONV, TOOLS

    rows = []
    for k in range(10, 13):
        conv = json.loads(json.dumps(CONV))
        conv[1]["content"] = f"Fix level{k}.js"
        rows.append({"id": f"h{k}", "messages": conv, "tools": TOOLS})
    src = WORK / "held.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rfile = WORK / "held.roles.jsonl"
    _run("omk_qad.rollouts", "--model", str(g4["student"]), "--in", str(src), "--out", str(rfile), "--seq", "64",
         "--stride", "32", "--history-reasoning", "keep")
    out = roles_cache / "evalset_held.pt"
    for f in ("evalset_held.pt", "evalset_held4.pt"):  # this test's own artifacts: built fresh every run
        (roles_cache / f).unlink(missing_ok=True)
    before = _snapshot(roles_cache)
    args = ("omk_qad.train", "--phase", "evalset", "--model", str(g4["teacher"]), "--cache", str(roles_cache),
            "--device", "cpu", "--eval-set", "held", "--eval-data", str(rfile), "--n-eval", "500", "--seq", "64")
    proc = _run(*args)
    after = _snapshot(roles_cache)
    assert {k: v for k, v in after.items() if k != out.name} == before  # the cache itself is untouched
    return {"rfile": rfile, "out": out, "args": args, "log": proc.stdout + proc.stderr}


def test_an_eval_set_reads_each_held_out_row_once_beside_an_untouched_cache(g4, roles_cache, held_set):
    from omk_qad.train import cache_ref, teacher_scored

    ev = torch.load(held_set["out"])
    rows = [json.loads(ln) for ln in held_set["rfile"].read_text().splitlines() if ln.strip()]
    assert len(rows) < 500  # --n-eval above the file: every row once, in file order, none repeated
    assert ev["x"].tolist() == [r["ids"] for r in rows] and ev["role"].tolist() == [r["role"] for r in rows]
    n_rows = len(rows)
    few = [*held_set["args"]]
    few[few.index("--eval-set") + 1], few[few.index("--n-eval") + 1] = "held4", "4"
    _run(*few)
    ev4 = torch.load(roles_cache / "evalset_held4.pt")
    spread = [i * n_rows // 4 for i in range(4)]  # spread over the whole file, not its first rows
    assert ev4["x"].tolist() == [rows[i]["ids"] for i in spread] and spread[-1] >= n_rows // 2
    assert ev["role"].shape == ev["x"].shape and ev["idx"].shape[:2] == (n_rows, 63)
    assert ev["identity"]["cache"] == cache_ref(roles_cache)
    assert f"{int(teacher_scored(ev['role'][:, 1:]).sum())} teacher-scored" in held_set["log"]
    again = _run(*held_set["args"])
    assert "kept: same rows and identity" in again.stdout + again.stderr
    other = [*held_set["args"]]
    other[other.index(str(g4["teacher"]))] = str(g4["student"])  # another teacher than the cache's
    proc = _run(*other, check=False)
    assert proc.returncode != 0 and "was made with other settings" in proc.stdout + proc.stderr


def test_an_eval_set_scores_only_the_teacher_scored_positions(g4, roles_cache, held_set):
    import shutil

    from omk_qad.train import teacher_scored

    ev = torch.load(held_set["out"])
    keep = teacher_scored(ev["role"][:, 1:])
    for name, garble in (("offmask", ~keep), ("onmask", keep)):  # garbage teacher targets where the mask says
        g = {**ev, "idx": ev["idx"].clone(), "val": ev["val"].clone()}
        g["idx"][garble] = 7
        g["val"][garble] = 0.0
        torch.save(g, roles_cache / f"evalset_{name}.pt")
    out = WORK / "run_evalset"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(roles_cache),
                "--out", str(out), "--steps", "1", "--eval-every", "1", "--eval-set", "held,offmask,onmask")
    evals = [json.loads(ln.split("EVAL ", 1)[1]) for ln in (proc.stdout + proc.stderr).splitlines()
             if ln.startswith("EVAL ")]
    assert evals and all(set(e["sets"]) == {"held", "offmask", "onmask"} for e in evals)
    for e in evals:
        held, off, on = (e["sets"][k] for k in ("held", "offmask", "onmask"))
        assert held["scored"] == int(keep.sum()) and "scored" not in e  # the cache's own eval keeps every position
        assert {k: off[k] for k in ("klk", "rkl", "jaccard", "top1")} == {k: held[k] for k in ("klk", "rkl",
                                                                                                "jaccard", "top1")}
        assert on["klk"] != held["klk"]


def test_sft_trains_on_role_windows_with_no_teacher_targets(g4, roles_cache):
    """--loss ce: rows come from --sft-data (role windows), CE on their scored positions; the cache only evaluates."""
    rfile = WORK / "conv.roles.jsonl"
    n = json.loads(rfile.with_suffix(".stats.json").read_text())["windows"]
    steps = max(1, n // 2)
    out = WORK / "run_sft"
    shutil.rmtree(out, ignore_errors=True)  # a finished run refuses a relaunch: its log would be the last run's
    efile = WORK / "conv_eval.roles.jsonl"  # held-out stand-in: another file (same windows; mechanics only)
    efile.write_bytes(rfile.read_bytes())
    efile.with_suffix(".stats.json").write_bytes(rfile.with_suffix(".stats.json").read_bytes())
    sft = ("--loss", "ce", "--sft-data", str(rfile), "--mtp", "freeze", "--sft-eval", str(efile))
    proc = _run("omk_qad.train", *TRAIN, *sft, "--model", str(g4["student"]), "--cache", str(roles_cache),
                "--out", str(out), "--steps", str(steps), "--eval-every", str(steps), check=False)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    text = (out / "stage1" / "log.txt").read_text()
    assert f"DATA sft {n} rows, order shuffle" in text
    assert "SRCLOSS window 32 rows, trained share {\"conv\": 1.0}" in text and "SRCLOSS step 1 fixmix" in text
    assert f"SFT_EVAL {n} held-out rows" in text
    evals = [json.loads(ln.split("EVAL ", 1)[1]) for ln in text.splitlines() if ln.startswith("EVAL ")]
    assert [e["tag"] for e in evals] == ["init", f"step{steps}"], evals
    assert all(set(e["families"]) == {"conv_eval"} and "klk" not in e and 0 < e["nll"] < 50 for e in evals), evals
    ck = _ckpt(out)
    assert ck["step"] == steps and all(torch.isfinite(t.float()).all() for t in ck["latents"].values())
    losses = [float(ln.split("loss=")[1].split()[0]) for ln in text.splitlines() if ln.startswith("step ")]
    assert losses and all(0 < q < 50 for q in losses), losses
    base = ("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(roles_cache), "--steps", "1")
    for extra, why in ((("--sft-data", str(rfile)), "--loss ce trains on --sft-data"),  # kl + sft data
                       (("--loss", "ce", "--mtp", "freeze"), "--loss ce trains on --sft-data"),
                       ((*sft, "--ce-weight", "1"), "count it twice"),
                       (("--loss", "ce", "--sft-data", str(rfile)), "--mtp freeze"),
                       (sft[:-2], "needs --sft-eval"),
                       ((*sft[:-1], str(rfile)), "are trained on"),
                       ((*sft, "--steps", str(n)), f"--sft-data holds {n} rows")):
        bad = _run(*base, *extra, "--out", str(WORK / "run_sft_refused"), check=False)
        assert bad.returncode != 0 and why in bad.stdout + bad.stderr, (extra, (bad.stdout + bad.stderr)[-800:])


def test_sft_needs_no_cache_eval_pt(g4, roles_cache):
    """An SFT run reads only the cache's metadata; a KD run still needs eval.pt (bm-893: lost in an archive)."""
    rfile = WORK / "conv.roles.jsonl"
    efile = WORK / "conv_eval.roles.jsonl"
    efile.write_bytes(rfile.read_bytes())
    efile.with_suffix(".stats.json").write_bytes(rfile.with_suffix(".stats.json").read_bytes())
    meta_only = WORK / "cache_meta_only"
    shutil.rmtree(meta_only, ignore_errors=True)
    meta_only.mkdir()
    for f in roles_cache.iterdir():
        if f.is_file() and f.suffix != ".pt":
            shutil.copy2(f, meta_only / f.name)
    assert not (meta_only / "eval.pt").exists() and (meta_only / "cache.json").exists()
    base = ("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(meta_only), "--steps", "1")
    out = WORK / "run_sft_meta_only"
    shutil.rmtree(out, ignore_errors=True)
    sft = ("--loss", "ce", "--sft-data", str(rfile), "--mtp", "freeze", "--sft-eval", str(efile))
    proc = _run(*base, *sft, "--out", str(out), "--eval-every", "1", check=False)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-3000:]
    text = (out / "stage1" / "log.txt").read_text()
    evals = [json.loads(ln.split("EVAL ", 1)[1]) for ln in text.splitlines() if ln.startswith("EVAL ")]
    assert [e["tag"] for e in evals] == ["init", "step1"] and all(0 < e["nll"] < 50 for e in evals), evals
    kd = _run(*base, "--out", str(WORK / "run_kd_meta_only"), check=False)
    assert kd.returncode != 0 and "eval.pt" in kd.stdout + kd.stderr, (kd.stdout + kd.stderr)[-800:]


def test_sft_epochs_append_seeded_passes(roles_cache):
    """--sft-epochs E (sft10): E passes back to back, pass 0 = the single-pass order, pass e seeded seed + e."""
    from omk_qad.train import SftRows

    rfile = str(WORK / "conv.roles.jsonl")
    seq = len(json.loads(open(rfile).readline())["ids"])
    one = SftRows([rfile], seq, "shuffle", 7)
    two = SftRows([rfile], seq, "shuffle", 7, epochs=2)
    n = len(one)
    assert len(two) == 2 * n and two.rows[:n] == one.rows
    assert sorted(two.rows[n:]) == sorted(one.rows) and two.rows[n:] == SftRows([rfile], seq, "shuffle", 8).rows
    assert SftRows([rfile], seq, "cache", 7, epochs=2).rows == list(range(n)) * 2


def test_sft_weights_scale_the_gradient_not_the_logged_loss(g4, roles_cache):
    """--sft-weights name=w (sft7): the row's CE is scaled for the gradient only; an unknown source is refused."""
    rfile = WORK / "conv.roles.jsonl"
    efile = WORK / "conv_eval.roles.jsonl"
    efile.write_bytes(rfile.read_bytes())
    efile.with_suffix(".stats.json").write_bytes(rfile.with_suffix(".stats.json").read_bytes())
    base = ("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(roles_cache), "--steps", "1",
            "--loss", "ce", "--sft-data", str(rfile), "--mtp", "freeze", "--sft-eval", str(efile), "--eval-every", "1")
    runs = {}
    for name, extra in (("w1", ()), ("w3", ("--sft-weights", "conv=3"))):
        out = WORK / f"run_sft_weights_{name}"
        shutil.rmtree(out, ignore_errors=True)
        _run(*base, *extra, "--out", str(out))
        text = (out / "stage1" / "log.txt").read_text()
        runs[name] = ([ln.split("loss=")[1].split()[0] for ln in text.splitlines() if ln.startswith("step ")],
                      _ckpt(out)["latents"], text)
    assert 'SFT_WEIGHTS {"conv": 3.0}' in runs["w3"][2]
    assert runs["w1"][0] == runs["w3"][0]  # the logged loss is unweighted
    assert any(not torch.equal(runs["w1"][1][k], runs["w3"][1][k]) for k in runs["w1"][1])  # the update is not
    bad = _run(*base, "--sft-weights", "nosuch=2", "--out", str(WORK / "run_sft_weights_bad"), check=False)
    assert bad.returncode != 0 and "is not a --sft-data source" in bad.stdout + bad.stderr


# ----------------------------------------------------------------------------------------- router_full (bm-898 sft11r)
ROUTER_FULL_SUFFIXES = ("router.proj.weight", "router.scale", "router.per_expert_scale",
                        "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                        "pre_feedforward_layernorm_2.weight", "post_feedforward_layernorm_1.weight",
                        "post_feedforward_layernorm_2.weight")


def test_router_full_scope_is_the_whole_router_block(g4):
    """``classes:router_full`` = 8 tensors per layer: router proj + its two gains + the five FFN-block norm scales."""
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import scope_params

    m, _ = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    names = scope_params(m, "classes:router_full")
    assert len(names) == 8 * N_LAYERS, names
    for suf in ROUTER_FULL_SUFFIXES:
        assert sum(n.endswith(suf) for n in names) == N_LAYERS, suf
    assert not any(("self_attn" in n or "embed" in n or "experts." in n or "input_layernorm" in n) for n in names)
    both = scope_params(m, "classes:attention,router_full")
    assert set(names) < set(both) and any("self_attn" in n for n in both)
    params = dict(m.named_parameters())
    assert all(params[n].ndim == 1 for n in names if not n.endswith("proj.weight"))  # the gains train as fp params


def test_router_full_trains_with_lb_loss_and_telemetry(g4):
    """router_full + experts, --lb-weight, --router-gate: fp params move, ROUTER lines per eval, lb logged, no trip."""
    from omk_qad.ptq import load_text_causal_lm
    from omk_qad.recovery import scope_params

    out = WORK / "run_router_full"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--steps", "4", "--eval-every", "2", "--save-every", "100", "--fp-lr", "1e-3",
                "--train-scope", "classes:router_full,routed_experts", "--lb-weight", "0.01",
                "--router-gate", "3.0,0.5,1.0")
    log = proc.stdout + proc.stderr
    assert "ROUTER_PROBE 6 routers, lb_weight 0.01, gate 3.0,0.5,1.0" in log, log[-2000:]
    assert "FP_GROUPS" in log and '"router_scale"' in log and '"per_expert_scale"' in log and '"ffn_norm"' in log
    routers = [json.loads(line[7:]) for line in log.splitlines() if line.startswith("ROUTER ")]
    assert len(routers) == 3, [line for line in log.splitlines() if "ROUTER" in line]  # init, step2, step4
    assert routers[0]["top_n"] == 1 and set(routers[0]["layers"]) == {str(i) for i in range(N_LAYERS)}
    assert all(0 < r["entropy_mean"] <= math.log(8) + 1e-6 for r in routers), routers
    assert "flip_mean" in routers[-1] and routers[0].get("flip_mean", 0.0) == 0.0
    steps = re.findall(r"step \d+ loss=.* lb ([\d.e+-]+) .*router_scale upd ([\d.e+-]+) drift ([\d.e+-]+)", log)
    assert steps, log[-3000:]
    assert all(float(lb) > 0 for lb, _, _ in steps) and any(float(u) > 0 for _, u, _ in steps), steps
    assert "ROUTER_GATE_STOP" not in log and "QAD_DONE" in log
    assert (out / "stage1" / "router_init.pt").exists()
    ck = _ckpt(out)
    m, _ = load_text_causal_lm(str(g4["student"]), "cpu", torch.float32)
    want = set(scope_params(m, "classes:router_full,routed_experts"))
    assert set(ck["latents"]) | set(ck["fp"]) == want
    assert all(any(n.endswith(suf) for suf in ROUTER_FULL_SUFFIXES[1:]) for n in ck["fp"]), sorted(ck["fp"])[:5]
    moved = [n for n, t in ck["fp"].items() if not torch.equal(t.float(), dict(m.named_parameters())[n])]
    assert len(moved) == len(ck["fp"]), f"{len(moved)}/{len(ck['fp'])} fp params moved"
    res = json.loads((out / "stage1" / "summary.json").read_text())["results"]
    assert all("router" in r for r in res) and res[-1]["router"]["flip_mean"] >= 0.0


def test_fp_lr_scale_rms_gives_every_tensor_the_same_relative_step(g4):
    """--fp-lr-scale rms: lr = fp_lr x rms per tensor, so a gain of 32 and a gain of 1 take the same RELATIVE step."""
    from omk_qad.recovery import fp_param_groups

    big, one = torch.nn.Parameter(torch.full((16,), 32.0)), torch.nn.Parameter(torch.ones(4))
    assert fp_param_groups([big, one], 1e-4) == [{"params": [big, one], "lr": 1e-4, "base": 1e-4}]
    groups = fp_param_groups([big, one], 1e-4, "rms")
    assert [g["params"] for g in groups] == [[big], [one]]
    assert groups[0]["base"] == pytest.approx(3.2e-3) and groups[1]["base"] == pytest.approx(1e-4)
    opt = torch.optim.AdamW(groups, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    (big.sum() * 3.0 + one.sum() * 0.01).backward()  # Adam's first step is lr x sign(grad), whatever the grad scale
    opt.step()
    rel = [float((p.detach() - v).abs().max() / v) for p, v in ((big, 32.0), (one, 1.0))]
    assert rel[0] == pytest.approx(1e-4, rel=1e-3) and rel[1] == pytest.approx(1e-4, rel=1e-3), rel
    with pytest.raises(SystemExit):
        fp_param_groups([torch.nn.Parameter(torch.zeros(3))], 1e-4, "rms")
    out = WORK / "run_fplr_rms"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--steps", "3", "--eval-every", "100", "--save-every", "2", "--fp-lr", "1e-3",
                "--fp-lr-scale", "rms", "--train-scope", "classes:router_full,routed_experts")
    log = proc.stdout + proc.stderr
    assert f"FP_LR scale rms: {N_LAYERS * 7} groups" in log and "QAD_DONE" in log, log[-2000:]
    # a resume rebuilds the groups and loads the saved AdamW state: the per-tensor peak lr must come back with it
    opt2 = torch.optim.AdamW(fp_param_groups([big, one], 1e-4, "rms"), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    for g in opt.param_groups:
        g["lr"] = g["base"] * 0.5  # mid-schedule
    opt2.load_state_dict(opt.state_dict())
    assert [g["base"] for g in opt2.param_groups] == [g["base"] for g in opt.param_groups]
    assert opt2.param_groups[0]["lr"] == pytest.approx(1.6e-3)


def test_fp_scope_does_not_change_the_latent_gradient_at_step_1(g4):
    """Step 1 runs on the SAME weights whatever the fp scope: loss and latent grad norm must not depend on it.

    sft11r (router_full) logged a latent grad norm 3.8x / 11x sft11's (router) on the same rows (2026-11-20).
    """
    seen = {}
    for name, scope in (("proj", "classes:router,routed_experts"), ("full", "classes:router_full,routed_experts")):
        out = WORK / f"run_fpscope_{name}"
        shutil.rmtree(out, ignore_errors=True)
        proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                    "--out", str(out), "--steps", "1", "--eval-every", "100", "--save-every", "100",
                    "--fp-lr", "1e-3", "--train-scope", scope)
        log = proc.stdout + proc.stderr
        m = re.search(r"step 1 loss=([\d.e+-]+) .* grad_norm ([\d.e+-]+) ", log)
        assert m, log[-2000:]
        seen[name] = (float(m.group(1)), float(m.group(2)))
    (l0, g0), (l1, g1) = seen["proj"], seen["full"]
    assert abs(l0 - l1) <= 1e-4 * abs(l0), seen
    assert abs(g0 - g1) <= 1e-2 * g0, seen


class _FakeRouter(torch.nn.Module):
    """Router-shaped module: (probabilities [T, E], top-k weights, top-k index), with a trainable projection."""

    def __init__(self, hidden: int, experts: int):
        """Linear router over ``hidden`` channels into ``experts`` experts, top-1."""
        super().__init__()
        self.proj = torch.nn.Linear(hidden, experts, bias=False)

    def forward(self, x):
        """Softmax probabilities and the top-1 pick."""
        p = torch.softmax(self.proj(x), -1)
        w, i = p.topk(1, -1)
        return p, w, i


class _FakeBlock(torch.nn.Module):
    """One MoE-ish layer: router + a dense map, run under non-reentrant gradient checkpointing like train.py."""

    def __init__(self, hidden: int, experts: int):
        """Build the router and the dense map."""
        super().__init__()
        self.router = _FakeRouter(hidden, experts)
        self.mlp = torch.nn.Linear(hidden, hidden)

    def _body(self, x):
        p, w, _i = self.router(x)
        return self.mlp(x) * w + p.sum(-1, keepdim=True) * 0  # keep the router on the graph

    def forward(self, x):
        """Checkpointed body (``use_reentrant=False``, as train.py enables it)."""
        return torch.utils.checkpoint.checkpoint(self._body, x, use_reentrant=False)


class _FakeMoE(torch.nn.Module):
    """``model.layers.N.router`` tree that router_modules() recognises (it keys on ``.layers.N.router``)."""

    def __init__(self, n_layers: int = 3, hidden: int = 8, experts: int = 4):
        """Stack ``n_layers`` fake blocks under ``model``."""
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_FakeBlock(hidden, experts) for _ in range(n_layers))

    def forward(self, x):
        """Run the blocks."""
        for blk in self.model.layers:
            x = blk(x)
        return x


def test_router_probe_does_not_collect_during_checkpoint_recompute():
    """pop_lb() ends the train window so the backward's checkpoint recompute passes do not refill the probe.

    A term collected from a recompute pass pinned every recomputed layer's activations: +30 GiB and a CUDA OOM at
    step 1 on the 20B student (2026-11-20).
    """
    import torch.utils.checkpoint

    from omk_qad.router_telemetry import RouterProbe

    torch.manual_seed(0)
    model = _FakeMoE()
    probe = RouterProbe(model)
    calls = []
    for i, m in probe.routers.items():
        m.register_forward_hook(lambda _m, _i, _o, i=i: calls.append((i, probe.mode)))
    x = torch.randn(5, 8, requires_grad=True)
    probe.begin_train()
    out = model(x)
    lb = probe.pop_lb()
    assert lb is not None and lb.requires_grad and probe.mode == "off" and probe._seen == {}
    assert [c for c in calls if c[1] == "train"] == [(0, "train"), (1, "train"), (2, "train")]
    torch.autograd.backward([out.sum(), 0.01 * lb], [None, None])  # the dual-root backward of train_step
    recompute = calls[3:]
    assert len(recompute) == 3 and all(mode == "off" for _, mode in recompute), calls  # recompute ran, hook was inert
    assert probe._seen == {}, probe._seen  # nothing collected from the recompute pass
    assert model.model.layers[0].router.proj.weight.grad is not None  # lb gradient reached the router
    probe.begin_train()  # the next micro-batch re-arms the window
    assert probe.mode == "train"
    probe.remove()


def test_router_gate_unit_and_trip(g4):
    """gate() on synthetic stats (deterministic); a wild fp LR under a tight gate stops a run with ROUTER_GATE.json."""
    from omk_qad.router_telemetry import LayerStats, RouterStats, gate, parse_gate, top_n_of

    assert top_n_of(98) == 8 and top_n_of(128) == 11 and top_n_of(8) == 1
    assert parse_gate("") is None and parse_gate("0") is None and parse_gate("1.5,0.85,0.25") == (1.5, 0.85, 0.25)
    with pytest.raises(SystemExit):
        parse_gate("0.9,0.85,0.25")
    mk = lambda top, ent: RouterStats({0: LayerStats([top, 1 - top], top, ent, None, 10)}, 1)  # noqa: E731
    init = mk(0.20, 0.60)
    assert gate(mk(0.25, 0.58), init, (1.5, 0.85, 0.25)) == []
    why = gate(mk(0.31, 0.50), init, (1.5, 0.85, 0.25))
    assert len(why) == 3 and "top-1 share 0.310 > 1.5 x init" in why[0] and "entropy 0.500 < 0.85" in why[1] \
        and "> 0.25 (init 0.200)" in why[2], why
    assert gate(mk(0.26, 0.60), mk(0.30, 0.60), (1.5, 0.85, 0.25)) == []  # init already above top_abs: no abs trip
    out = WORK / "run_router_gate_trip"
    shutil.rmtree(out, ignore_errors=True)
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--steps", "4", "--eval-every", "2", "--save-every", "100", "--fp-lr", "3.0",
                "--warmup", "0", "--train-scope", "classes:router_full", "--router-gate", "1.001,0.999,1.0",
                check=False)
    log = proc.stdout + proc.stderr
    assert "ROUTER_GATE_STOP step" in log, log[-3000:]
    g = json.loads((out / "stage1" / "ROUTER_GATE.json").read_text())
    assert g["reasons"] and g["gate"] == [1.001, 0.999, 1.0] and g["init"]["flip_mean"] == 0.0
    assert not (out / "stage1" / "final.pt").exists() and (out / "stage1" / "last.pt").exists()
    assert json.loads((out / "stage1" / "state.json").read_text())["status"] == "gated"


def test_export_writes_fp_params_in_fp32_so_a_small_scale_change_survives(tmp_path):
    """A trained scale near 32 moved by 0.01 must reach the export: bf16 spacing there is 0.25."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from omk_qad.recovery import export_bf16

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    scale = torch.full((8,), 32.0, dtype=torch.bfloat16)
    w = torch.randn(4, 8).to(torch.bfloat16)
    save_file({"model.layers.0.router.scale": scale, "model.layers.0.router.proj.weight": w,
               "model.norm.weight": torch.ones(8, dtype=torch.bfloat16)}, src / "model.safetensors",
              metadata={"format": "pt"})
    moved = scale.float() + 0.01
    assert torch.equal(moved.to(torch.bfloat16), scale)  # the premise: bf16 cannot hold the change
    new_w = (w.float() * 1.5).to(torch.bfloat16)
    ck = tmp_path / "ck.pt"
    torch.save({"step": 3, "latents": {"model.layers.0.router.proj.weight": new_w},
                "fp": {"model.layers.0.router.scale": moved}}, ck)
    rep = export_bf16(src, str(ck), out, lambda _m: None)
    assert rep["fp32_keys"] == ["model.layers.0.router.scale"]
    assert rep["replaced"] == 2 and rep["unchanged_verified"] == 1
    with safe_open(out / "model.safetensors", "pt") as f:
        got = f.get_tensor("model.layers.0.router.scale")
        assert got.dtype == torch.float32 and torch.equal(got, moved)
        gw = f.get_tensor("model.layers.0.router.proj.weight")
        assert gw.dtype == torch.bfloat16 and torch.equal(gw, new_w)
        assert f.get_tensor("model.norm.weight").dtype == torch.bfloat16
