# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of bf16 recovery mode (omk_qad.recovery) on a tiny Gemma-4 26B-A4B-shaped model.

Artifacts go to $OMK_QAD_WORK (persistent disk; /tmp is tmpfs on solidpc).
"""

import json
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
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_teacher"]),
                "--out", str(out), "--steps", "8", "--eval-every", "8", "--save-every", "4", check=False)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0 and "is finished and immutable" not in log:
        raise AssertionError(log[-3000:])
    if proc.returncode == 0:
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
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(g4["cache_self"]),
                "--out", str(out), "--steps", "2", "--eval-every", "2", "--stop-at", "1", check=False)
    log = proc.stdout + proc.stderr
    assert "LOSSLESS_GATE klk=" in log or "RESUME" in log, log[-3000:]


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
    proc = _run("omk_qad.train", *TRAIN, "--model", str(g4["student"]), "--cache", str(roles_cache),
                "--out", str(out), "--steps", "4", "--eval-every", "4", "--ul-weight", "0.5", check=False)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0 or "is finished and immutable" in log, log[-3000:]
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
