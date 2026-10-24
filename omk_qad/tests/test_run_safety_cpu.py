# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the run-safety rules (runctl) wired into ``train --phase cache`` and ``--phase train``.

Cache: a drain stops at a shard boundary, a relaunch continues without recomputing finished shards, a relaunch with
another --topk is REFUSED (the done state is kept), a shard whose sidecar disagrees is refused. Train: one process per
run dir, a resume with other run-critical args is refused unless ``--allow-change`` names them.
"""

import fcntl
import json
import shutil
import signal
import subprocess
import sys

from omk_qad.tests.test_qad_cpu import CPU_ENV, LLAMA_CPP, WORK, _train, setup  # noqa: F401


def _cache_cmd(model, lay, cache, text, *extra):
    return [sys.executable, "-m", "omk_qad.train", "--phase", "cache", "--model", str(model), "--cache", str(cache),
            "--layout", str(lay), "--device", "cpu", "--data", str(text), "--eval-data", str(text), "--n-seq", "8",
            "--n-eval", "2", "--seq", "64", "--shard", "4", "--llama-cpp", str(LLAMA_CPP), *extra]


def _run_cache(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=CPU_ENV, check=False)


def _mtimes(cache):
    return {p.name: p.stat().st_mtime_ns for p in sorted(cache.glob("train_*.pt"))}


def test_cache_drain_resume_refuse(setup):  # noqa: F811
    model, lay = setup["moe"]
    c = WORK / "cache_safety"
    if c.exists():
        shutil.rmtree(c)  # test scratch of this test only
    cmd = _cache_cmd(model, lay, c, setup["text"])
    # drain requested before the first shard is done (the "eval:" line comes after the control is installed)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=CPU_ENV)
    out = []
    assert p.stdout is not None
    for line in p.stdout:
        out.append(line)
        if line.startswith("eval: "):
            p.send_signal(signal.SIGUSR1)
    assert p.wait(timeout=600) == 0, "".join(out)[-3000:]
    log = "".join(out)
    assert "CACHE_DRAINED after 4/8" in log and not (c / "cache.json").exists()
    assert json.loads((c / "state.json").read_text())["status"] == "drained"
    assert list(_mtimes(c)) == ["train_0000.pt"] and (c / "train_0000.json").exists()
    first = _mtimes(c)

    r = _run_cache(cmd)  # relaunch: continues, shard 0 is not recomputed
    assert r.returncode == 0 and "CACHE_DONE" in r.stdout, (r.stdout + r.stderr)[-3000:]
    assert _mtimes(c)["train_0000.pt"] == first["train_0000.pt"] and len(_mtimes(c)) == 2
    assert "eval.pt kept" in r.stdout
    done = _mtimes(c)
    assert json.loads((c / "state.json").read_text())["status"] == "done"

    r = _run_cache([*cmd, "--topk", "8"])  # other targets into the same dir: refused, nothing touched
    assert r.returncode != 0 and "RUN_REFUSED cache" in r.stdout + r.stderr and "topk: 64 -> 8" in r.stdout + r.stderr
    st = json.loads((c / "state.json").read_text())
    assert st["status"] == "done" and "RUN_REFUSED" in st["last_refusal"]
    assert _mtimes(c) == done

    side = json.loads((c / "train_0001.json").read_text())
    (c / "train_0001.json").write_text(json.dumps({**side, "x_sha": "0" * 16}))  # a shard of other rows
    r = _run_cache(cmd)
    assert r.returncode != 0 and "train_0001.pt was made with another identity or other rows" in r.stdout + r.stderr


def test_train_lock_and_resume_identity(setup):  # noqa: F811
    model, lay = setup["moe"]
    out = WORK / "run_safety"
    if out.exists():
        shutil.rmtree(out)  # test scratch of this test only
    (out / "stage1").mkdir(parents=True)
    with (out / "stage1" / "run.lock").open("a+") as held:  # another trainer holds the run dir
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.write("pid 999 since test\n")
        held.flush()
        p = _train(model, lay, setup["cache"], out, check=False)
        assert p.returncode != 0 and "RUN_REFUSED train" in p.stdout + p.stderr and "pid 999" in p.stdout + p.stderr

    (out / "stage1" / "PAUSE").touch()
    p = _train(model, lay, setup["cache"], out)
    assert "PAUSED at step 1" in p.stdout + p.stderr
    p = _train(model, lay, setup["cache"], out, "--lr", "2e-3", check=False)
    assert p.returncode != 0 and "RUN_REFUSED" in p.stdout + p.stderr and "lr" in p.stdout + p.stderr
    st = json.loads((out / "stage1" / "state.json").read_text())
    assert st["status"] != "failed" and "RUN_REFUSED" in st["last_refusal"]
    p = _train(model, lay, setup["cache"], out, "--lr", "2e-3", "--allow-change", "lr")
    assert "RESUME from step 1" in p.stdout + p.stderr and "ALLOWED CHANGE lr" in p.stdout + p.stderr
    assert json.loads((out / "stage1" / "state.json").read_text())["status"] == "done"
    hist = json.loads((out / "stage1" / "identity.json").read_text())["history"]
    assert hist and "lr" in hist[-1]["changed"]
