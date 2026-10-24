# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""W&B switch rules of omk_qad.train.Telemetry: the key is the switch, an explicit ask is a requirement.

Test names are their docs; a fake wandb module stands in, nothing reaches the network.
"""

import argparse
import sys
import types
from pathlib import Path

import pytest

from omk_qad import train as T


def _args(**kw) -> argparse.Namespace:
    base = {"wandb_project": "", "wandb_name": "", "wandb_entity": "", "no_wandb": False, "stage": 1, "branch": "",
            "out": "/runs/m6/iq2"}
    return argparse.Namespace(**{**base, **kw})


class _Run:
    url = "https://wandb.example/run"

    def __init__(self, **kw):
        self.kw, self.rows, self.done = kw, [], False

    def log(self, data, step):
        self.rows.append((step, data))

    def finish(self):
        self.done = True


@pytest.fixture
def fake_wandb(monkeypatch):
    mod = types.ModuleType("wandb")
    mod.runs = []

    def init(**kw):
        mod.runs.append(_Run(**kw))
        return mod.runs[-1]

    mod.init = init
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


@pytest.fixture(autouse=True)
def no_ambient_key(monkeypatch, tmp_path):
    for v in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # netrc.netrc() reads ~/.netrc: an empty home has no key
    monkeypatch.delenv("NETRC", raising=False)


def test_no_key_is_off(tmp_path, fake_wandb):
    assert T.Telemetry(_args(), tmp_path, {}).wb is None and not fake_wandb.runs


def test_explicit_ask_without_key_refuses(tmp_path, fake_wandb):
    with pytest.raises(SystemExit, match="asks for W&B"):
        T.Telemetry(_args(wandb_project="p"), tmp_path, {})


def test_off_switches_win_over_a_key(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    assert T.Telemetry(_args(no_wandb=True), tmp_path, {}).wb is None
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert T.Telemetry(_args(), tmp_path, {}).wb is None
    assert not fake_wandb.runs


def test_key_without_package_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    monkeypatch.setitem(sys.modules, "wandb", None)  # import wandb -> ImportError
    with pytest.raises(SystemExit, match="omk-modelopt env is incomplete"):
        T.Telemetry(_args(), tmp_path, {})


def test_netrc_key_turns_it_on(tmp_path, fake_wandb):
    (tmp_path / ".netrc").write_text(f"machine {T.WANDB_HOST}\n  login user\n  password {'n' * 40}\n")
    (tmp_path / ".netrc").chmod(0o600)
    assert T.Telemetry(_args(), tmp_path, {}).wb is not None


def test_run_id_is_stable_per_run_dir_and_logs_flatten(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    run = tmp_path / "stage2" / "b1"
    run.mkdir(parents=True)
    a = _args(stage=2, branch="b1")
    t = T.Telemetry(a, run, {"lr": 1e-5})
    again = T.Telemetry(a, run, {"lr": 1e-5}).wb
    assert t.wb.kw["id"] == again.kw["id"] and t.wb.kw["resume"] == "allow"  # a relaunch reattaches
    assert t.wb.kw["name"] == "m6-iq2-stage2-b1" and t.wb.kw["project"] == "omk-qad"
    assert t.wb.kw["id"] != T.Telemetry(_args(), tmp_path / "stage1", {}).wb.kw["id"]
    t.log({"train/loss": 0.5, "train/note": "dropped"}, 3)
    t.eval({"tag": "step3", "klk": 0.4, "families": {"eval_code": {"klk": 0.2}}, "stats": {"x": 1}}, 3)
    t.finish()
    assert t.wb is None and again is not None
    rows = fake_wandb.runs[0].rows
    assert rows[0] == (3, {"train/loss": 0.5})
    assert rows[1] == (3, {"eval/klk": 0.4, "eval/eval_code/klk": 0.2})
    assert Path(fake_wandb.runs[0].kw["dir"]) == run


def test_resume_rewinds_to_the_checkpoint_step(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    t = T.Telemetry(_args(), tmp_path, {}, resume_step=40)
    rid = T.Telemetry(_args(), tmp_path, {}).wb.kw["id"]
    assert t.wb.kw["resume_from"] == f"{rid}?_step=40" and "id" not in t.wb.kw
    t.log({"train/loss": 1.0}, 41)
    assert fake_wandb.runs[0].rows == [(41, {"train/loss": 1.0})]


def test_resume_without_rewind_skips_the_rows_wandb_holds(tmp_path, fake_wandb, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k" * 40)
    plain = fake_wandb.init

    def init(**kw):
        if "resume_from" in kw:
            raise RuntimeError("rewind not enabled")
        r = plain(**kw)
        r.step = 46  # the killed run logged up to step 45, its checkpoint is step 40
        return r

    fake_wandb.init = init
    t = T.Telemetry(_args(), tmp_path, {}, resume_step=40)
    assert t.wb.kw["resume"] == "allow"
    for step in (41, 45, 46):
        t.log({"train/loss": float(step)}, step)
    assert t.wb.rows == [(46, {"train/loss": 46.0})]
