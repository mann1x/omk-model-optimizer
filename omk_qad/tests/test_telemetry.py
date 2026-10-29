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
    fake_wandb.leftover = False  # real wandb: a failed rewind keeps its stream until teardown()

    def teardown():
        fake_wandb.leftover = False

    def init(**kw):
        if "resume_from" in kw:
            fake_wandb.leftover = True
            raise RuntimeError("rewind not enabled")
        if fake_wandb.leftover:
            raise RuntimeError(f"run ID {kw['id']} is in use")
        r = plain(**kw)
        r.step = 46  # the killed run logged up to step 45, its checkpoint is step 40
        return r

    fake_wandb.init, fake_wandb.teardown = init, teardown
    t = T.Telemetry(_args(wandb_project="omk-qad"), tmp_path, {}, resume_step=40)  # explicit: a failure refuses
    assert t.wb.kw["resume"] == "allow"
    for step in (41, 45, 46):
        t.log({"train/loss": float(step)}, step)
    assert t.wb.rows == [(46, {"train/loss": 46.0})]


def test_source_loss_tracks_each_source_and_a_fixed_mix():
    src = ["web", "web", "code", "code", "code", "code"]
    sl = T.SourceLoss(src, trained=list(range(6)), window=2)
    assert sl.share == {"web": 1 / 3, "code": 2 / 3}
    first = sl.row([0, 2], [2.4, 0.3])
    assert first["train/src/web"] == 2.4 and first["train/src/code"] == 0.3
    assert first["train/loss_fixmix"] == pytest.approx(2.4 / 3 + 0.3 * 2 / 3)
    nxt = sl.row([3, 4], [0.2, 0.1])  # an all-code batch: the raw loss drops, the fixed mix only by code's share
    assert "train/src/web" not in nxt and nxt["train/src/code"] == pytest.approx(0.15)
    assert nxt["train/src_avg/code"] == pytest.approx(0.15)  # window 2: the 0.3 has left
    assert nxt["train/loss_fixmix"] == pytest.approx(2.4 / 3 + 0.15 * 2 / 3)
    assert sl.row([0, 1], [1.0]) == {}  # losses that do not pair with the rows log nothing


def test_source_loss_waits_for_every_source_before_the_fixed_mix():
    sl = T.SourceLoss(["a", "b", "c"], trained=[0, 1, 2], window=4)
    assert "train/loss_fixmix" not in sl.row([0, 1], [1.0, 2.0])
    assert "train/loss_fixmix" in sl.row([2], [3.0])


def test_cache_row_sources_redraws_the_seeded_interleave():
    import random

    data, n_seq, shard, seed = "dir/pool.roles.jsonl=3,dir/text.txt=1", 20, 8, 5
    rng, picks = random.Random(seed), []
    for s0 in range(0, n_seq, shard):  # as _phase_cache draws them
        picks += rng.choices(range(2), [3.0, 1.0], k=min(shard, n_seq - s0))
    rps = {"dir/pool.roles.jsonl": picks.count(0), "dir/text.txt": picks.count(1)}
    meta = {"args": {"data": data, "seed": seed}, "n_seq": n_seq, "shard": shard, "rows_per_source": rps}
    assert T.cache_row_sources(meta) == [("pool", "text")[k] for k in picks]
    assert T.cache_row_sources({**meta, "rows_per_source": {**rps, "dir/text.txt": rps["dir/text.txt"] + 1}}) is None
    assert T.cache_row_sources({**meta, "args": {"seed": seed}}) is None


def test_source_names_are_short_and_distinct():
    assert T.source_names(["a/sft_x_train.roles.jsonl", "b/sft_x_train.roles.jsonl", "c/web.txt"]) == \
        ["sft_x_train", "sft_x_train#1", "web"]
