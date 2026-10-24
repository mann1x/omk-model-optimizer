# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of omk_qad.runctl: atomic writes, the run lock, identity refusal, control modes and signals."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from omk_qad.runctl import (
    Control,
    RunLock,
    State,
    atomic_json,
    atomic_write,
    check_identity,
    file_id,
    parse_allow,
    publish_verified,
)


def test_atomic_write_leaves_no_temp(tmp_path):
    p = tmp_path / "a.json"
    atomic_json(p, {"x": 1})
    atomic_write(p, b"raw")
    assert p.read_bytes() == b"raw"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["a.json"]


def test_lock_refuses_second_holder_and_names_it(tmp_path):
    held = RunLock(tmp_path / "run.lock", "train")
    with pytest.raises(SystemExit, match=rf"RUN_REFUSED train: .*pid {os.getpid()}"):
        RunLock(tmp_path / "run.lock", "train")
    held.release()
    RunLock(tmp_path / "run.lock").release()          # free again after the with block


def test_identity_first_write_then_refuse_then_allow(tmp_path):
    m = tmp_path / "identity.json"
    check_identity(m, {"bs": 8, "teacher": "a"})
    check_identity(m, {"bs": 8, "teacher": "a"})       # same settings: fine
    with pytest.raises(SystemExit, match=r"RUN_REFUSED .*bs: 8 -> 16"):
        check_identity(m, {"bs": 16, "teacher": "a"})
    with pytest.raises(SystemExit, match="teacher"):    # allowing bs does not allow teacher
        check_identity(m, {"bs": 16, "teacher": "b"}, parse_allow(["bs"]))
    with pytest.raises(SystemExit, match="bs"):         # k=v allows only that value
        check_identity(m, {"bs": 32, "teacher": "a"}, parse_allow(["bs=16"]))
    check_identity(m, {"bs": 16, "teacher": "a"}, parse_allow(["bs=16"]), log=lambda *_: None)
    man = json.loads(m.read_text())
    assert man["identity"]["bs"] == 16 and man["history"][0]["changed"] == {"bs": [8, 16]}


def test_file_id_uses_sidecar_and_hashes_small_files(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    assert file_id(f)["sha256"].startswith("ba7816bf")
    (tmp_path / "w.bin.sha256").write_text("deadbeef  w.bin\n")
    assert file_id(f)["sha256"] == "deadbeef"


def test_control_boundary_pause_resume_and_drain(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    ctl = Control(tmp_path / "c", log=lambda *_: None)
    try:
        assert ctl.boundary() == "run"
        (tmp_path / "c").write_text("pause\n")
        hits = []

        def resume():
            time.sleep(0.3)
            (tmp_path / "c").write_text("run\n")

        threading.Thread(target=resume, daemon=True).start()
        t0 = time.time()
        assert ctl.boundary(on_pause=lambda: hits.append(1), poll=0.05) == "run"
        assert hits == [1] and time.time() - t0 >= 0.25
        (tmp_path / "c").write_text("drain\n")
        assert ctl.boundary() == "drain"
    finally:
        ctl.restore()
    assert signal.getsignal(signal.SIGTERM) is before   # handlers restored


def test_control_resets_a_stale_mode_at_start(tmp_path):
    (tmp_path / "c").write_text("drain\n")
    ctl = Control(tmp_path / "c", log=lambda *_: None)
    try:
        assert ctl.mode() == "run"
    finally:
        ctl.restore()


def test_state_is_merged_and_atomic(tmp_path):
    st = State(tmp_path / "s.json", run="r1")
    st.update(status="running", step=1)
    st.update(step=2)
    d = json.loads((tmp_path / "s.json").read_text())
    assert d["run"] == "r1" and d["status"] == "running" and d["step"] == 2 and d["pid"] == os.getpid()


def test_first_sigterm_drains_second_stops_at_once(tmp_path):
    code = textwrap.dedent(f"""
        import os, signal, sys, time
        from omk_qad.runctl import Control
        c = Control({str(tmp_path / 'c')!r}, log=lambda *a: print(*a, flush=True))
        os.kill(os.getpid(), signal.SIGTERM)
        print("MODE", c.mode(), flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        print("NOT REACHED", flush=True)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert "MODE drain" in r.stdout and "NOT REACHED" not in r.stdout and r.returncode == 130


def test_publish_verified_renames_only_a_verified_build(tmp_path):
    out = tmp_path / "m.gguf"

    def good(part):
        part.write_bytes(b"gguf")
        part.with_suffix(".export.json").write_text(json.dumps({"out": str(part)}))
        part.with_suffix(".imatrix.gguf").write_bytes(b"im")
        return {"out": str(part)}

    (tmp_path / "m.partial.gguf").write_bytes(b"left by a kill")
    rep = publish_verified(out, good, sidecars=(".imatrix.gguf",))
    assert rep == {"out": str(out), "ok": True} and out.read_bytes() == b"gguf"
    assert json.loads((tmp_path / "m.export.json").read_text()) == {"out": str(out), "ok": True}
    assert (tmp_path / "m.imatrix.gguf").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.export.json", "m.gguf", "m.gguf.lock", "m.imatrix.gguf"]

    def bad(part):
        part.write_bytes(b"wrong")
        part.with_suffix(".export.json").write_text(json.dumps({"out": str(part)}))
        raise SystemExit("VERIFY FAILED")

    out2 = tmp_path / "n.gguf"
    with pytest.raises(SystemExit, match="VERIFY FAILED"):
        publish_verified(out2, bad)
    assert not out2.exists() and (tmp_path / "n.verify-failed.gguf").read_bytes() == b"wrong"
    assert json.loads((tmp_path / "n.verify-failed.export.json").read_text())["ok"] is False
    assert out.read_bytes() == b"gguf"  # an earlier good export is never touched by a failed one


def test_multi_forwards_signals_to_every_child(tmp_path):
    from omk_qad.multi import _launch

    child = textwrap.dedent("""
        import pathlib, signal, sys, time
        out = pathlib.Path(sys.argv[1])
        signal.signal(signal.SIGTERM, lambda *a: (out.write_text("drained"), sys.exit(0)))
        out.write_text("up")
        time.sleep(60)
    """)
    jobs = [(n, [sys.executable, "-c", child, str(tmp_path / f"{n}.st")], dict(os.environ), tmp_path / f"{n}.log")
            for n in ("a", "b")]
    before = signal.getsignal(signal.SIGTERM)

    def kick():
        for _ in range(200):
            if all((tmp_path / f"{n}.st").exists() for n in ("a", "b")):
                break
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)  # to the driver only: it must reach both children

    threading.Thread(target=kick, daemon=True).start()
    t0 = time.time()
    res = _launch(jobs)
    assert time.time() - t0 < 30 and {n: r["rc"] for n, r in res.items()} == {"a": 0, "b": 0}
    assert all((tmp_path / f"{n}.st").read_text() == "drained" for n in ("a", "b"))
    assert signal.getsignal(signal.SIGTERM) is before
