# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of omk_qad.replay generate (resume and --redo-capped).

Resume skips answered prompts; --redo-capped re-asks a row that hit a lower length cap and archives the old pair
OUTSIDE the capture dir (rollouts --capture scans it recursively).
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from omk_qad.replay import generate


def _server(finish="stop"):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = {"choices": [{"finish_reason": finish, "message": {"content": "ok"}}],
                   "usage": {"completion_tokens": 7}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _resp(finish, tokens):
    body = {"choices": [{"finish_reason": finish, "message": {"content": ""}}], "usage": {"completion_tokens": tokens}}
    return {"at": "t", "status": 200, "body": json.dumps(body), "complete": True, "error": None}


def _args(tmp_path, prompts, **kw):
    a = {"prompts": str(prompts), "url": "", "model": "m", "out": str(tmp_path / "cap"), "max_tokens": 32768,
         "parallel": 1, "timeout": 30, "limit": 0, "log_every": 10, "redo_capped": False, "allow_change": []}
    a.update(kw)
    return SimpleNamespace(**a)


def _setup(tmp_path):
    prompts = tmp_path / "p.jsonl"
    rows = [{"id": f"code/{n}", "body": {"messages": [{"role": "user", "content": n}]}} for n in ("a", "b", "c")]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cap = tmp_path / "cap" / "code"
    cap.mkdir(parents=True)
    for n, rec in (("a", _resp("length", 16384)), ("b", _resp("stop", 900)), ("c", _resp("length", 32768))):
        (cap / f"{n}.json").write_text(json.dumps({"body": {"max_tokens": 16384}, "n": n}))
        (cap / f"{n}.response.json").write_text(json.dumps(rec))
    return prompts, cap


def test_resume_skips_capped_without_flag(tmp_path):
    prompts, _ = _setup(tmp_path)
    srv, seen = _server()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv.server_port}"))
    finally:
        srv.shutdown()
    assert seen == []
    assert not (tmp_path / "cap.capped").exists()


def test_redo_capped_reasks_only_lower_cap_and_archives(tmp_path):
    prompts, cap = _setup(tmp_path)
    old_a = (cap / "a.response.json").read_text()
    srv, seen = _server()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv.server_port}", redo_capped=True))
    finally:
        srv.shutdown()
    # only "a" (capped at 16384 < 32768) is re-asked; "b" finished, "c" already hit the current cap
    assert len(seen) == 1 and seen[0]["messages"][0]["content"] == "a" and seen[0]["max_tokens"] == 32768
    arch = tmp_path / "cap.capped" / "16384" / "code"
    assert (arch / "a.response.json").read_text() == old_a
    assert json.loads((arch / "a.json").read_text())["n"] == "a"
    new = json.loads((cap / "a.response.json").read_text())
    assert new["complete"] and json.loads(new["body"])["choices"][0]["finish_reason"] == "stop"
    assert json.loads((cap / "a.json").read_text())["body"]["max_tokens"] == 32768
    assert not list((tmp_path / "cap").glob("**/*capped*"))


def _gated_server():
    """Stub that holds each response until `release` is set; `arrived` fires when a request is in."""
    seen, arrived, release = [], threading.Event(), threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            arrived.set()
            release.wait(10)
            data = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                               "usage": {"completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen, arrived, release


def _fresh_prompts(tmp_path, n=3):
    prompts = tmp_path / "p.jsonl"
    rows = [{"id": f"code/{i}", "body": {"messages": [{"role": "user", "content": str(i)}]}} for i in range(n)]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return prompts


def _state(tmp_path):
    return json.loads((tmp_path / "cap.state.json").read_text())


def test_drain_finishes_inflight_then_stops(tmp_path, capsys):
    prompts = _fresh_prompts(tmp_path)
    srv, seen, arrived, release = _gated_server()

    def driver():
        arrived.wait(10)
        (tmp_path / "cap.control").write_text("drain\n")
        release.set()

    threading.Thread(target=driver, daemon=True).start()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv.server_port}"))
    finally:
        srv.shutdown()
    assert len(seen) == 1                                   # the in-flight request was finished, no new one started
    assert json.loads((tmp_path / "cap/code/0.response.json").read_text())["complete"]
    assert not (tmp_path / "cap/code/1.json").exists()
    assert _state(tmp_path)["status"] == "drained"
    assert "GENERATE_DRAINED" in capsys.readouterr().out
    # restart: the control file is reset to run and the rest is asked; the drained row is not re-asked
    release.set()
    srv2, seen2 = _server()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv2.server_port}"))
    finally:
        srv2.shutdown()
    assert [b["messages"][0]["content"] for b in seen2] == ["1", "2"]
    assert _state(tmp_path)["status"] == "done"


def test_pause_then_resume_without_restart(tmp_path):
    import time

    prompts = _fresh_prompts(tmp_path)
    srv, seen, arrived, release = _gated_server()
    log = {}

    def driver():
        arrived.wait(10)
        (tmp_path / "cap.control").write_text("pause\n")
        release.set()
        for _ in range(100):
            if (tmp_path / "cap.state.json").exists() and _state(tmp_path)["status"] == "paused":
                break
            time.sleep(0.1)
        log["paused_seen"] = len(seen)
        log["paused_status"] = _state(tmp_path)["status"]
        time.sleep(2.5)                                     # stays paused: nothing new is asked
        log["after_wait"] = len(seen)
        (tmp_path / "cap.control").write_text("run\n")

    threading.Thread(target=driver, daemon=True).start()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv.server_port}"))
    finally:
        srv.shutdown()
    assert log == {"paused_seen": 1, "paused_status": "paused", "after_wait": 1}
    assert len(seen) == 3 and _state(tmp_path)["status"] == "done"


def test_second_instance_is_refused(tmp_path):
    import fcntl

    import pytest

    prompts = _fresh_prompts(tmp_path, 1)
    (tmp_path / "cap").mkdir()
    with (tmp_path / "cap.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.write("pid 999 since test\n")
        held.flush()
        with pytest.raises(SystemExit, match="RUN_REFUSED replay generate"):
            generate(_args(tmp_path, prompts, url="http://127.0.0.1:9"))


def test_torn_response_is_moved_aside_and_reasked(tmp_path):
    prompts = _fresh_prompts(tmp_path, 1)
    d = tmp_path / "cap" / "code"
    d.mkdir(parents=True)
    (d / "0.json").write_text("{}")
    (d / "0.response.json").write_text('{"at": "t", "complete": tr')   # killed mid-write by an older writer
    srv, seen = _server()
    try:
        generate(_args(tmp_path, prompts, url=f"http://127.0.0.1:{srv.server_port}"))
    finally:
        srv.shutdown()
    assert len(seen) == 1
    assert (tmp_path / "cap.bad" / "code" / "0.response.json").read_text().endswith("tr")
    assert json.loads((d / "0.response.json").read_text())["complete"]
    assert not list(d.glob("*.tmp"))


def test_restart_with_another_model_is_refused(tmp_path):
    import pytest

    prompts = _fresh_prompts(tmp_path, 1)
    srv, _ = _server()
    try:
        url = f"http://127.0.0.1:{srv.server_port}"
        generate(_args(tmp_path, prompts, url=url))
        with pytest.raises(SystemExit, match=r"RUN_REFUSED replay generate: .*model: 'm' -> 'other'"):
            generate(_args(tmp_path, prompts, url=url, model="other"))
        generate(_args(tmp_path, prompts, url=url, model="other", allow_change=["model"]))
    finally:
        srv.shutdown()


def test_prompts_skip_an_empty_user_turn(tmp_path):
    """A masked RL-Ultra row (empty question until fill_placeholders.py) is counted, never sampled (bug-843)."""
    from omk_qad.replay import prompts

    def row(text):
        return {"agent_ref": {"name": "math_with_judge_simple_agent"},
                "responses_create_params": {"input": [{"role": "user", "content": text}]}}

    src = tmp_path / "rlvr.jsonl"
    src.write_text("".join(json.dumps(row(t)) + "\n" for t in ("", "  ", "What is 2 + 2?")))
    out = tmp_path / "p.jsonl"
    prompts(SimpleNamespace(gate="", no_gate=True, src=str(src), seed=1, per_category=5, gate_frac=0.1,
                            out=str(out)))
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert [r["body"]["messages"][-1]["content"] for r in rows] == ["What is 2 + 2?"]
    census = json.loads(out.with_suffix(".manifest.json").read_text())["census"]
    assert census["math:empty_user_turn"] == 2
