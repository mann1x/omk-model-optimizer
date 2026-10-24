# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the rollouts run-safety paths: the resumable /apply-template cache and the live-capture refusal."""

import fcntl
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from omk_qad.rollouts import TemplateVerifier, refuse_live_capture


def _server(template="T1", fail_first=0, fail_always=False):
    calls = {"apply": 0, "template": template, "fail_first": fail_first, "fail_always": fail_always}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"chat_template": calls["template"]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls["apply"] += 1
            if calls["fail_always"] or calls["apply"] <= calls["fail_first"]:
                return self._send(500, {"error": "busy"})
            self._send(200, {"prompt": "P:" + body["messages"][0]["content"]})

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls, f"http://127.0.0.1:{srv.server_port}"


def _body(t):
    return {"messages": [{"role": "user", "content": t}]}


def test_verify_retries_caches_and_resumes(tmp_path):
    cache = tmp_path / "x.roles.jsonl.verify.jsonl"
    srv, calls, url = _server(fail_first=1)
    try:
        v = TemplateVerifier(url, cache, retries=(0, 0), log=lambda *_: None)
        assert v.matches(_body("a"), "P:a") and not v.matches(_body("b"), "P:wrong")
        assert calls["apply"] == 3  # one failure retried, then a and b
        v.fh.close()
        v2 = TemplateVerifier(url, cache, retries=(0, 0), log=lambda *_: None)
        assert v2.matches(_body("a"), "P:a") and not v2.matches(_body("b"), "P:wrong")
        assert calls["apply"] == 3  # both answered from the cache
        v2.fh.close()
        calls["template"] = "T2"  # the server now renders with another template: the cache is not reused
        v3 = TemplateVerifier(url, cache, retries=(0, 0), log=lambda *_: None)
        assert v3.matches(_body("a"), "P:a") and calls["apply"] == 4
        v3.fh.close()
        assert len(list(tmp_path.glob("*.verify.jsonl.stale-*"))) == 1
    finally:
        srv.shutdown()


def test_verify_failure_stops_and_keeps_earlier_answers(tmp_path):
    cache = tmp_path / "v.jsonl"
    srv, calls, url = _server()
    try:
        v = TemplateVerifier(url, cache, retries=(0,), log=lambda *_: None)
        assert v.matches(_body("a"), "P:a")
        calls["fail_always"] = True
        with pytest.raises(SystemExit, match="1 answers are cached, rerun the same command to resume"):
            v.matches(_body("b"), "P:b")
        v.fh.close()
        cache.write_text(cache.read_text() + '{"key": "torn')  # a build killed mid-line
        calls["fail_always"] = False
        v2 = TemplateVerifier(url, cache, retries=(0,), log=lambda *_: None)
        before = calls["apply"]
        assert v2.matches(_body("a"), "P:a") and calls["apply"] == before  # cached
        assert v2.matches(_body("b"), "P:b") and calls["apply"] == before + 1
    finally:
        srv.shutdown()


def test_live_capture_is_refused(tmp_path):
    root = tmp_path / "cap"
    root.mkdir()
    refuse_live_capture(root)  # an SDK capture: no lock, no state file
    with (tmp_path / "cap.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.write("pid 7 since t\n")
        held.flush()
        with pytest.raises(SystemExit, match=r"still being written \(pid 7"):
            refuse_live_capture(root)
    (tmp_path / "cap.state.json").write_text(json.dumps({"status": "drained"}))
    with pytest.raises(SystemExit, match="status is 'drained'"):
        refuse_live_capture(root)
    (tmp_path / "cap.state.json").write_text(json.dumps({"status": "done"}))
    refuse_live_capture(root)
