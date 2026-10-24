# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D103
"""CPU tests of the GGUFTeacher child lifecycle, against a fake ``llama_topk`` speaking the same pipe protocol.

A hung child is killed by the per-sequence deadline (never a wait forever); ``close()`` stops a child that ignores
n=0 and SIGTERM; a parent that dies hard (``os._exit``, as the second SIGTERM does) takes the child with it.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch

from omk_qad.gguf_teacher import GGUFTeacher

FAKE = textwrap.dedent("""\
    #!{py}
    import os, signal, struct, sys, time
    mode, k = os.environ["FAKE_MODE"], int(sys.argv[sys.argv.index("--topk") + 1])
    if mode == "deaf":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.stderr.write("loading\\nLLAMA_TOPK_READY n_vocab=10\\n"); sys.stderr.flush()
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    while True:
        h = inp.read(4)
        if len(h) < 4:
            break
        (n,) = struct.unpack("i", h)
        if n == 0:
            if mode == "deaf":
                while True:
                    time.sleep(1)
            break
        inp.read(4 * n + n - 1)
        if mode in ("hang", "orphan"):
            while True:
                time.sleep(1)
        for _ in range(n - 1):
            out.write(struct.pack("i", 1) * k + struct.pack("f", -0.5) * k + struct.pack("f", -1.0))
        out.flush()
""")


def _fake(tmp_path):
    exe = tmp_path / "llama_topk"
    exe.write_text(FAKE.format(py=sys.executable))
    exe.chmod(0o755)
    return exe


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = open(f"/proc/{pid}/stat").read().split()[2] if os.path.exists(f"/proc/{pid}/stat") else "X"
    return st != "Z"


def test_answer_and_clean_close(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "ok")
    t = GGUFTeacher("m.gguf", 16, 4, 10, binary=str(_fake(tmp_path)), log=lambda *_: None)
    idx, val, nll = t(torch.zeros(2, 5, dtype=torch.int64))
    assert idx.shape == (2, 4, 4) and (idx == 1).all() and nll == pytest.approx(8.0)
    t.close()
    t.close()  # twice is fine
    assert t.p.returncode == 0


def test_hung_child_is_killed_by_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    t = GGUFTeacher("m.gguf", 16, 4, 10, binary=str(_fake(tmp_path)), log=lambda *_: None, seq_timeout=1.0)
    t0 = time.time()
    with pytest.raises(SystemExit, match="no answer within 1s, killed"):
        t(torch.zeros(1, 5, dtype=torch.int64))
    assert time.time() - t0 < 20 and t.p.poll() is not None


def test_close_stops_a_child_that_ignores_n0_and_sigterm(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "deaf")
    t = GGUFTeacher("m.gguf", 16, 4, 10, binary=str(_fake(tmp_path)), log=lambda *_: None)
    t.p.wait = _short_wait(t.p.wait)  # the 120 s / 30 s grace periods, shortened for the test
    t.close()
    assert t.p.poll() is not None


def _short_wait(wait):
    return lambda timeout=None: wait(timeout=min(timeout or 1, 1))


def test_child_dies_with_a_hard_killed_parent(tmp_path):
    exe = _fake(tmp_path)
    code = textwrap.dedent(f"""
        import os, sys, torch, threading
        from omk_qad.gguf_teacher import GGUFTeacher
        t = GGUFTeacher("m.gguf", 16, 4, 10, binary={str(exe)!r}, log=lambda *_: None)
        print(t.p.pid, flush=True)
        threading.Thread(target=t, args=(torch.zeros(1, 5, dtype=torch.int64),), daemon=True).start()
        import time; time.sleep(0.5)
        os._exit(130)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": os.getcwd(), "FAKE_MODE": "orphan"})
    child = int(r.stdout.split()[0])
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child), f"llama_topk {child} outlived its parent"
