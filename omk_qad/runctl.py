# SPDX-License-Identifier: Apache-2.0
"""Run control shared by every long omk_qad step: safe to pause, resume, stop and restart.

The rules (docs: backup_models/docs/plans/plan_omk_qad_run_safety.md):

  R1 resumable           finished units are never redone, unfinished ones never silently skipped
  R2 graceful stop       drain at a unit boundary (unit written, sentinel, exit 0); a 2nd SIGTERM stops at once
  R3 pause / resume      in-process, for anything long
  R4 atomic outputs      temp + rename; a reload never sees a torn file
  R5 one process         per output dir (``RunLock``), the refusal names the holder
  R6 identity            a restart with settings that made the existing outputs different is REFUSED unless the key
                         is passed to ``--allow-change`` (logged, recorded in the manifest history)
  R7 visible state       ``State`` json + sentinel lines

  atomic_write / atomic_torch_save   temp file unique per process (two writers can never rename each other's)
  RunLock(path)                      flock; ``SystemExit("RUN_REFUSED ...")`` when another process holds it
  Control(path)                      ``<path>`` holds run | pause | drain; SIGUSR1 = drain, SIGUSR2 = toggle pause,
                                     SIGTERM / SIGINT = drain then (2nd) stop at once; ``boundary()`` blocks while
                                     paused and returns the mode to act on; ``restore()`` puts the old handlers back
  State(path)                        ``update(status=..., **fields)`` -> atomic json with pid + time
  check_identity(path, current, allow)
                                     first run writes the manifest; later runs compare and refuse on a difference
  file_id(path)                      cheap content identity of a (possibly huge) file
  publish_verified(out, build)       build under a temp name, rename into place only when it verified
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

MODES = ("run", "pause", "drain")


def now() -> str:
    """UTC timestamp, the format every omk_qad log line uses."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def atomic_write(path: str | Path, data: str | bytes) -> None:
    """Write ``data`` to ``path`` via a per-process temp file + fsync + rename."""
    path = Path(path)
    tmp = _tmp(path)
    with open(tmp, "wb") as fh:
        fh.write(data.encode() if isinstance(data, str) else data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_json(path: str | Path, obj: Any) -> None:
    """``atomic_write`` of ``json.dumps(obj, indent=1)``."""
    atomic_write(path, json.dumps(obj, indent=1, default=str))


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    """``torch.save`` via a per-process temp file + rename."""
    import torch

    path = Path(path)
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def file_id(path: str | Path, full_below: int = 64 << 20) -> dict:
    """Content identity: sha256 for files below ``full_below``, else size + sha256 of the first and last 4 MiB.

    A ``.sha256`` sidecar (first token) is used when present, so a known artifact is not re-hashed.
    """
    p = Path(path)
    side = p.with_name(p.name + ".sha256")
    st = p.stat()
    if side.exists():
        tok = side.read_text().split()
        if tok:
            return {"size": st.st_size, "sha256": tok[0]}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        if st.st_size < full_below:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
            return {"size": st.st_size, "sha256": h.hexdigest()}
        h.update(fh.read(4 << 20))
        fh.seek(-(4 << 20), os.SEEK_END)
        h.update(fh.read(4 << 20))
    return {"size": st.st_size, "head_tail_sha256": h.hexdigest()}


class RunLock:
    """Exclusive flock on ``path`` for the life of the process (or the ``with`` block)."""

    def __init__(self, path: str | Path, what: str = ""):
        """Take the lock or exit with RUN_REFUSED naming the holder."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.seek(0)
            holder = self.fh.read().strip() or "unknown holder"
            self.fh.close()
            raise SystemExit(f"RUN_REFUSED {what or self.path}: another process holds {self.path} ({holder})")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid {os.getpid()} host {socket.gethostname()} since {now()} cmd {' '.join(sys.argv)[:300]}\n")
        self.fh.flush()

    def release(self) -> None:
        """Drop the lock (also dropped by the OS when the process exits)."""
        if not self.fh.closed:
            self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class Control:
    """Run mode from a control file and signals. pause / drain never interrupt a unit: they act at ``boundary()``."""

    def __init__(self, path: str | Path, name: str = "run", log=print):
        """Reset the control file to run and install the signal handlers."""
        self.path = Path(path)
        self.name, self.log = name, log
        self.term = 0
        self.pending: list[str] = []  # signal-handler log lines, printed at the next mode() (print is not reentrant)
        if self.path.exists():
            old = self.path.read_text().strip()
            if old != "run":
                self.log(f"{now()} {name}: control file said {old!r} at start -> reset to run (a restart is explicit)")
        atomic_write(self.path, "run\n")
        self.prev = {s: signal.signal(s, h) for s, h in ((signal.SIGUSR1, self._drain), (signal.SIGUSR2, self._toggle),
                                                         (signal.SIGTERM, self._term), (signal.SIGINT, self._term))}

    def restore(self) -> None:
        """Put back the signal handlers that were installed before this Control."""
        for s, h in self.prev.items():
            signal.signal(s, h)

    def set(self, mode: str, why: str, _from_signal: bool = False) -> None:
        """Change the mode (written to the control file so ``control status`` and other readers agree)."""
        atomic_write(self.path, mode + "\n")
        msg = f"{now()} {self.name}: {why} -> {mode}"
        if _from_signal:  # a handler can interrupt a print of the main thread: never print from it
            self.pending.append(msg)
        else:
            self.log(msg)

    def _drain(self, *_):
        self.set("drain", "SIGUSR1", True)

    def _toggle(self, *_):
        self.set("run" if self._read() == "pause" else "pause", "SIGUSR2", True)

    def _term(self, signum, _frame):
        self.term += 1
        if self.term > 1:
            os.write(2, f"{now()} {self.name}: second {signal.Signals(signum).name}: stopping now "
                        "(the unit in progress is redone on the next run)\n".encode())
            os._exit(130)
        self.set("drain", f"{signal.Signals(signum).name} (send again to stop at once)", True)

    def mode(self) -> str:
        """Current mode: run | pause | drain (logs what signal handlers did since the last call)."""
        while self.pending:
            self.log(self.pending.pop(0))
        return self._read()

    def _read(self) -> str:
        try:
            m = self.path.read_text().strip()
        except OSError:
            m = "run"
        return m if m in MODES else "run"

    def boundary(self, on_pause=None, poll: float = 2.0) -> str:
        """Call between units. Blocks while paused (``on_pause()`` once on entry); returns "run" or "drain"."""
        m = self.mode()
        if m == "pause":
            if on_pause:
                on_pause()
            self.log(f"{now()} {self.name}: PAUSED (control file {self.path} -> run to resume, or SIGUSR2)")
            while (m := self.mode()) == "pause":
                time.sleep(poll)
            self.log(f"{now()} {self.name}: {'RESUMED' if m == 'run' else 'drain while paused'}")
        return m


class State:
    """Atomic state json: ``update(status=..., **fields)`` merges and rewrites."""

    def __init__(self, path: str | Path, **fixed):
        """``fixed`` fields are written on every update; fields of an earlier run's file are kept until replaced."""
        self.path = Path(path)
        try:
            old = json.loads(self.path.read_text())
        except (OSError, ValueError):
            old = {}
        self.data: dict = {**old, "pid": os.getpid(), **fixed}

    def refused_or_failed(self, e: BaseException) -> None:
        """Record why a run ended abnormally: a RUN_REFUSED relaunch leaves the earlier run's status alone."""
        if isinstance(e, SystemExit) and e.code in (0, None):
            return
        msg = f"{type(e).__name__}: {e}"[:800]
        if isinstance(e, SystemExit) and str(e.code).startswith("RUN_REFUSED"):
            self.update(last_refusal=msg)
        else:
            self.update(status="failed", error=msg)

    def update(self, **fields) -> None:
        """Merge ``fields`` and rewrite the file."""
        self.data.update(fields, at=now())
        atomic_json(self.path, self.data)


def publish_verified(out: str | Path, build, sidecars: tuple[str, ...] = (), report: str = ".export.json"):
    """Build ``out`` (and its ``.with_suffix`` sidecars) under a temp name; the final name appears only on success.

    ``build(part)`` writes ``<stem>.partial<suffix>`` and raises on a failed verify. Success: sidecars, then the file
    itself, are renamed into place (the file last: its name is the completion marker) and the ``report`` json gets
    ``ok: true`` + the final path. Failure (any exception, a kill included on the next run): the partial set is kept
    as ``<stem>.verify-failed<suffix>`` with ``ok: false``. One process per ``out`` (``<out>.lock``).
    """
    out = Path(out)
    part = out.with_name(f"{out.stem}.partial{out.suffix}")
    failed = out.with_name(f"{out.stem}.verify-failed{out.suffix}")
    lock = RunLock(out.with_name(out.name + ".lock"), f"export {out}")
    names = (*sidecars, report)

    def move(src: Path, dst: Path, ok: bool) -> None:
        for suf in names:
            if src.with_suffix(suf).exists():
                os.replace(src.with_suffix(suf), dst.with_suffix(suf))
        rep = dst.with_suffix(report)
        if rep.exists():
            atomic_json(rep, {**json.loads(rep.read_text()), "out": str(out if ok else dst), "ok": ok})
        if src.exists():
            os.replace(src, dst)

    try:
        for f in (part, *(part.with_suffix(x) for x in names)):  # leftovers of a killed export: never complete
            if f.exists():
                f.unlink()
        try:
            res = build(part)
        except BaseException:
            move(part, failed, False)
            raise
        move(part, out, True)
        if isinstance(res, dict):
            res.update(out=str(out), ok=True)
        return res
    finally:
        lock.release()


def parse_allow(items: list[str] | None) -> dict[str, str | None]:
    """``--allow-change k`` or ``k=v`` -> {k: v or None}. ``k=v`` only allows that exact new value."""
    out: dict[str, str | None] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip() if _ else None
    return out


def check_identity(path: str | Path, current: dict, allow: dict[str, str | None] | None = None,
                   what: str = "run", log=print) -> dict:
    """First call writes ``current`` as the identity manifest; later calls refuse any difference not allowed.

    Returns the (possibly updated) manifest. An allowed change is logged and appended to ``history``.
    """
    path = Path(path)
    allow = allow or {}
    cur = json.loads(json.dumps(current, default=str))
    if not path.exists():
        atomic_json(path, {"identity": cur, "created": now(), "history": []})
        return {"identity": cur}
    man = json.loads(path.read_text())
    old = man.get("identity", {})
    diff = {k: (old.get(k), cur.get(k)) for k in sorted(set(old) | set(cur)) if old.get(k) != cur.get(k)}
    bad = {k: v for k, v in diff.items()
           if k not in allow or (allow[k] is not None and str(v[1]) != allow[k])}
    if bad:
        lines = "; ".join(f"{k}: {a!r} -> {b!r}" for k, (a, b) in bad.items())
        raise SystemExit(f"RUN_REFUSED {what}: existing outputs in {path.parent} were made with different settings "
                         f"({lines}). Use a new output dir, or pass --allow-change {' '.join(bad)} if mixing them "
                         "is intended.")
    if diff:
        for k, (a, b) in diff.items():
            log(f"{now()} {what}: ALLOWED CHANGE {k}: {a!r} -> {b!r}")
        man.setdefault("history", []).append({"at": now(), "changed": {k: list(v) for k, v in diff.items()}})
        man["identity"] = cur
        atomic_json(path, man)
    return man
