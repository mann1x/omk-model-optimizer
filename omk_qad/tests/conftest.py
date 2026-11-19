# SPDX-License-Identifier: Apache-2.0
"""omk_qad test-suite hooks: live failure log, suites, impact selection, resume (2026-11-19).

The full CPU suite (345 tests) is ~2.5 h of mostly ``python -m omk_qad.train`` subprocesses, and a change rarely
touches more than a few files of it. This conftest makes the suite a set of tools instead of one monolith:

**Suites** -- every test file belongs to one or more suites (``SUITES`` below; an unlisted file fails collection so the
table cannot rot). ``--suite kfac`` / ``--suite kernels,ternary`` keeps those files only; ``--list-suites`` prints the
table. ``fast`` is computed, not listed: the files that launch no subprocess. Single tests work as always
(``pytest omk_qad/tests/test_q8_native.py::test_x``).

**Impact** -- ``--impact`` (uncommitted worktree), ``--impact REV`` (``git diff REV``) or ``--impact-files P...``
selects the tests the change can reach through the import graph (``omk_qad/tests/impact.py``). Files reachable only
through a weak (in-function) edge are *extended*: of those, only the tests that name the changed kernel's qtype run;
``--impact-weak`` runs them in full. A change to the harness itself (conftest / impact.py / pyproject) selects all.

**Resume** -- ``--resume RUN.jsonl``: every test result is appended to RUN.jsonl as it lands; a rerun with the same
``--resume`` file deselects every test that already passed in it. Pause = stop the process group (the runner's
``pause`` / ``cont``: SIGSTOP / SIGCONT, the train subprocesses included); a killed run resumes where it was.
Independently, every result is appended to ``$OMK_QAD_WORK/durations.jsonl`` -- the suite's accumulating profile
(``omk_test.sh profile`` reads it).

**Live failures** -- each failure is written THE MOMENT IT HAPPENS: ``OMK_TEST_FAILED <phase> <nodeid>`` + traceback
tail on stderr and in ``$OMK_PYTEST_FAILLOG`` (default ``$OMK_QAD_WORK/failures.log``), bracketed by
``OMK_TEST_SESSION_START`` / ``_END`` lines, so a killed run still names its failures (2026-11-19: four ``F`` at 45 %
and a ``timeout 3000`` kill).

Upstream's ``pyproject.toml`` asks for plugins the ``omk-modelopt`` env does not have: run with ``-o addopts=""``
(the runner ``omk_qad/tests/omk_test.sh`` does). One pytest session per ``$OMK_QAD_WORK`` (bug-968).
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

_DEFAULT_WORK = "/srv/dev-disk-by-uuid-f8b1803e-334f-4f4b-af3b-f802bb6883c5/backup_models/omk_qad_work/pytest"
_TRACEBACK_LINES = 30
_HERE = Path(__file__).resolve().parent

# file -> suites. A file may serve several (test_q8_native tests the Q8_0 codec AND the K-FAC rounder's use of it).
SUITES: dict[str, tuple[str, ...]] = {
    "test_iq2_native.py": ("kernels", "iq2"),
    "test_native_quants.py": ("kernels", "iq3", "iq4"),
    "test_q8_native.py": ("kernels", "q8", "kfac"),
    "test_kquant_native.py": ("kernels", "kquant", "kfac"),  # Q4_K / Q5_K / Q6_K CUDA encoders (bm-901)
    "test_kfac_round_ggml.py": ("kfac", "kernels"),
    "test_kfac_cpu.py": ("kfac",),
    "test_seed_latents.py": ("kfac", "qad"),
    "test_ternary.py": ("ternary",),
    "test_layout_floors.py": ("layout",),
    "test_losses.py": ("losses",),
    "test_qad_cpu.py": ("qad",),
    "test_run_safety_cpu.py": ("qad", "runctl"),
    "test_telemetry.py": ("qad",),
    "test_e2e_cpu.py": ("qad", "gguf"),
    "test_gguf_eog_cpu.py": ("gguf", "qad"),
    "test_gguf_teacher_cpu.py": ("gguf",),
    "test_recovery_cpu.py": ("recovery",),
    "test_replay_cpu.py": ("recovery", "runctl"),
    "test_rollouts_cpu.py": ("recovery", "rollouts"),
    "test_rollouts_safety_cpu.py": ("recovery", "rollouts"),
    "test_runctl_cpu.py": ("runctl",),
}
SUITE_DOC = {
    "kernels": "native encoders vs libggml (IQ2/IQ3/IQ4/Q8_0; CUDA tests skip without a GPU)",
    "iq2": "IQ2_XXS/XS/S", "iq3": "IQ3_XXS/S", "iq4": "IQ4_XS/NL", "q8": "Q8_0",
    "kfac": "K-FAC factors, rounder, seeded latents", "ternary": "T1 ternary codec + ternarizers",
    "layout": "layout planning floors", "losses": "omk_qad.losses",
    "qad": "the QAD trainer end to end (subprocess runs on tiny models; the slow core)",
    "gguf": "GGUF export / EOG metadata / GGUF teacher", "recovery": "bf16 recovery mode, replay, rollouts",
    "rollouts": "rollout -> token roles", "runctl": "run lock, control modes, run-safety rules",
    "fast": "(computed) every file that launches no subprocess",
}


def _work() -> Path:
    return Path(os.environ.get("OMK_QAD_WORK", _DEFAULT_WORK))


def _faillog() -> Path:
    p = os.environ.get("OMK_PYTEST_FAILLOG")
    return Path(p) if p else _work() / "failures.log"


def _append(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(text)
    except OSError as e:  # a missing / read-only work volume must not turn into a second failure
        sys.stderr.write(f"OMK_TEST log {path} unavailable ({e})\n")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _is_fast(path: Path) -> bool:
    try:
        return "subprocess" not in path.read_text()
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """--suite / --impact / --resume (see the module docstring)."""
    g = parser.getgroup("omk_qad", "omk_qad suite selection and resume")
    g.addoption("--suite", action="append", default=[], metavar="NAME[,NAME]",
                help="keep only these suites (repeatable; --list-suites shows them)")
    g.addoption("--list-suites", action="store_true", help="print the suite table and exit")
    g.addoption("--impact", nargs="?", const="worktree", default=None, metavar="REV",
                help="keep the tests the changed files reach (default: uncommitted worktree; or a git rev)")
    g.addoption("--impact-files", nargs="*", default=None, metavar="PATH", help="explicit changed paths for --impact")
    g.addoption("--impact-weak", action="store_true", help="run the extended (weak-edge) files in full too")
    g.addoption("--resume", default=None, metavar="RUN.jsonl",
                help="append results here; deselect tests already passed in it")


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers; honour --list-suites; expose --resume to the report hook through the env."""
    config.addinivalue_line("markers", "suite(name): omk_qad suite membership (set from conftest.SUITES)")
    config.addinivalue_line("markers", "fast: launches no subprocess")
    resume = config.getoption("--resume")
    if resume:
        os.environ["_OMK_PYTEST_RESUME"] = str(Path(resume).resolve())
    if config.getoption("--list-suites"):
        names = sorted({s for v in SUITES.values() for s in v} | {"fast"})
        for s in names:
            files = sorted(f for f, v in SUITES.items() if s in v) if s != "fast" else sorted(
                f for f in SUITES if _is_fast(_HERE / f))
            print(f"{s:10s} {SUITE_DOC.get(s, ''):75s}\n{'':10s} {' '.join(files)}")
        pytest.exit("listed", returncode=0)


def _passed_in(run_file: Path) -> set[str]:
    """Nodeids whose LAST call record in the run file passed (or whose setup skipped)."""
    last: dict[str, str] = {}
    if not run_file.exists():
        return set()
    for ln in run_file.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if "nodeid" not in r:
            continue
        if r.get("when") == "call" or (r.get("when") == "setup" and r.get("outcome") == "skipped"):
            last[r["nodeid"]] = r["outcome"]
    return {n for n, o in last.items() if o in ("passed", "skipped")}


def _impact(config: pytest.Config) -> dict | None:
    rev, files = config.getoption("--impact"), config.getoption("--impact-files")
    if rev is None and not files:
        return None
    from omk_qad.tests.impact import analyse, changed_paths

    return analyse(changed_paths(None if rev in (None, "worktree") else rev, files))


def _names_qtype(item: pytest.Item, tokens: list[str]) -> bool:
    if any(t in item.nodeid for t in tokens):
        return True
    try:
        src = inspect.getsource(item.function)  # type: ignore[attr-defined]
    except (OSError, TypeError, AttributeError):
        return False
    return any(t in src for t in tokens)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tag suite markers; apply --suite, --impact, --resume; report what was kept."""
    want = {s for v in config.getoption("--suite") for s in v.split(",") if s}
    unknown = want - {s for v in SUITES.values() for s in v} - {"fast"}
    if unknown:
        raise pytest.UsageError(f"--suite: unknown {sorted(unknown)}; --list-suites shows the table")
    imp = _impact(config)
    weak_ok = config.getoption("--impact-weak")
    resume = config.getoption("--resume")
    passed = _passed_in(Path(resume)) if resume else set()
    fast_cache: dict[str, bool] = {}
    keep, drop_suite, drop_imp, drop_res = [], [], [], []
    for it in items:
        fname = Path(str(it.fspath)).name
        if fname not in SUITES:
            raise pytest.UsageError(f"{fname} is not in conftest.SUITES -- add it (every test file needs a suite)")
        fast = fast_cache.setdefault(fname, _is_fast(Path(str(it.fspath))))
        for s in SUITES[fname]:
            it.add_marker(pytest.mark.suite(s))
        if fast:
            it.add_marker(pytest.mark.fast)
        if want and not (want & set(SUITES[fname]) or ("fast" in want and fast)):
            drop_suite.append(it)
            continue
        if imp is not None and not imp["full"]:
            selected = fname in imp["selected"]
            extended = fname in imp["extended"] and (weak_ok or _names_qtype(it, imp["qtype_tokens"]))
            if not (selected or extended):
                drop_imp.append(it)
                continue
        if it.nodeid in passed:
            drop_res.append(it)
            continue
        keep.append(it)
    deselected = drop_suite + drop_imp + drop_res
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep
    parts = [f"OMK_SELECT kept {len(keep)} of {len(keep) + len(deselected)}"]
    if want:
        parts.append(f"suite={','.join(sorted(want))} (-{len(drop_suite)})")
    if imp is not None:
        parts.append("impact=FULL (harness changed)" if imp["full"] else
                     f"impact: changed={','.join(imp['changed_modules'] + imp['changed_tests']) or '-'} "
                     f"selected={len(imp['selected'])} files extended={len(imp['extended'])} files (-{len(drop_imp)})")
    if resume:
        parts.append(f"resume={resume} (-{len(drop_res)} already passed)")
    sys.stderr.write("  ".join(parts) + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------------------------------------------------
# live failure log + per-test progress / durations
# ---------------------------------------------------------------------------------------------------------------------
def pytest_sessionstart(session: pytest.Session) -> None:
    """Header line per run: time, argv, failures-log path (also printed, so a reader knows where to grep)."""
    line = f"OMK_TEST_SESSION_START {_now()} pid {os.getpid()} argv {' '.join(sys.argv)}\n"
    _append(_faillog(), line)
    resume = session.config.getoption("--resume")
    if resume:
        _append(Path(resume), json.dumps({"session_start": _now(), "pid": os.getpid(), "argv": sys.argv}) + "\n")
    sys.stderr.write(f"omk_qad tests: live failure log {_faillog()}\n")
    sys.stderr.flush()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Append every result (progress + durations profile); on a failure, print and log the traceback head at once."""
    rec = {"ts": _now(), "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
           "duration": round(report.duration, 3)}
    if report.when == "call" or report.failed or report.skipped:
        _append(_work() / "durations.jsonl", json.dumps(rec) + "\n")
        resume = os.environ.get("_OMK_PYTEST_RESUME")
        if resume:
            _append(Path(resume), json.dumps(rec) + "\n")
    if not report.failed:
        return
    phase = "ERROR" if report.when != "call" else "FAILED"
    head = str(report.longrepr).splitlines()
    tb = "\n".join(head[-_TRACEBACK_LINES:]) if len(head) > _TRACEBACK_LINES else "\n".join(head)
    text = f"OMK_TEST_FAILED {phase} {report.nodeid} ({report.duration:.1f}s)\n{tb}\n"
    sys.stderr.write("\n" + text)
    sys.stderr.flush()
    _append(_faillog(), f"{_now()} {text}\n")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """End line: a log without it after a START is a killed run."""
    _append(_faillog(), f"OMK_TEST_SESSION_END {_now()} exitstatus {int(exitstatus)} "
            f"failed {session.testsfailed} collected {session.testscollected}\n")
    resume = session.config.getoption("--resume")
    if resume:
        _append(Path(resume), json.dumps({"session_end": _now(), "exitstatus": int(exitstatus),
                                          "failed": session.testsfailed}) + "\n")
