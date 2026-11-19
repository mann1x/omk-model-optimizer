# SPDX-License-Identifier: Apache-2.0
"""Which omk_qad tests does a change touch? -- the import graph decides, not a hand-kept list (2026-11-19).

The full CPU suite is 345 tests and ~2.5 h, almost all of it ``python -m omk_qad.train`` subprocesses. A change to one
kernel (``q8.py`` + ``kernels/q8.cu``), to the K-FAC rounder or to the ternary codec must not cost that; the full suite
is for the end of a development stint, an upstream sync, or an explicit ask. This module computes the impacted test
set from the code itself:

* **modules**: every ``omk_qad/*.py`` (name = stem); ``omk_qad/kernels/<stem>.{cu,cpp,cuh}`` belong to ``<stem>``.
* **edges** (AST, every ``import`` node, nested ones included): module A -> module B when A imports B. An import
  at module level is a **strong** edge (A cannot load without B). An import inside a function is a **weak** edge (A
  uses B on some paths only: ``fakequant`` reaching ``ternary``, ``kfac_round`` reaching ``q8`` via qtype dispatch).
* **tests**: a test file depends on the modules it imports, the modules it launches as ``python -m omk_qad.X`` (string
  literals), and, transitively, the deps of sibling test modules it imports (``tests.test_qad_cpu``).
* **closure**: changed modules C. Strong-impacted = fixpoint of "imports a member strongly". Weak-impacted = fixpoint
  over strong+weak edges, minus the strong set. A test file is *selected* when its deps meet C or the strong set;
  *extended* when they meet only the weak set. Extended files run in full only with ``--impact-weak``; otherwise only
  their tests that NAME the changed kernel's qtype (``Q8_0``, ``IQ2_``, ...) are kept (see conftest).
* changed test file -> itself; changed ``conftest.py`` / ``impact.py`` / ``pyproject.toml`` -> everything; anything
  outside ``omk_qad/`` -> nothing.

CLI::

    python -m omk_qad.tests.impact                 # uncommitted changes (git status) -> selection
    python -m omk_qad.tests.impact --rev HEAD~3    # a commit range
    python -m omk_qad.tests.impact --files omk_qad/q8.py omk_qad/kernels/q8.cu
    python -m omk_qad.tests.impact --graph         # dump the module graph (strong / weak edges)
    python -m omk_qad.tests.impact --json          # machine-readable (what conftest consumes)

Used by ``conftest.py`` for ``pytest --impact [REV] [--impact-weak]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent  # omk_qad/
REPO = PKG.parent
TESTS = PKG / "tests"
KERNEL_EXT = (".cu", ".cpp", ".cuh", ".h")
# a kernel module's qtype vocabulary: a test that names one of these is a direct test of that module even when the
# file reaches it only through the fakequant / kfac_round dispatch (a weak edge)
QTYPE_TOKENS: dict[str, tuple[str, ...]] = {
    "q8": ("Q8_0",),
    "iq2": ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M"),
    "iq3": ("IQ3_XXS", "IQ3_S", "IQ3_M"),
    "iq4": ("IQ4_XS", "IQ4_NL"),
    "kquant": ("Q4_K", "Q5_K", "Q6_K", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S"),
    "ternary": ("PQ2_0", "PQ1_0", "TQ1_0", "TQ2_0", "ternar"),
}
# files whose change voids every selection (the selection machinery itself, or the test harness)
EVERYTHING = {"omk_qad/tests/conftest.py", "omk_qad/tests/impact.py", "pyproject.toml", "omk_qad/__init__.py"}
_MOD_IN_STR = re.compile(r"omk_qad\.([a-z_][a-z0-9_]*)")


def _modules() -> set[str]:
    return {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}


def _import_edges(path: Path, pkg_depth: int = 0) -> tuple[set[str], set[str]]:
    """(strong, weak) omk_qad module names imported by a file. Nested (in-function) imports are weak."""
    tree = ast.parse(path.read_text(), filename=str(path))
    strong: set[str] = set()
    weak: set[str] = set()

    def names_of(node: ast.AST) -> set[str]:
        out: set[str] = set()
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1:  # relative: `from .x import y` / `from . import x` / `from ..x import y` (tests)
                if node.module:
                    out.add(node.module.split(".")[0])
                else:
                    out |= {a.name for a in node.names}
            elif node.module and node.module.split(".")[0] == "omk_qad":
                parts = node.module.split(".")
                if len(parts) >= 2:
                    out.add(parts[1])
                else:
                    out |= {a.name for a in node.names}  # `from omk_qad import x`
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "omk_qad" and len(parts) >= 2:
                    out.add(parts[1])
        return out

    def walk(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            nested = in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                (weak if in_func else strong).update(names_of(child))
            walk(child, nested)

    walk(tree, False)
    return strong, weak


def module_graph() -> dict[str, dict[str, set[str]]]:
    """{module: {"strong": {deps}, "weak": {deps}}} over omk_qad/*.py (self-edges and non-modules dropped)."""
    mods = _modules()
    g: dict[str, dict[str, set[str]]] = {}
    for m in sorted(mods):
        s, w = _import_edges(PKG / f"{m}.py")
        s = {d for d in s if d in mods and d != m}
        w = {d for d in w if d in mods and d != m} - s
        g[m] = {"strong": s, "weak": w}
    return g


def test_deps() -> dict[str, set[str]]:
    """{test file name: omk_qad modules it exercises} -- imports, `-m omk_qad.X` literals, sibling test imports."""
    mods = _modules()
    files = sorted(p for p in TESTS.glob("test_*.py"))
    direct: dict[str, set[str]] = {}
    siblings: dict[str, set[str]] = {}
    for p in files:
        s, w = _import_edges(p)
        deps = {d for d in (s | w) if d in mods}
        text = p.read_text()
        deps |= {m for m in _MOD_IN_STR.findall(text) if m in mods}
        sib = set(re.findall(r"tests\.(test_[a-z0-9_]+)", text)) | {
            d for d in (s | w) if d.startswith("test_")}
        direct[p.name] = deps
        siblings[p.name] = {f"{x}.py" for x in sib if (TESTS / f"{x}.py").exists()}
    out: dict[str, set[str]] = {}
    for name in direct:
        seen, todo, acc = set(), [name], set()
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            acc |= direct.get(n, set())
            todo.extend(siblings.get(n, set()))
        out[name] = acc
    return out


def reverse_closure(graph: dict[str, dict[str, set[str]]], changed: set[str]) -> tuple[set[str], set[str]]:
    """(strong_impacted, weak_impacted): modules that import a changed module, transitively, by edge strength."""

    def fix(use_weak: bool) -> set[str]:
        hit = set(changed)
        while True:
            more = {m for m, e in graph.items()
                    if m not in hit and (e["strong"] & hit or (use_weak and e["weak"] & hit))}
            if not more:
                return hit - changed
            hit |= more

    strong = fix(False)
    weak = fix(True) - strong
    return strong, weak


def changed_paths(rev: str | None, files: list[str] | None) -> list[str]:
    """Repo-relative paths: explicit --files, or `git diff --name-only REV`, or the uncommitted worktree."""
    if files:
        return [str(Path(f).resolve().relative_to(REPO)) if Path(f).is_absolute() else f for f in files]
    if rev:
        out = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", rev], capture_output=True, text=True,
                             check=True).stdout
        return [ln for ln in out.splitlines() if ln]
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True, check=True).stdout
    paths = []
    for ln in out.splitlines():
        p = ln[3:]
        if " -> " in p:
            p = p.split(" -> ", 1)[1]
        paths.append(p)
    return paths


def path_to_module(p: str) -> str | None:
    """omk_qad/x.py -> x; omk_qad/kernels/x.cu -> x; else None."""
    parts = Path(p).parts
    if len(parts) >= 2 and parts[0] == "omk_qad":
        if len(parts) == 2 and parts[1].endswith(".py") and parts[1] != "__init__.py":
            return Path(parts[1]).stem
        if len(parts) == 3 and parts[1] == "kernels" and parts[2].endswith(KERNEL_EXT):
            return Path(parts[2]).stem
    return None


def analyse(paths: list[str]) -> dict:
    """The selection for a list of changed repo-relative paths."""
    graph = module_graph()
    deps = test_deps()
    mods = set(graph)
    changed_mods = {m for m in (path_to_module(p) for p in paths) if m and m in mods}
    changed_tests = {Path(p).name for p in paths if Path(p).parent == Path("omk_qad/tests") and Path(p).name in deps}
    everything = sorted(p for p in paths if p in EVERYTHING)
    ignored = sorted(p for p in paths if p not in EVERYTHING and not path_to_module(p)
                     and Path(p).name not in changed_tests)
    strong, weak = reverse_closure(graph, changed_mods)
    hit = changed_mods | strong
    selected = sorted(set(changed_tests) | {t for t, d in deps.items() if d & hit})
    extended = sorted(t for t, d in deps.items() if t not in selected and d & weak)
    tokens = sorted({tok for m in changed_mods for tok in QTYPE_TOKENS.get(m, ())})
    return {
        "changed_paths": sorted(paths), "changed_modules": sorted(changed_mods), "changed_tests": sorted(changed_tests),
        "everything": everything, "ignored": ignored,
        "strong_impacted": sorted(strong), "weak_impacted": sorted(weak),
        "selected": selected, "extended": extended, "qtype_tokens": tokens,
        "full": bool(everything),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: print the selection for the worktree / a rev / explicit files, or dump the graph."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rev", help="git diff --name-only REV (default: the uncommitted worktree)")
    ap.add_argument("--files", nargs="*", help="explicit changed paths (repo-relative or absolute)")
    ap.add_argument("--graph", action="store_true", help="print the module graph and the test deps and exit")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.graph:
        g = module_graph()
        for m, e in g.items():
            strong, weak = " ".join(sorted(e["strong"])) or "-", " ".join(sorted(e["weak"])) or "-"
            print(f"{m:14s} strong: {strong:40s} weak: {weak}")
        print()
        for t, d in test_deps().items():
            print(f"{t:30s} {' '.join(sorted(d))}")
        return 0
    r = analyse(changed_paths(a.rev, a.files))
    if a.json:
        print(json.dumps(r, indent=1))
        return 0
    print(f"changed modules : {' '.join(r['changed_modules']) or '-'}")
    print(f"changed tests   : {' '.join(r['changed_tests']) or '-'}")
    if r["everything"]:
        print(f"FULL SUITE      : {' '.join(r['everything'])} changed")
    print(f"strong impacted : {' '.join(r['strong_impacted']) or '-'}")
    print(f"weak impacted   : {' '.join(r['weak_impacted']) or '-'}")
    print(f"ignored paths   : {len(r['ignored'])} (outside omk_qad)")
    print(f"SELECTED  ({len(r['selected']):2d}) : {' '.join(r['selected']) or '-'}")
    print(f"EXTENDED  ({len(r['extended']):2d}) : {' '.join(r['extended']) or '-'}"
          + (f"   (kept now: tests naming {' '.join(r['qtype_tokens'])})" if r["qtype_tokens"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
