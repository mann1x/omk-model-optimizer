# Testing omk_qad — suites, impact selection, resumable runs

The omk_qad suite is 345 tests in 20 files. Run whole, on the CPU, it takes about 2.5 h, almost all of it in the
8 files that train tiny models through `python -m omk_qad.train` subprocesses (149 tests). Nobody should pay that for
a change to one kernel. The suite is therefore a set of tools, and the **full run is reserved for three moments**:
the end of a development stint, an upstream sync, and an explicit ask. Everything else runs the impacted tests only.

Everything below lives in `omk_qad/tests/`: `conftest.py` (selection + logging hooks), `impact.py` (the change →
tests map), `omk_test.sh` (the runner). The env is `omk-modelopt`; `-o addopts=""` is mandatory (upstream's addopts
want plugins the env does not have) and the runner adds it.

## Validating a change: impact first

```bash
omk_qad/tests/omk_test.sh impact                 # uncommitted worktree -> which tests the change reaches
omk_qad/tests/omk_test.sh impact --rev HEAD~3    # a commit range
omk_qad/tests/omk_test.sh impact --files omk_qad/q8.py omk_qad/kernels/q8.cu
omk_qad/tests/omk_test.sh run q8fix --impact     # run exactly that selection (resumable run named "q8fix")
```

The map is computed from the code, never hand-kept. `impact.py` parses every `import` in `omk_qad/*.py` and in the
test files (including `python -m omk_qad.X` subprocess literals and sibling test imports). An import at module level
is a **strong** edge; an import inside a function is a **weak** edge (`fakequant` → `ternary`, `kfac_round` → `q8`
through qtype dispatch). A changed module selects the test files whose dependencies meet it or its strong closure
in full; files reached only through weak edges are *extended*: of those, only the tests that name the changed
kernel's qtype run (`Q8_0`, `IQ2_XS`, ...). `--impact-weak` runs the extended files in full. A change to the harness
itself (conftest, impact.py, pyproject) selects everything. `omk_qad/kernels/<x>.cu|.cpp` count as module `<x>`.

What a change costs today (files selected / extended):

| change | selected | extended (qtype-named tests only) |
|---|---|---|
| `q8.py` + `kernels/q8.cu` | test_q8_native, test_gguf_eog_cpu | test_kfac_round_ggml, test_iq2_native |
| `kfac_round.py` | test_q8_native, test_kfac_round_ggml, test_iq2_native | — |
| `ternary.py` | test_ternary, test_qad_cpu, test_native_quants, test_iq2_native, test_run_safety_cpu | 7 files |
| `train.py` | 8 files (every subprocess-training file) | test_kfac_cpu |

`train.py` is the hub: a change there legitimately reaches most of the slow core.

## Suites

```bash
omk_qad/tests/omk_test.sh suites                 # the table
omk_qad/tests/omk_test.sh run k --suite kfac     # one suite
omk_qad/tests/omk_test.sh run kt --suite kernels,ternary
omk_qad/tests/omk_test.sh run one omk_qad/tests/test_q8_native.py::test_reference_matches_libggml   # one test
```

Suites (`conftest.SUITES`, a file may belong to several): `kernels` (iq2, iq3, iq4, q8), `kfac`, `ternary`,
`layout`, `losses`, `qad` (the trainer end to end — the slow core), `gguf`, `recovery` (+ `rollouts`), `runctl`, and
the computed `fast` (every file that launches no subprocess: 196 tests). A test file not in the table fails
collection, so the table cannot rot. Any pytest argument passes through (`-k`, `-x`, nodeids, `--durations`).

## Runs: resume, pause, stop

Every run has a name and a progress file; every test result lands in it the moment it finishes.

```bash
T=omk_qad/tests/omk_test.sh
$T run full                 # the whole suite, detached; log + progress under $OMK_QAD_WORK/runs/full.*
$T status                   # alive / PAUSED / ended, counts, last test, log tail
$T pause full               # SIGSTOP the process group (pytest AND its train subprocess) -- frees the CPU
$T cont full                # SIGCONT
$T stop full                # SIGINT: the current test finishes, the summary is written
$T resume full              # same argv, skips every test that already passed in full.jsonl
```

`--resume RUN.jsonl` is a plain pytest option too. A killed run (OOM, lost ssh, host reboot) resumes the same way.
One pytest session per `$OMK_QAD_WORK` at a time (bug-968): the runner refuses to start while a run is alive.

## Profile

Every result is also appended to `$OMK_QAD_WORK/durations.jsonl`, so the suite's cost is measured, not guessed:

```bash
omk_qad/tests/omk_test.sh profile 30        # 30 slowest tests + per-file minutes and shares
omk_qad/tests/omk_test.sh profile 30 full   # from one run's progress file
```

## Failures

Each failure is written as it happens (`OMK_TEST_FAILED <phase> <nodeid>` + traceback tail) on stderr and in
`$RUNS/NAME.failures.log` (or `$OMK_PYTEST_FAILLOG` / `$OMK_QAD_WORK/failures.log` for a bare pytest call), bracketed
by `OMK_TEST_SESSION_START` / `_END`: a start without an end is a killed run.

## When the full suite runs

1. End of a development stint (before the publish / the merge of the stint's commits).
2. After an upstream sync (`omk/docs/upstream.md`).
3. On demand.

Between those, `--impact`. The CUDA kernel tests (`kernels` suite) need the GPU visible: `OMK_TEST_GPU=1`.
