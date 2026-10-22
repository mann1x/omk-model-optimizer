# Tracking upstream (NVIDIA/Model-Optimizer)

## Branch model

| branch | what | rule |
|---|---|---|
| `main` | **pure mirror** of `upstream/main` | fast-forward only, never committed to |
| `omk-main` (default) | upstream + omk's additive tree | only `omk_qad/`, `omk/`, `.github/README.md`, `.github/workflows/omk-*` may differ from upstream |
| `omk/<topic>` | work branches | merge into `omk-main` |

Because omk never edits an upstream file, merging upstream can only conflict inside files upstream itself changed
twice. The repo's landing page is `.github/README.md` (GitHub shows it before the root `README.md`), so the root
README stays NVIDIA's, byte for byte.

Remotes: `origin` = `mann1x/omk-model-optimizer`, `upstream` = `NVIDIA/Model-Optimizer` with push disabled
(`git remote set-url --push upstream DISABLED-no-push-to-nvidia`).

## Monitoring

```bash
bash omk/scripts/upstream_status.sh               # report
bash omk/scripts/upstream_status.sh --sync-main   # also fast-forward the main mirror and push it
```

It prints how many upstream commits `omk-main` lacks, the **watch list** (the modelopt modules `omk_qad` imports,
derived from the code), the upstream commits and files that touch it, and checks the additive rule (exit 1 if
broken, or if an upstream workflow is active). The watch list today: `modelopt/torch/quantization` (incl. `ggml/` encoders, `extensions.py`,
`ggml/common.py`) and `modelopt/torch/export`.

## Merging upstream into omk-main

1. `upstream_status.sh` — read what moved on the watch list.
2. `git switch omk-main && git merge upstream/main` (merge, not rebase: `omk-main` is published).
3. If the watch list moved, especially `modelopt/torch/quantization/ggml/*`: the encoders training and export rely
   on may have changed. Run the CPU suite **and** a bit-exactness check before trusting new exports:
   `CUDA_VISIBLE_DEVICES= python -m pytest -o addopts="" omk_qad/tests/`, then re-export one trained student and
   compare its GGUF tensor bytes with the previous export (`omk_qad.qad_export` already verifies each file against
   gguf-py's reference decoders; the byte comparison shows whether the encoding itself changed).
4. Rebuild the env if pins moved (`setup_conda_env.sh --update`).
5. Upstream's CI workflows are **disabled** on this repo (they need NVIDIA's runners: GPU / regression jobs
   queue forever, and schedules burn Actions minutes). A merge can add new ones, active by default;
   `upstream_status.sh` flags them. Only `omk-gitleaks` runs.

## Merge log

- **2026-10-22, upstream `d4f2c671f`** (merge `44b3cee29`, from `8a8dceb60`): 45 commits, 10 on the watch list (all
  GGML encoders, `ggml/common.py`, `f1d68fd4d` Megatron-Bridge GGML PTQ/QAD/export). No conflicts; the only pin change
  is a docs extra (no env rebuild). CPU suite 74 passed / 4 skipped (same as before the merge). Bit-exactness: the M6
  Qwen3.5-4B iq2 (IQ2_XXS) stage-1 student re-exported with the pre-merge code (control) and with the merged code:
  both byte-identical to the shipped GGUF, 441/441 tensors and every metadata key. Only the IQ2_XXS export path is
  covered by that byte check; the other encoders by the CPU suite.

  The checks: `omk_qad_work/m6/upstream_bytecheck*.sh` + `gguf_bytes_diff.py` in backup_models. Do **not** nest a
  `resource-lock.sh` claim inside a `claude-hooks-res run` job: it is a wrapper over the same claims and deadlocks.
