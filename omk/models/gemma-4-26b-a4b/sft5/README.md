# Gemma-4 26B-A4B-it bf16: sft5 render and launcher (bm-889)

sft5 trains on the sft4 corpus (`../sft4/`) re-rendered with one change to the chat template: in assistant tool calls,
string arguments that contain a newline are written after the other arguments. Every value is byte-identical; only
the order of the arguments in the rendered call changes. Paths are bs2's as run
(`/mnt/sdc/omk-model-optimizer/g4_recovery`, work dir `sft4_build`).

## Why

The sft4 step 200 gate (2026-11-02) produced a new form of the swallowed-argument defect: an `intent` string closed by
a newline and the next field name (`... */\nnew_text:`) instead of the closing delimiter. The base model has none in
10 gate runs; sft2 and sft3 had the older form, not this one.

The shipped Gemma-4 template sorts call arguments by name (`dictsort`). lambda kimi's `write_file` writes whole files,
which end in a newline, and `content` sorts before `path`, so training showed "value ends with a newline, then another
field" 4,882 times. The sft2/sft3 corpus shows it 206 times and sft1's opencode 7 times. Kimi is also the only sft4
source with newlines inside call strings (1.04M lines, about 65,000 of them YAML/env `key: value` lines). Measured with
`analysis/newline_census.py`, `newline_where.py` and `yaml_lines_census.py`, on trained positions only.

## What changed

- `corpus/chat_template_mlast2.jinja`: the shipped template with `format_argument(..., ml_last=False)`. Only the
  assistant call-argument site passes `ml_last=True`. It writes single-line keys first and multi-line string keys
  last, each in name order, at the top level and inside nested objects and lists. Tool declarations and tool
  responses render as before.
- `corpus/chat_template_mlast.jinja`: first version, top level only. Its gate failed on 37 `delegate_task` calls
  whose nested `context` came before `goal`; kept as the record of that render.
- `corpus/sanitize_kimi_nl.py`: an earlier attempt that stripped edge newlines from kimi values. The owner rejected it
  because it changes file contents. Not rendered or trained; its counts are in `lambda_kimi_pre2r_nl.sanitize.json`.

## Pipeline

1. `corpus/run_render_v3_mlast2.sh` renders the 14 files listed in `jobs_v3_mlast2.txt`, with the same inputs and
   flags as the sft4 corpus, via `render_sft4_mlast.py` (`render_sft4.py` plus `MLAST_TPL`).
2. `corpus/sft5_gates.py` checks:
   - all 14 files are complete
   - the 12 non-kimi files are byte-identical to the sft4 corpus
   - kimi has the same rows
   - no trained multi-line value is followed by a single-line field

   It then writes `CORPUS_MANIFEST.json`.
3. `sft2_overlap.py` writes `OVERLAP_OK` (no held-out text in training).
4. `train/sft5_chain.sh` runs 1 to 3, then:
   - stops sft4 with one SIGTERM
   - deletes sft4's checkpoints and the step 200 export weights (owner's decision; logs, json and W&B kept)
   - launches `train/sft5_train.sh`

## Result of the render

- 2,495 windows and 33,933,689 scored tokens, against 2,495 and 33,933,726 for sft4. The difference is in kimi, where
  reordered arguments tokenize slightly differently at their boundaries.
- Kimi trained multi-line values: 6,870 in total.
  - 6,414 are last in their call or object.
  - 456 are followed by another multi-line value: `patch` carries both old and new text, so one must come first.
  - None is followed by a single-line field.

## Training

Same as sft4: omk_qad with `--loss ce` window length from the data, `--mode bf16 --bf16-latent fp16`, scope attention,
dense MLP, router and routed experts, `--bs 1 --seq 32768`, one epoch = 2,495 steps, peak lr 1e-2/steps = 4.01e-6,
cosine, warmup 10, eval every 200 steps on the corpus's own held-out splits, checkpoints every 100 (steps 100, 200
and 400 kept). Run dir `runs/g4_bf16_sft5`, W&B `bm889-g4-bf16-sft5`.

## Known gaps

- The served template still sorts by name. History turns at inference show the shipped order, and the model's own
  calls can come in any order; the parser accepts any order.
- The gate checks the training data, not the model. Whether the defect is gone has to be measured on a gate run of
  an sft5 checkpoint.
