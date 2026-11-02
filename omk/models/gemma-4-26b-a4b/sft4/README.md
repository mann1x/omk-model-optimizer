# Gemma-4 26B-A4B-it bf16: sft4 corpus and launcher (bm-889)

The scripts that built the sft4 SFT corpus and launched its training on bs2 on 2026-11-02, with their provenance
records. No data is stored here: the parquet sources come from the Hugging Face revisions in `corpus/REVISIONS.txt`
(sha256 in `corpus/SHA256SUMS`), and each rendered file's sha256 is in `corpus/CORPUS_MANIFEST.json`.

Paths are bs2's as run: `/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build` (work dir),
`/mnt/sdc/ml/models/gemma-4-26B-A4B-it` (tokenizer and model), `/mnt/sdc/omk-model-optimizer/venv` (Python).

## Corpus

Train: 2,495 windows of 32,768 tokens, 33.9M scored tokens. Each source keeps its own eval split.

| Source | Train rows | Scored tokens | Origin |
|---|---|---|---|
| interstellarninja/hermes_reasoning_tool_use | 23,775 | 15.21M | upstream rows, validated and filtered |
| lambda/hermes-agent-reasoning-traces, config `kimi` | 2,715 | 13.03M | upstream rows, validated and filtered |
| ifchat_v1, ifchat_v2, multilingual, science, cp | 3,303 | 5.69M | sft2 corpus rows (Nemotron prompts, Gemma-4-regenerated thinking), re-rendered at 32k |

## Pipeline (run in this order, from the work dir)

1. **Validate** every hermes-family source with one rule set: `validate_hermes.py NAME PARQUET validated`.
   A row fails on the first of: tag imbalance in an assistant or tool message, a call that is not JSON or names a
   tool not offered or has non-object arguments, a tool message without a call (or the reverse), no final answer.
   Passing rows carry per-turn `think_ok` (a `<think>` of at least 20 characters) and flags (thinking loop,
   duplicate, Python-literal tool response). Reports: `validated/*.report.json`.
2. **Select**: `select_rows.py` keeps passing rows where every assistant turn thinks, with no loop, duplicate or
   Python-literal response. Counts: `selection.json`.
3. **Convert** to the renderer's chat format: `build_conv.py NAME PARQUET [allow-preamble]`, through an-finetune's
   `replay_normalize.convert_hermes` and sft2's `sft2_jobs.split_tool_responses` (neither is in this repo), then
   re-validated. The split is by sha256 of the first user message, so variants of one prompt stay on one side (a
   random row split leaked 3.2% of interstellarninja eval positions). `allow-preamble` keeps call turns that carry
   visible text; the renderer moves that text into the turn's reasoning. Counts: `conv_stats/`.
4. **Render** at 32k: `render_sft4.py IN OUT HISTORY TOOL_PREAMBLE` (`run_render.sh` with `jobs.txt`, then
   `run_render_v2.sh` with `jobs_v2.txt` for the regrouped hermes splits). It calls `omk_qad.rollouts.build` with
   `--pack` and drops segments over 32k instead of windowing them. Stats: `render_stats/`.
5. **Gate**: `sft2_overlap.py RENDER_DIR` writes `OVERLAP_OK` only when no scored eval token ends a 200-gram seen
   in train. `find_rt.py` found the one Kimi row whose render does not survive a tokenizer round-trip.
6. **Train**: `train/sft4_train.sh` checks the manifest hashes, `OVERLAP_OK` and free disk, then runs `omk_qad.train`.

## Decisions (the owner's, 2026-11-02)

- Kimi config, not GLM-5.1. Same validator: GLM-5.1 passes 0 of 7,055 rows (96.4% of later turns have an empty
  `<think>`; 6,312 rows end without a final answer). `census_glm51.py` measured it.
- Kimi rows without a final answer, with a bad call or with any turn without thinking are dropped.
- Kimi call turns with visible text beside the call keep that text, moved into the turn's reasoning
  (`--tool-preamble reasoning`). Left as content, the Gemma-4 template renders it after the call.
- Dropped sources: Jofthomas/hermes-function-calling-thinking-V1 (3,511 of 3,570 rows write calls as Python dicts)
  and all NousResearch/hermes-function-calling-v1 configs (no thinking).
- 32k windows. The old tool-calling sources of the sft2/sft3 corpus (agentic, opencode, swe, math and the 3,875-row
  hermes cut) are out: their trained turns lacked thinking 15% to 49% of the time. `corpus_quality.py` measured it.

## Training (sft4_train.sh)

omk_qad at commit 20a3e2e58 semantics (`--loss ce` window length from the data; one 32k eval row per forward).
`--mode bf16 --bf16-latent fp16`, scope attention, dense MLP, router and routed experts, `--bs 1 --seq 32768`,
2,495 steps (one epoch), peak lr 4.01e-6 (1e-2 / steps), cosine, warmup 10, CE loss, eval every 200 steps on the
corpus's own eval splits, checkpoints every 100 (step 100, 200 and 400 snapshots kept), divergence guard ratio 3.
First steps on an RTX PRO 6000 (96 GB): about 64 s per step, 77.9 GiB peak.

## Known gaps

- `rollouts --in` renders without `preserve_thinking`, so the thinking of tool-call turns in earlier user segments is
  not in a row's context, while the server shows it. Scored turns are unaffected.
- interstellarninja: the renderer refused 1,237 of 33,560 segments whose per-turn render is not a prefix of the
  segment render; they are dropped, not trained.
- 5 Kimi eval rows were dropped for overlap: the token that opens the thinking channel after a harness tool output
  that is identical in a train row (`kimi_pre2r_eval_overlap_drop.json`).
