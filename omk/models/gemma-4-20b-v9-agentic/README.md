# Gemma-4 20B v9-agentic (sft12) — the 98-expert prune of the agentic 26B, retrained, quantized with K-FAC

**Published:** [ManniX-ITA/gemma-4-20B-v9-agentic](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic)
(BF16 safetensors, the **sft12** build, replaced 2026-11-16) ·
[ManniX-ITA/gemma-4-20B-v9-agentic-GGUF](https://huggingface.co/ManniX-ITA/gemma-4-20B-v9-agentic-GGUF)
(K-FAC ladder + `imatrix.dat`; IQ2_XXS from the seeded QAD) · Ollama `mannix/gemma-4-20B-v9-agentic`.

**What it is.** [gemma-4-26B-A4B-agentic](../gemma-4-26b-a4b/README.md) (the sft11 loop-free recovery of
`google/gemma-4-26B-A4B-it`) with 30 of its 128 experts per layer removed by the published v9-agentic competence map,
**no shared-expert upweight**, then one round of short-trace agentic SFT (sft12) on the pruned model's own weights.
30 layers × 98 experts, 20.45 B parameters, same tokenizer, chat template, attention and norms as the 26B. Earlier
releases under this name (2026-10-17, built from the stock `-it` with the mandatory `α = 1.2` shared-FFN upweight) are
superseded; their numbers on the model card are kept only as the previous build's reference.

| | |
|---|---|
| mode | [BF16 recovery](../../docs/bf16-recovery.md) SFT (`--mode bf16 --loss ce`), scope `dense_mlp,router,routed_experts` |
| quantization | [K-FAC rounding](../../docs/kfac-rounding.md) of every `llama-quantize` tier (AC imatrix + Gemma-4 attention rule); IQ2_XXS = [QAD](../../docs/qad.md) seeded from the K-FAC rounding |
| hosts | bs2 (one RTX PRO 6000 Blackwell 96 GB per job), pod 60948837 (2× RTX 3090) for the Q6_K suite |
| tasks | bm-893 (model), bm-895 (K-FAC), bm-897 (ladders + publish) |

## Why

The 98-expert prune of the stock `-it` (the 2026-10 v9-agentic) and its first pruned-sft11 variants "struggle to
converge" in agent runs: too many failed tool calls, long ruminations. The 2026-11-13/14 A/B on eleven2go (xollama,
Q4_K_M, the owner's own agent sessions) walked through the map variants (refA, A_sft11, hub_k12, ml_k8) and the
shared-expert upweight (α 1.2 vs none): the **ml_k8 map without upweight** was the best of them but still weaker
than the 26B. sft12 trains that model back toward the 26B on the agentic families it serves, with short, convergent
traces only (a cut turn teaches a turn that never ends).

## 1. Weights: the prune (`mlk8-noa`)

- source: the sft11 final of the 26B agentic (`/mnt/sdc/bm889-exports/gemma-4-26B-A4B-agentic`, the published BF16).
- drop map: omnimergekit `recipes/gemma4/v9_agentic/maps/v9ml_ladder/ml_k8.json`, sha256 `84f2bd96…` — the map of the
  published 20B v9-agentic (refA + 8 multilingual/IFEval rescues), unchanged.
- tool: omnimergekit `scripts/build_v9_agentic_sft11_quick.sh inputs/ml_k8.json mlk8-noa 1 none`: `expert_drop.py`
  on the sft11 weights, **`ALPHA=none`** — `router_shared_upweight.py` is not run, the shared expert stays exactly as in
  sft11 (`shared_alpha.txt` = `none`); official chat template; sft11's generation / preprocessor config.
- The "shared alpha 1.2" variants are dead ends for this model: the α-1.2 builds of refA / A_sft11 / ml_k8 were
  regenerated without it on 2026-11-14 at the owner's word, and a last test applying α 1.2 to the finished sft12
  (`mlk8-noa-sft12-sa120`, Q4_K_M, quick imatrix) "does not work" (owner, 2026-11-16 08:23) and was removed.
- [`../gemma-4-26b-a4b/v9a_sft11_mlk8_98e/`](../gemma-4-26b-a4b/v9a_sft11_mlk8_98e/): the full sft11 recipe
  (2,108 steps, lr 4.74e-6, attention included) re-run on this prune; stopped at step 748 — code families regressed
  while the long chat families it cannot recover dominated the loss. Its step-600 export was A/B-tested on eleven2go
  and superseded by sft12. sft12 starts from `mlk8-noa` itself, not from that run.

## 2. sft12 — corpus

[`train/sft12_select.py`](train/sft12_select.py) selects segments from sft11's rendered train windows
(`sft11_build/render`, `--history-reasoning strip --pack`, one assistant turn scored per segment) and re-packs them with
the renderer's own `Packer` (16,384-token windows, `<bos>` per segment):

- **families kept**: `agentic`, `opencode` (1 in 3 of its segments: `--opencode-frac 0.33`), `hermes`, `kimi31b` —
  the agentic map's job, short turns — plus an **ifchat anchor** (`ifchat_v1` + `ifchat_v2`) sized to 10 % of the
  estimated loss (held-out init nll × scored tokens) for IFEval.
- **dropped**: `multilingual`, `science`, `cp`, `math`, `swe` — long turns and capacity the prune removed (98e-init vs
  128e-sft11 held-out nll +0.15…+0.41 on the long chat families, ~0 on code).
- **no long traces**: a segment whose scored tokens exceed 1,024 is dropped, not truncated (agentic 617 of 5,427,
  opencode 452 of 2,772, hermes 157 of 3,875, kimi31b 21 of 693, ifchat_v1 1,311 of 1,446). Every kept segment ends on
  `<turn|>` or `<|tool_response>`.
- result `render_oc33`: **1,710 windows → 855 steps at bs 2**, 3.06 M scored tokens; estimated loss share agentic
  64 %, hermes 15 %, ifchat 10 %, kimi31b 6 %, opencode 5 % (`selection.json`). Eval files are hard-linked from sft11
  (the same 20 held-out files, 217 windows, 1,165,554 scored positions), so nll is comparable with sft11 and the mlk8
  run.

## 3. sft12 — training

[`train/sft12_train.sh`](train/sft12_train.sh) (bs2 GPU1, 2026-11-14 19:58Z → 2026-11-15 14:52Z):

```bash
python -m omk_qad.train --phase train --mode bf16 --model <mlk8-noa> --cache <cache_teacher_r4 metadata> --out runs/g4_bf16_sft12_98e_render_oc33 \
  --train-scope classes:dense_mlp,router,routed_experts --bs 2 --steps 855 --lr 2e-6 --fp-lr 2e-6 --warmup 50 \
  --seq 16384 --loss ce --sft-data "<render_oc33/*.train.roles.jsonl>" --sft-eval "<sft11 eval files>" --mtp freeze \
  --eval-every 100 --save-every 100 --snapshot-every 100 --snapshot-keep 2 --guard-ratio 3 \
  --wandb-project omk-qad --wandb-name bm893-g4-98e-sft12-render_oc33
```

- **attention frozen** (scope = dense MLP, router, routed experts: the tensors the prune disturbed); lr 2e-6 (sft11:
  4.74e-6), warmup 50, cosine; one epoch; ~45 s/step, 68.6 GiB.
- **stop rule** (launcher watcher): held-out nll of any code family (`opencode`, `swe`, `cp`) above init + 0.01, `kimi31b`
  above init + 0.02 → pause. The run paused once at step 100 (kimi31b +0.0104 under the first +0.01 limit); the
  step-200 eval showed it was eval noise on 8 rows (+0.002) and the run resumed from step 100 with the +0.02 limit
  (`sft12_train.sh.v1_lim001` is the first launcher).
- step 400 was a wobble (the 9 `sft_mix_*` held-out families rose 0.008–0.022 nll), reversed at 500; snapshots 400,
  500 and `last.pt` kept, `final.pt` = step 855 (36.15 GB).

Held-out nll, init → step 855 (sft11 on the 26B for reference: init 0.864, final 0.926 overall):

| | overall | agentic | opencode | kimi31b | swe | cp | hermes | ifchat_v1 | ifchat_v2 | multilingual | math | science |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| init (`mlk8-noa`) | 1.060 | 0.428 | 0.222 | 0.377 | 0.294 | 0.231 | 0.243 | 0.721 | 0.657 | 0.529 | 0.219 | 0.423 |
| **step 855** | **0.977** | **0.392** | 0.225 | 0.379 | 0.296 | 0.231 | **0.232** | **0.619** | **0.554** | **0.468** | 0.219 | **0.385** |

Overall ppl 2.885 → 2.658. The code families stay within +0.003 of init (the point of the stop rule); the agentic,
hermes and ifchat families recover a third to a half of the prune's loss. W&B `omk-qad/bm893-g4-98e-sft12-render_oc33`.

## 4. Export and publish

- [`train/sft12_export_q4km.sh`](train/sft12_export_q4km.sh): `omk_qad.train --phase export --mode bf16` writes the HF
  dir `gemma-4-20B-v9-agentic-sft11-mlk8-noa-sft12/` from `final.pt`: **180 tensors replaced** (30 layers × experts
  gate_up + down, dense gate/up/down, router), **833 verified byte-identical** to `mlk8-noa`, 9 shards, 40.9 GB, 73 s
  (`omk_bf16_export.json`); then a quick Q4_K_M (calibration_datav5 imatrix) for the eleven2go A/B.
- [`publish/publish_sft12_weights.py`](publish/publish_sft12_weights.py): `SHA256SUMS` → model card notice →
  `upload_folder` to the private repo (stale shards of the 2026-10 build deleted) → `super_squash_history` → size
  verify. Published 2026-11-16 12:28Z, one squashed commit.

## 5. GGUF: F16, AC imatrix

[`quant/sft12_ac_imatrix.sh`](quant/sft12_ac_imatrix.sh): omnimergekit `scripts/quantize_gguf.py --imatrix-only`
with llama.cpp **b12386** (the build every file below is made with): F16 GGUF (39.8 GB, 658 tensors, no MTP head,
`eos = 106`, official template embedded) and the imatrix on the **AtomicChat calib-corpora recipe
`gemma-4-20b-v9-agentic`** (`calib_train.txt`, the same 3,043-document / 4.91 M-token corpus as the 26B's, rendered
through the Gemma-4 template), **all 9,590 chunks, `--parse-special`**, computed on this build:
`imatrix_ac.dat`, 88.5 MB, sha256 `3b23762e…` — published as `imatrix.dat` in the GGUF repo. Every tier of the ladder
carries its `quantize.imatrix.*` keys (file, 9,590 chunks, 295 entries).

## 6. K-FAC ladder

Method: [`docs/kfac-rounding.md`](../../docs/kfac-rounding.md). This model's inputs:

- **factors** — [`quant/kfac_collect_20b.sh`](quant/kfac_collect_20b.sh): `omk_qad.kfac --windows 512 --seq 2048
  --seed 1 --layers-per-pass 3 --down-g diag --dense --drop-grow 10` on the 20B bf16, token windows from the **26B QAD
  cache** (same tokenizer; the labels are sampled from the 20B itself) → `factors_w512_s2048_g10`, 59 GB, 76 min on
  bs2 GPU0 (2026-11-16 12:58–14:14Z).
- **eval reference** — [`quant/build_eval_from_gguf.py`](quant/build_eval_from_gguf.py): the 26B cache's 44 held-out
  role rows with this model's own F16 GGUF top-64 at every scored position (`llama_topk` b12386) → `eval_20b.pt`,
  256,853 scored positions. A quant is then scored against its own unquantised model (same kernels both sides).
- **tier** — [`quant/kfac_ladder_twins.sh 20b <TIER> all`](quant/kfac_ladder_twins.sh): PTQ = `llama-quantize
  --imatrix imatrix_ac.dat --tensor-type 'blk\.\d+\.attn_(q|k|v|output)\.weight=<q8_0|q6_k>'` (Q8_0 from Q4_K_M up,
  Q6_K below) → `kfac_round --stems all --qtype auto --ref ptq.gguf` → both scored (`ptq.eval.json`,
  `kfac.eval.json`, `LT_TWIN` line) → `ptq.gguf` removed (owner's `.rm_ptq_granted`). A `kfac.gguf` without
  `KFACR_DONE` in its log is quarantined, never scored or shipped (bug-949).
- **lanes** — [`quant/kfac_ladder_lane.sh`](quant/kfac_ladder_lane.sh) runs the tier list (disk-gated, skips a tier
  another lane is on), [`quant/kfac_lanes_after_cache.sh`](quant/kfac_lanes_after_cache.sh) starts the full lanes once
  the QAD cache frees GPU0. Ladder: Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, IQ4_XS, IQ4_NL, Q3_K_M, IQ3_M, IQ3_XS,
  IQ3_XXS, IQ2_M, IQ2_S, IQ2_XS (K-FAC) + IQ2_XXS (QAD, §7).
- **archive / publish** — [`quant/kfac_ladder_archive.sh`](quant/kfac_ladder_archive.sh) (solidpc, sha-verified copy of
  every scored tier) and [`quant/kfac_publish_bs2.py`](quant/kfac_publish_bs2.py) (bs2: upload as
  `gemma-4-20B-v9-agentic-<TIER>.gguf`, verify the hub LFS sha256, `.sha256` sidecar, `imatrix.dat`, the model card's
  tier table; the bs2 copy is removed only once the hub and solidpc copies verify).

Twins so far (44 rows, top-256 KL vs this model's F16; identical size per pair):

| tier | size | llama-quantize + AC imatrix | K-FAC | ΔKL |
|---|---:|---:|---:|---:|
| Q4_K_M (attention Q8_0) | 13.78 GB | 0.00981 / 0.9753 | **0.00822 / 0.9781** | −16.1 % |
| Q6_K (attention Q8_0) | 18.07 GB | 0.00311 / 0.9862 | **0.00292 / 0.9868** | −6.0 % |

The complete table is regenerated on the GGUF card as tiers land. The **Q6_K K-FAC** file is the one the canonical
9-bench suite runs on (pod 60948837, two 3090s, omnimergekit `scripts/eval_v9a_q6k_full_pod.sh kfac`, llama.cpp
b12386 + the PEG-degrade patch, greedy with `presence_penalty 0.025`): HumanEval 98.17, HumanEval+ 90.85 at the time of
writing; the rest is on the model card when it finishes.

## 7. IQ2_XXS: QAD seeded from the K-FAC rounding

[`qad/qad20_seeded_E_chain.sh`](qad/qad20_seeded_E_chain.sh) — one chain, unattended, bs2 (started 2026-11-16 20:12Z):

0. weights check (`SHA256SUMS`);
1. **teacher cache** on GPU0: `omk_qad.train --phase cache` on the 11 sft11 role families (the 26B QAD's data and
   weights, 4,217 windows × 16,384, top-64, 44 eval rows, `--train-scope gemma4 --mtp train`), ~3 h; it also writes
   this model's own corpus imatrix (`cache/imatrix.gguf`, first 256 windows) — the one imatrix every later stage shares;
2. **layout E6**: `omk_qad.layout --rest IQ2_M --fixed routed_experts=iq2_xxs --no-mtp --override-regex
   'ffn_down_exps\.weight$=IQ4_NL' 'ffn_gate\.weight$=Q4_K' 'ffn_up\.weight$=Q4_K' 'ffn_down\.weight$=IQ4_NL'
   'attn_{q,k,v,output}\.weight$=Q6_K'` (checked: 30 IQ2_XXS, 60 IQ4_NL, 60 Q4_K, attention only Q6_K);
3. **PTQ reference** `step1/E6.gguf`: `llama-quantize --imatrix cache/imatrix.gguf --token-embedding-type q5_K` + the
   same tensor types, preset IQ2_M;
4. **seed**: `kfac_round --qtype IQ2_XXS --method gptq --ref E6.gguf --save-latents round/E6_latents` (requires
   `KFACR_DONE` + 30 latent files), then both E6 and the rounding scored;
5. **seed gate**: untrained export with `--init-latents` → `SEED_LATENTS … 30 gate_up`, attention bytes identical,
   routed IQ2_XXS bytes ≤ 0.5 % apart, |Δklk| ≤ 0.0005 vs the rounding's score → `QAD20_GATE_PASS`, else the chain stops;
6. **train** on GPU1 (after the 26B run ends, GPU1 < 2 GB, ≥ 150 GB disk): the 26B qadE6 recipe — `--seq 4096
   --n-seq 16000 --bs 1 --accum 1 --steps 600 --lr 2e-6 --fp-lr 2e-6 --min-lr 1e-7 --warmup 50 --beta1 0 --beta2 0.999
   --clip 1.0 --loss kl --anchor-as-teacher --mtp-weight 0.3 --refresh 4 --ggml-refresh 4 --eval-every 100
   --save-every 100 --snapshot-every 0 --init-latents round/E6_latents --imatrix cache/imatrix.pt
   --imatrix-convention msq --ggml-encode Q8_0 --latent-dtype fp16 --export-only embedding --topk 64`, W&B
   `omk-qad-20b-v9agentic-iq2-qad` / `qadE6_kfacseed_r4`;
7. **export** `qadE6_kfacseed_s600.gguf` (+ sha + score) → `QAD20_DONE`; published as `gemma-4-20B-v9-agentic-IQ2_XXS.gguf`.

The 26B's run of the same recipe is in [`../gemma-4-26b-a4b/kfac/`](../gemma-4-26b-a4b/kfac/) with its numbers.

## Known gaps

- The K-FAC factors use the 26B cache's token windows (the 20B had no cache of its own at the time); the 20B QAD cache
  now exists and a 20B-window collection would be the clean repeat.
- The Q5_K_S twin of 2026-11-16 19:52Z (`LT_TWIN 0.0054 vs 0.0054`) is void: a killed rounder's partial file was
  scored (bug-949); the tier is rebuilt.
- F16 and `mmproj` on the GGUF repo: the F16 is replaced by this build's; the SigLIP projector is untouched by the
  prune and the SFT (vision tower not trained), the 2026-10 `mmproj-…-F16.gguf` stays valid.
- Vision and the Ollama `vision-`/`mtp-` tag ladder are not re-evaluated for this build.
