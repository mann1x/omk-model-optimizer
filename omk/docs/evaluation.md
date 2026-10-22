# Evaluation protocol

Every model section reports the same three measurements on GGUF files, so a trained student, a stock llama.cpp quant
and the full-precision reference sit on one size / quality curve.

## 1. KLD against the full-precision GGUF

- Reference: the model's **F16 GGUF**, converted once; its logits are saved once (`kld_base_<ref>_c512_n100.bin`) and
  reused for every file.
- Text: `wikitext-2-raw/wiki.test.raw`, **100 chunks × 512 tokens**.
- Tool: `llama-perplexity --kl-divergence` via omnimergekit `scripts/gguf_kld.py`, on a pinned llama.cpp build (the
  Qwen3.5-4B runs: `311d4211bf`). Reported: mean KLD, same top-p %, PPL of the quant.
- Files stock llama.cpp cannot load (PQ2_0 / PTQ1_0) are measured with **opencoti's** build of the same tool against
  the same saved reference. Before trusting such a row, run that tool on the stock F16 (expect KLD ≈ 0) and on a stock
  quant (expect the stock tool's number). Qwen3.5-4B: F16 0.00036 / 98.86 %, Q4_K_M 0.0272 vs 0.0271.
- `omk_qad.ternary_gate unfold` (ternary tensors as F32, no prism keys) gives a stock-runnable twin of a ternary
  file; its KLD must match the real file's (Qwen3.5-4B: 0.9116 vs 0.9117).

KLD on wiki text measures generic fidelity. It does **not** rank coding ability between heavily quantized files (see
the Qwen3.5-4B IQ2_XS re-encode), so it is never the only column.

## 2. HumanEval-164 and HumanEval+

lm-eval-harness, served by `llama-server` (or the opencoti engine for ternary files), raw text completion:

```bash
lm_eval --model local-completions \
  --model_args "model=NAME,base_url=http://127.0.0.1:PORT/v1/completions,num_concurrent=2,tokenizer_backend=huggingface,tokenizer=BASE_MODEL_DIR,max_gen_toks=2048" \
  --tasks humaneval   # and humaneval_plus
  --batch_size 1 --gen_kwargs "temperature=0.0,top_p=1.0" --confirm_run_unsafe_code \
  --use_cache CACHE/NAME.TASK --log_samples --output_path OUT
```

- **Greedy** (`temperature 0, top_p 1`). The task's own 1,024-token cap and stop list apply.
- Server: `-ngl 99 -fa on -c 8192 -np 2`. The tokenizer is the **base model's** HF dir.
- Score = `pass@1,create_test` from `results_*.json`.
- Every run's samples are checked (`omk/scripts/he_samples.py`): count of empty, < 5-char, fenced, think-tag
  generations, length percentiles, > 3,000-char runaways. A flag is investigated before the score is reported.
- n = 164: one problem is 0.61 points; the binomial standard error at ~50 % is ~3.8 points. Differences below that
  are noise.
- Engine control: a stock quant is also run on the second engine, so an engine's own effect on a row is bounded
  (Qwen3.5-4B Q4_K_M: 62.80 llama.cpp vs 60.37 opencoti).

## 3. Stock baselines

`quantize_gguf.py` (omnimergekit) defaults for each tier, the **v5 calibration imatrix** (`calibration_datav5.txt`,
128 chunks, `--parse-special`) computed once on the same F16 and kept next to the quants, same llama.cpp build. The
imatrix, the F16 and the KLD reference are reused across tiers (the script checks each before regenerating).

## Collecting

Each model directory has an `evals/collect_results.py` that reads the run directory (KLD jsonl, opencoti logs,
lm-eval results) and writes `results.json` + `results.md`. The model README embeds `results.md` between
`results:begin` / `results:end` markers; tables are regenerated, never hand-edited.
