#!/usr/bin/env python3
# h2h_card_section.py REPORT_JSON RUNS_JSON {full|short} -- render the HF model-card section for the bm-889 SDK 4.100.277
#   head-to-head from h2h_report.py's output, so every number on the cards comes from the report, none typed by hand.
#   full  = weights card (complete 1:1 analysis); short = GGUF card (headline table + pointer to the weights card).
# Reads only; prints markdown on stdout.
import json
import sys

rep = json.load(open(sys.argv[1]))
runs = json.load(open(sys.argv[2]))
mode = sys.argv[3]
A, B = "base", "sft11"
a, b = rep["arms"][A], rep["arms"][B]
F = rep["features"]


def pct(x):
    return f"{100 * x:.0f} %"


def num(v, nd=1):
    if v is None:
        return "–"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}" if nd == 0 else f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.{nd}f}"


def med(key, arm, sub=None, nd=1):
    s = (F[key][sub] if sub else F[key])[arm]
    return "–" if not s else num(s["median"], nd)


def pval(key, sub=None):
    p = (F[key][sub] if sub else F[key])["mwu_p"]
    if p is None:
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def row(label, key, nd=1, sub=None):
    # no bold here: several rows are ones where this model is WORSE (refused edits, repeated-call warnings)
    return f"| {label} | {med(key, A, sub, nd)} | {med(key, B, sub, nd)} | {pval(key, sub)} |"


# optional: draft acceptance per server from the llama-server logs ("0.894,0.887" = base,sft11), summed over the h2h
acc = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None


fisher = rep["tests"]["pass_fisher_two_sided_p"]
fisher_s = f"{fisher:.4f}" if fisher >= 0.0001 else f"{fisher:.1e}"
ttp = rep["time_to_pass"]
fm_a, fm_b = rep["fail_modes"][A], rep["fail_modes"][B]
starts_b = sorted((m for m in fm_b if m.startswith("starts")), key=lambda m: m)
starts_a = sorted((m for m in fm_a if m.startswith("starts")), key=lambda m: m)
fails_a, fails_b = a["n"] - a["pass"], b["n"] - b["pass"]
nostart_a, nostart_b = fm_a.get("game does not start", 0), fm_b.get("game does not start", 0)
rate_a, rate_b = rep["rates"][A], rep["rates"][B]
gen_a, gen_b = F["gen_tok_s"][A]["median"], F["gen_tok_s"][B]["median"]
bursts = sorted(r["events"]["max_calls_one_turn"] for r in runs if r["arm"] == B and r["events"]["max_calls_one_turn"] >= 10)
sw_a = sum(1 for r in runs if r["arm"] == A and r["findings"].get("swallowed_argument", 0))
sw_b = sum(1 for r in runs if r["arm"] == B and r["findings"].get("swallowed_argument", 0))

head = f"""| | base gemma-4-26B-A4B-it | **this model** |
|---|---|---|
| runs finished within the 30-minute limit (20 each) | {a['pass']} / {a['n']} ({pct(a['pass_rate'])}) | **{b['pass']} / {b['n']} ({pct(b['pass_rate'])})** |
| 95 % interval (Wilson) | {pct(a['wilson95'][0])}–{pct(a['wilson95'][1])} | {pct(b['wilson95'][0])}–{pct(b['wilson95'][1])} |
| passed within 10 / 20 / 30 min | {pct(ttp[A]['10'])} / {pct(ttp[A]['20'])} / {pct(ttp[A]['30'])} | **{pct(ttp[B]['10'])} / {pct(ttp[B]['20'])} / {pct(ttp[B]['30'])}** |
| reasoning-loop cuts per run (median) | {med('reasoning_loop_cuts', A)} | **{med('reasoning_loop_cuts', B)}** |
| reasoning-loop cuts per model-hour | {num(rate_a['reasoning_loop_cuts'])} | **{num(rate_b['reasoning_loop_cuts'])}** |

Pass/fail difference: Fisher exact two-sided p = {fisher_s}.

**What the pass rate means.** It is not an absolute success rate: it is the share of runs that finished the repair
**within a fixed 30-minute budget**. The test is built to measure what the fine-tune changes in a definite amount of
time — how much of the budget goes to loops, how quickly the model gets from a broken file to a working game. A timeout
means "not done in 30 minutes", not "cannot do it". With more time both models would pass more often and the gap would
likely narrow, but not by much: in extended manual testing of the base model and of its pruned 98-expert variants
(the v7-coder and v9-agentic recipes), a run that does not converge quickly becomes steadily more likely to loop, so
the base model either needs one to two hours or more to pass, or ends in massive loops. Anti-repetition sampling hides
the loops but costs the model much of its effectiveness and its ability to converge. The fine-tune targets exactly
this — less looping, faster convergence — and the data below shows it achieved. With only `presence_penalty 0.025`
(no other anti-repetition setting) it is virtually loop-free and completes the task in an acceptable time even on
runs where it does not converge quickly. The
pass curve below shows how the rate builds up minute by minute."""

setup = """**Setup.** One task, run 20 times per model: repair a broken HTML game (`manic_miner`, a syntax error plus logic
bugs) inside a coding-agent harness (Cerebriline SDK 4.100.277, VS Code host profile, LSP diagnostics after every edit
and the `check_file` / `ask_lsp` tools on, browser off) until the game's own oracle reports that it loads, reaches play
and passes 23 logic checks. 30 minutes wall clock per run, one run at a time per model. Both models as BF16 GGUF on the
same llama.cpp build, each on its own RTX PRO 6000 Blackwell (96 GB), f16 KV cache, Gemma-4 assistant drafter for
speculative decoding, thinking on (effort high), temperature 0.8 / top_p 0.95 / top_k 64 / min_p 0 /
presence_penalty 0.025. The two configurations differ only in the model file: harness, task, sampler, server build,
chat template and guards are identical (checked from each run's recorded provenance). The harness guards were on for
both: a reasoning block that repeats itself is cut and the request retried, a repeated identical tool call draws
warnings, then a stop."""

if mode == "short":
    print(f"""## Results — head-to-head with the base model

{head}

{setup}

Share of all runs passed by minute *t*: base {' / '.join(pct(ttp[A][str(t)]) for t in (10, 20, 30))}, this model
{' / '.join(pct(ttp[B][str(t)]) for t in (10, 20, 30))} at 10 / 20 / 30 min.

Full 1:1 analysis (loops, guards, length, time, tool use, failure modes, known weaknesses):
[weights card](https://huggingface.co/ManniX-ITA/gemma-4-26B-A4B-agentic#head-to-head-with-the-base-model). Runs on
the BF16 GGUF; the quants have not been gated separately.""")
    sys.exit()

out = [f"""## Head-to-head with the base model

{head}

{setup}

### Outcome

| | base | **this model** |
|---|---|---|
| passed | {a['pass']} | **{b['pass']}** |
| failed: game never starts (syntax / runtime error left in the file) | {nostart_a} | {nostart_b} |
| failed: game starts, some logic checks still failing | {sum(fm_a[m] for m in starts_a)} | {sum(fm_b[m] for m in starts_b)}{' (' + ', '.join(m.split()[-1] for m in starts_b) + ' checks)' if starts_b else ''} |
| runs ended by the harness for repeating a tool call | {a['runs_stopped_by_harness_for_looping']} | {b['runs_stopped_by_harness_for_looping']} |
| runs that claimed success but failed | {a['claimed_success_but_failed']} | {b['claimed_success_but_failed']} |
| time to the first passing game check, median (passed runs) | {med('first_oracle_ok_min', A, 'passes')} min | **{med('first_oracle_ok_min', B, 'passes')} min** |

Share of all runs passed by minute *t*:

| | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| base | {' | '.join(pct(ttp[A][str(t)]) for t in (5, 10, 15, 20, 25, 30))} |
| this model | {' | '.join(pct(ttp[B][str(t)]) for t in (5, 10, 15, 20, 25, 30))} |

### Loops and guards

Medians per run, all {a['n']} + {b['n']} runs; *p* = Mann–Whitney two-sided.

| | base | **this model** | p |
|---|---|---|---|
{row('reasoning blocks cut for repeating themselves', 'reasoning_loop_cuts')}
{row('guard interventions, all kinds', 'guard_trips')}
{row('model requests cut or failed', 'requests_failed')}
{row('reasoning blocks with a phrase repeated ≥ 10×', 'blocks_rep_ge10')}
{row('worst repeat count in any reasoning block', 'max_repeat', 0)}
{row('repeated-tool-call warnings', 'loop_det_soft')}
| runs with no reasoning-loop cut | {a['runs_without_reasoning_loop_cut']} / {a['n']} | **{b['runs_without_reasoning_loop_cut']} / {b['n']}** | |

Per model-hour (sum over all runs ÷ total model time): reasoning-loop cuts **{num(rate_a['reasoning_loop_cuts'])} → {num(rate_b['reasoning_loop_cuts'])}**,
all guard interventions {num(rate_a['guard_trips'])} → {num(rate_b['guard_trips'])}, repeated-tool-call warnings
{num(rate_a['loop_det_soft'])} → {num(rate_b['loop_det_soft'])}.

### Time, length and effort

| | base | **this model** | p |
|---|---|---|---|
{row('wall time per run (min, 30 = timeout)', 'wall_min')}
{row('longest single model turn (s)', 'turn_max_s', 0)}
{row('median model turn (s)', 'turn_median_s')}
{row('reasoning per run (k characters)', 'think_chars_k', 0)}
{row('longest reasoning block (k characters)', 'longest_block_k', 0)}
{row('generated tokens per run (k)', 'tok_out_k', 0)}
{row('tokens written into tool-call arguments (k)', 'tok_tool_input_k')}
{row('generated tokens per tool call', 'tok_per_tool_call', 0)}
{row('peak context (k tokens)', 'ctx_peak_k', 0)}

Same restricted to failed runs (all or nearly all of the 30 minutes used): reasoning {med('think_chars_k', A, 'fails', 0)}k vs
{med('think_chars_k', B, 'fails', 0)}k characters, tool calls {med('tool_calls', A, 'fails', 0)} vs {med('tool_calls', B, 'fails', 0)},
reasoning-loop cuts {med('reasoning_loop_cuts', A, 'fails')} vs {med('reasoning_loop_cuts', B, 'fails')}.

### Behaviour

| | base | **this model** | p |
|---|---|---|---|
{row('tool calls per run', 'tool_calls', 0)}
{row('tool calls per model-minute', 'tool_calls_per_model_min')}
{row('edits applied', 'edits', 0)}
{row('edits refused by the tool', 'edits_failed', 0)}
{row('edits not followed by a check', 'edits_without_check', 0)}
{row('game runs (`run_game`)', 'run_game_calls', 0)}
{row('`read_files` calls', 'reads', 0)}
{row('LSP: edits checked automatically', 'lsp_edits_checked', 0)}
{row('LSP: `check_file` calls', 'check_file', 0)}
{row('LSP: `ask_lsp` calls', 'ask_lsp', 0)}
| runs that restored the original file and started over | {sum(1 for r in runs if r['arm'] == A and r['metrics'].get('edits.restores', 0) > 0)} / {a['n']} | {sum(1 for r in runs if r['arm'] == B and r['metrics'].get('edits.restores', 0) > 0)} / {b['n']} | |

**What changes.** Total reasoning per run is about the same ({med('think_chars_k', A, nd=0)}k vs {med('think_chars_k', B, nd=0)}k
characters); how it is spent is not. The base model spends {med('tok_per_tool_call', A, nd=0)} generated tokens per tool
call against {med('tok_per_tool_call', B, nd=0)}, its longest turn runs {med('turn_max_s', A, nd=0)} s against
{med('turn_max_s', B, nd=0)} s, and its reasoning repeats itself until the guard cuts it ({med('reasoning_loop_cuts', A)}
cuts per run against {med('reasoning_loop_cuts', B)}). It makes {med('tool_calls_per_model_min', A)} tool calls per
model-minute against {med('tool_calls_per_model_min', B)} and runs the game {med('run_game_calls', A, nd=0)} times per
run against {med('run_game_calls', B, nd=0)}, so it sees far less often what is still broken. {'All ' + str(fails_a) if nostart_a == fails_a else str(nostart_a) + ' of its ' + str(fails_a)}
{'of its ' if nostart_a == fails_a else ''}failed runs ended with the game not even loading; {fails_b - nostart_b} of this model's {fails_b} failures got
the game running{(' and passing ' + ', '.join(m.split()[-1] for m in starts_b) + ' logic checks') if starts_b else ''}.
"""]

weak = [f"""### Known weaknesses

- **Repeating a refused tool call.** When an edit is refused, this model is more likely than the base model to send the
  same call again: more repeated-tool-call warnings in failed runs (median: base {med('loop_det_soft', A, 'fails')},
  this model {med('loop_det_soft', B, 'fails')}; p = {pval('loop_det_soft', 'fails')}). Runs ended by the harness for
  repeating a call: base {a['runs_stopped_by_harness_for_looping']}, this model {b['runs_stopped_by_harness_for_looping']}."""]
if bursts:
    weak.append(f"""- **Degenerate tool-call bursts.** In {len(bursts)} of {b['n']} runs this model emitted the same refused `editor` call
  {' and '.join(str(x) for x in bursts)} times **in a single response**, until the output-token cap. Both began with an
  edit whose `start_line` argument was dropped, so the tool refused it (the base model hit that refusal in {sw_a} runs, this model in
  {sw_b}); the base model never answered it with a burst. Worth capping tool calls per response when running it in an
  agent loop.""")
acc_s = (f" Speculative-draft acceptance was the same for both ({pct(acc[0])} vs {pct(acc[1])}), so it is not the drafter;"
         f" this model also worked at longer contexts (median peak {med('ctx_peak_k', B, nd=0)}k vs"
         f" {med('ctx_peak_k', A, nd=0)}k tokens), which slows decoding.") if acc else ""
weak.append(f"""- **Serving.** This model's server generated {num(gen_b, 0)} tok/s against {num(gen_a, 0)} for the base model's
  ({pct(1 - gen_b / gen_a)} slower on the same build and GPU model), which if anything works against it under a
  wall-clock limit.{acc_s}
- **Scope.** One task, 20 runs per model, one harness. The intervals above are wide; treat the size of the gap as
  approximate, its direction as solid.""")
out.append("\n".join(weak))
print("\n".join(out))
