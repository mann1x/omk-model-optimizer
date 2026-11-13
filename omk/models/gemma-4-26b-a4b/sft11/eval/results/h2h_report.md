# Head-to-head, SDK 4.100.277: base vs sft11

## Outcome

| | base | sft11 |
|---|---|---|
| pass | **3 / 20** (15 %) | **12 / 20** (60 %) |
| 95 % Wilson interval | 5–36 % | 39–78 % |
| verdicts | {'FAIL': 1, 'PASS': 3, 'TIMEOUT': 16} | {'FAIL': 1, 'PASS': 12, 'TIMEOUT': 7} |
| claimed success but failed | 0 | 0 |
| runs that used restore_file (start over from the original) | 7 | 12 |
| runs with no reasoning-loop cut | 0 | 9 |
| runs with no guard trip at all | 0 | 0 |
| runs that reached a hard repeated-call warning | 1 | 2 |
| runs stopped by the harness for looping | 1 | 1 |
| runs with a burst of >=10 tool calls in one response | 0 | 2 |
| model-hours / wall-hours spent | 9.08 / 9.35 | 6.27 / 6.53 |

Fisher exact (two-sided) on pass/fail: p = 0.0079

### Failure modes (last run_game of each run)

| mode | base | sft11 |
|---|---|---|
| game does not start | 17 | 6 |
| pass | 3 | 12 |
| starts, oracle 17/23 | 0 | 1 |
| starts, oracle 21/23 | 0 | 1 |

Best partial oracle score reached in failed runs: base –; sft11 ['20/23', '21/23']

### Pass curve (share of all runs passed by minute t)

| | 5 min | 10 min | 15 min | 20 min | 25 min | 30 min |
|---|---|---|---|---|---|---|
| base | 0 % | 0 % | 0 % | 15 % | 15 % | 15 % |
| sft11 | 5 % | 25 % | 40 % | 50 % | 55 % | 60 % |

### Loop rates per model-hour (sum over runs / sum of model time)

| | base | sft11 |
|---|---|---|
| reasoning-loop cuts | 18.1 | 2.9 |
| guard trips (all) | 24.8 | 17.7 |
| repeated-call warnings | 6.1 | 9.9 |
| repeated-call last warnings | 0.22 | 0.48 |
| blocks >=10x | 4.6 | 3.8 |
| edit oscillations | 3.7 | 9.7 |

### Time

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| wall time per run (min) | 30.2 [30.2–30.2] | 21.0 [9.8–30.2] | 0.05 | 30.2 | 30.2 | 0.17 |
| model time (min) | 29.4 [27.8–29.8] | 20.0 [9.5–28.7] | 0.00 | 29.5 | 29.1 | 0.15 |
| tool time (min) | 0.11 [0.08–0.13] | 0.20 [0.16–0.29] | 0.00 | 0.10 | 0.36 | 0.00 |
| median turn (s) | 22.0 [10.4–48.8] | 9.1 [5.7–13.0] | 0.01 | 32.7 | 18.4 | 0.21 |
| longest turn (s) | 213 [179–316] | 108 [76.0–146] | 0.00 | 229 | 131 | 0.28 |
| time to first passing oracle (min, passes only) | 16.9 [16.5–17.2] | 11.5 [6.9–16.9] | 0.28 | – | – | – |

### Loops and guards (per run)

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| reasoning-loop guard cuts | 8.0 [7.0–10.2] | 1.0 [0.0–1.0] | 0.00 | 9.0 | 1.0 | 0.00 |
| guard trips, all kinds | 9.5 [7.0–11.2] | 4.0 [1.8–8.0] | 0.00 | 10.0 | 8.5 | 0.32 |
| repeated-tool-call warnings (soft) | 0.50 [0.0–1.5] | 1.5 [0.75–3.2] | 0.12 | 0.0 | 6.0 | 0.01 |
| repeated-tool-call last warnings (hard) | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.59 | 0.0 | 0.0 | 0.22 |
| reasoning blocks with a unit repeated >=10x | 2.0 [1.8–3.0] | 1.0 [0.0–1.2] | 0.03 | 2.0 | 1.0 | 0.53 |
| reasoning blocks with a unit repeated >=20x | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.57 | 0.0 | 0.0 | 1.000 |
| worst repeat count in any block | 12.5 [10.8–16.5] | 10.0 [9.0–13.2] | 0.05 | 13.0 | 14.0 | 0.73 |
| reasoning chars inside >=10x blocks (k) | 49.1 [41.2–88.7] | 19.3 [0.0–34.1] | 0.00 | 49.7 | 26.0 | 0.09 |
| edit-oscillation detector firings | 0.50 [0.0–1.5] | 1.0 [0.0–3.0] | 0.23 | 0.0 | 6.0 | 0.01 |
| identical failed call re-sent | 0.50 [0.0–1.5] | 1.0 [0.0–3.0] | 0.23 | 0.0 | 6.0 | 0.01 |
| turns ended without a call (nudged) | 0.0 [0.0–0.0] | 1.0 [0.0–1.0] | 0.00 | 0.0 | 0.0 | 0.01 |
| most tool calls in one response | 1.0 [1.0–1.0] | 1.0 [1.0–1.0] | 0.08 | 1.0 | 1.0 | 0.04 |
| model requests cut or failed | 9.0 [8.0–11.2] | 1.0 [0.0–2.0] | 0.00 | 9.0 | 2.0 | 0.00 |

### Length

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| reasoning chars (k) | 226 [194–257] | 230 [113–356] | 0.95 | 232 | 416 | 0.03 |
| reasoning blocks | 33.0 [23.0–41.0] | 46.5 [39.2–57.5] | 0.00 | 26.0 | 52.5 | 0.03 |
| longest reasoning block (k chars) | 38.2 [32.7–58.2] | 29.7 [23.2–37.3] | 0.01 | 38.5 | 34.8 | 0.11 |
| median reasoning block (chars) | 470 [366–957] | 811 [650–1,586] | 0.01 | 513 | 2,580 | 0.00 |
| generated tokens (k) | 121 [99.5–148] | 128 [76.4–205] | 0.56 | 132 | 209 | 0.00 |
| reasoning tokens (k) | 92.6 [83.4–105] | 95.0 [49.0–136] | 0.99 | 93.4 | 161 | 0.05 |
| reasoning share of generated tokens | 0.89 [0.79–0.95] | 0.85 [0.71–0.88] | 0.03 | 0.89 | 0.77 | 0.10 |
| tokens written into tool-call inputs (k) | 12.1 [5.1–29.6] | 22.6 [8.9–36.5] | 0.17 | 12.8 | 49.0 | 0.02 |
| tool-input tokens per tool call | 287 [141–1,642] | 355 [151–603] | 0.95 | 306 | 478 | 0.50 |
| prompt tokens processed (k) | 1,181 [559–1,815] | 2,785 [1,654–4,042] | 0.00 | 692 | 3,264 | 0.02 |
| peak context (k tokens) | 84.3 [70.7–121] | 110 [87.2–134] | 0.26 | 93.6 | 133 | 0.04 |
| context compactions | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.16 | 0.0 | 0.0 | 0.35 |
| final answer length (chars) | 0.0 [0.0–0.0] | 21.0 [0.0–21.0] | 0.01 | 0.0 | 0.0 | 0.17 |

### Behaviour

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| tool calls with a swallowed argument | 0.0 [0.0–2.2] | 0.0 [0.0–2.2] | 0.86 | 0.0 | 7.5 | 0.06 |
| turns (model requests) | 38.5 [27.0–48.2] | 49.0 [40.8–61.8] | 0.02 | 37.0 | 56.0 | 0.10 |
| tool calls | 27.5 [18.5–39.2] | 50.5 [43.8–67.8] | 0.00 | 24.0 | 69.0 | 0.00 |
| tool calls per model-minute | 0.94 [0.63–1.4] | 3.3 [2.6–6.2] | 0.00 | 0.87 | 2.3 | 0.00 |
| generated tokens per tool call | 3,866 [2,671–7,248] | 2,100 [1,126–2,646] | 0.00 | 4,083 | 3,125 | 0.09 |
| failed tool calls | 3.5 [2.8–5.2] | 8.5 [4.5–16.5] | 0.00 | 3.0 | 19.5 | 0.00 |
| edits | 5.0 [3.8–7.0] | 10.0 [8.0–13.0] | 0.00 | 5.0 | 11.0 | 0.07 |
| edits refused / failed | 3.0 [1.8–5.2] | 7.0 [4.5–16.5] | 0.00 | 3.0 | 19.5 | 0.00 |
| edits not followed by a check | 3.5 [1.8–6.2] | 5.5 [3.8–8.2] | 0.08 | 3.0 | 8.5 | 0.25 |
| restore_file (start over) | 0.0 [0.0–1.0] | 1.0 [0.0–1.0] | 0.08 | 0.0 | 1.0 | 0.15 |
| run_game calls | 2.0 [1.0–3.0] | 5.0 [3.0–6.0] | 0.00 | 1.0 | 3.0 | 0.07 |
| read_files calls | 7.5 [5.0–11.0] | 13.0 [10.5–17.0] | 0.01 | 5.0 | 13.5 | 0.26 |
| grep calls | 2.0 [1.0–6.0] | 4.0 [0.0–6.2] | 0.92 | 2.0 | 0.0 | 0.08 |

### LSP

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| check_file calls (LSP) | 1.0 [0.0–2.0] | 0.50 [0.0–2.0] | 0.58 | 1.0 | 0.50 | 0.44 |
| ask_lsp calls (LSP) | 0.0 [0.0–1.0] | 0.0 [0.0–0.0] | 0.35 | 0.0 | 0.0 | 0.10 |
| edits auto-checked by the LSP | 7.5 [7.0–10.2] | 17.5 [12.5–25.8] | 0.00 | 7.0 | 35.0 | 0.00 |
| edits that got an LSP diagnostics report | 1.5 [1.0–3.0] | 2.5 [0.75–5.0] | 0.34 | 1.0 | 3.5 | 0.16 |
| edits the LSP flagged as regressions | 1.0 [0.75–1.2] | 1.0 [0.0–3.0] | 0.31 | 1.0 | 3.0 | 0.09 |

### Output hygiene

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| tool-call parse recoveries | 0.0 [0.0–1.0] | 0.0 [0.0–1.0] | 1.000 | 0.0 | 1.0 | 0.22 |
| leaked control markers | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 1.000 | 0.0 | 0.0 | 1.000 |
| unclosed thinking blocks | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 1.000 | 0.0 | 0.0 | 1.000 |
| output-cap hits | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.76 | 0.0 | 0.0 | 0.41 |
| empty completions | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 1.000 | 0.0 | 0.0 | 1.000 |

### Serving (parity check: same GPU model, same server)

Median [p25–p75] per run, all runs; Mann–Whitney two-sided p. Failed-runs-only medians on the right.

| metric | base | sft11 | p | base fails | sft11 fails | p (fails) |
|---|---|---|---|---|---|---|
| server generation speed (tok/s) | 176 [173–196] | 168 [164–171] | 0.00 | 176 | 165 | 0.00 |
| server prompt speed (tok/s) | 2,537 [2,500–2,611] | 2,537 [2,466–2,576] | 0.34 | 2,533 | 2,484 | 0.14 |
| time to first token (ms) | 963 [893–1,261] | 1,015 [1,001–1,069] | 0.47 | 954 | 1,010 | 0.43 |

### Tool-use profile (all runs, share of calls)

| tool | base calls | base share | sft11 calls | sft11 share |
|---|---|---|---|---|
| editor | 183 | 31 % | 998 | 62 % |
| read_files | 176 | 30 % | 297 | 18 % |
| run_commands | 57 | 10 % | 118 | 7 % |
| grep | 68 | 12 % | 77 | 5 % |
| search_codebase | 32 | 5 % | 44 | 3 % |
| check_file | 24 | 4 % | 21 | 1 % |
| list_files | 19 | 3 % | 25 | 2 % |
| restore_file | 8 | 1 % | 17 | 1 % |
| sed | 2 | 0 % | 11 | 1 % |
| ask_lsp | 7 | 1 % | 5 | 0 % |
| task_progress | 2 | 0 % | 4 | 0 % |
| awk | 4 | 1 % | 0 | 0 % |

### Guard trips by kind (sum)

| guard | base | sft11 |
|---|---|---|
| edit_verification.nudge | 3 | 9 |
| loop_detection.hard | 2 | 3 |
| loop_detection.soft | 55 | 62 |
| loop_detection.stopped | 1 | 1 |
| nudge.no_tool_call | 0 | 18 |
| reasoning_loop.cut | 164 | 18 |
| reasoning_loop.cut.argument_loop | 0 | 2 |
| reasoning_loop.cut.reasoning_loop | 164 | 16 |

### Failed tool calls by class (sum)

| class | base | sft11 |
|---|---|---|
| /workspace/manic_miner.html changed since you last read it, | 2 | 11 |
| Command exited with code N | 0 | 3 |
| Duplicated instead of replaced | 0 | 1 |
| Error reading file | 0 | 2 |
| Invalid end_column | 1 | 0 |
| Invalid start_column | 0 | 1 |
| Missing required argument `file_path` | 5 | 5 |
| No change | 26 | 29 |
| No replacement performed | 7 | 33 |
| Read before editing | 7 | 42 |
| Read before replacing | 0 | 1 |
| This call names `end_line | 35 | 675 |
| Tool call editor was cut off by the output budget | 4 | 4 |
| `insert_line` adds text at a boundary and `start_line` repla | 0 | 7 |
| `start_column`/`end_column` need `start_line` to say which l | 1 | 0 |
| grep failed | 4 | 5 |
| not modified | 1 | 3 |

### Per run

| arm | run | verdict | min | cuts | trips | ≥10x | worst | think k | tok k | calls | edits | run_game | end state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 001 | TIMEOUT | 30.2 | 8 | 9 | 2 | 19 | 140 | 155 | 14 | 4 | 1 | game does not start |
| base | 002 | TIMEOUT | 30.2 | 6 | 9 | 2 | 12 | 124 | 135 | 19 | 2 | 1 | game does not start |
| base | 003 | TIMEOUT | 30.2 | 7 | 7 | 3 | 13 | 331 | 170 | 14 | 7 | 1 | game does not start |
| base | 004 | PASS | 17.7 | 6 | 7 | 2 | 11 | 163 | 76 | 43 | 11 | 4 | pass |
| base | 005 | TIMEOUT | 30.2 | 7 | 34 | 2 | 24 | 216 | 105 | 54 | 4 | 4 | game does not start |
| base | 006 | TIMEOUT | 30.2 | 11 | 11 | 1 | 18 | 231 | 117 | 28 | 5 | 3 | game does not start |
| base | 007 | TIMEOUT | 30.2 | 7 | 12 | 2 | 16 | 186 | 174 | 27 | 3 | 3 | game does not start |
| base | 008 | PASS | 17.2 | 2 | 3 | 0 | 8 | 241 | 124 | 37 | 7 | 4 | pass |
| base | 009 | TIMEOUT | 30.2 | 7 | 7 | 5 | 28 | 315 | 145 | 19 | 4 | 1 | game does not start |
| base | 010 | FAIL | 27.1 | 9 | 26 | 3 | 13 | 266 | 133 | 37 | 6 | 1 | game does not start |
| base | 011 | TIMEOUT | 30.2 | 10 | 10 | 2 | 10 | 277 | 92 | 24 | 2 | 3 | game does not start |
| base | 012 | TIMEOUT | 30.2 | 12 | 13 | 2 | 10 | 255 | 104 | 42 | 10 | 2 | game does not start |
| base | 013 | TIMEOUT | 30.2 | 11 | 11 | 0 | 8 | 220 | 100 | 42 | 5 | 3 | game does not start |
| base | 014 | TIMEOUT | 30.2 | 12 | 15 | 2 | 12 | 135 | 53 | 13 | 0 | 1 | game does not start |
| base | 015 | TIMEOUT | 30.2 | 8 | 10 | 3 | 15 | 196 | 157 | 17 | 5 | 1 | game does not start |
| base | 016 | TIMEOUT | 30.2 | 9 | 9 | 4 | 18 | 256 | 106 | 39 | 9 | 2 | game does not start |
| base | 017 | TIMEOUT | 30.2 | 11 | 11 | 1 | 15 | 209 | 93 | 24 | 2 | 1 | game does not start |
| base | 018 | PASS | 16.3 | 5 | 5 | 3 | 12 | 195 | 98 | 40 | 7 | 4 | pass |
| base | 019 | TIMEOUT | 30.2 | 7 | 7 | 3 | 12 | 258 | 181 | 14 | 6 | 1 | game does not start |
| base | 020 | TIMEOUT | 30.2 | 9 | 9 | 0 | 9 | 257 | 132 | 35 | 9 | 2 | game does not start |
| sft11 | 001 | TIMEOUT | 30.2 | 3 | 21 | 4 | 20 | 445 | 204 | 51 | 10 | 2 | game does not start |
| sft11 | 002 | PASS | 16.9 | 1 | 5 | 3 | 13 | 262 | 123 | 67 | 18 | 7 | pass |
| sft11 | 003 | TIMEOUT | 30.2 | 2 | 4 | 1 | 14 | 430 | 208 | 54 | 16 | 3 | game does not start |
| sft11 | 004 | PASS | 4.6 | 0 | 2 | 0 | 7 | 66 | 31 | 38 | 10 | 5 | pass |
| sft11 | 005 | PASS | 26.0 | 1 | 1 | 1 | 13 | 339 | 168 | 70 | 13 | 5 | pass |
| sft11 | 006 | PASS | 10.4 | 0 | 1 | 0 | 9 | 143 | 82 | 27 | 5 | 3 | pass |
| sft11 | 007 | TIMEOUT | 30.2 | 0 | 6 | 1 | 10 | 476 | 211 | 97 | 25 | 11 | starts, oracle 21/23 |
| sft11 | 008 | PASS | 7.7 | 0 | 2 | 0 | 9 | 105 | 49 | 44 | 13 | 6 | pass |
| sft11 | 009 | PASS | 5.7 | 0 | 1 | 0 | 5 | 81 | 39 | 36 | 8 | 5 | pass |
| sft11 | 010 | PASS | 5.4 | 0 | 2 | 0 | 6 | 76 | 35 | 50 | 9 | 6 | pass |
| sft11 | 011 | FAIL | 26.6 | 2 | 14 | 3 | 15 | 205 | 187 | 332 | 0 | 1 | game does not start |
| sft11 | 012 | TIMEOUT | 30.2 | 1 | 6 | 0 | 6 | 402 | 200 | 84 | 19 | 8 | starts, oracle 17/23 |
| sft11 | 013 | TIMEOUT | 30.3 | 1 | 13 | 4 | 16 | 290 | 210 | 36 | 6 | 4 | game does not start |
| sft11 | 014 | TIMEOUT | 30.2 | 1 | 8 | 1 | 11 | 442 | 235 | 47 | 12 | 3 | game does not start |
| sft11 | 015 | PASS | 18.4 | 0 | 1 | 1 | 10 | 254 | 127 | 64 | 12 | 4 | pass |
| sft11 | 016 | PASS | 13.2 | 1 | 4 | 2 | 11 | 175 | 86 | 43 | 10 | 6 | pass |
| sft11 | 017 | PASS | 23.5 | 4 | 8 | 1 | 10 | 255 | 127 | 51 | 9 | 5 | pass |
| sft11 | 018 | PASS | 13.9 | 0 | 2 | 0 | 9 | 183 | 106 | 44 | 9 | 5 | pass |
| sft11 | 019 | TIMEOUT | 30.2 | 1 | 9 | 1 | 14 | 95 | 223 | 338 | 2 | 1 | game does not start |
| sft11 | 020 | PASS | 7.9 | 0 | 1 | 1 | 10 | 116 | 57 | 44 | 8 | 8 | pass |

### Basis parity

- base: models ['gemma-4-26b-a4b-it-bf16-eos106'], SDK ['4.100.277'], samplers ['{"min_p": 0, "name": "owner-8244-2026-11-12", "owner": "arm", "presence_penalty": 0.025, "repeat_last_n": 0, "repeat_penalty": 1, "temperature": 0.8, "top_k": 64, "top_p": 0.95}']
- sft11: models ['gemma-4-26b-a4b-it-bm889-sft11-sfinal-bf16-eos106'], SDK ['4.100.277'], samplers ['{"min_p": 0, "name": "owner-8244-2026-11-12", "owner": "arm", "presence_penalty": 0.025, "repeat_last_n": 0, "repeat_penalty": 1, "temperature": 0.8, "top_k": 64, "top_p": 0.95}']
