| student | status | at | lr | loss | klk | jaccard | top1 | ppl | klk:agent_eval | klk:cp_reasoning.eval | klk:eval_code | klk:eval_general | klk:eval_tech | klk:exercism_sql.eval | klk:opencode.eval | klk:swe_agentless.eval | init_klk | gguf_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iq2 | final | step2400 | 1e-05 | jaccard | 0.4836 | 0.2087 | 0.8489 | 3.699 | 1.124 | 0.3379 | 0.262 | 0.3012 | 0.2953 | 0.3325 | 0.8663 | 0.3493 | 1.166 | 1581 |
| tern | final | step2400 | 6e-05 | kl | 0.4629 | 0.3043 | 0.7897 | 4.002 | 0.4003 | 0.5091 | 0.4934 | 0.6372 | 0.3627 | 0.337 | 0.5087 | 0.4551 | 8.029 | 1144 |

References (not the table's metric or eval rows unless stated):
- bm860_r1: {"note": "full-vocab KL code 0.732 / tech 0.382 / general 0.723 / agent 0.507; HE 10.37% (base 59.76%)"}
