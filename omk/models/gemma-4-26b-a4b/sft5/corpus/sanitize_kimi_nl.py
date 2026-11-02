#!/usr/bin/env python
"""bm-889 sft4 (owner 2026-11-02 16:35 CET): sanitize lambda kimi newlines so its tool calls match sft1/sft2/sft3.
sft4 s200 run 002 closed a string argument with a real newline instead of <|"|>. Kimi was the only sft4 source with
newlines inside call strings, and 5,077 of 33,198 values ended with a newline right before the closing delimiter
(sft1 opencode 7 of 57,866; sft2/3 opencode 194 of 16,507), 139 started with one (others 0).
Rule: every string value in an assistant tool call (nested too) loses its leading blank lines and its trailing
newline(s) plus trailing whitespace. Interior newlines stay (code/config). Then the post-conversion checks of
build_conv.py, plus: no call string value starts or ends with a newline.
In:  conv_v2/lambda_kimi_pre2r.train.conv.jsonl, conv_v2/lambda_kimi_pre2r_ov.eval.conv.jsonl (as the chosen render)
Out: conv_v3/lambda_kimi_pre2r_nl.{train,eval}.conv.jsonl, conv_v3/lambda_kimi_pre2r_nl.sanitize.json"""
import json, re, collections
S = '/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/'
LEAD = re.compile(r'^(?:[ \t]*\r?\n)+'); TRAIL = re.compile(r'(?:\r?\n[ \t]*)+$')
C = collections.Counter(); byf = collections.Counter(); ex = []
def clean(v, path):
    if isinstance(v, str):
        n = TRAIL.sub('', LEAD.sub('', v))
        if n != v:
            C['values_changed'] += 1; byf[path] += 1
            if v != v.lstrip('\r\n') and LEAD.match(v): C['lead_removed'] += 1
            if TRAIL.search(v): C['trail_removed'] += 1
            if not n: C['became_empty'] += 1
            if len(ex) < 6: ex.append([path, repr(v[-60:]), repr(n[-60:])])
        return n
    if isinstance(v, dict): return {k: clean(x, f'{path}.{k}') for k, x in v.items()}
    if isinstance(v, list): return [clean(x, path + '[]') for x in v]
    return v
def strings(v):
    if isinstance(v, str): yield v
    elif isinstance(v, dict):
        for x in v.values(): yield from strings(x)
    elif isinstance(v, list):
        for x in v: yield from strings(x)
def check(msgs, tools):
    A = [m for m in msgs if m['role'] == 'assistant']
    calls = [c for m in A for c in m.get('tool_calls') or []]
    if any(not isinstance((c.get('function') or c).get('arguments'), dict) for c in calls): return 'args_not_object'
    if any(len((m.get('reasoning_content') or '').strip()) < 20 for m in A): return 'reasoning_lost'
    if sum(m['role'] == 'tool' for m in msgs) != len(calls): return 'tool_count'
    if msgs[-1]['role'] != 'assistant' or msgs[-1].get('tool_calls') or not (msgs[-1].get('content') or '').strip(): return 'no_final_answer'
    if calls and not tools: return 'no_tools'
    for c in calls:
        for s in strings((c.get('function') or c).get('arguments')):
            if s.startswith(('\n', '\r')) or s.endswith(('\n', '\r')): return 'newline_at_edge'
    return None
for split, src in (('train', 'conv_v2/lambda_kimi_pre2r.train.conv.jsonl'), ('eval', 'conv_v2/lambda_kimi_pre2r_ov.eval.conv.jsonl')):
    with open(S + src) as f, open(S + f'conv_v3/lambda_kimi_pre2r_nl.{split}.conv.jsonl', 'w') as o:
        for l in f:
            d = json.loads(l); C[f'{split}_in'] += 1; n0 = C['values_changed']
            for m in d['messages']:
                if m['role'] != 'assistant': continue
                for c in m.get('tool_calls') or []:
                    fn = c.get('function') or c
                    fn['arguments'] = clean(fn['arguments'], fn.get('name', '?'))
            if C['values_changed'] > n0: C[f'{split}_rows_changed'] += 1
            why = check(d['messages'], d.get('tools'))
            if why: C[f'{split}_reject_{why}'] += 1; continue
            C[f'{split}_out'] += 1
            o.write(json.dumps(d, ensure_ascii=False) + '\n')
rep = {'counts': dict(C), 'by_field': dict(byf.most_common()), 'examples': ex}
json.dump(rep, open(S + 'conv_v3/lambda_kimi_pre2r_nl.sanitize.json', 'w'), indent=1)
print(json.dumps(rep['counts'])); print('by field', byf.most_common(10))
for e in ex: print('EX', e)
print('SANITIZE_DONE')
