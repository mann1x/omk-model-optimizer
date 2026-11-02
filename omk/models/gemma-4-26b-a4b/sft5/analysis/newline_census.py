#!/usr/bin/env python
"""bm-889 2026-11-02: per-source census of NEWLINE handling in trained (role>0) tool calls and text.
sft4 s200 run 002: a string argument closed by a real newline instead of <|"|>. Field-name agnostic.
Reads *.train.roles.jsonl; prints one row per source; writes newline_census.json."""
import json, glob, os, re, sys, collections
import numpy as np
from transformers import AutoTokenizer
G = '/mnt/sdc/omk-model-optimizer/g4_recovery/'
T = AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
Q, CO, CC, CHO, CHC = 52, 48, 49, 100, 101
ESC = re.compile(r'(?<!\\)\\n')          # backslash-n as two characters
FIELD_AFTER_NL = re.compile(r'\n\s*[A-Za-z_]\w*\s*:')
def dec(a): return T.decode(a, skip_special_tokens=False)
def scan(path):
    r = collections.Counter(); ex = collections.defaultdict(list)
    for l in open(path):
        d = json.loads(l); ids = np.array(d['ids']); role = np.array(d['role']); sc = role > 0
        r['windows'] += 1; r['scored_tok'] += int(sc.sum())
        # trained tool calls
        for i in np.nonzero((ids == CO) & sc)[0]:
            j = np.nonzero(ids[i:] == CC)[0]
            if not len(j): r['call_unclosed'] += 1; continue
            j = i + int(j[0]); r['calls'] += 1
            qs = list(np.nonzero(ids[i:j] == Q)[0] + i)
            if len(qs) % 2: r['call_odd_delims'] += 1; ex['call_odd_delims'].append(dec(ids[i:j+1].tolist())[-300:])
            # outside-string segments: between call open/after a closing delim and the next opening delim
            bounds = [i + 1] + [q + 1 for q in qs[1::2]]; ends = qs[0::2] + [j]
            for a, b in zip(bounds, ends):
                seg = dec(ids[a:b].tolist())
                if '\n' in seg:
                    r['sep_newline'] += 1
                    if len(ex['sep_newline']) < 4: ex['sep_newline'].append(repr(seg[-160:]))
            for a, b in zip(qs[0::2], qs[1::2]):
                v = dec(ids[a + 1:b].tolist()); r['str_vals'] += 1
                if '\n' in v: r['val_real_nl'] += 1
                if ESC.search(v):
                    r['val_escaped_nl'] += 1
                    if len(ex['val_escaped_nl']) < 4: ex['val_escaped_nl'].append(repr(v[:160]))
                if '\n' in v and ESC.search(v): r['val_mixed_nl'] += 1
                if v.endswith('\n'): r['val_ends_nl'] += 1
                if v.startswith('\n'): r['val_starts_nl'] += 1
                if FIELD_AFTER_NL.search(v):
                    r['val_nl_then_field'] += 1
                    if len(ex['val_nl_then_field']) < 4: ex['val_nl_then_field'].append(repr(v[:200]))
        # trained text outside calls: escaped newlines in reasoning / answer
        txt = dec(ids[sc].tolist())
        r['scored_esc_nl'] += len(ESC.findall(txt))
        r['scored_real_nl'] += txt.count('\n')
    return r, ex
sets = []
for f in sorted(glob.glob(G + 'sft4_build/CHOSEN_RENDER/*.train.roles.jsonl')): sets.append(('sft4:' + os.path.basename(f).split('.')[0], f))
for f in sorted(glob.glob(G + 'sft2_build/render_v3/*.train.roles.jsonl')): sets.append(('v3:' + os.path.basename(f).split('.')[0], f))
for f in sorted(glob.glob(G + 'sft_build/sft_*_train.roles.jsonl')): sets.append(('sft1:' + os.path.basename(f)[4:].replace('_train.roles.jsonl', ''), f))
out = {}
K = ['calls', 'call_odd_delims', 'sep_newline', 'str_vals', 'val_real_nl', 'val_escaped_nl', 'val_mixed_nl', 'val_ends_nl', 'val_starts_nl', 'val_nl_then_field', 'scored_esc_nl', 'scored_real_nl']
print('source'.ljust(30) + ' '.join(k[:13].rjust(13) for k in K), flush=True)
for lab, f in sets:
    r, ex = scan(f); out[lab] = {'counts': r, 'examples': ex}
    print(lab.ljust(30) + ' '.join(str(r[k]).rjust(13) for k in K), flush=True)
json.dump(out, open('newline_census.json', 'w'), indent=1, default=str)
print('CENSUS_DONE', flush=True)
