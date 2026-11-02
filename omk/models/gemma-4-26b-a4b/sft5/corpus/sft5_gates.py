#!/usr/bin/env python
"""bm-889 sft5: gate render_32k_v3_mlast2 before training. Exit 0 + GATES_OK, else GATES_FAIL <why>.
1 all 14 stats complete  2 non-kimi files byte-identical to render_32k_v2_full (the template change is a no-op there)
3 kimi rows/segments equal to v2_full  4 no trained call has a multi-line string value followed by a single-line field
5 writes CORPUS_MANIFEST.json (same fields as render_32k_v2_full's)"""
import json, glob, os, sys, hashlib, collections
import numpy as np
from transformers import AutoTokenizer
S = '/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/'; N = S + 'render_32k_v3_mlast2/'; O = S + 'render_32k_v2_full/'
def fail(w): print('GATES_FAIL', w); sys.exit(1)
st = {os.path.basename(f).replace('.roles.stats.json', ''): json.load(open(f)) for f in glob.glob(N + '*.roles.stats.json')}
if len(st) != 14: fail(f'{len(st)} stats files, want 14')
for k, v in st.items():
    if not v.get('complete'): fail(f'{k} incomplete')
old = {os.path.basename(f).replace('.roles.stats.json', ''): json.load(open(f)) for f in glob.glob(O + '*.roles.stats.json')}
for k in st:
    if k.startswith('lambda_kimi'):
        for q in ('rows', 'segments'):
            if q in old[k] and st[k].get(q) != old[k].get(q): fail(f'{k} {q} {st[k].get(q)} vs {old[k].get(q)}')
    elif st[k]['sha256'] != old[k]['sha256']: fail(f'{k} differs from v2_full ({st[k]["sha256"][:12]} vs {old[k]["sha256"][:12]})')
print('non-kimi identical to v2_full: 12/12; kimi rows', {k: st[k].get('rows') for k in st if 'kimi' in k})
T = AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it'); Q, CO, CC = 52, 48, 49
c = collections.Counter()
for l in open(N + 'lambda_kimi_pre2r.train.roles.jsonl'):
    d = json.loads(l); ids = np.array(d['ids']); sc = np.array(d['role']) > 0
    for i in np.nonzero((ids == CO) & sc)[0]:
        j = i + int(np.nonzero(ids[i:] == CC)[0][0]); qs = list(np.nonzero(ids[i:j] == Q)[0] + i)
        vals = [(T.decode(ids[a+1:b].tolist()), b) for a, b in zip(qs[0::2], qs[1::2])]
        for n, (v, b) in enumerate(vals):
            if '\n' not in v: continue
            c['ml_values'] += 1
            if ids[b+1] == CC or T.decode(ids[b+1:b+2].tolist()).startswith('}'): c['ml_last'] += 1; continue
            nxt_ml = n + 1 < len(vals) and '\n' in vals[n+1][0] and vals[n+1][1] > b
            # a following non-string field (number/bool) is single-line too
            c['ml_then_ml' if nxt_ml else 'ml_then_single'] += 1
            if v.endswith('\n'): c['ends_nl_then_field'] += 1
print('kimi trained multi-line values', dict(c))
if c['ml_then_single']: fail(f"{c['ml_then_single']} multi-line values still followed by a single-line field")
m = {'name': 'bm889-sft5-corpus', 'chosen_by_owner': '2026-11-02 16:35-16:44 CET: keep tool data valid, change how calls are rendered (multi-line value last), all datasets; launch sft5 with it',
     'seq': 32768, 'overlap_gate': 'sft2_overlap.py OVERLAP_OK', 'template': 'sft4_build/chat_template_mlast2.jinja',
     'template_sha256': hashlib.sha256(open(S + 'chat_template_mlast2.jinja', 'rb').read()).hexdigest(),
     'from': 'render_32k_v2_full inputs and flags (jobs_v3_mlast2.txt)', 'kimi_multiline_values': dict(c), 'files': {}}
tw = ts = 0
for f in sorted(glob.glob(N + '*.roles.jsonl')):
    k = os.path.basename(f).replace('.roles.jsonl', ''); s = st[k]
    m['files'][os.path.basename(f)] = {'sha256': hashlib.sha256(open(f, 'rb').read()).hexdigest(), 'rows': s.get('rows'), 'windows': s.get('windows'),
        'scored': s.get('scored', sum((s.get('tokens_by_role') or [0])[1:])), 'dropped_long_segments': s.get('dropped_long_segments'), 'refused_not_prefix': s.get('refused_not_prefix')}
    if '.train' in k: tw += s.get('windows') or 0; ts += s.get('scored', sum((s.get('tokens_by_role') or [0])[1:]))
m['train_windows'] = tw; m['train_scored'] = ts
json.dump(m, open(N + 'CORPUS_MANIFEST.json', 'w'), indent=1)
print('MANIFEST train windows', tw, 'scored', ts); print('GATES_OK')
