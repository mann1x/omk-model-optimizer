"""bm-889: per source, `key: value` / `key:` lines (YAML/env style) inside TRAINED tool-call string values."""
import json, re, sys, glob, os, collections
import numpy as np
from transformers import AutoTokenizer
T = AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
Q, CO, CC = 52, 48, 49
G = '/mnt/sdc/omk-model-optimizer/g4_recovery/'
PYKW = re.compile(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with|return|lambda|async|case|match|default)\b')
KV = re.compile(r'^\s*-?\s*[A-Za-z_][\w.-]*\s*:(\s|$)')
def scan(f):
    r = collections.Counter()
    for l in open(f):
        d = json.loads(l); ids = np.array(d['ids']); role = np.array(d['role']); sc = role > 0
        for i in np.nonzero((ids == CO) & sc)[0]:
            j = np.nonzero(ids[i:] == CC)[0]
            if not len(j): continue
            j = i + int(j[0]); qs = list(np.nonzero(ids[i:j] == Q)[0] + i)
            for a, b in zip(qs[0::2], qs[1::2]):
                v = T.decode(ids[a+1:b].tolist()); r['vals'] += 1; r['val_tok'] += b - a - 1
                if '\n' not in v: continue
                lines = v.split('\n'); r['lines'] += len(lines)
                kv = [x for x in lines if KV.match(x) and not PYKW.match(x) and '(' not in x.split(':')[0]]
                r['kv_lines'] += len(kv)
                if len(kv) >= 3: r['yaml_vals'] += 1
    return r
sets = []
for f in sorted(glob.glob(G + 'sft4_build/CHOSEN_RENDER/*.train.roles.jsonl')): sets.append(('sft4:' + os.path.basename(f).split('.')[0], f))
for f in sorted(glob.glob(G + 'sft2_build/render_v3/*.train.roles.jsonl')): sets.append(('v3:' + os.path.basename(f).split('.')[0], f))
for f in sorted(glob.glob(G + 'sft_build/sft_*_train.roles.jsonl')): sets.append(('sft1:' + os.path.basename(f)[4:].replace('_train.roles.jsonl', ''), f))
print(f"{'source':28s} {'str vals':>9s} {'val tok':>10s} {'lines':>9s} {'key: lines':>10s} {'yaml vals':>9s} {'key:/1k val tok':>15s}", flush=True)
for lab, f in sets:
    r = scan(f)
    if not r['vals']: continue
    print(f"{lab:28s} {r['vals']:9d} {r['val_tok']:10d} {r['lines']:9d} {r['kv_lines']:10d} {r['yaml_vals']:9d} {1000*r['kv_lines']/max(r['val_tok'],1):15.2f}", flush=True)
print('YAML_DONE')
