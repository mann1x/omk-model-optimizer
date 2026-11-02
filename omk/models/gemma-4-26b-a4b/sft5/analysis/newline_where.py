import json, re, sys, collections, random
import numpy as np
from transformers import AutoTokenizer
T = AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
Q, CO, CC, CHO, CHC = 52, 48, 49, 100, 101
ESC = re.compile(r'(?<!\\)\\n')
LINEBREAK_ESC = re.compile(r'(?<!\\)\\n(?=\s*(import |from |def |class |#|return |if |for |print\(|    |\w+ = ))')
f = sys.argv[1]; random.seed(0)
r = collections.Counter(); ctx = collections.defaultdict(list)
for l in open(f):
    d = json.loads(l); ids = np.array(d['ids']); role = np.array(d['role']); sc = role > 0
    region = np.zeros(len(ids), dtype=int)  # 0 answer, 1 thought, 2 call, 3 call string
    st = None
    for k, t in enumerate(ids):
        if t == CHO: st = 'th'
        elif t == CHC: st = None
        if t == CO: st2 = True
    # mark regions by scanning
    inth = False; incall = False; instr = False
    for k, t in enumerate(ids):
        if t == CHO: inth = True
        if t == CO: incall = True; instr = False
        if incall and t == Q: instr = not instr
        region[k] = 3 if (incall and instr) else 2 if incall else 1 if inth else 0
        if t == CHC: inth = False
        if t == CC: incall = False; instr = False
    for reg in (0, 1, 2, 3):
        m = sc & (region == reg)
        if not m.any(): continue
        # contiguous spans
        idx = np.nonzero(m)[0]; spans = np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1)
        for s in spans:
            txt = T.decode(ids[s].tolist(), skip_special_tokens=False)
            n = len(ESC.findall(txt)); nb = len(LINEBREAK_ESC.findall(txt))
            r[f'esc_{reg}'] += n; r[f'lbesc_{reg}'] += nb; r[f'realnl_{reg}'] += txt.count('\n')
            if reg == 3 and n and '\n' not in txt: r['val_escaped_only'] += 1
            for mm in list(LINEBREAK_ESC.finditer(txt))[:1]:
                if random.random() < 0.05 and len(ctx[reg]) < 6:
                    ctx[reg].append(repr(txt[max(0, mm.start() - 80):mm.end() + 60]))
names = {0: 'answer', 1: 'thought', 2: 'call-syntax', 3: 'call-string'}
for reg in range(4):
    print(f"{names[reg]:12s} escaped \\n {r[f'esc_{reg}']:6d}  as-linebreak {r[f'lbesc_{reg}']:6d}  real \\n {r[f'realnl_{reg}']:8d}")
print('call string spans with escaped and no real newline:', r['val_escaped_only'])
for reg, v in ctx.items():
    print('==', names[reg])
    for x in v: print('   ', x[:260])
