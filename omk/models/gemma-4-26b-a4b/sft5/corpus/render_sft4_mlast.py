#!/usr/bin/env python
"""bm-889 sft4 render: omk_qad.rollouts.build at --seq 32768 --pack; segments longer than the window are DROPPED, not
windowed (a window would start mid-conversation without system prompt and tools). Same output and stats format as
rollouts --in (complete, sha256, windows), plus dropped_long_segments / dropped_long_tokens.
usage: render_sft4_mlast.py (render_sft4.py + chat_template_mlast.jinja) IN.conv.jsonl OUT.roles.jsonl HISTORY TOOL_PREAMBLE"""
import json,sys
from pathlib import Path
sys.path.insert(0,'/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft3_code')
from omk_qad import rollouts as R
from omk_qad.runctl import atomic_json, atomic_write
from transformers import AutoTokenizer
inp,out,hist,pre=sys.argv[1:5]; SEQ=32768
M='/mnt/sdc/ml/models/gemma-4-26B-A4B-it'
tok=AutoTokenizer.from_pretrained(M)
import os; TPL=os.environ.get("MLAST_TPL","/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/chat_template_mlast.jinja"); tok.chat_template=open(TPL).read()  # bm-889: multi-line call arguments last
rows=[json.loads(l) for l in open(inp) if l.strip()]
win,stats=R.build(tok,rows,SEQ,SEQ//2,R.CODE_ARGS,hist,True,0,pre)
keep=[w for w in win if '/w' not in w['id']]
long=[w for w in win if '/w' in w['id']]
segs={w['id'].rsplit('/w',1)[0] for w in long}
tb=[0]*R.N_ROLES
for w in keep:
    for q in w['role']: tb[q]+=1
data=''.join(json.dumps(w)+'\n' for w in keep)
stats.update(windows=len(keep),tokens_by_role=tb,dropped_long_segments=len(segs),dropped_long_windows=len(long),sha256=R._sha(data),complete=True)
atomic_write(out,data)
atomic_json(Path(out).with_suffix('.stats.json'),{**stats,'args':dict(model=M,inp=inp,out=out,seq=SEQ,stride=SEQ//2,code_args=','.join(R.CODE_ARGS),history_reasoning=hist,pack=True,tool_preamble=pre,max_tokens=0,long_segments='dropped',chat_template=TPL)})
print(json.dumps({k:v for k,v in stats.items() if k!='args'}),flush=True)
