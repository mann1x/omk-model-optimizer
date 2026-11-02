#!/usr/bin/env python
"""bm-889 sft4: selected hermes rows -> OpenAI conv rows (omk_qad.rollouts --in), re-validated after conversion.

Conversion = sft2's hermes path: an-finetune replay_normalize.convert_hermes, then sft2_jobs.split_tool_responses.
After conversion every row must still have: reasoning_content of >= 20 chars on every assistant turn, the same number
of tool calls as the raw row with object arguments, one tool message per call, and a last assistant turn with text and
no call. ``preamble`` = an assistant turn with visible text beside its calls (owner rule 2026-10-30: rejected).
Split: sha256(first user message) % 50 == 0 -> eval (a prompt's variants stay on one side; random row split leaked
3.2% of interstellarninja eval positions). Rows in kimi_pre2r_roundtrip_bad.json (tokenizer round-trip fails) are dropped. Out: conv/<name>.{train,eval}.conv.jsonl, conv/<name>.conv.stats.json."""
import hashlib,json,re,sys,collections
sys.path.insert(0,'/srv/ml/an-finetune/docker'); sys.path.insert(0,'/mnt/sdc/omk-model-optimizer/g4_recovery/sft2_build')
from replay_normalize import convert_hermes
from sft2_jobs import split_tool_responses
import pyarrow.parquet as pq
S='/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/'
name,path=sys.argv[1],sys.argv[2]
ALLOW_PRE=len(sys.argv)>3 and sys.argv[3]=='allow-preamble'
OUT=name+('_pre2r' if ALLOW_PRE else '')
keep=[json.loads(l) for l in open(S+f'selected/{name}.keep.jsonl')]
rows=pq.read_table(S+path).to_pylist()
TC=re.compile(r'<tool_call>',re.S)
import os
BAD=set(json.load(open(S+'kimi_pre2r_roundtrip_bad.json'))) if os.path.exists(S+'kimi_pre2r_roundtrip_bad.json') else set()
C=collections.Counter(); out={'train':open(S+f'conv_v2/{OUT}.train.conv.jsonl','w'),'eval':open(S+f'conv_v2/{OUT}.eval.conv.jsonl','w')}
for k in keep:
    r=rows[k['index']]
    raw_calls=sum(len(TC.findall(m['value'])) for m in r['conversations'] if m['from'] in ('gpt','assistant','model'))
    msgs,tools=convert_hermes(r)
    if not msgs: C['convert_none']+=1; continue
    msgs=split_tool_responses(msgs)
    A=[m for m in msgs if m['role']=='assistant']
    why=None
    calls=sum(len(m.get('tool_calls') or []) for m in A)
    if calls!=raw_calls: why='calls_lost'
    elif any(not isinstance((c.get('function') or c).get('arguments'),dict) for m in A for c in m.get('tool_calls') or []): why='args_not_object'
    elif any(len((m.get('reasoning_content') or '').strip())<20 for m in A): why='reasoning_lost'
    elif sum(m['role']=='tool' for m in msgs)!=calls: why='tool_count'
    elif msgs[-1]['role']!='assistant' or msgs[-1].get('tool_calls') or not (msgs[-1].get('content') or '').strip(): why='no_final_answer'
    elif calls and not tools: why='no_tools'
    elif not ALLOW_PRE and any(m.get('tool_calls') and (m.get('content') or '').strip() for m in A): why='preamble'
    C[why or 'OK']+=1
    if why: continue
    rid=f"{name}/{k['id']}"
    if rid in BAD: C['roundtrip_bad']+=1; continue
    first=next((m.get('content') or '' for m in msgs if m['role']=='user'),'')
    split='eval' if int(hashlib.sha256(first.encode()).hexdigest(),16)%50==0 else 'train'  # group: one prompt, one side
    C[split]+=1
    out[split].write(json.dumps({'id':rid,'messages':msgs,'tools':tools},ensure_ascii=False)+'\n')
for f in out.values(): f.close()
json.dump(dict(C),open(S+f'conv_v2/{OUT}.conv.stats.json','w'),indent=1); print(OUT,dict(C),flush=True)
