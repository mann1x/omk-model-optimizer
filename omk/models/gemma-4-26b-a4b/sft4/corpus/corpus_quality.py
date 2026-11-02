"""Per source: rows, trained turns, thinking quality of trained turns, full-conversation token length."""
import json,glob,os,collections,statistics as st,re
from transformers import AutoTokenizer
T=AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
G='/mnt/sdc/omk-model-optimizer/'
def q(v,p): v=sorted(v); return v[int(p*(len(v)-1))] if v else 0
files=[(('v3:'+os.path.basename(f).split('.')[0]),f) for f in sorted(glob.glob(G+'g4_recovery/sft2_build/data_v3/*.train.conv.jsonl'))]
files+=[(('sft1:'+m+':'+os.path.basename(f).split('.')[0]),f) for m in ('mix_v1','mix_v2') for f in sorted(glob.glob(G+f'data/{m}/*.train.conv.jsonl'))]
for lab,f in files:
    rows=0; turns=0; calls=0; rl=[]; tiny=collections.Counter(); ntiny=0; empty=0; toks=[]; okstart=0
    for l in open(f):
        d=json.loads(l); rows+=1; txt=[]
        for m in d['messages']:
            c=m.get('content') or ''
            if isinstance(c,list): c=' '.join(x.get('text','') for x in c if isinstance(x,dict))
            r=m.get('reasoning_content') or ''
            txt.append(c); txt.append(r)
            for tc in m.get('tool_calls') or []: txt.append(json.dumps((tc.get('function') or tc).get('arguments')))
            if m['role']=='assistant' and m.get('score',True):
                turns+=1; calls+=1 if m.get('tool_calls') else 0
                rs=r.strip(); rl.append(len(rs))
                if not rs: empty+=1
                elif len(rs)<=40: ntiny+=1; tiny[rs[:30]]+=1
                if re.match(r'^(ok|okay)[,. ]',rs.lower()) and len(rs)<200: okstart+=1
        if rows<=400 or rows%5==0: toks.append(len(T(' '.join(txt),add_special_tokens=False)['input_ids']))
    print(f"{lab:34s} rows {rows:5d} trained_turns {turns:6d} call {calls:5d} | think empty {100*empty/max(turns,1):5.1f}% <=40ch {100*ntiny/max(turns,1):5.1f}% ok<200 {100*okstart/max(turns,1):4.1f}% think_chars p10 {q(rl,.1):6d} p50 {q(rl,.5):6d} p90 {q(rl,.9):6d} | row_tok p50 {q(toks,.5):6d} p90 {q(toks,.9):6d} max {max(toks or [0]):7d} >16k {100*sum(t>16384 for t in toks)/max(len(toks),1):4.1f}% >32k {100*sum(t>32768 for t in toks)/max(len(toks),1):4.1f}% | tiny {tiny.most_common(3)}",flush=True)
