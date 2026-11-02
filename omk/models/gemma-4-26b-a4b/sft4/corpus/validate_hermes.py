#!/usr/bin/env python
"""bm-889 sft4: one validator for every hermes-family source (sharegpt rows, <think>/<tool_call>/<tool_response>).

Row checks (a row fails on the first):
  tools_missing      no tool list (tools column, or <tools> in the system prompt) although the row calls tools
  bad_call           a <tool_call> that is not JSON, names a tool not offered, or has non-object arguments
  unbalanced_tags    <think>/<tool_call>/<tool_response> open and close counts differ in an assistant or tool message
  orphan_response    a tool message with no call before it, or a call turn not followed by a tool message
  no_final_answer    the row does not end on an assistant turn without a call and with visible text
Flags kept on passing rows: per assistant turn think_ok (a <think> of >= 20 chars at its start), thinking loops
(a sentence of >= 25 chars repeated >= 4 times in one turn), duplicate rows (same text), python-literal responses.
Out: <out>/<name>.report.json, <out>/<name>.pass.jsonl (index, id, tokens, think_ok per assistant turn, flags)."""
import ast,collections,hashlib,json,re,sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer
T=AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
TC=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S); TH=re.compile(r'^\s*<think>(.*?)</think>',re.S)
TOOLS=re.compile(r'<tools>\s*(.*?)\s*</tools>',re.S); SENT=re.compile(r'[^.!?\n]{25,}[.!?\n]')
ASSIST={'gpt','model','assistant'}
def lit(s):
    try: return json.loads(s),'json'
    except Exception:
        try: return ast.literal_eval(s),'python'
        except Exception: return None,None
def names(x):
    if isinstance(x,dict): x=x.get('tools',[x])
    out=set()
    for e in x if isinstance(x,list) else []:
        if isinstance(e,dict):
            e=e.get('function',e)
            if isinstance(e,dict) and 'name' in e: out.add(e['name'])
    return out
def q(v,p): v=sorted(v); return v[int(p*(len(v)-1))] if v else 0
name,path,out=sys.argv[1],sys.argv[2],sys.argv[3]
rows=pq.read_table(path).to_pylist()
C=collections.Counter(); F=collections.Counter(); keep=[]; seen=set(); toks=[]; th_turns=collections.Counter()
for i,r in enumerate(rows):
    conv=[(m.get('from',m.get('role')),m.get('value',m.get('content')) or '') for m in r['conversations']]
    tools=None
    if r.get('tools'):
        x,_=lit(r['tools']); tools=names(x) if x is not None else None
    if tools is None or not tools:
        for role,v in conv:
            if role=='system':
                for blk in TOOLS.findall(v):
                    x,_=lit(blk)
                    if x: tools=names(x); break
    why=None; pend=0; thinks=[]; loops=0; pyresp=0; ncalls=0
    for k,(role,v) in enumerate(conv):
        for a,b in (('<think>','</think>'),('<tool_call>','</tool_call>'),('<tool_response>','</tool_response>')):
            if role!='system' and v.count(a)!=v.count(b): why=why or 'unbalanced_tags'  # a system prompt may name the tags
        if role in ASSIST:
            if pend: why=why or 'orphan_response'
            mt=TH.match(v); th=mt.group(1).strip() if mt else ''
            thinks.append(len(th)>=20)
            if th and max(collections.Counter(s.strip() for s in SENT.findall(th)).values() or [0])>=4: loops+=1
            calls=TC.findall(v); ncalls+=len(calls)
            for c in calls:
                j,kind=lit(c)
                if kind!='json' or not isinstance(j,dict) or not isinstance(j.get('arguments'),dict): why=why or 'bad_call'; continue
                if not tools: why=why or 'tools_missing'
                elif j.get('name') not in tools: why=why or 'bad_call'
            pend=len(calls)
        elif role=='tool':
            if not pend: why=why or 'orphan_response'
            pend=0
            if "{'" in v and '{"' not in v: pyresp+=1
    last_role,last=conv[-1]
    vis=TH.sub('',last,count=1).strip() if last_role in ASSIST else ''
    if last_role not in ASSIST or '<tool_call>' in last or not vis: why=why or 'no_final_answer'
    C[why or 'PASS']+=1
    if why: continue
    txt=''.join(v for _,v in conv); h=hashlib.sha256(txt.encode()).hexdigest()
    flags=[]
    if h in seen: flags.append('duplicate')
    seen.add(h)
    if loops: flags.append('think_loop')
    if pyresp: flags.append('python_response')
    if not ncalls: flags.append('no_tool_call')
    n=len(T(txt,add_special_tokens=False)['input_ids']); toks.append(n)
    full=all(thinks) and bool(thinks)
    th_turns['full' if full else ('none' if not any(thinks) else 'partial')]+=1
    for f in flags: F[f]+=1
    keep.append(dict(index=i,id=r.get('id',f'{name}/{i}'),tokens=n,think_ok=thinks,flags=flags))
rep=dict(name=name,path=path,rows=len(rows),outcome=dict(C),passing_thinking=dict(th_turns),passing_flags=dict(F),
         tokens=dict(p50=q(toks,.5),p90=q(toks,.9),max=max(toks or [0]),le16k=sum(t<=16384 for t in toks),le32k=sum(t<=32768 for t in toks),le64k=sum(t<=65536 for t in toks)))
json.dump(rep,open(f'{out}/{name}.report.json','w'),indent=1)
with open(f'{out}/{name}.pass.jsonl','w') as f:
    for k in keep: f.write(json.dumps(k)+'\n')
print(json.dumps(rep),flush=True)
