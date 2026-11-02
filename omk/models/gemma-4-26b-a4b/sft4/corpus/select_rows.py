"""bm-889 sft4 selection (owner 2026-11-02): rows that passed validate_hermes.py, every assistant turn thinking,
no thinking loop, no duplicate, no python-literal tool response. Out: selected/<name>.keep.jsonl + selection.json."""
import json,collections
S='/mnt/sdc/omk-model-optimizer/g4_recovery/sft4_build/'
res={}
for name in ('lambda_kimi','interstellarninja'):
    k=[json.loads(l) for l in open(S+f'validated/{name}.pass.jsonl')]
    why=collections.Counter(); keep=[]
    for r in k:
        w=None
        if not (r['think_ok'] and all(r['think_ok'])): w='turn_without_thinking'
        elif 'think_loop' in r['flags']: w='think_loop'
        elif 'duplicate' in r['flags']: w='duplicate'
        elif 'python_response' in r['flags']: w='python_response'
        why[w or 'KEEP']+=1
        if not w: keep.append(r)
    with open(S+f'selected/{name}.keep.jsonl','w') as f:
        for r in keep: f.write(json.dumps(r)+'\n')
    t=[r['tokens'] for r in keep]; t.sort()
    res[name]=dict(validated=len(k),**why,tokens_total=sum(t),p50=t[len(t)//2],p90=t[int(.9*(len(t)-1))],max=t[-1],
                   le16k=sum(x<=16384 for x in t),le32k=sum(x<=32768 for x in t),le64k=sum(x<=65536 for x in t),
                   with_calls=sum('no_tool_call' not in r['flags'] for r in keep))
    print(name,res[name])
json.dump(res,open(S+'selection.json','w'),indent=1)
