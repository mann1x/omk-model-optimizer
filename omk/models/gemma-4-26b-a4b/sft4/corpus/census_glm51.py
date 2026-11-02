"""Validation census of lambda/hermes-agent-reasoning-traces glm-5.1: thinking, tool-call format, pairing, length."""
import json,re,collections
import pyarrow.parquet as pq
from transformers import AutoTokenizer
T=AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
t=pq.read_table('/mnt/sdc/omk-model-optimizer/data/hermes_lambda_glm51/glm-5.1.train.0000.parquet').to_pylist()
TC=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S); TH=re.compile(r'^\s*<think>(.*?)</think>',re.S)
def q(v,p): v=sorted(v); return v[int(p*(len(v)-1))] if v else 0
C=collections.Counter(); thl=[]; first=[]; later=[]; toks=[]; turns=[]; cat=collections.Counter(); badrow=collections.Counter()
for r in t:
    conv=r['conversations']; cat[r['category']]+=1
    try: tools={x['name'] for x in json.loads(r['tools'])}
    except Exception: tools=set(); C['tools_unparsable']+=1
    roles=[m['from'] for m in conv]; ok=True; pend=0; gi=0
    for k,m in enumerate(conv):
        v=m['value']
        if m['from']=='gpt':
            gi+=1; C['gpt_turns']+=1
            mt=TH.match(v)
            if not mt: C['gpt_no_think']+=1; th=''
            else: th=mt.group(1).strip()
            (first if gi==1 else later).append(len(th)); thl.append(len(th))
            if not th: C['think_empty']+=1
            elif len(th)<=40: C['think_le40']+=1
            if v.count('<think>')>1: C['multi_think']+=1; ok=False
            calls=TC.findall(v)
            if v.count('<tool_call>')!=v.count('</tool_call>'): C['call_unbalanced']+=1; ok=False
            for c in calls:
                C['calls']+=1
                try:
                    j=json.loads(c)
                    if not isinstance(j.get('arguments'),dict): C['call_args_not_obj']+=1; ok=False
                    if j.get('name') not in tools: C['call_unknown_tool']+=1; ok=False
                except Exception: C['call_bad_json']+=1; ok=False
            pend=len(calls)
            if not calls: C['gpt_text_turns']+=1
        elif m['from']=='tool':
            n=v.count('<tool_response>')
            C['tool_msgs']+=1
            if pend==0: C['tool_without_call']+=1; ok=False
            pend=0
    if conv[-1]['from']!='gpt': C['ends_not_gpt']+=1; ok=False
    elif '<tool_call>' in conv[-1]['value']: C['ends_on_call']+=1; ok=False
    if roles.count('human')!=1: C['multi_human_rows']+=1
    if not ok: badrow['bad']+=1
    turns.append(roles.count('gpt'))
    toks.append(len(T(''.join(m['value'] for m in conv),add_special_tokens=False)['input_ids']))
n=len(t)
print('rows',n,'categories',cat.most_common(8))
print(dict(C))
print('rows failing a structural check',badrow['bad'],'(%.1f%%)'%(100*badrow['bad']/n))
print('gpt turns/row p50 %d p90 %d max %d'%(q(turns,.5),q(turns,.9),max(turns)))
print('think chars: all p10 %d p50 %d p90 %d | first turn p10 %d p50 %d | later turns p10 %d p50 %d p90 %d'%(q(thl,.1),q(thl,.5),q(thl,.9),q(first,.1),q(first,.5),q(later,.1),q(later,.5),q(later,.9)))
print('think empty %.1f%% of gpt turns (first turns %.1f%%, later %.1f%%), <=40ch %.1f%%'%(100*C['think_empty']/C['gpt_turns'],100*sum(x==0 for x in first)/len(first),100*sum(x==0 for x in later)/max(len(later),1),100*C['think_le40']/C['gpt_turns']))
for L in (16384,32768,49152,65536,131072): print('rows <= %d tok: %d (%.1f%%)'%(L,sum(x<=L for x in toks),100*sum(x<=L for x in toks)/n))
print('row tokens p50 %d p90 %d max %d'%(q(toks,.5),q(toks,.9),max(toks)))
