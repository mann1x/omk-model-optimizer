import json,sys
sys.path.insert(0,'/mnt/sdc/omk-model-optimizer/g4_recovery/omk_sft3_code')
from omk_qad import rollouts as R
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/mnt/sdc/ml/models/gemma-4-26B-A4B-it')
bad=[]
for i,l in enumerate(open('conv/lambda_kimi_pre2r.train.conv.jsonl')):
    r=json.loads(l)
    try: R.build(tok,[r],32768,16384,R.CODE_ARGS,'keep',True,0,'reasoning')
    except ValueError as e: bad.append((i,r['id'],str(e)[:80]))
print(len(bad),bad[:10])
json.dump([b[1] for b in bad],open('kimi_pre2r_roundtrip_bad.json','w'))
