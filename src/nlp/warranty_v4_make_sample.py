"""Generate disjoint random samples for population-prevalence estimation.
Run after `warranty_semantic_v4.py` and `warranty_v4_weak_search.py`.
All labels must be independently reviewed by a human before estimating.
"""
import csv
import io
import random
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs_warranty'

def load(name):
 with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(f))
core=load('warranty_v4_core_matches.csv')
context=load('warranty_v4_candidates.csv')
weak=load('warranty_v4_weak_model_discoveries.csv')
seen={int(r['review_id']) for s in [core,context,weak] for r in s}
zip_paths=[ROOT/'data'/'train subset.zip',ROOT.parent/'data'/'train subset.zip',ROOT/'train subset.zip',Path.home()/'Downloads'/'train subset.zip']
p=next((x for x in zip_paths if x.exists()),None)
if p is None: raise FileNotFoundError('train subset.zip missing')
with zipfile.ZipFile(p) as z:
 inner=next(x for x in z.namelist() if x.endswith('.csv'))
 rows=list(csv.DictReader(io.TextIOWrapper(z.open(inner),encoding='utf-8-sig')))
remaining=sorted(set(range(len(rows)))-seen)
rng=random.Random(20261101)
result=[]
for name,source,n in [('core',core,50),('contextual_candidates',context,60),('weak_model_candidates',weak,40)]:
 for r in rng.sample(source,n):
  result.append({'stratum':name,'review_id':r['review_id'],'asin':r.get('asin',''), 'overall':r.get('overall',''),'source_rule':r.get('primary_rule',r.get('matched_text','')),'full_text':r['full_text'],'human_label':'','reason':''})
for i in rng.sample(remaining,100):
 r=rows[i];result.append({'stratum':'unmatched','review_id':i,'asin':r.get('asin',''),'overall':r.get('overall',''),'source_rule':'none','full_text':r.get('full_text',''), 'human_label':'','reason':''})
rng.shuffle(result)
assert len({r['review_id'] for r in result})==len(result)==250
with (OUT/'warranty_v4_stratified_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print('Created',len(result),'stratified sample rows. Labels BLANK; must be annotated.')
