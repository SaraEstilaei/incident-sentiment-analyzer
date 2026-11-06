"""Optional weakly supervised contextual retrieval (not a validated classifier).
Requires numpy and scikit-learn. Run after warranty_semantic_v4.py.
    python src/nlp/warranty_v4_weak_search.py
Training labels are automatically inferred from regex, NOT ground truth.
All discovered reviews require independent annotation.
"""
import csv
import json
import io
import random
import re
import zipfile
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs_warranty"
zip_paths = [ROOT / 'data' / 'train subset.zip', ROOT.parent/'data'/'train subset.zip',ROOT/'train subset.zip',Path.home()/'Downloads'/'train subset.zip']
zip_path = next((p for p in zip_paths if p.is_file()),None)
if zip_path is None:
    raise FileNotFoundError('train subset.zip missing')
with zipfile.ZipFile(zip_path) as z:
    inner=next(n for n in z.namelist() if n.endswith('.csv'))
    rows=list(csv.DictReader(io.TextIOWrapper(z.open(inner),encoding='utf-8-sig')))
texts=[r.get('full_text') or '' for r in rows]
redact=re.compile(r'\b(?:warranty|guarantee|warranties|warrantee|warrantees|warrenty|protection\s+plan)\b',re.I)
core=np.array([i for i,t in enumerate(texts) if redact.search(t)],dtype=int)
core_ids=set(core.tolist())
noncore=np.array([i for i in range(len(texts)) if i not in core_ids],dtype=int)
rng=np.random.default_rng(55)
neg=rng.choice(noncore,9000,replace=False)
vec=TfidfVectorizer(min_df=2,max_features=35000,ngram_range=(1,2),stop_words='english',sublinear_tf=True,dtype=np.float32)
X=vec.fit_transform([redact.sub(' ',t)[:3500] for t in texts])
train=np.concatenate([core,neg])
y=np.concatenate([np.ones(len(core)),np.zeros(len(neg))])
clf=LogisticRegression(max_iter=300,class_weight='balanced',C=0.5)
clf.fit(X[train],y)
p=clf.predict_proba(X[noncore])[:,1]
order=np.argsort(-p)[:200]
known=set()
with (OUT/'warranty_v4_candidates.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):known.add(int(r['review_id']))
discoveries=[]
for rank,k in enumerate(order,1):
    i=int(noncore[k])
    if i in known:
        continue
    discoveries.append(dict(
        review_id=i,rank=rank,weak_model_score=round(float(p[k]),4),
        asin=rows[i].get('asin',''),overall=rows[i].get('overall',''),
        full_text=texts[i],human_label='',review_notes=''
    ))
with (OUT/'warranty_v4_weak_model_discoveries.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(discoveries[0]));w.writeheader();w.writerows(discoveries)
summary_path=OUT/'warranty_v4_summary.json'
summary=json.loads(summary_path.read_text(encoding='utf-8'))
core_count=summary['core_rule_matches']
rule_count=summary['new_unverified_candidates']
all_ids=set()
all_asins=set()
for name in ['warranty_v4_core_matches.csv','warranty_v4_candidates.csv','warranty_v4_weak_model_discoveries.csv']:
 with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   all_ids.add(int(row['review_id']))
   if row['asin']:all_asins.add(row['asin'])
summary.update(weak_model_only_unverified_candidates=len(discoveries),
    total_cumulative_unverified_retrieval_candidates=rule_count+len(discoveries),
    expanded_exploratory_union_with_model=len(all_ids),
    exploratory_union_with_model_products=len(all_asins))
summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('Weak-model-only additional candidates (unverified):',len(discoveries))
print('Saved:',OUT/'warranty_v4_weak_model_discoveries.csv')
