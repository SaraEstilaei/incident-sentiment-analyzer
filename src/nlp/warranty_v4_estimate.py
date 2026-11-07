"""Estimate prevalence from a randomly selected, human-labelled, stratified sample.
Enter 0/1 in `human_label` of warranty_v4_stratified_review.csv first.
"""
import csv, math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'outputs_warranty'/'warranty_v4_stratified_review.csv'
samples=defaultdict(list)
with p.open(encoding='utf-8-sig',newline='') as f:
 for r in csv.DictReader(f):
  if r['human_label'].strip() in ('0','1'):
   samples[r['stratum']].append(int(r['human_label']))
N={'core':1027,'contextual_candidates':573,'weak_model_candidates':125,'unmatched':50000-1725}
if any(len(samples[g])<20 for g in N):
 print('Need at least 20 completed labels per stratum before estimating.')
 for g in N: print(g, len(samples[g]),'labels /',N[g],'reviews')
 raise SystemExit(1)

def wilson(k,n,z=1.96):
 p=k/n; q=z*z
 center=(p+q/(2*n))/(1+q/n)
 rad=z*math.sqrt(p*(1-p)/n+q/(4*n*n))/(1+q/n)
 return center-rad,center+rad

print('Human labelled, stratified estimate (approximate; sampling uncertainty only):')
total=low=high=0.0
for g,n_total in N.items():
 y=samples[g]; n=len(y); k=sum(y);pr=k/n;lo,hi=wilson(k,n)
 print(g, 'n=',n,'positives=',k,'positive_rate=',round(pr,3), 'Wilson95%=', (round(lo,3),round(hi,3)))
 total+=n_total*pr; low+=n_total*lo;high+=n_total*hi
print('Estimated warranty-related reviews:',round(total),'/ 50000')
print('Approximate conservative interval from stratum Wilson bounds:',round(low),'to',round(high))
print('Warning: sensitive to imperfect human labels and the scope of the warranty definition.')
