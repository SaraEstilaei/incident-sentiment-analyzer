# Warranty NLP - Team Handoff

## Objective
Identify Amazon reviews mentioning product warranty, protection
plans, or related manufacturer services, then compute mean overall
ratings for each product.

## Dataset
- Source: Train_subset_50k.csv
- Reviews: 50000

## Methods
1. Regex keyword baseline
2. Word2Vec and FastText for candidate discovery
3. Expanded Regex with spelling variants
4. Protection-plan phrase expansion
5. Context-aware semantic candidate mining
6. Weakly supervised TF-IDF retrieval
7. Sentence-BERT candidate ranking

## Retrieval results
- Conservative core matches: 1027
- Products in core output: 945
- Additional context-rule candidates: 573
- Additional weak-model candidates: 125
- Total retrieved union: 1725
- Core coverage: 2.05%
- Exploratory retrieval coverage: 3.45%
- Candidates ranked with SBERT: 698

## Preliminary evaluation
The previous evaluation used 30 purposefully selected samples
with preliminary labels, not an independently verified test set.

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Regex | 0.800 | 0.889 | 0.842 |
| Expanded Regex | 0.800 | 0.889 | 0.842 |
| FastText | 0.643 | 1.000 | 0.783 |
| Word2Vec | 0.429 | 1.000 | 0.600 |
| Sentence-BERT | 0.000 | 0.000 | 0.000 |

These scores do not validate the complete V4 pipeline.
The FastText result may vary when its model is retrained.

## Human validation
- Stratified sample size: 250
- Confirmed positive labels: 0
- Confirmed negative labels: 0
- Unlabeled samples: 250

A reliable estimate of true warranty-related prevalence
cannot yet be reported without independent validation.

## Recommended outputs
- Conservative product statistics:
  outputs_warranty/warranty_v4_core_products.csv
- Conservative matched reviews:
  outputs_warranty/warranty_v4_core_matches.csv
- Additional candidates:
  outputs_warranty/warranty_v4_candidates.csv
  outputs_warranty/warranty_v4_weak_model_discoveries.csv
- Semantic ranking:
  outputs_warranty/warranty_v4_sbert_ranked.csv

## Limitations
- The 1,725 retrieved reviews are not 1,725 verified positives.
- Manufacturer replacements and RMAs may be unrelated to warranty.
- Sentence-BERT ranking scores are not probabilities.
- Average overall rating measures general product satisfaction
  among matched reviews, not warranty-specific sentiment.
- Population prevalence and recall remain unverified.

## Delivery status
Retrieval, preliminary evaluation, product statistics, and
documentation are ready for team review.
Independent validation remains follow-up work.


## Product-level reliability

- Products with one review: 879
- Products with two reviews: 50
- Products with three or more reviews: 16

These averages are descriptive and represent overall
product ratings among warranty-related reviews.
Products with very few reviews have less reliable averages.
