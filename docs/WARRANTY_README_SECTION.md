## Warranty Aspect Retrieval (Araz — NLP, Problem 2)

**Goal:** Retrieve reviews about product warranties, guarantees, protection plans, and closely related manufacturer repair/replacement services; then aggregate the **overall review rating** by product (`asin`). This is warranty-aspect **retrieval**, not warranty-specific sentiment classification.

### Method

1. **Interpretable baseline:** Regex for `warranty`, `guarantee`, spelling variants, and `protection plan`.
2. **Vocabulary exploration:** Word2Vec and FastText for semantically related terms and spelling variation.
3. **Recall-oriented discovery:** Context-aware phrase rules plus weakly supervised TF-IDF / logistic regression. The latter uses noisy regex-derived labels, not human ground truth.
4. **Semantic triage:** Sentence-BERT ranks new review candidates by similarity to warranty-related descriptions; cosine scores are **not** calibrated probabilities.
5. **Error analysis:** Separate literal warranty references from generic claims such as “I guarantee you…” and examine RMA/returns ambiguity.
6. **Product aggregation:** Count matched reviews per `asin` and compute mean `overall` (general product rating) within the matched set.

### Results on `Train_subset_50k.csv`

| Output | Reviews | Status |
|---|---:|---|
| Original two-keyword regex | 979 | Keyword hits |
| Conservative V2/V4 core | **1,027** | Rule-matched, not fully human-verified |
| Additional context-rule candidates | 573 | Unverified |
| Additional weak-model-only candidates | 125 | Unverified |
| **Total candidate union** | **1,725** | 3.45% of 50,000; **not confirmed positives** |
| Stricter context-audited subset | 975 | Provisional sensitivity analysis, not validated replacement |

The core maps to **945 products**; **879 of 945 (93.02%)** have only one matched review, so product averages are descriptive and often unstable. The stricter subset maps to 896 products. A stratified validation sample of 250 reviews is prepared but has **0 independent human labels** at handoff. Consequently, the true prevalence and validated precision/recall of V4 are unknown; neither 1,027 nor 1,725 is an established count of all genuinely warranty-related reviews.

### Reproduce / inspect

Requires the project training-data archive `train subset.zip` containing `Train_subset_50k.csv` (not committed to Git). Place it under `data/` or `~/Downloads` as described in the scripts.

```bash
python -m pip install -r requirements-warranty.txt
python src/nlp/warranty_semantic_v4.py
python src/nlp/warranty_v4_weak_search.py
python src/nlp/warranty_v4_make_sample.py
# Optional: run notebooks/nlp/warranty_araz.ipynb and
# notebooks/nlp/warranty_v4_review.ipynb for detailed exploration / SBERT ranking.
```

**Do not run `warranty_v4_estimate.py` as a final prevalence estimate before human labelling:** it requires at least 20 verified labels in each of four sampling strata.

**Key artifacts:**

- Main notebook: `notebooks/nlp/warranty_araz.ipynb`
- SBERT review notebook: `notebooks/nlp/warranty_v4_review.ipynb`
- Product statistics: `outputs_warranty/warranty_v4_core_products.csv`
- Broader candidate pool: `outputs_warranty/warranty_v4_candidates.csv` and `outputs_warranty/warranty_v4_weak_model_discoveries.csv`
- Detailed handoff: `docs/WARRANTY_V4_TEAM_HANDOFF.md`
- Error-analysis caveats: `docs/WARRANTY_V4_QA_ADDENDUM.md`

**Suggested team-members table entry:** `**Araz** | **NLP warranty aspect retrieval (Problem 2):** regex/Word2Vec/FastText exploration, context-aware and weak-supervised retrieval, Sentence-BERT candidate ranking, product-level overall-rating aggregation, error analysis, and transparent limitations.`

> **Integration:** Append this section to the existing `dev` README instead of replacing it. The `dev` README contains the team's CV documentation, while the warranty feature branch still has an empty top-level `README.md`.
