# Warranty-related Review Retrieval — V4 Evidence Report

## Question
How many of the 50,000 Amazon electronics reviews discuss product warranties, protection plans, or closely related manufacturer repair/replacement services?

**Important:** There is no fully human-annotated 50,000-row ground truth. Accordingly, the exact number of true warranty-related reviews is currently **unknown**. A keyword hit count or model-retrieved count must not be called the true prevalence.

## Reproducible findings on `Train_subset_50k.csv`

| Scope | Review count | % of 50,000 | Distinct ASINs | Verification status |
|---|---:|---:|---:|---|
| Regex V2 core (spelling variants + `protection plan`) | 1,027 | 2.054% | 945 | Rule matches, not all manually checked |
| Additional RMA / plan / repair-related contextual candidates | 573 | 1.146% | — | Not independently verified |
| Exploratory core + contextual rule candidates | 1,600 | 3.200% | 1,419 | Not a validated classifier |
| New candidates from weakly supervised text retrieval beyond those rules | 125 | 0.250% | — | Not independently verified |
| Exploratory **union of all** methods | 1,725 | 3.450% | 1,517 | **Candidate pool only, NOT ground truth** |

Distinct ASINs are computed from matched reviews, and `overall` means general product rating in matched reviews, *not* a directly labeled warranty-satisfaction sentiment.

## Method

1. Begin with the interpretable Regex V2: `warranty`, `guarantee`, selected spelling variants, and `protection plan`.
2. Mine additional context-aware candidates using RMA, SquareTrade, repair authorizations, paid plans, replacement at no charge, manufacturer repairs, support-related replacement language, and selected other after-sales phrases. Avoid naive expansion with `support`, `replacement`, `claim`, `insurance`, and `coverage` alone.
3. Train an **optional weak-supervision retrieval model**: mask exact V2 seed terms in texts, use V2 matches as *noisy positives* and 9,000 randomly drawn nonmatches as *noisy negatives*, fit TF-IDF word unigrams/bigrams + logistic regression, and retrieve the top 200 non-core reviews; 125 are novel relative to the contextual rules. These scores are for ranking only and are **not calibrated probabilities or a validated semantic classifier**.
4. Optionally rank the 698 candidate reviews with the pretrained Sentence-BERT model in the adjacent notebook; do not infer truth from its raw cosine similarity or a fixed uncalibrated threshold.
5. Human-annotate a **random stratified sample** from (a) core, (b) context candidates, (c) weak-model-only candidates, and (d) nonmatches to estimate true population prevalence, including undetected cases. The sample CSV leaves `human_label` blank deliberately.

## Targeted error analysis

A known positive in the 30-review sample (review ID `22552`) refers to the paid `Rescue` policy for data protection but does not use the original warranty seed terms. Adding narrowly tailored `Rescue policy` / `data recovery plan` candidate rules recovered that missed positive **without blindly matching** generic `policy`, `rescue`, or `data recovery`. This change affects candidate retrieval only and requires separate verification.

## Review of actual examples

- Positive-likely: manufacturer/seller support replacing a device after a failure (e.g. Anker, Corsair, Samsung-related repair process); `RMA` often refers to after-sales claims but sometimes a regular return.
- Positive-likely: SquareTrade protection programs and explicit `protection plan` expressions.
- False-positive risk: `extended coverage` is often Wi-Fi signal range; `insurance policy` is often figurative; `replacement` can refer to a new alternative product/part; `claims` can refer to product advertising claims; `support` may refer to software or hardware compatibility.

## Quality check on previous 30 preliminarily labelled reviews

This 30-review set was purposefully sampled and the labels were preliminary, not independently adjudicated. It **cannot** estimate population metrics or total true prevalence.

| Method | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| V2 Core | 8 | 2 | 1 | 0.800 | 0.889 | 0.842 |
| Core + 573 contextual candidates | 9 | 5 | 0 | 0.643 | 1.000 | 0.783 |
| Core + 573 + 125 additional model candidates | 9 | 5 | 0 | 0.643 | 1.000 | 0.783 |

**Interpretation:** Blind expansion worsened F1 on the preliminary sample but recovered the one known missed positive through targeted investigation of `Rescue policy` / `data recovery plan`. It cannot be declared the improved production extractor just because it returns more hits.

## Recommendation to mentor/team

Use V2 (1,027 *rule matches*) as the conservative reference **for now**. Use V4's 698 additional candidates as a separate inspection queue. The exact number of genuinely relevant reviews requires independent human labels; do not report 1,725 as verified. When sufficient labels exist, estimate prevalence from the `warranty_v4_stratified_review.csv` sample using `warranty_v4_estimate.py`. Only add candidate classes to a final extractor if their label-reviewed precision meets the team's agreed threshold, and rerun per-product statistics using that final approved set.

## Output guide

- `warranty_v4_core_matches.csv`: 1,027 V2 matches, preserving review IDs.
- `warranty_v4_core_products.csv`: product-level mean overall among these matched reviews.
- `warranty_v4_candidates.csv`: 573 *unverified* context-rule candidates, their matched evidence, and blank human label.
- `warranty_v4_weak_model_discoveries.csv`: 125 *unverified* additional weak-model-only candidates.
- `warranty_v4_priority_review.csv`: 198 prioritized candidate cases (115 direct-term and 80 randomly sampled contextual).
- `warranty_v4_random_unmatched_audit.csv`: 150 random nonmatches relative to core/context rules (exploratory only).
- `warranty_v4_stratified_review.csv`: 250 randomly sampled, mutually exclusive records across four strata; use this file for population estimation after human annotation.
- `warranty_v4_EXPLORATORY_products.csv`: product aggregates across core + 573 candidates, **not a validated final output**.
- `warranty_v4_summary.json`: counts and warnings.

## Caveats

- Weak-supervision training labels are noisy and the model was not independently validated; its scores are rankings, not probabilities.
- V2 also includes some ambiguous usages (`guarantee` as a verb), so 1,027 should not be described as 1,027 definitely true positives.
- Some genuine warranty-related reviews can be absent from **all** candidate methods. Hence 1,027 and 1,725 are *not proven lower/upper bounds* for true prevalence.
- Agreement about the annotation policy is essential: does “warranty-related” include ordinary returns and retailer refunds, or only warranty/guarantee/paid protection and manufacturer repair/replacement claims? Define this before labelling.
- Versioned output files preserve prior results; no destructive overwrite of prior V2/V3 outputs.
