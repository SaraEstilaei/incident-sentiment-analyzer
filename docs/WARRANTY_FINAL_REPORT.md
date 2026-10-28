# Warranty Review Analysis — Final Report

## Overview
Analyzed 50,000 preprocessed Amazon electronics reviews
to identify warranty-related feedback.

## Approach
- Built a Regex baseline using `warranty` and `guarantee`.
- Explored related terms using Word2Vec and FastText.
- Checked spelling variants and candidate contexts.
- Tested Sentence-BERT semantic similarity.
- Extended Regex with selected spelling variants.

## Results
- Baseline matches: 979
- Expanded matches: 1010
- Additional matches: 31
- Products identified: 929

## Preliminary Evaluation

30 manually reviewed samples: 9 positive and 21 negative.

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Regex | 0.800 | 0.889 | 0.842 |
| Expanded Regex | 0.800 | 0.889 | 0.842 |
| FastText | 0.643 | 1.000 | 0.783 |
| Word2Vec | 0.429 | 1.000 | 0.600 |
| Sentence-BERT | 0.000 | 0.000 | 0.000 |

## Selected Approach
Expanded Regex was selected as a practical preliminary approach.
It covers additional spelling variants without reducing F1
on the evaluation sample.

## Limitations
- Evaluation uses only 30 purposefully sampled reviews.
- Initial labels require independent verification.
- The 31 additional matches were not all manually verified.
- Sentence-BERT used an uncalibrated similarity threshold.
- Results are based on the 50k subset, not the full dataset.
- Overall ratings approximate warranty satisfaction;
  they do not directly measure warranty-specific sentiment.

## Outputs
- `expanded_warranty_matches.csv`
- `expanded_warranty_product_stats.csv`
- `expanded_warranty_summary.json`
- `model_comparison.csv`
- `warranty_final_evaluation.json`
