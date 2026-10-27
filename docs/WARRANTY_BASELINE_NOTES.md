# Warranty Review Analysis

## Dataset
Train_subset_50k.csv — 50,000 preprocessed reviews.

## Keyword Baseline
- Keywords: warranty, guarantee
- Clean text matches: 965
- Full text matches: 979
- Additional summary matches: 14
- Products with matched reviews: 903

## Word Embeddings
- Word2Vec: 70 suggestions, 68 unique terms
- FastText: 60 suggestions, 52 unique terms
- Sentence-BERT: 90 reviews scored
- Co-occurrence analysis: 61 terms
- Spelling variants: 26 additional matches

## Findings
- FastText retrieved spelling variants such as warrenty and warrantee.
- Some embedding suggestions are unrelated to warranty.
- Cosine similarity and co-occurrence do not prove relevance.
- Full text is preferred for the keyword baseline.

## Status
- Regex baseline: completed
- Word2Vec candidate extraction: completed
- FastText experiment: completed
- Sentence-BERT scoring: completed
- Manual labeling: pending
- Precision, recall and F1 comparison: pending
- Final model selection: pending

## Next
Manually label the evaluation sample and compare all methods
using the same ground-truth labels.
