# Project 2 — GitHub review snapshot (2026-11-06)

Repository: https://github.com/SaraEstilaei/incident-sentiment-analyzer

## Branches and integration status

| Location | Evidence found | Integration note |
|---|---|---|
| `dev`: `README.md` | Team overview, project scope and Sara's detailed Computer Vision experiment notes; 'Araz' and three other team members still show 'To be updated'. | Add the warranty README section after syncing with `dev`. |
| `dev`: `requirements.txt` | Includes torch, torchvision, numpy, pandas, scikit-learn, optuna, matplotlib, seaborn, cv2, PIL, dotenv, huggingface-hub, tqdm, joblib. | Primarily CV dependencies; NLP extras are in `requirements-warranty.txt` on the warranty branch. |
| `dev`: `notebooks/01` to `07` | CV acquisition, EDA, MobileNetV2, EfficientNet-B0, temporal-motion ablations and feature selection. | Sara's README states final CV test mAP 0.8119 on 23 videos. |
| `Model/NLP_02_DeBERTa` | `NLP_02_deberta_final.ipynb`; logged validation micro-F1 0.7418 on 16k. | Exists in separate branch, not present in `dev` tree reviewed. |
| `Model/NLP_03_nlptown` | `nlp-03-nlptown-final.ipynb`; logged validation micro-F1 0.7231 on 16k. | Separate branch, not present in `dev` tree reviewed. |
| `Model/NLP_04_ensemble` | `NLP_04_ensemble.ipynb`; logged test micro-F1 0.7439 (8k), vs DeBERTa 0.7369 and NLPTown 0.7206 on same test. Exported 20k-row `q2_submission.csv` within notebook. | Notebook and outputs may require access to separate Drive/Kaggle artifacts; verify before calling it a reproducible end-to-end submission. |
| `Model/NLP_02_DeBERTa_for_ans` | Notebook generating 20k DeBERTa prediction probabilities for held-out data. | Separate branch. |
| `feature/araz-warranty-baseline` | Warranty notebooks/scripts, 50k review retrieval, core 1,027, union 1,725 candidates, strict 975 sensitivity set. | PR #16 to `dev` **open** at review time. |

## What needs to be coordinated for the team presentation

1. Confirm the final tasks/ownership of Amirali, Kiana and Ramtin; current `dev` README has placeholders, so the repository does not establish their exact work allocation.
2. Ask the owner of rating prediction to confirm final validation/test split, micro-F1 and submission artifact; do not confuse 8k labelled test in a notebook with the hidden competition scoring set.
3. Add Araz's warranty section to the existing `dev` README without overwriting Sara's content.
4. Integrate the sentiment notebook branches into `dev` via reviewed PRs, or state explicitly they remain branch-local.
5. Consolidate `requirements.txt` + `requirements-warranty.txt` only after confirming installation on the intended kernel/Colab runtime.
6. Presentation: show CV and NLP rating separately, and distinguish warranty retrieval and general-star aggregation from warranty-specific sentiment.

## Sources checked

- [`dev/README.md`](https://github.com/SaraEstilaei/incident-sentiment-analyzer/blob/dev/README.md)
- [`dev/requirements.txt`](https://github.com/SaraEstilaei/incident-sentiment-analyzer/blob/dev/requirements.txt)
- [PR #16](https://github.com/SaraEstilaei/incident-sentiment-analyzer/pull/16)
- [DeBERTa experiment](https://github.com/SaraEstilaei/incident-sentiment-analyzer/blob/Model/NLP_02_DeBERTa/NLP_02_deberta_final.ipynb)
- [NLPTown experiment](https://github.com/SaraEstilaei/incident-sentiment-analyzer/blob/Model/NLP_03_nlptown/nlp-03-nlptown-final.ipynb)
- [Ensemble experiment](https://github.com/SaraEstilaei/incident-sentiment-analyzer/blob/Model/NLP_04_ensemble/NLP_04_ensemble.ipynb)
