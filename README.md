# Incident & Sentiment Analyzer

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/SaraEstilaei/incident-sentiment-analyzer.git)
[![Google Drive](<https://img.shields.io/badge/Google%20Drive-Data%20%26%20Artifacts-green?logo=googledrive>)](https://drive.google.com/drive/folders/1v-m-iUyZs6zxk0QtIHGCy2vL7q2Kaykb?usp=drive_link)

This repository contains the source code, experimental pipelines, and model evaluation artifacts for **Project 2 of the Quera AI Bootcamp (Summer 1405)**. The project addresses two core real-world machine learning challenges:

1. **Problem 1: Video Crash Detection** (Computer Vision)
2. **Problem 2: Amazon Reviews Sentiment Analysis** (Natural Language Processing)

---

## Team Members & Task Allocation

| Member                    | Primary Focus & Contributions                                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sara Estilaei**   | **Computer Vision (Problem 1):** End-to-end crash detection pipeline (Notebooks `01`, `03`, `06`, `06b`, `07`), frame extraction, feature caching, hyperparameter optimization, and multi-backbone cross-validation. |
| **Amirali Shahabi** | *To be updated*                                                                                                                                                                                                                    |
| **Kiana Ghaznavi**  | *To be updated*                                                                                                                                                                                                                    |
| **Ramtin Karimi**   | *To be updated*                                                                                                                                                                                                                    |
| **Araz**            | *To be updated*                                                                                                                                                                                                                    |

---

## Project Overview

### 1. Problem 1: Video Crash Detection

- **Objective:** Build a robust binary classification system to detect vehicular accidents from short video clips.
- **Approach:** To handle compute and memory constraints on standard hardware/Colab, videos are represented as sequences of uniformly sampled frames. Features are extracted using deep convolutional backbones and aggregated temporally for video-level prediction.

### 2. Problem 2: Product Review Sentiment Analysis

- **Objective:** Predict user rating scores (1 to 5) from raw Amazon customer reviews of electronics products.
- **Components:**
  - **Exploratory Data Analysis (EDA):** Class distribution inspection, text-length filtering, top reviewer analysis, and word clouds.
  - **Aspect-Based Analysis:** Semantic retrieval of warranty- and guarantee-related reviews using word embeddings / LLM representations.
  - **Sentiment Modeling:** Multi-class classification evaluated via micro-averaged F1 score on an unlabelled test split.

---

## Video Crash Detection Pipeline (Sara Estilaei)

I designed, implemented, and benchmarked the entire computer vision pipeline from initial data acquisition through final cross-validation model selection.

#### 1. Data Ingestion & Dev Sample (`01_download_and_extract.ipynb`)

- Authenticated with Hugging Face using secure token management (`.env`).
- Downloaded and validated metadata files (`solution.csv`, `sample_submission.csv`, `time_to_accident_test_map.csv`, `evaluate_submission.py`).
- Built a balanced 50-video development sample (25 crash / 25 non-crash) indexed in `videos_metadata.csv` to ensure unbiased early prototyping.

#### 2. MobileNetV2 Baseline (`03_cv_baseline_training.ipynb`)

- Fine-tuned an ImageNet-pretrained MobileNetV2 with a binary classification head on frame representations.
- Optimized hyperparameters via Optuna (8 trials) against validation macro-F1.
- Reached a tuned macro-F1 of **0.6703** on 10 validation videos, highlighting high variance and confirming the necessity for robust cross-validation.

#### 3. EfficientNet-B0 Pipeline & HPO (`06_cv_efficientnet_training.ipynb`)

- Extracted 10 uniformly spaced frames across 150 videos with verified train/val/test split separation (zero data leakage).
- Built an EfficientNet-B0 model with a custom classification head (`Dropout` + `Linear(1280, 1)`).
- Conducted Optuna hyperparameter optimization (12 trials) targeting validation mAP, achieving **0.7176** (retrained score: **0.7025**).

#### 4. Temporal Granularity & Motion Caching (`06b_effnet_30frames_motion.ipynb`)

- Scaled frame extraction to 30 frames per video (150 videos).
- Precomputed and cached frozen 1280-d appearance representations (`(150, 30, 1280)`) and frame-to-frame pixel motion arrays (`(150, 29)`).
- Evaluated feature ablations via repeated stratified 5-fold CV:
  - *Mean-pooled appearance alone:* **0.740 ± 0.065 mAP**
  - *Appearance + motion:* **0.701 ± 0.050 mAP**
  - *Finding:* Frame appearance features alone proved more stable and predictive than basic pixel-level motion statistics.

#### 5. Multi-Backbone Grid Search & Final Model (`07_cv_feature_comparison.ipynb`)

- Extracted cached frozen representations across 4 backbones: **EfficientNet-B0, EfficientNet-B3, ConvNeXt-Tiny, ResNet50**.
- Ran an exhaustive grid search over 127 train+val videos: 4 backbones × 4 pooling strategies (`mean`, `mean+max`, `mean+max+std`, `frame_top5`) × 3 regularization parameters ($C$).
- **Final Model Selection:** Selected **EfficientNet-B0 + frame_top5 + C = 0.1** based on cross-validation stability and generalizability (CV mean AP: **0.7349**).
- **Test Set Evaluation (23 videos):**
  - **Test mAP:** **0.8119** (Official scorer exact match: `0.811933`)
  - **95% Bootstrap Confidence Interval:** `[0.5988, 0.9671]` (2000 resamples)
- **Saved Artifacts:** `outputs/final_model.joblib`, `outputs/submission.csv`, `outputs/solution_test_only.csv`.

---

## Links & Resources

- **Source Code Repository:** [GitHub Repository](https://github.com/SaraEstilaei/incident-sentiment-analyzer.git)
- **Datasets, Feature Caches & Models:** [Google Drive Folder](https://drive.google.com/drive/folders/1v-m-iUyZs6zxk0QtIHGCy2vL7q2Kaykb?usp=drive_link)
