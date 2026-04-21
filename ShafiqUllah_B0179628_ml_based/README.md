# Student 2 — ML-Based Hallucination Detector

**MSc Project: Detecting and Analysing Hallucinations in Large Language Model Outputs**  
University of the West of Scotland — School of Computing, Engineering and Physical Sciences

---

## Overview

This is the **individual contribution** of Student 2 to the group's comparative hallucination
detection study. It implements a classical machine learning-based hallucination detector using
rich feature engineering across four complementary signal families.

The system is designed to be **directly comparable** with Student 1's rule-based approach —
using the same datasets, evaluation metrics, and output format.

---

## Project Structure

```
student2_ml_based/
├── src/
│   ├── main.py                  # Demo runner (quick test of predictions)
│   ├── train.py                 # Full training pipeline
│   ├── evaluation.py            # Evaluation on labelled dataset
│   ├── interpretability.py      # Feature importance + SHAP analysis
│   ├── build_dataset.py         # Rebuild combined dataset from raw sources
│   ├── pipeline.py              # MLHallucinationPipeline (main interface)
│   ├── feature_pipeline.py      # Feature extraction + TF-IDF + scaler
│   ├── models.py                # RF, XGBoost, LightGBM, SVM definitions
│   ├── trainer.py               # Cross-validation + Optuna tuning
│   ├── config.py                # Settings dataclass
│   ├── utils.py                 # Text normalization helpers
│   └── modules/
│       ├── lexical_features.py      # 16 lexical features
│       ├── semantic_features.py     # 9 semantic features
│       ├── linguistic_features.py   # 13 linguistic features
│       └── confidence_features.py   # 13 confidence/certainty features
├── data/
│   ├── raw/                     # Raw datasets (TruthfulQA, HaluEval, FEVER)
│   └── processed/
│       └── combined_dataset.json    # Shared with Student 1
├── models/                      # Saved trained models
├── results/                     # Evaluation outputs
├── tests/
│   └── quick_test.py            # Unit tests (no trained model needed)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests (no model needed)

```bash
cd src
python ../tests/quick_test.py
```

### 3. Run feature demo (no model needed)

```bash
cd src
python main.py --features-only
```

### 4. Train all models

```bash
cd src
python train.py
```

With Optuna hyperparameter tuning:
```bash
python train.py --tune --n-trials 50
```

Train specific models only:
```bash
python train.py --models random_forest xgboost
```

### 5. Evaluate on test set

```bash
cd src
python evaluation.py
```

### 6. Feature importance analysis

```bash
cd src
python interpretability.py --top-n 30
python interpretability.py --shap    # requires shap library
```

---

## Feature Engineering

The system extracts **51 handcrafted features** across four modules plus TF-IDF:

| Group | Features | Examples |
|-------|----------|---------|
| Lexical | 16 | TF-IDF, TTR, hedge ratio, word count |
| Semantic | 9 | Answer-knowledge cosine sim, intra-coherence |
| Linguistic | 13 | FK grade, passive voice, readability |
| Confidence | 13 | Perplexity proxy, certainty words, vagueness |
| TF-IDF | 3000 | N-gram representation of answer text |

---

## Models

Four classifiers are trained and compared:

| Model | Notes |
|-------|-------|
| Random Forest | Robust, built-in feature importance |
| XGBoost | High performance gradient boosting |
| LightGBM | Efficient, scalable gradient boosting |
| SVM (RBF) | Non-linear decision boundary |

Best model is selected by validation F1-score and saved to `models/best_model.pkl`.

---

## Results (Test Set — 4,000 samples)

| Metric | Student 2 (XGBoost) | Student 1 (Rule-based) |
|--------|---------------------|------------------------|
| Accuracy | **0.9785** | 0.8659 |
| Precision | **0.9795** | 0.9263 |
| Recall | **0.9775** | 0.7951 |
| F1-score | **0.9785** | 0.8557 |
| ROC-AUC | **0.9956** | 0.8836 |
| Avg inference (ms) | 59.6 | 19.8 |

Student 2's ML approach outperforms Student 1's rule-based system on all accuracy metrics,
at the cost of higher inference time and requiring an offline training phase.

## Evaluation Metrics

Consistent with Student 1 and the group's shared protocol:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Per-source breakdown (HaluEval, TruthfulQA)
- Confusion matrix (TP, TN, FP, FN)
- Average inference time (ms)

---

## Datasets

Shared with Student 1:

| Dataset | Type | Samples |
|---------|------|---------|
| HaluEval | QA hallucinations | ~19,000 |
| TruthfulQA | Factual QA | ~739 |
| FEVER | Claim verification | Optional |

Combined balanced dataset: `data/processed/combined_dataset.json`  
(10,000 factual + 10,000 hallucinated = 20,000 total)

---

## Comparing with Student 1

Both pipelines expose identical interfaces:

<!-- ```python
# Student 1 (Rule-based)
from pipeline import RuleBasedHallucinationPipeline
s1 = RuleBasedHallucinationPipeline()
result = s1.predict(answer="...", question="...", knowledge="...")

# Student 2 (ML-based)
from pipeline import MLHallucinationPipeline
s2 = MLHallucinationPipeline.load("models/best_model.pkl", "models/feature_pipeline.pkl")
result = s2.predict(answer="...", question="...", knowledge="...") -->

# Both return:
# result["final_result"]["label"]       → 0 or 1
# result["final_result"]["final_score"] → probability [0,1]
```

---

## References

- Manakul et al. (2023) SelfCheckGPT
- Varshney et al. (2023) Hallucination validation via low-confidence detection
- Breiman (2001) Random Forests
- Chen & Guestrin (2016) XGBoost
- Lundberg & Lee (2017) SHAP
- Reimers & Gurevych (2019) Sentence-BERT
