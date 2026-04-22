# Manzoor Ahmed B01798278 — Rule-Based Hallucination Detector

## Overview
This project implements Rule-Based Hallucination Detector:
- **Module 1** — Temporal consistency checking (27 anachronism rules, BC-year detection, event ordering)
- **Module 2** — Entity verification (50-country capitals database, person/role contradictions)
- **Module 3** — Contradiction detection (18 antonym pairs, inline negation)
- **Module 4** — Numerical plausibility validation (age, weight, population, percentage ranges)
- **Module 5** — Citation validation (DOI/arXiv format, suspicious journal names)
- **Module 6** — Knowledge grounding (answer vs knowledge passage comparison)

**Results:** Accuracy=0.8659 | Precision=0.9263 | Recall=0.7951 | F1=0.8557 | ROC-AUC=0.8836

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Commands (all run from inside src/)

```bash
cd src
```

### Run demo (16 built-in examples)
```bash
python main.py
```

### Run quick test (30 cases — must all pass)
```bash
python ../tests/quick_test.py
```

### Build evaluation dataset
```bash
python build_dataset.py
```

### Run full evaluation
```bash
python evaluation.py ../data/processed/combined_dataset.json
```

### Run grid search (weight optimisation)
```bash
python grid_search.py ../data/processed/combined_dataset.json
```

### Run prediction system (file + terminal mode)
```bash
python predict_file.py
```

---


## Dataset Files
```
data/raw/truthfulqa/TruthfulQA.csv
data/raw/halueval/qa_data.json
data/raw/fever/train.jsonl
```

---

## Output Files
```
results/metrics.json              — accuracy, precision, recall, F1, ROC-AUC, inference time
results/predictions.csv           — per-sample predictions (label, score, triggered modules)
results/errors.csv                — FP and FN cases for error analysis
results/grid_search_results.json  — optimal weights from grid search
```

---

## Expected Results
```
Accuracy  : 0.8659
Precision : 0.9263
Recall    : 0.7951
F1        : 0.8557
ROC-AUC   : 0.8836
Avg ms    : 3.70
TP: 7951   TN: 9367   FP: 633   FN: 2049
```

---

## Config (src/config.py)

| Setting                 | Default | Description                              |
|-------------------------|---------|------------------------------------------|
| `final_label_threshold` | 0.06    | Score threshold — lower = more sensitive |
| `knowledge weight`      | 0.30    | Highest — knowledge module is primary driver |
| `entity weight`         | 0.18    | Capital cities and role contradictions   |
| `contradiction weight`  | 0.16    | Antonym pairs and negation               |
| `temporal weight`       | 0.12    | Anachronisms and date ordering           |
| `numerical weight`      | 0.14    | Range validation                         |
| `citation weight`       | 0.10    | DOI/arXiv format checks                  |


# Project Directory Structure

MANZOOR-AHMED-B01798278/
│
├── src/                        # Main source code
│   ├── main.py                 # Runs demo examples
│   ├── pipeline.py             # Core logic (connects all modules)
│   ├── config.py               # Settings (weights, thresholds)
│   ├── evaluation.py           # Runs full dataset evaluation
│   ├── grid_search.py          # Finds best weights
│   ├── build_dataset.py        # Builds dataset
│   │
│   └── modules/                # Individual detection modules
│       ├── temporal_checker.py        # Checks time/date errors
│       ├── entity_checker.py          # Checks facts (countries, people)
│       ├── contradiction_checker.py   # Detects logical conflicts
│       ├── numerical_checker.py       # Checks numbers
│       ├── citation_checker.py        # Validates references
│       └── knowledge_checker.py       # Compares with context
│
├── data/                       # Dataset folder
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned dataset (used for testing)
│
├── results/                    # Output files
│   ├── metrics.json            # Final performance scores
│   ├── predictions.csv         # Predictions for each sample
│   ├── errors.csv              # False positives & negatives
│   └── grid_search_results.json # Best weights
│
├── README.md                   # Project documentation
└── requirements.txt            # Required libraries