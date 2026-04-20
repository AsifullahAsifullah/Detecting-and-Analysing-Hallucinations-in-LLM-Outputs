# Asif Ullah B01797967 — Deep Learning Hallucination Detector

**MSc Project: Detecting and Analysing Hallucinations in Large Language Model Outputs**  
University of the West of Scotland — School of Computing, Engineering and Physical Sciences

---

## Overview

This is my **individual contribution** of to the group's comparative hallucination
detection study. It implements a transformer-based hallucination detector by fine-tuning
BERT, RoBERTa, and DeBERTa on the shared group dataset.

---

## Project Structure

```
AsifUllah_B01797967/
├── src/
│   ├── main.py                  # Demo runner
│   ├── train.py                 # Fine-tuning pipeline (all architectures)
│   ├── evaluation.py            # Evaluation on labelled dataset
│   ├── attention_analysis.py    # Attention-based interpretability
│   ├── predict_file.py          # Interactive prediction tool
│   ├── build_dataset.py         # Rebuild dataset from raw sources
│   ├── pipeline.py              # DeepLearningHallucinationPipeline
│   ├── model.py                 # TransformerClassifier (encoder + head)
│   ├── dataset.py               # PyTorch Dataset + DataLoader
│   ├── trainer.py               # Fine-tuning loop, AdamW, early stopping
│   ├── config.py                # Settings dataclass
│   └── utils.py                 # Text helpers, seed setting
├── data/
│   ├── raw/                     # TruthfulQA, HaluEval, FEVER
│   └── processed/
│       └── combined_dataset.json   # Shared Dataset
├── models/                      # Saved fine-tuned models (.pt)
├── results/                     # Evaluation outputs
├── tests/
│   └── quick_test.py            # Unit tests (no model download needed)
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
cd tests
python3 quick_test.py
```

### 3. Fine-tune BERT (recommended first run)


```bash
cd src
python3 train.py --arch bert
```
```bash
python3 train.py --arch roberta
```

```bash
python3 train.py --arch deberta
```


Fine-tune all three architectures:
```bash
python3 train.py --arch all
python3 train.py --arch all --max-samples 50 --epochs 2
```

Quick test on small subset:
```bash
python3 train.py --arch bert --max-samples 50 --epochs 2
python3 train.py --arch roberta --max-samples 50 --epochs 2
python3 train.py --arch deberta --max-samples 50 --epochs 2
```

### 4. Demo predictions

```bash
cd src
python3 main.py
python3 main.py --arch roberta --model-path ../models/roberta_best.pt # for specific model
```

### 5. Interactive prediction tool

```bash
cd src
python3 predict_file.py
python3 predict_file.py --model-path ../models/bert_best.pt --model-name bert-base-uncased # for specific model


```

### 6. Full evaluation

```bash
cd src
python3 evaluation.py
python3 evaluation.py --model-path ../models/bert_best.pt --model-name bert-base-uncased # for specific model

```

### 7. Attention interpretability analysis

```bash
cd src
python3 attention_analysis.py
python3 attention_analysis.py --arch bert --n-samples 20 # for specific model with some samples

```

---

## Architectures

| Model   | HuggingFace ID            | Size | Notes                             |
| ------- | ------------------------- | ---- | --------------------------------- |
| BERT    | bert-base-uncased         | 110M | Bidirectional contextual encoding |
| RoBERTa | roberta-base              | 125M | Improved BERT training            |
| DeBERTa | microsoft/deberta-v3-base | 86M  | Disentangled attention            |

Best architecture is selected by validation F1 and saved to `models/<arch>_best.pt`.

---

## Training Details

- **Optimiser:** AdamW with weight decay (0.01)
- **Schedule:** Linear warmup (10%) + linear decay
- **Loss:** Binary cross-entropy
- **Early stopping:** patience=2 on validation loss
- **Splits:** 60% train / 20% val / 20% test (stratified, same as S1 & S2)
- **Max sequence length:** 256 tokens

---


## References

- Devlin et al. (2019) BERT
- Liu et al. (2019) RoBERTa
- He et al. (2021) DeBERTa
- Chen et al. (2023) Hallucination detection with DeBERTa
- Vaswani et al. (2017) Attention is all you need
