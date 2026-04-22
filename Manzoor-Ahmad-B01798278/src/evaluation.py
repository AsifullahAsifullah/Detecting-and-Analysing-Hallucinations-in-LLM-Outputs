# src/evaluation.py
import sys
import json
import time
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from pipeline import RuleBasedHallucinationPipeline


def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(dataset_path: str) -> dict:
    data = load_dataset(dataset_path)
    pipeline = RuleBasedHallucinationPipeline()

    y_true, y_pred, y_scores = [], [], []
    times, error_types = [], []
    triggered_modules_all = []

    print(f"Evaluating on {len(data)} samples from {dataset_path}...")
    t_global = time.time()

    for i, row in enumerate(data):
        t0 = time.time()

        text = str(row.get("text", "")).strip()
        question = str(row.get("question", "")).strip()
        knowledge = str(row.get("knowledge", "")).strip()

        parts = []
        if knowledge:
            parts.append(f"Knowledge: {knowledge}")
        if question:
            parts.append(f"Question: {question}")
        parts.append(f"Answer: {text}")

        model_input = "\n".join(parts)

        result = pipeline.predict(model_input)
        elapsed = time.time() - t0

        pred = result["final_result"]["label"]
        score = result["final_result"]["final_score"]
        true_label = row["label"]

        y_true.append(true_label)
        y_pred.append(pred)
        y_scores.append(score)
        times.append(elapsed)

        triggered_modules_all.append(
            [t["module"] for t in result["final_result"]["triggered_modules"]]
        )

        if true_label == 1 and pred == 1:
            et = "TP"
        elif true_label == 0 and pred == 0:
            et = "TN"
        elif true_label == 0 and pred == 1:
            et = "FP"
        else:
            et = "FN"
        error_types.append(et)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(data)} done  ({time.time() - t_global:.1f}s elapsed)")

    roc = None
    if len(set(y_true)) > 1:
        roc = round(roc_auc_score(y_true, y_scores), 4)

    metrics = {
        "total_samples": len(data),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": roc,
        "avg_inference_ms": round(sum(times) / len(times) * 1000, 2),
        "total_time_s": round(time.time() - t_global, 2),
    }

    ec = Counter(error_types)
    metrics["TP"] = ec["TP"]
    metrics["TN"] = ec["TN"]
    metrics["FP"] = ec["FP"]
    metrics["FN"] = ec["FN"]

    per_source: dict = {}
    for row, et in zip(data, error_types):
        src = row.get("source", "unknown")
        per_source.setdefault(src, {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
        per_source[src][et] += 1
    metrics["per_source"] = per_source

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df["prediction"] = y_pred
    df["score"] = y_scores
    df["error_type"] = error_types
    df["triggered_modules"] = [", ".join(m) if m else "" for m in triggered_modules_all]

    df.to_csv(results_dir / "predictions.csv", index=False)
    df[df["error_type"].isin(["FP", "FN"])].to_csv(results_dir / "errors.csv", index=False)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 55}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 55}")
    print(f"  Samples   : {metrics['total_samples']}")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1_score']}")
    print(f"  ROC-AUC   : {metrics['roc_auc']}")
    print(f"  Avg ms    : {metrics['avg_inference_ms']}")
    print(f"\n  TP:{ec['TP']}  TN:{ec['TN']}  FP:{ec['FP']}  FN:{ec['FN']}")
    print("\n  Per-source breakdown:")
    for src, counts in per_source.items():
        total = sum(counts.values())
        correct = counts["TP"] + counts["TN"]
        print(
            f"    {src:<14}: acc={correct / total:.3f}  "
            f"TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}"
        )
    print("\n  Saved → results/metrics.json")
    print("         → results/predictions.csv")
    print("         → results/errors.csv")

    return metrics


if __name__ == "__main__":
    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/processed/combined_dataset.json"
    )
    run_evaluation(dataset)