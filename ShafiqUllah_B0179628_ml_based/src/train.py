# src/train.py


import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from feature_pipeline import FeaturePipeline
from models import build_model, save_model, get_probabilities
from trainer import cross_validate, tune_hyperparameters
from pipeline import MLHallucinationPipeline
from config import SETTINGS


def load_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_dataset(data: list, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """Split data into train / val / test with stratification."""
    labels = [r["label"] for r in data]

    # First split off test set
    train_val, test = train_test_split(
        data, test_size=test_size,
        stratify=labels, random_state=random_state
    )
    # Then split val from train
    train_val_labels = [r["label"] for r in train_val]
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=adjusted_val,
        stratify=train_val_labels, random_state=random_state
    )
    return train, val, test


def print_split_info(train, val, test):
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        labels = [r["label"] for r in split]
        c = Counter(labels)
        print(f"  {name}: {len(split)} samples  (0={c[0]}, 1={c[1]})")


def evaluate_on_split(model, X, y, split_name="Split"):
    """Quick evaluation of a model on a data split."""
    y_pred = model.predict(X)
    y_prob = get_probabilities(model, X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc = roc_auc_score(y, y_prob) if len(set(y)) > 1 else None
    roc_str = f"{roc:.4f}" if roc is not None else "N/A"
    print(f"  [{split_name}] Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
          f"F1={f1:.4f}  ROC-AUC={roc_str}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}


def main():
    parser = argparse.ArgumentParser(description="Train ML hallucination detector")
    parser.add_argument("--dataset", default="../data/processed/combined_dataset.json")
    parser.add_argument("--models", nargs="+",
                        default=["random_forest", "xgboost", "lightgbm"],
                        choices=["random_forest", "xgboost", "lightgbm", "svm"])
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-tfidf", action="store_true", help="Disable TF-IDF features")
    parser.add_argument("--tfidf-features", type=int, default=3000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default="../models")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML-Based Hallucination Detector")
    print("  Training Pipeline")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1] Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"  Total samples: {len(data)}")

    # 2. Split
    print("\n[2] Splitting dataset...")
    train_data, val_data, test_data = split_dataset(
        data, test_size=args.test_size,
        val_size=0.1, random_state=42
    )
    print_split_info(train_data, val_data, test_data)

    # 3. Feature extraction
    print("\n[3] Extracting features...")
    tfidf_features = 0 if args.no_tfidf else args.tfidf_features
    feature_pipeline = FeaturePipeline(
        tfidf_max_features=tfidf_features,
        tfidf_ngram_range=(1, 2),
    )

    t_feat = time.time()
    X_train = feature_pipeline.fit_transform(train_data)
    X_val = feature_pipeline.transform(val_data)
    X_test = feature_pipeline.transform(test_data)
    print(f"  Feature extraction time: {time.time() - t_feat:.1f}s")

    y_train = np.array([r["label"] for r in train_data])
    y_val = np.array([r["label"] for r in val_data])
    y_test = np.array([r["label"] for r in test_data])

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    trained_models = {}

    # 4. Train each model
    for model_name in args.models:
        print(f"\n[4] {model_name.upper()}")
        print("-" * 40)

        best_params = None

        # Optional: Cross-validation first
        print(f"  Running {args.cv_folds}-fold cross-validation...")
        cv_results = cross_validate(
            model_name, X_train, y_train,
            n_folds=args.cv_folds, random_state=42
        )
        print(f"  CV F1: {cv_results.get('f1_mean', 0):.4f} ± {cv_results.get('f1_std', 0):.4f}")
        print(f"  CV ROC-AUC: {cv_results.get('roc_auc_mean', 0):.4f}")

        # Optional: Optuna tuning
        if args.tune:
            print(f"  Tuning hyperparameters ({args.n_trials} trials)...")
            best_params = tune_hyperparameters(
                model_name, X_train, y_train,
                n_trials=args.n_trials, n_folds=3, random_state=42
            )

        # Train final model
        print(f"  Training final model...")
        t0 = time.time()
        model = build_model(model_name, best_params)
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        val_metrics = evaluate_on_split(model, X_val, y_val, "Validation")
        test_metrics = evaluate_on_split(model, X_test, y_test, "Test")

        # Save model
        model_path = models_dir / f"{model_name}.pkl"
        save_model(model, str(model_path))

        trained_models[model_name] = {
            "model": model,
            "val_f1": val_metrics["f1"],
            "test_metrics": test_metrics,
            "cv_results": cv_results,
            "train_time": train_time,
            "params": best_params or {},
        }
        all_results[model_name] = {
            "cv": cv_results,
            "val": val_metrics,
            "test": test_metrics,
            "train_time_s": train_time,
        }

    # 5. Select best model
    print("\n[5] Model Comparison")
    print("-" * 40)
    print(f"  {'Model':<20} {'Val F1':>10} {'Test F1':>10} {'Test AUC':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, info in trained_models.items():
        tf1 = info["test_metrics"]["f1"]
        tauc = info["test_metrics"]["roc_auc"] or 0.0
        print(f"  {name:<20} {info['val_f1']:>10.4f} {tf1:>10.4f} {tauc:>10.4f}")

    best_name = max(trained_models, key=lambda n: trained_models[n]["val_f1"])
    best_model = trained_models[best_name]["model"]
    print(f"\n  Best model: {best_name} (Val F1={trained_models[best_name]['val_f1']:.4f})")

    # 6. Save best model + feature pipeline
    print("\n[6] Saving best model and feature pipeline...")
    best_model_path = str(models_dir / "best_model.pkl")
    pipeline_path = str(models_dir / "feature_pipeline.pkl")
    save_model(best_model, best_model_path)
    feature_pipeline.save(pipeline_path)

    # 7. Save all results
    with open(results_dir / "training_results.json", "w") as f:
        # Remove non-serializable model objects
        serializable = {}
        for name, info in all_results.items():
            serializable[name] = info
        serializable["best_model"] = best_name
        json.dump(serializable, f, indent=2)
    print(f"  Training results saved → results/training_results.json")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model: {best_name}")
    print(f"  Model saved: {best_model_path}")
    print(f"  Feature pipeline: {pipeline_path}")
    print("=" * 60)
    print("\nNext steps:")
    print("  python evaluation.py  — run full evaluation on test set")
    print("  python main.py        — run demo predictions")


if __name__ == "__main__":
    main()
