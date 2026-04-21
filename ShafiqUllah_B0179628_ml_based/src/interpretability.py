# src/interpretability.py


import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from models import load_model
from feature_pipeline import FeaturePipeline


# Feature group prefixes (for grouped analysis)
FEATURE_GROUPS = {
    "Lexical":    "lex_",
    "Semantic":   "sem_",
    "Linguistic": "ling_",
    "Confidence": "conf_",
    "TF-IDF":     "tfidf_",
}


def get_feature_importances(model, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
    return {}


def group_importances(
    importances: Dict[str, float],
    feature_groups: Dict[str, str] = FEATURE_GROUPS,
) -> Dict[str, float]:
    """Aggregate importances by feature group."""
    group_totals = defaultdict(float)
    group_counts = defaultdict(int)

    for feat_name, imp in importances.items():
        assigned = False
        for group, prefix in feature_groups.items():
            if feat_name.startswith(prefix):
                group_totals[group] += imp
                group_counts[group] += 1
                assigned = True
                break
        if not assigned:
            group_totals["Other"] += imp
            group_counts["Other"] += 1

    return dict(group_totals)


def print_top_features(importances: Dict[str, float], top_n: int = 20):
    """Print top N most important features."""
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} Features by Importance:")
    print(f"  {'Feature':<45} {'Importance':>12}")
    print(f"  {'-'*45} {'-'*12}")
    for name, imp in sorted_feats[:top_n]:
        print(f"  {name:<45} {imp:>12.6f}")


def print_group_importances(group_imps: Dict[str, float]):
    
    total = sum(group_imps.values()) or 1.0
    sorted_groups = sorted(group_imps.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Group Contributions:")
    print(f"  {'Group':<15} {'Total Importance':>18} {'% of Total':>12}")
    print(f"  {'-'*15} {'-'*18} {'-'*12}")
    for group, imp in sorted_groups:
        pct = 100.0 * imp / total
        print(f"  {group:<15} {imp:>18.6f} {pct:>11.1f}%")


def run_shap_analysis(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_dir: str = "../results",
    max_samples: int = 500,
):
    
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Skipping SHAP analysis.")
        print("  Install with: pip install shap")
        return

    print(f"\n  Running SHAP analysis on {min(len(X_sample), max_samples)} samples...")

    X_shap = X_sample[:max_samples]

    model_type = type(model).__name__
    if "RandomForest" in model_type or "XGB" in model_type or "LGBM" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        
        background = shap.kmeans(X_shap, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_shap[:100])[:, :, 1]

    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importances = {
        name: float(val) for name, val in zip(feature_names, mean_abs_shap)
    }

    
    out_path = Path(output_dir) / "shap_importances.json"
    with open(out_path, "w") as f:
        json.dump(
            dict(sorted(shap_importances.items(), key=lambda x: x[1], reverse=True)),
            f, indent=2
        )
    print(f"  SHAP importances saved → {out_path}")

    
    print_top_features(shap_importances, top_n=20)

    
    group_imps = group_importances(shap_importances)
    print_group_importances(group_imps)

    return shap_importances


def analyze(
    model_path: str = "../models/best_model.pkl",
    pipeline_path: str = "../models/feature_pipeline.pkl",
    dataset_path: str = "../data/processed/combined_dataset.json",
    top_n: int = 20,
    run_shap: bool = False,
    output_dir: str = "../results",
):
    """Run full interpretability analysis."""
    import json as _json

    print("=" * 60)
    print("  FEATURE IMPORTANCE & INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    
    print(f"\n  Loading model: {model_path}")
    model = load_model(model_path)
    print(f"  Loading feature pipeline: {pipeline_path}")
    fp = FeaturePipeline.load(pipeline_path)

    feature_names = fp.get_feature_names()
    print(f"  Total features: {len(feature_names)}")

    # Tree-based feature importances
    print("\n  Extracting tree-based feature importances...")
    importances = get_feature_importances(model, feature_names)

    if importances:
        print_top_features(importances, top_n=top_n)
        group_imps = group_importances(importances)
        print_group_importances(group_imps)

        # Save
        out_path = Path(output_dir) / "feature_importances.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            _json.dump(
                dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)),
                f, indent=2
            )
        print(f"\n  Importances saved → {out_path}")
    else:
        print("  Model does not expose feature_importances_ (e.g., SVM).")
        print("  Use --shap for model-agnostic explanations.")


    if run_shap:
        print("\n  Loading dataset sample for SHAP...")
        with open(dataset_path, "r") as f:
            data = _json.load(f)
        sample = data[:1000]
        X_sample = fp.transform(sample)
        run_shap_analysis(model, X_sample, feature_names, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../models/best_model.pkl")
    parser.add_argument("--pipeline", default="../models/feature_pipeline.pkl")
    parser.add_argument("--dataset", default="../data/processed/combined_dataset.json")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    parser.add_argument("--output-dir", default="../results")
    args = parser.parse_args()

    analyze(
        model_path=args.model,
        pipeline_path=args.pipeline,
        dataset_path=args.dataset,
        top_n=args.top_n,
        run_shap=args.shap,
        output_dir=args.output_dir,
    )
