# src/grid_search.py

import sys
import json
import itertools
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


WEIGHT_STEPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
MODULE_KEYS = [
    "knowledge",
    "entity",
    "temporal",
    "contradiction",
    "numerical",
    "citation",
]
SUM_TOLERANCE = 0.01


def _valid_combos(steps: List[float], n: int, target: float = 1.0) -> List[Tuple[float, ...]]:
    return [
        combo
        for combo in itertools.product(steps, repeat=n)
        if abs(sum(combo) - target) <= SUM_TOLERANCE
    ]


def _predict_with_weights(
    module_scores: List[Dict[str, float]],
    weights: Dict[str, float],
    threshold: float,
) -> List[int]:
    preds = []
    for ms in module_scores:
        weighted = sum(ms.get(k, 0.0) * weights.get(k, 0.0) for k in MODULE_KEYS)
        preds.append(1 if weighted >= threshold else 0)
    return preds


def _tune_threshold(
    module_scores: List[Dict[str, float]],
    y_true: List[int],
    weights: Dict[str, float],
    thresholds: List[float],
) -> Tuple[float, float]:
    best_thresh = thresholds[0]
    best_f1 = -1.0

    for t in thresholds:
        preds = _predict_with_weights(module_scores, weights, t)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def _build_model_input(row: Dict) -> str:
    text = str(row.get("text", "")).strip()
    question = str(row.get("question", "")).strip()
    knowledge = str(row.get("knowledge", "")).strip()

    parts = []
    if knowledge:
        parts.append(f"Knowledge: {knowledge}")
    if question:
        parts.append(f"Question: {question}")
    parts.append(f"Answer: {text}")

    return "\n".join(parts)


def run_grid_search(
    dataset_path: str,
    max_samples: int = 400,
    optimize_for: str = "f1",
    threshold_steps: int = 20,
    verbose: bool = True,
) -> Dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)
    pos = [r for r in data if r.get("label") == 1]
    neg = [r for r in data if r.get("label") == 0]

    n_pos = min(len(pos), max_samples // 2)
    n_neg = min(len(neg), max_samples // 2)

    subset = random.sample(pos, n_pos) + random.sample(neg, n_neg)
    random.shuffle(subset)

    y_true = [r["label"] for r in subset]

    if verbose:
        print(f"Grid search on {len(subset)} samples ({n_pos} hallucinated + {n_neg} factual)")
        print(f"Optimising: {optimize_for}")

    if verbose:
        print("Pre-computing module scores...")

    from pipeline import RuleBasedHallucinationPipeline
    pipeline = RuleBasedHallucinationPipeline()

    module_scores: List[Dict[str, float]] = []
    t0 = time.time()

    for i, row in enumerate(subset):
        model_input = _build_model_input(row)
        result = pipeline.predict(model_input)
        ms = {
            k: result["module_results"].get(k, {"score": 0.0}).get("score", 0.0)
            for k in MODULE_KEYS
        }
        module_scores.append(ms)

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(subset)} pre-computed ({time.time() - t0:.1f}s)")

    if verbose:
        print(f"Pre-computation done in {time.time() - t0:.1f}s")

    combos = _valid_combos(WEIGHT_STEPS, len(MODULE_KEYS))
    thresholds = [round(0.01 + i * (0.19 / threshold_steps), 3) for i in range(threshold_steps + 1)]

    if verbose:
        print(
            f"Searching {len(combos)} weight combinations × {len(thresholds)} thresholds = "
            f"{len(combos) * len(thresholds):,} evaluations..."
        )

    metric_fns = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    }
    metric_fn = metric_fns.get(optimize_for, metric_fns["f1"])

    best_score = -1.0
    best_weights = None
    best_threshold = None
    results = []

    t1 = time.time()

    for combo in combos:
        weights = dict(zip(MODULE_KEYS, combo))
        tuned_threshold, _ = _tune_threshold(module_scores, y_true, weights, thresholds)
        preds = _predict_with_weights(module_scores, weights, tuned_threshold)
        score = metric_fn(y_true, preds)

        row = {
            "weights": weights,
            "threshold": tuned_threshold,
            "score": round(score, 4),
        }
        results.append(row)

        if score > best_score:
            best_score = score
            best_weights = weights
            best_threshold = tuned_threshold

    elapsed = time.time() - t1
    results.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print(f"\nGrid search complete in {elapsed:.1f}s")
        print("=" * 50)
        print(f"Best {optimize_for}: {best_score:.4f}")
        print(f"Best threshold: {best_threshold}")
        print("Best weights:")
        for k, v in best_weights.items():
            print(f"  {k:<15}: {v}")

        print("\nTop 5 configurations:")
        for r in results[:5]:
            weights_str = "  ".join(f"{k}={v}" for k, v in r["weights"].items())
            print(f"  {optimize_for}={r['score']:.4f}  thresh={r['threshold']}  {weights_str}")

    return {
        "best_weights": best_weights,
        "best_threshold": best_threshold,
        "best_score": round(best_score, 4),
        "optimize_for": optimize_for,
        "n_samples": len(subset),
        "n_combinations": len(combos),
        "elapsed_s": round(elapsed, 2),
        "top_results": results[:10],
    }


def apply_best_weights(grid_result: Dict) -> None:
    w = grid_result["best_weights"]
    t = grid_result["best_threshold"]

    print("\n# Apply these to src/config.py")
    print("@dataclass")
    print("class Weights:")
    for k, v in w.items():
        print(f"    {k:<15}: float = {v}")
    print()
    print("@dataclass")
    print("class Thresholds:")
    print(f"    final_label_threshold: float = {t}")


if __name__ == "__main__":
    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../data/processed/combined_dataset.json"
    )

    result = run_grid_search(
        dataset_path=dataset,
        max_samples=3000,
        optimize_for="f1",
        threshold_steps=50,
        verbose=True,
    )

    apply_best_weights(result)

    out = Path("../results/grid_search_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nFull results saved to {out}")