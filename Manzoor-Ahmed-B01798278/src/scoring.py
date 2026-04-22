# src/scoring.py
from typing import Dict, Any
from config import SETTINGS


def aggregate_module_scores(module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    weights = SETTINGS.weights

    def get_score(module: str) -> float:
        try:
            score = float(module_results.get(module, {}).get("score", 0.0))
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.0

    knowledge = get_score("knowledge")
    entity = get_score("entity")
    temporal = get_score("temporal")
    contradiction = get_score("contradiction")
    numerical = get_score("numerical")
    citation = get_score("citation")

    
    weighted_score = (
        knowledge * weights.knowledge
        + entity * weights.entity
        + temporal * weights.temporal
        + contradiction * weights.contradiction
        + numerical * weights.numerical
        + citation * weights.citation
    )

    
    total_weight = (
        weights.knowledge
        + weights.entity
        + weights.temporal
        + weights.contradiction
        + weights.numerical
        + weights.citation
    )

    if total_weight > 0:
        weighted_score = weighted_score / total_weight

    weighted_score = max(0.0, min(1.0, weighted_score))

    
    threshold = SETTINGS.thresholds.final_label_threshold

    if contradiction >= 0.60:
        label = 1
    elif entity >= 0.60:
        label = 1
    elif knowledge >= 0.60:
        label = 1
    elif numerical >= 0.70:
        label = 1

    elif (
        (knowledge >= 0.40 and entity >= 0.40)
        or (knowledge >= 0.40 and contradiction >= 0.35)
        or (entity >= 0.40 and contradiction >= 0.35)
        or (knowledge >= 0.45 and numerical >= 0.35)
    ):
        label = 1
    else:
        label = 1 if weighted_score >= threshold else 0

    
    triggered = []
    module_order = ["knowledge", "entity", "temporal", "contradiction", "numerical", "citation"]

    for name in module_order:
        result = module_results.get(name, {})
        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        flag = result.get("flag", None)
        label_flag = result.get("label", 1 if score > 0 else 0)

        is_triggered = (
            flag == 1
            or label_flag == 1
            or score >= 0.30
        )

        if is_triggered:
            triggered.append({
                "module": name,
                "reason": result.get("reason", ""),
                "evidence": result.get("evidence", []),
                "score": round(score, 4),
            })

    triggered.sort(key=lambda x: x["score"], reverse=True)


    if label == 1:
        if triggered:
            top_modules = ", ".join(t["module"] for t in triggered[:3])
            explanation = f"Hallucination likely detected due to signals from: {top_modules}."
        else:
            explanation = "Hallucination likely detected from ensemble score."
    else:
        explanation = "No strong hallucination signal detected."

    return {
        "final_score": round(weighted_score, 4),
        "label": label,
        "triggered_modules": triggered,
        "explanation": explanation,
    }