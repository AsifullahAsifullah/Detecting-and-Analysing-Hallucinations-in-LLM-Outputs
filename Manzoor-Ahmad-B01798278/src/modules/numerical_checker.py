# src\modules\numerical_checker.py
import re
from typing import Dict, Any, List, Tuple
from config import SETTINGS


def _get_range(key: str) -> Tuple[float, float]:
    return SETTINGS.plausible_ranges[key]


def check_numerical(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    evidence: List[str] = []
    reason_parts: List[str] = []
    score = 0.0

    for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
        value = float(p)
        low, high = _get_range("percentage")
        if value < low or value > high:
            score += 1.0
            evidence.append(f"{value}%")
            reason_parts.append(f"Invalid percentage value: {value}%.")

    for age in re.findall(r"(\d{1,4})\s+years?\s+old", text_lower):
        value = int(age)
        low, high = _get_range("human_age")
        if value < low or value > high:
            score += 0.9
            evidence.append(f"{value} years old")
            reason_parts.append(f"Implausible human age: {value} years old.")

    if "population" in text_lower or "people" in text_lower:
        for val in re.findall(r"(\d+(?:\.\d+)?)\s+billion", text_lower):
            value = float(val)
            low, high = _get_range("earth_population_billion")
            if value < low or value > high:
                score += 0.9
                evidence.append(f"{value} billion")
                reason_parts.append(f"Implausible population figure: {value} billion.")

    if "human" in text_lower or "person" in text_lower or "patient" in text_lower:
        for val in re.findall(r"(\d+(?:\.\d+)?)\s*kg", text_lower):
            value = float(val)
            low, high = _get_range("human_weight_kg")
            if value < low or value > high:
                score += 0.9
                evidence.append(f"{value} kg")
                reason_parts.append(f"Implausible human weight: {value} kg.")

    kg_vals = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*kg", text_lower)]
    lb_vals = [float(m) for m in re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pounds?)", text_lower)]
    if kg_vals and lb_vals:
        expected_lb = kg_vals[0] * 2.205
        ratio = lb_vals[0] / (expected_lb + 1e-9)
        if not (0.8 < ratio < 1.2):
            score += 0.65
            evidence += [f"{kg_vals[0]}kg", f"{lb_vals[0]}lb"]
            reason_parts.append(
                f"Unit inconsistency: {kg_vals[0]} kg ≠ {lb_vals[0]} lbs "
                f"(conversion ratio off by {abs(1 - ratio) * 100:.0f}%)."
            )

    for val in re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?celsius", text_lower):
        value = float(val)
        low, high = _get_range("temperature_c")
        if value < low or value > high:
            score += 0.8
            evidence.append(f"{value}°C")
            reason_parts.append(
                f"Implausible Celsius temperature for everyday context: {value}°C "
                f"(expected range: {low}–{high}°C)."
            )

    flag = 1 if score > 0 else 0
    return {
        "module":   "numerical",
        "flag":     flag,
        "score":    min(round(score, 4), 1.0),
        "reason":   " | ".join(reason_parts) if reason_parts else "No numerical issue detected.",
        "evidence": evidence,
    }
