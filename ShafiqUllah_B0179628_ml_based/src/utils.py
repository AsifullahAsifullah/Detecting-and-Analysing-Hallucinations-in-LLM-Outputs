# src/utils.py
import re
from typing import List


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division returning default when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
