# src/modules/confidence_features.py
"""
Confidence & Certainty Feature Extractor
=========================================
Extracts proxy features for model confidence and uncertainty:
- Perplexity proxy (via unigram log-probability estimate)
- Hedge word frequency
- Certainty word frequency
- Vague quantifier frequency
- Response length signals
- Overconfidence indicators
"""

import re
import math
from collections import Counter
from typing import Dict, List

from utils import normalize_text, safe_divide


# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "likely", "might",
    "could", "seems", "appears", "suggest", "believe", "think",
    "approximately", "around", "about", "roughly", "unclear",
    "uncertain", "unknown", "allegedly", "reportedly", "supposedly",
    "often", "sometimes", "occasionally", "generally", "usually",
    "typically", "tend", "tends", "tended",
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "always", "never", "every", "all", "none",
    "guaranteed", "proven", "confirmed", "established", "without doubt",
    "unquestionably", "invariably", "necessarily", "must", "will",
}

VAGUE_QUANTIFIERS = {
    "some", "many", "few", "several", "various", "numerous", "most",
    "lots", "plenty", "enough", "little", "much", "more", "less",
    "any", "certain", "specific", "particular",
}

OVERCONFIDENCE_PHRASES = [
    r"\bit is a fact\b",
    r"\bscientifically proven\b",
    r"\beveryone knows\b",
    r"\bwidely known\b",
    r"\bwithout question\b",
    r"\bno doubt\b",
    r"\bundeniably\b",
    r"\bobviously\b",
]

DISCLAIMER_PHRASES = [
    r"\bi('m| am) not sure\b",
    r"\bi('m| am) not certain\b",
    r"\bi don't know\b",
    r"\bit'?s? (possible|unclear|uncertain)\b",
    r"\bmight be wrong\b",
    r"\bcould be incorrect\b",
    r"\bapproximately\b",
]


# ---------------------------------------------------------------------------
# Perplexity proxy
# ---------------------------------------------------------------------------

def _unigram_perplexity_proxy(tokens: List[str]) -> float:
    """
    Proxy for perplexity using unigram entropy.
    Higher entropy (more uniform distribution) = higher perplexity proxy.
    Hallucinated text tends to have either very high or very low token entropy.
    Returns a value in [0, 1].
    """
    if not tokens:
        return 0.5

    counts = Counter(tokens)
    n = len(tokens)
    probs = [c / n for c in counts.values()]
    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    # Max entropy for n unique tokens = log2(n)
    max_entropy = math.log2(max(len(counts), 2))
    normalised = safe_divide(entropy, max_entropy)
    return min(normalised, 1.0)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_confidence_features(text: str) -> Dict[str, float]:
    """
    Extract confidence-related features from a single text.

    Returns a flat dict of feature_name -> float.
    """
    text = normalize_text(text)
    if not text:
        return _empty_confidence_features()

    tokens = re.findall(r"\b\w+\b", text.lower())
    n_tokens = max(len(tokens), 1)

    # --- Hedge words ---
    hedge_count = sum(1 for t in tokens if t in HEDGE_WORDS)
    hedge_ratio = safe_divide(hedge_count, n_tokens)

    # --- Certainty words ---
    certainty_count = sum(1 for t in tokens if t in CERTAINTY_WORDS)
    certainty_ratio = safe_divide(certainty_count, n_tokens)

    # --- Vague quantifiers ---
    vague_count = sum(1 for t in tokens if t in VAGUE_QUANTIFIERS)
    vague_ratio = safe_divide(vague_count, n_tokens)

    # --- Overconfidence phrase count ---
    overconf_count = sum(
        1 for pat in OVERCONFIDENCE_PHRASES
        if re.search(pat, text, re.IGNORECASE)
    )
    overconf_ratio = min(overconf_count / 3.0, 1.0)

    # --- Disclaimer phrase count ---
    disclaimer_count = sum(
        1 for pat in DISCLAIMER_PHRASES
        if re.search(pat, text, re.IGNORECASE)
    )
    disclaimer_ratio = min(disclaimer_count / 3.0, 1.0)

    # --- Perplexity proxy ---
    perplexity_proxy = _unigram_perplexity_proxy(tokens)

    # --- Uncertainty signal: high hedge + low certainty ---
    uncertainty_signal = min(hedge_ratio * 5 + vague_ratio * 2, 1.0)

    # --- Overconfidence signal ---
    overconfidence_signal = min(certainty_ratio * 5 + overconf_ratio, 1.0)

    # --- Response length normalised ---
    response_length_norm = min(math.log1p(n_tokens) / 8.0, 1.0)

    # --- Hedge-certainty balance ---
    # Very high certainty with little supporting content = possible hallucination
    hedge_certainty_ratio = safe_divide(certainty_count, max(hedge_count + certainty_count, 1))

    # --- Token repetition (proxy for circular reasoning) ---
    token_counts = Counter(tokens)
    max_repeat = max(token_counts.values()) if token_counts else 1
    repetition_max_norm = min(max_repeat / 10.0, 1.0)

    return {
        "conf_hedge_ratio": min(hedge_ratio * 10, 1.0),
        "conf_certainty_ratio": min(certainty_ratio * 10, 1.0),
        "conf_vague_ratio": min(vague_ratio * 10, 1.0),
        "conf_overconf_ratio": overconf_ratio,
        "conf_disclaimer_ratio": disclaimer_ratio,
        "conf_perplexity_proxy": perplexity_proxy,
        "conf_uncertainty_signal": uncertainty_signal,
        "conf_overconfidence_signal": overconfidence_signal,
        "conf_response_length_norm": response_length_norm,
        "conf_hedge_certainty_balance": hedge_certainty_ratio,
        "conf_token_repetition_max": repetition_max_norm,
        "conf_hedge_count_norm": min(hedge_count / 5.0, 1.0),
        "conf_certainty_count_norm": min(certainty_count / 5.0, 1.0),
    }


def _empty_confidence_features() -> Dict[str, float]:
    return {
        "conf_hedge_ratio": 0.0,
        "conf_certainty_ratio": 0.0,
        "conf_vague_ratio": 0.0,
        "conf_overconf_ratio": 0.0,
        "conf_disclaimer_ratio": 0.0,
        "conf_perplexity_proxy": 0.0,
        "conf_uncertainty_signal": 0.0,
        "conf_overconfidence_signal": 0.0,
        "conf_response_length_norm": 0.0,
        "conf_hedge_certainty_balance": 0.0,
        "conf_token_repetition_max": 0.0,
        "conf_hedge_count_norm": 0.0,
        "conf_certainty_count_norm": 0.0,
    }
