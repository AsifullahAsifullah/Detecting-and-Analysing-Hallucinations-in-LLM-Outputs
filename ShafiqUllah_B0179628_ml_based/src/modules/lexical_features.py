# src/modules/lexical_features.py
"""
Lexical Feature Extractor
=========================
Extracts surface-level textual features from LLM outputs:
- TF-IDF n-gram representations
- Word / sentence count statistics
- Vocabulary richness (type-token ratio)
- Punctuation and special character density
- Hedge word and certainty word frequencies
"""

import re
import math
from typing import Dict, List

from utils import normalize_text, safe_divide, split_sentences


# ---------------------------------------------------------------------------
# Hedge / certainty word lists (from config via caller, or defaults here)
# ---------------------------------------------------------------------------
DEFAULT_HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "probably", "likely", "might",
    "could", "seems", "appears", "suggest", "believe", "think",
    "approximately", "around", "about", "roughly", "unclear",
    "uncertain", "unknown", "allegedly", "reportedly",
]

DEFAULT_CERTAINTY_WORDS = [
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "undoubtedly", "always", "never", "every", "all", "none",
    "guaranteed", "proven", "confirmed", "established",
]


def extract_lexical_features(
    text: str,
    hedge_words: List[str] = None,
    certainty_words: List[str] = None,
) -> Dict[str, float]:
    """
    Extract lexical features from a single text sample.

    Returns a flat dict of feature_name -> float value.
    All values are normalized to [0, 1] where possible.
    """
    hedge_words = hedge_words or DEFAULT_HEDGE_WORDS
    certainty_words = certainty_words or DEFAULT_CERTAINTY_WORDS

    text = normalize_text(text)
    if not text:
        return _empty_lexical_features()

    tokens = re.findall(r"\b\w+\b", text.lower())
    sentences = split_sentences(text)

    n_tokens = len(tokens)
    n_sentences = max(len(sentences), 1)
    n_chars = len(text)

    # --- Basic counts ---
    word_count = n_tokens
    sentence_count = n_sentences
    avg_word_length = safe_divide(sum(len(t) for t in tokens), n_tokens)
    avg_sentence_length = safe_divide(n_tokens, n_sentences)

    # --- Vocabulary richness (type-token ratio) ---
    unique_tokens = set(tokens)
    ttr = safe_divide(len(unique_tokens), n_tokens)

    # --- Punctuation density ---
    punctuation_chars = re.findall(r"[^\w\s]", text)
    punctuation_density = safe_divide(len(punctuation_chars), n_chars)

    # --- Special characters ---
    special_chars = re.findall(r"[^a-zA-Z0-9\s.,!?;:'\"-]", text)
    special_char_density = safe_divide(len(special_chars), n_chars)

    # --- Hedge words ---
    hedge_count = sum(1 for t in tokens if t in hedge_words)
    hedge_ratio = safe_divide(hedge_count, n_tokens)

    # --- Certainty words ---
    certainty_count = sum(1 for t in tokens if t in certainty_words)
    certainty_ratio = safe_divide(certainty_count, n_tokens)

    # --- Function word ratio (approximate) ---
    function_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "must", "can", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "up", "about", "into",
        "through", "during", "before", "after", "above", "below", "and",
        "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "no", "it", "its", "this", "that", "these", "those",
        "he", "she", "they", "we", "you", "i", "me", "him", "her", "us",
        "them", "my", "your", "his", "our", "their",
    }
    function_word_count = sum(1 for t in tokens if t in function_words)
    function_word_ratio = safe_divide(function_word_count, n_tokens)

    # --- Digit ratio ---
    digits = re.findall(r"\d", text)
    digit_ratio = safe_divide(len(digits), n_chars)

    # --- Uppercase ratio (ALL-CAPS words) ---
    upper_words = [t for t in re.findall(r"\b[A-Z]{2,}\b", text)]
    upper_word_ratio = safe_divide(len(upper_words), n_tokens)

    # --- Response length (log-normalised) ---
    log_length = math.log1p(n_tokens) / 10.0  # normalised loosely

    # --- Repetition ratio (duplicate tokens / total) ---
    from collections import Counter
    token_counts = Counter(tokens)
    repeated_tokens = sum(1 for cnt in token_counts.values() if cnt > 1)
    repetition_ratio = safe_divide(repeated_tokens, len(unique_tokens))

    return {
        "lex_word_count": min(word_count / 200.0, 1.0),
        "lex_sentence_count": min(sentence_count / 20.0, 1.0),
        "lex_avg_word_length": min(avg_word_length / 10.0, 1.0),
        "lex_avg_sentence_length": min(avg_sentence_length / 40.0, 1.0),
        "lex_ttr": ttr,
        "lex_punctuation_density": min(punctuation_density * 10, 1.0),
        "lex_special_char_density": min(special_char_density * 20, 1.0),
        "lex_hedge_ratio": min(hedge_ratio * 10, 1.0),
        "lex_certainty_ratio": min(certainty_ratio * 10, 1.0),
        "lex_function_word_ratio": function_word_ratio,
        "lex_digit_ratio": min(digit_ratio * 10, 1.0),
        "lex_upper_word_ratio": min(upper_word_ratio * 5, 1.0),
        "lex_log_length": min(log_length, 1.0),
        "lex_repetition_ratio": repetition_ratio,
        "lex_hedge_count": min(hedge_count / 5.0, 1.0),
        "lex_certainty_count": min(certainty_count / 5.0, 1.0),
    }


def _empty_lexical_features() -> Dict[str, float]:
    return {
        "lex_word_count": 0.0,
        "lex_sentence_count": 0.0,
        "lex_avg_word_length": 0.0,
        "lex_avg_sentence_length": 0.0,
        "lex_ttr": 0.0,
        "lex_punctuation_density": 0.0,
        "lex_special_char_density": 0.0,
        "lex_hedge_ratio": 0.0,
        "lex_certainty_ratio": 0.0,
        "lex_function_word_ratio": 0.0,
        "lex_digit_ratio": 0.0,
        "lex_upper_word_ratio": 0.0,
        "lex_log_length": 0.0,
        "lex_repetition_ratio": 0.0,
        "lex_hedge_count": 0.0,
        "lex_certainty_count": 0.0,
    }
