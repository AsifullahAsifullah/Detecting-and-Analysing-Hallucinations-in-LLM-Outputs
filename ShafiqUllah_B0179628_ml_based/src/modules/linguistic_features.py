# src/modules/linguistic_features.py
"""
Linguistic Feature Extractor
=============================
Extracts syntactic and readability features from LLM outputs:
- POS tag distributions (noun/verb/adj ratios)
- Sentence complexity indicators
- Readability scores (Flesch-Kincaid proxy, Gunning Fog proxy)
- Passive voice frequency
- Syntactic depth proxy

Uses regex-based patterns as a lightweight spaCy fallback.
If spaCy is installed and enabled, richer POS features are used.
"""

import re
import math
from typing import Dict, List

from utils import normalize_text, safe_divide, split_sentences


# ---------------------------------------------------------------------------
# Regex-based POS proxies (no spaCy needed)
# ---------------------------------------------------------------------------

# Common verb endings
VERB_PATTERN = re.compile(
    r"\b(\w+(?:ed|ing|ise|ize|ify|ated|ates|ating))\b", re.IGNORECASE
)
# Common adjective endings
ADJ_PATTERN = re.compile(
    r"\b(\w+(?:ful|less|ous|ive|able|ible|al|ic|ish|ent|ant))\b", re.IGNORECASE
)
# Common adverb endings
ADV_PATTERN = re.compile(r"\b(\w+ly)\b", re.IGNORECASE)
# Passive voice proxy: "was/were/is/are/been + past participle"
PASSIVE_PATTERN = re.compile(
    r"\b(was|were|is|are|been|be|being)\s+\w+ed\b", re.IGNORECASE
)
# Conjunction words
CONJ_PATTERN = re.compile(
    r"\b(and|but|or|nor|so|yet|because|although|while|since|unless|until|"
    r"whereas|however|therefore|moreover|furthermore|consequently)\b", re.IGNORECASE
)


def syllable_count(word: str) -> int:
    """Approximate syllable count for readability scores."""
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 3:
        return 1
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Subtract silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _readability_flesch_kincaid(tokens: List[str], sentences: List[str]) -> float:
    """
    Flesch-Kincaid Grade Level proxy.
    Higher grade = more complex.
    Returns normalised [0, 1].
    """
    n_words = len(tokens)
    n_sentences = max(len(sentences), 1)
    n_syllables = sum(syllable_count(t) for t in tokens)

    if n_words == 0:
        return 0.0

    asl = n_words / n_sentences  # avg sentence length
    asw = n_syllables / n_words   # avg syllables per word

    fk_grade = 0.39 * asl + 11.8 * asw - 15.59
    # Typical range: 0–18 (college grad). Normalise to [0,1].
    return min(max(fk_grade / 18.0, 0.0), 1.0)


def _readability_gunning_fog(tokens: List[str], sentences: List[str]) -> float:
    """
    Gunning Fog Index proxy.
    Returns normalised [0, 1].
    """
    n_words = len(tokens)
    n_sentences = max(len(sentences), 1)
    # Complex words: 3+ syllables
    complex_words = [t for t in tokens if syllable_count(t) >= 3]

    if n_words == 0:
        return 0.0

    fog = 0.4 * (n_words / n_sentences + 100 * len(complex_words) / n_words)
    # Typical range: 6–17. Normalise to [0,1].
    return min(max((fog - 6) / 11.0, 0.0), 1.0)


def extract_linguistic_features(text: str) -> Dict[str, float]:
    """
    Extract linguistic features from a single text.

    Returns a flat dict of feature_name -> float.
    """
    text = normalize_text(text)
    if not text:
        return _empty_linguistic_features()

    tokens = re.findall(r"\b\w+\b", text)
    sentences = split_sentences(text)
    n_tokens = max(len(tokens), 1)
    n_sentences = max(len(sentences), 1)

    # --- POS proxy counts ---
    verb_matches = VERB_PATTERN.findall(text)
    adj_matches = ADJ_PATTERN.findall(text)
    adv_matches = ADV_PATTERN.findall(text)
    conj_matches = CONJ_PATTERN.findall(text)
    passive_matches = PASSIVE_PATTERN.findall(text)

    verb_ratio = safe_divide(len(verb_matches), n_tokens)
    adj_ratio = safe_divide(len(adj_matches), n_tokens)
    adv_ratio = safe_divide(len(adv_matches), n_tokens)
    conj_ratio = safe_divide(len(conj_matches), n_tokens)
    passive_ratio = safe_divide(len(passive_matches), n_sentences)

    # --- Sentence length variance ---
    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    mean_sl = sum(sent_lengths) / n_sentences
    variance_sl = sum((l - mean_sl) ** 2 for l in sent_lengths) / n_sentences
    std_sl = math.sqrt(variance_sl)
    norm_std_sl = min(std_sl / 20.0, 1.0)

    # --- Sentence complexity: words per sentence ---
    avg_sent_length = mean_sl / 40.0  # normalise by typical max

    # --- Readability ---
    fk_grade = _readability_flesch_kincaid(tokens, sentences)
    fog_index = _readability_gunning_fog(tokens, sentences)

    # --- Complex word ratio (3+ syllables) ---
    complex_words = [t for t in tokens if syllable_count(t) >= 3]
    complex_word_ratio = safe_divide(len(complex_words), n_tokens)

    # --- Negation frequency ---
    negation_words = {"not", "no", "never", "neither", "nor", "none", "nobody",
                      "nothing", "nowhere", "hardly", "barely", "scarcely"}
    neg_count = sum(1 for t in tokens if t.lower() in negation_words)
    negation_ratio = min(safe_divide(neg_count, n_tokens) * 10, 1.0)

    # --- Question sentences ---
    q_sentences = [s for s in sentences if s.strip().endswith("?")]
    question_ratio = safe_divide(len(q_sentences), n_sentences)

    # --- Exclamation sentences ---
    excl_sentences = [s for s in sentences if s.strip().endswith("!")]
    exclamation_ratio = safe_divide(len(excl_sentences), n_sentences)

    return {
        "ling_verb_ratio": min(verb_ratio * 3, 1.0),
        "ling_adj_ratio": min(adj_ratio * 3, 1.0),
        "ling_adv_ratio": min(adv_ratio * 5, 1.0),
        "ling_conj_ratio": min(conj_ratio * 5, 1.0),
        "ling_passive_ratio": min(passive_ratio, 1.0),
        "ling_avg_sent_length": min(avg_sent_length, 1.0),
        "ling_sent_length_std": norm_std_sl,
        "ling_fk_grade": fk_grade,
        "ling_fog_index": fog_index,
        "ling_complex_word_ratio": min(complex_word_ratio * 3, 1.0),
        "ling_negation_ratio": negation_ratio,
        "ling_question_ratio": question_ratio,
        "ling_exclamation_ratio": exclamation_ratio,
    }


def _empty_linguistic_features() -> Dict[str, float]:
    return {
        "ling_verb_ratio": 0.0,
        "ling_adj_ratio": 0.0,
        "ling_adv_ratio": 0.0,
        "ling_conj_ratio": 0.0,
        "ling_passive_ratio": 0.0,
        "ling_avg_sent_length": 0.0,
        "ling_sent_length_std": 0.0,
        "ling_fk_grade": 0.0,
        "ling_fog_index": 0.0,
        "ling_complex_word_ratio": 0.0,
        "ling_negation_ratio": 0.0,
        "ling_question_ratio": 0.0,
        "ling_exclamation_ratio": 0.0,
    }
