# src/modules/semantic_features.py
"""
Semantic Feature Extractor
==========================
Extracts meaning-level features from LLM outputs:
- Cosine similarity between answer and knowledge/question
- Sentence-to-sentence coherence (intra-text consistency)
- Topic mixing proxy (vocabulary overlap)
- Semantic anomaly indicators

When sentence-transformers is not installed, falls back to
TF-IDF cosine similarity (lightweight but effective).
"""

import re
import math
from typing import Dict, List, Optional

from utils import normalize_text, safe_divide, split_sentences


# ---------------------------------------------------------------------------
# Cosine similarity helpers (pure Python / scipy-free)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vector(tokens: List[str], vocab: Dict[str, int]) -> List[float]:
    """Simple TF vector (normalised)."""
    from collections import Counter
    counts = Counter(tokens)
    vec = [0.0] * len(vocab)
    for term, idx in vocab.items():
        vec[idx] = counts.get(term, 0)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))


def cosine_similarity_texts(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF vectors."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    vocab = {t: i for i, t in enumerate(set(tokens_a + tokens_b))}
    vec_a = _tfidf_vector(tokens_a, vocab)
    vec_b = _tfidf_vector(tokens_b, vocab)
    return _cosine_sim(vec_a, vec_b)


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_semantic_features(
    answer: str,
    question: str = "",
    knowledge: str = "",
) -> Dict[str, float]:
    """
    Extract semantic features.

    Parameters
    ----------
    answer   : The LLM-generated answer text.
    question : The question (optional).
    knowledge: The reference knowledge/context (optional).

    Returns
    -------
    Dict of feature_name -> float.
    """
    answer = normalize_text(answer)
    question = normalize_text(question)
    knowledge = normalize_text(knowledge)

    if not answer:
        return _empty_semantic_features()

    feats: Dict[str, float] = {}

    # 1. Answer–knowledge similarity
    if knowledge:
        feats["sem_ans_knowledge_sim"] = cosine_similarity_texts(answer, knowledge)
    else:
        feats["sem_ans_knowledge_sim"] = 0.5  # neutral when no knowledge

    # 2. Answer–question similarity
    if question:
        feats["sem_ans_question_sim"] = cosine_similarity_texts(answer, question)
    else:
        feats["sem_ans_question_sim"] = 0.5

    # 3. Intra-answer sentence coherence
    sentences = split_sentences(answer)
    if len(sentences) >= 2:
        sims = []
        for i in range(len(sentences) - 1):
            s = cosine_similarity_texts(sentences[i], sentences[i + 1])
            sims.append(s)
        feats["sem_intra_coherence_mean"] = sum(sims) / len(sims)
        feats["sem_intra_coherence_min"] = min(sims)
        feats["sem_intra_coherence_std"] = _std(sims)
    else:
        feats["sem_intra_coherence_mean"] = 1.0
        feats["sem_intra_coherence_min"] = 1.0
        feats["sem_intra_coherence_std"] = 0.0

    # 4. Vocabulary overlap (answer vs knowledge)
    ans_tokens = set(_tokenize(answer))
    know_tokens = set(_tokenize(knowledge)) if knowledge else set()
    if know_tokens:
        overlap = len(ans_tokens & know_tokens)
        union = len(ans_tokens | know_tokens)
        feats["sem_vocab_overlap_jaccard"] = safe_divide(overlap, union)
    else:
        feats["sem_vocab_overlap_jaccard"] = 0.5

    # 5. Answer vs question overlap
    q_tokens = set(_tokenize(question)) if question else set()
    if q_tokens:
        overlap_q = len(ans_tokens & q_tokens)
        union_q = len(ans_tokens | q_tokens)
        feats["sem_vocab_overlap_q_jaccard"] = safe_divide(overlap_q, union_q)
    else:
        feats["sem_vocab_overlap_q_jaccard"] = 0.5

    # 6. Topic mixing proxy:
    #    High unique word ratio relative to length may indicate hallucination "topic drift"
    ans_toks = _tokenize(answer)
    feats["sem_unique_word_ratio"] = safe_divide(len(set(ans_toks)), max(len(ans_toks), 1))

    # 7. Knowledge contradiction proxy:
    #    Low similarity to knowledge but high confidence markers = likely hallucination
    if knowledge:
        dissimilarity = 1.0 - feats["sem_ans_knowledge_sim"]
        feats["sem_knowledge_dissimilarity"] = dissimilarity
    else:
        feats["sem_knowledge_dissimilarity"] = 0.5

    return feats


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _empty_semantic_features() -> Dict[str, float]:
    return {
        "sem_ans_knowledge_sim": 0.0,
        "sem_ans_question_sim": 0.0,
        "sem_intra_coherence_mean": 0.0,
        "sem_intra_coherence_min": 0.0,
        "sem_intra_coherence_std": 0.0,
        "sem_vocab_overlap_jaccard": 0.0,
        "sem_vocab_overlap_q_jaccard": 0.0,
        "sem_unique_word_ratio": 0.0,
        "sem_knowledge_dissimilarity": 0.0,
    }
