# src/modules/contradiction_checker.py

import re
from typing import Dict, Any, List


NEGATION_PAIRS = [
    (r"\bis alive\b", r"\b(died|is dead|passed away)\b"),
    (r"\bwon the match\b", r"\blost the match\b"),
    (r"\bis true\b", r"\bis false\b"),
    (r"\bthe meeting happened\b", r"\bthe meeting never happened\b"),
    (r"\balways\b", r"\bnever\b"),
    (r"\ball\b", r"\bnone\b"),
    (r"\beveryone\b", r"\bno one\b"),
    (r"\beverywhere\b", r"\bnowhere\b"),
    (r"\bincreased\b", r"\b(decreased|dropped|fell|declined)\b"),
    (r"\bmore than\b", r"\bless than\b"),
    (r"\blargest\b", r"\bsmallest\b"),
    (r"\bhighest\b", r"\blowest\b"),
    (r"\bexists\b", r"\bdoes not exist\b"),
    (r"\bwas founded\b", r"\bwas never founded\b"),
    (r"\bwas created\b", r"\bwas never created\b"),
    (r"\bcan always\b", r"\bcan never\b"),
    (r"\bwill always\b", r"\bwill never\b"),
    (r"\bhas always\b", r"\bhas never\b"),
    (r"\bmale\b", r"\bfemale\b"),
    (r"\btrue\b", r"\bnot true\b"),
]

INLINE_NEGATION = re.compile(
    r"\b([a-z][a-z\s]{7,40})\b.{0,80}\b(?:not|never)\s+\1\b",
    re.IGNORECASE | re.DOTALL,
)

INLINE_STOPLIST = {
    "to be", "napoleon", "sold", "the game", "da game",
    "the name", "the word", "the fact", "the case",
    "the capital", "the city", "the country", "the answer",
}

QA_ANSWER_BLOCK = re.compile(
    r"question:\s*(.*?)\s*answer:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

COMPARATIVE_CONFLICTS = [
    (r"\blarger than\b", r"\bsmaller than\b"),
    (r"\bolder than\b", r"\byounger than\b"),
    (r"\bhigher than\b", r"\blower than\b"),
    (r"\bmore populous than\b", r"\bless populous than\b"),
]


def _clean_pattern_text(pattern: str) -> str:
    return re.sub(r"\\b|[()]", "", pattern).strip()


def _extract_answer_text(text: str) -> str:
    match = QA_ANSWER_BLOCK.search(text)
    if not match:
        return text
    return match.group(2).strip()


def check_contradiction(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    answer_text = _extract_answer_text(text).lower()

    evidence: List[str] = []
    reason_parts: List[str] = []
    score = 0.0

    for pos_pat, neg_pat in NEGATION_PAIRS:
        if re.search(pos_pat, text_lower) and re.search(neg_pat, text_lower):
            pos_d = _clean_pattern_text(pos_pat)
            neg_d = _clean_pattern_text(neg_pat)
            score += 0.8
            evidence += [pos_d, neg_d]
            reason_parts.append(
                f"Possible contradiction: '{pos_d}' and '{neg_d}' both present."
            )

    seen = set()
    for match in INLINE_NEGATION.finditer(text):
        phrase = match.group(1).strip().lower()
        phrase = re.sub(r"\s+", " ", phrase)

        if phrase in INLINE_STOPLIST or phrase in seen:
            continue
        if len(phrase.split()) < 2:
            continue

        seen.add(phrase)
        score += 0.7
        evidence.append(phrase)
        reason_parts.append(
            f"Inline negation conflict: '{phrase}' appears both affirmed and denied."
        )

    qa_patterns = [
            re.compile(r"\b([a-z][a-z\s]{3,30})\b\s+(?:but|however|yet)\s+(?:is|was|are)?\s*not\s+\1\b", re.IGNORECASE),
            re.compile(r"\b([a-z][a-z\s]{3,30})\b\s+and\s+not\s+\1\b", re.IGNORECASE),
            re.compile(r"\b([a-z][a-z\s]{3,30})\b\s+or\s+not\s+\1\b", re.IGNORECASE),
        ]
    for pattern in qa_patterns:
        for match in pattern.finditer(answer_text):
            phrase = re.sub(r"\s+", " ", match.group(1).strip().lower())
            if phrase in INLINE_STOPLIST or len(phrase.split()) < 1:
                continue
            score += 0.75
            evidence.append(phrase)
            reason_parts.append(
                f"QA-style contradiction detected around phrase '{phrase}'."
            )

    for pos_pat, neg_pat in COMPARATIVE_CONFLICTS:
        if re.search(pos_pat, text_lower) and re.search(neg_pat, text_lower):
            pos_d = _clean_pattern_text(pos_pat)
            neg_d = _clean_pattern_text(neg_pat)
            score += 0.7
            evidence += [pos_d, neg_d]
            reason_parts.append(
                f"Comparative contradiction: '{pos_d}' and '{neg_d}' both present."
            )

    flag = 1 if score > 0 else 0
    return {
        "module": "contradiction",
        "flag": flag,
        "score": min(round(score, 4), 1.0),
        "reason": " | ".join(reason_parts) if reason_parts else "No contradiction issue detected.",
        "evidence": evidence,
    }