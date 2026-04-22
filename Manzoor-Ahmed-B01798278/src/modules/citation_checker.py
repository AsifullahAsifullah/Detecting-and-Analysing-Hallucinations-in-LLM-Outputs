# src\modules\citation_checker.py

import re
from typing import Dict, Any, List


DOI_PATTERN   = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_NEW     = re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\b",       re.IGNORECASE)
ARXIV_OLD     = re.compile(r"\barXiv:\s*[a-z\-]+/\d{7}(?:v\d+)?\b",        re.IGNORECASE)
CITATION_LANG = re.compile(
    r"\b(according to|cited in|as stated in|published in|as per|"
    r"et al\.|referenced in|\(\d{4}\)|doi:|arxiv:)\b",
    re.IGNORECASE,
)

SUSPICIOUS_JOURNALS: List[str] = [
    r"journal of totally real",
    r"journal of advanced studies in \w+",
    r"international journal of \w+ science and technology",
    r"global journal of \w+ research",
    r"world journal of \w+",
    r"american journal of \w+ innovation",
    r"open access journal of \w+",
    r"journal of emerging \w+ research",
    r"universal journal of \w+",
    r"interdisciplinary journal of \w+ excellence",
    r"journal of pure and applied \w+ sciences",
    r"annals of \w+ advancement",
    r"international research journal of \w+",
]


def check_citation(text: str) -> Dict[str, Any]:
    evidence: List[str]     = []
    reason_parts: List[str] = []
    score = 0.0
    text_lower = text.lower()

    doi_mentioned   = bool(re.search(r"\bdoi\b",   text, re.IGNORECASE))
    arxiv_mentioned = bool(re.search(r"\barxiv\b", text, re.IGNORECASE))
    valid_dois      = DOI_PATTERN.findall(text)
    all_arxiv       = [m.group() for m in ARXIV_NEW.finditer(text)] + \
                      [m.group() for m in ARXIV_OLD.finditer(text)]

    if doi_mentioned and not valid_dois:
        score += 0.7
        evidence.append("DOI mentioned but pattern invalid")
        reason_parts.append("Citation mentions DOI but no valid DOI pattern found.")

    if arxiv_mentioned and not all_arxiv:
        score += 0.7
        evidence.append("arXiv mentioned but pattern invalid")
        reason_parts.append("Citation mentions arXiv but no valid identifier found.")

    for pattern in SUSPICIOUS_JOURNALS:
        m = re.search(pattern, text_lower)
        if m:
            score += 0.5
            evidence.append(f"Suspicious journal: {m.group()}")
            reason_parts.append(f"Suspicious or likely predatory journal: '{m.group()}'.")

    if (CITATION_LANG.search(text)
            and not valid_dois and not all_arxiv
            and not doi_mentioned and not arxiv_mentioned):
        score += 0.3
        evidence.append("Unverifiable citation")
        reason_parts.append(
            "Text contains citation language but no DOI or arXiv identifier."
        )

    flag = 1 if score > 0 else 0
    return {
        "module":           "citation",
        "flag":             flag,
        "score":            min(round(score, 4), 1.0),
        "reason":           " | ".join(reason_parts) if reason_parts else "No citation issue detected.",
        "evidence":         evidence,
        "valid_dois_found": valid_dois,
        "arxiv_ids_found":  all_arxiv,
    }
