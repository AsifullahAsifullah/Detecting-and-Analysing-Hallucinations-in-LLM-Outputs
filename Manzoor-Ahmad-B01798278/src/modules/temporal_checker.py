# src\modules\temporal_checker.py
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime

CURRENT_YEAR = datetime.now().year

YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")

BC_YEAR_PATTERN = re.compile(r"\b(\d{1,4})\s*(?:B\.?C\.?E?\.?)\b", re.IGNORECASE)


_EP = r"{verb}(?:\s+(?!in\s+\d{{4}})[\w,]+){{0,5}}\s+in\s+(\d{{4}})"

EVENT_PATTERNS: Dict[str, str] = {
    "born":        _EP.format(verb="born"),
    "died":        _EP.format(verb="died"),
    "published":   _EP.format(verb="published"),
    "founded":     _EP.format(verb="founded"),
    "released":    _EP.format(verb="released"),
    "established": _EP.format(verb="established"),
    "invented":    _EP.format(verb="invented"),
    "discovered":  _EP.format(verb="discovered"),
}


ANACHRONISM_RULES: List[Tuple[str, int, str]] = [
    (r"\binternet\b",                                  1983, "internet"),
    (r"\bworld wide web\b|\bwww\b",                    1991, "World Wide Web"),
    (r"\bsmartphone\b",                                2007, "smartphone"),
    (r"\biphone\b",                                    2007, "iPhone"),
    (r"\btwitter\b|\btweet(?:ed|ing)?\b",              2006, "Twitter"),
    (r"\bfacebook\b",                                  2004, "Facebook"),
    (r"\bchatgpt\b|\bgpt-[34]\b",                      2022, "ChatGPT/GPT"),
    (r"\bwikipedia\b",                                 2001, "Wikipedia"),
    (r"\bemail\b|\be-mail\b",                          1971, "email"),
    (r"\btelevision\b|\btv\b",                         1920, "television"),
    (r"\baeroplane\b|\bairplane\b|\bflight\b",          1903, "airplane"),
    (r"\bradar\b",                                     1935, "radar"),
    (r"\bnuclear\b|\batomic bomb\b",                   1945, "nuclear weapons"),
    (r"\bpenicillin\b",                                1928, "penicillin"),
    (r"\belectricity\b",                               1879, "commercial electricity"),
    (r"\btelephone\b",                                 1876, "telephone"),
    (r"\bsteam engine\b",                              1712, "steam engine"),
    (r"\bphotograph\b|\bcamera\b",                     1839, "photography"),
    (r"\bcomputer\b",                                  1940, "computer"),
    (r"\blaptop\b|\bnotebook computer\b",              1981, "laptop computer"),
    (r"\bmobile phone\b|\bcell phone\b",               1973, "mobile phone"),
    (r"\bsocial media\b",                              2004, "social media"),
    (r"\brobot\b",                                     1920, "robot"),
    (r"\bsatellite\b",                                 1957, "satellite"),
    (r"\bnuclear power\b|\bnuclear plant\b",            1954, "nuclear power"),
    (r"\bworld war ii\b|\bwwii\b|\bsecond world war\b", 1939, "World War II"),
    (r"\bworld war i\b|\bwwi\b|\bfirst world war\b",   1914, "World War I"),
]

PAST_TENSE_VERBS = re.compile(
    r"\b(was born|died|published|founded|established|invented|discovered|"
    r"graduated|won|lost|defeated|signed|wrote|created|built)\b",
    re.IGNORECASE,
)


def extract_years(text: str) -> List[int]:
    """Return all 4-digit AD years found in text."""
    return [int(y) for y in YEAR_PATTERN.findall(text)]


def has_bc_year(text: str) -> bool:
    """Return True if any BC/BCE year reference is present."""
    return bool(BC_YEAR_PATTERN.search(text))


def check_temporal(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    years      = extract_years(text)
    bc_present = has_bc_year(text)

    evidence: List[str]     = []
    reason_parts: List[str] = []
    score = 0.0

    extracted_events: Dict[str, int] = {}
    for event, pattern in EVENT_PATTERNS.items():
        m = re.search(pattern, text_lower)
        if m:
            extracted_events[event] = int(m.group(1))

    born      = extracted_events.get("born")
    died      = extracted_events.get("died")
    published = extracted_events.get("published")
    founded   = extracted_events.get("founded")

    if born and died and born > died:
        score += 1.0
        evidence     += [f"born={born}", f"died={died}"]
        reason_parts.append("Temporal contradiction: birth year is after death year.")

    if died and published and published > died:
        score += 0.8
        evidence     += [f"died={died}", f"published={published}"]
        reason_parts.append(
            "Possible contradiction: publication year appears after death year."
        )

    if founded and born and founded < born:
        score += 0.7
        evidence     += [f"born={born}", f"founded={founded}"]
        reason_parts.append(
            "Temporal anomaly: organisation founded before person's birth year."
        )

    future_years = [y for y in years if y > CURRENT_YEAR]
    if future_years and PAST_TENSE_VERBS.search(text):
        score += 0.9
        evidence     += [str(y) for y in future_years]
        reason_parts.append(
            f"Future year(s) {future_years} used in past-tense event claim."
        )

    for kw_pattern, earliest_year, label in ANACHRONISM_RULES:
        if not re.search(kw_pattern, text_lower):
            continue
        if bc_present:
            bc_matches = BC_YEAR_PATTERN.findall(text)
            score += 0.85
            evidence     += [label, f"{bc_matches[0]} BC"]
            reason_parts.append(
                f"Anachronism: '{label}' referenced in a BC-year context "
                f"({bc_matches[0]} BC; earliest plausible: {earliest_year} AD)."
            )
            break 
        for y in years:
            if y < earliest_year:
                score += 0.8
                evidence     += [label, str(y)]
                reason_parts.append(
                    f"Anachronism: '{label}' referenced alongside year {y} "
                    f"(earliest plausible: {earliest_year})."
                )
                break  

    for y in years:
        if y > CURRENT_YEAR + 10:
            score += 0.5
            evidence.append(str(y))
            reason_parts.append(f"Suspiciously far-future year referenced: {y}.")

    flag = 1 if score > 0 else 0
    return {
        "module":   "temporal",
        "flag":     flag,
        "score":    min(round(score, 4), 1.0),
        "reason":   " | ".join(reason_parts) if reason_parts else "No temporal issue detected.",
        "evidence": evidence,
    }
