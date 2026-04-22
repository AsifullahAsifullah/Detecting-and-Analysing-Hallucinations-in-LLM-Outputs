# src/modules/entity_checker.py
import re
from typing import Dict, Any, List, Tuple
from config import SETTINGS

try:
    import spacy
except ImportError:
    spacy = None

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None and spacy is not None and SETTINGS.enable_spacy_ner:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = None
    return _NLP


CAPITALS_DB: Dict[str, str] = {
    **{k: v["capital"] for k, v in SETTINGS.known_facts.items() if "capital" in v},
    "United Kingdom": "London", "UK": "London",
    "United States": "Washington", "USA": "Washington", "America": "Washington",
    "Australia": "Canberra", "Canada": "Ottawa",
    "China": "Beijing", "India": "New Delhi",
    "Brazil": "Brasilia", "Russia": "Moscow",
    "Turkey": "Ankara", "Egypt": "Cairo",
    "Argentina": "Buenos Aires", "Mexico": "Mexico City",
    "Saudi Arabia": "Riyadh", "South Africa": "Pretoria",
    "Nigeria": "Abuja", "Kenya": "Nairobi",
    "Poland": "Warsaw", "Netherlands": "Amsterdam",
    "Belgium": "Brussels", "Sweden": "Stockholm",
    "Norway": "Oslo", "Denmark": "Copenhagen",
    "Switzerland": "Bern", "Austria": "Vienna",
    "Portugal": "Lisbon", "Greece": "Athens",
    "Hungary": "Budapest", "Czech Republic": "Prague",
    "Romania": "Bucharest", "Ukraine": "Kyiv",
    "Israel": "Jerusalem", "Iran": "Tehran",
    "Iraq": "Baghdad", "Afghanistan": "Kabul",
    "Bangladesh": "Dhaka", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Indonesia": "Jakarta",
    "Malaysia": "Kuala Lumpur", "Philippines": "Manila",
    "South Korea": "Seoul", "North Korea": "Pyongyang",
    "New Zealand": "Wellington",
}

ROLE_CONTRADICTIONS: Dict[str, List[str]] = {
    "albert einstein": [
        "footballer", "singer", "actor", "musician",
        "cricketer", "swimmer", "chef", "soldier", "carpenter"
    ],
    "nelson mandela": [
        "american president", "president of the usa",
        "president of america", "president of united states"
    ],
    "william shakespeare": ["american", "french", "german", "italian", "spanish"],
    "marie curie": ["footballer", "general", "soldier", "king", "queen"],
    "napoleon": ["british general", "american general", "chinese general"],
    "isaac newton": ["footballer", "singer", "chef"],
    "charles darwin": ["footballer", "singer", "king", "emperor"],
}

_BIRTH_DEATH = re.compile(
    r"\b(born|died|passed away|was born|was raised|was educated)\b",
    re.IGNORECASE,
)


def extract_candidate_capital_claims(text: str) -> List[Tuple[str, str]]:
    claims: List[Tuple[str, str]] = []

    patterns = [
        re.compile(
            r"capital of ([A-Z][a-zA-Z\s]+?) is ([A-Z][a-zA-Z\s]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"question:\s*what is the capital of ([A-Z][a-zA-Z\s]+?)\?\s*answer:\s*([A-Z][a-zA-Z\s]+)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"question:\s*which city is the capital of ([A-Z][a-zA-Z\s]+?)\?\s*answer:\s*([A-Z][a-zA-Z\s]+)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for pattern in patterns:
        for a, b in pattern.findall(text):
            claims.append((a.strip(), b.strip()))

    return claims


def extract_entities_spacy(text: str) -> List[str]:
    nlp = get_nlp()
    if nlp is None:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def check_entity(text: str) -> Dict[str, Any]:
    evidence: List[str] = []
    reason_parts: List[str] = []
    score = 0.0

    for country, claimed_capital in extract_candidate_capital_claims(text):
        true_capital = (
            CAPITALS_DB.get(country)
            or CAPITALS_DB.get(country.title())
            or CAPITALS_DB.get(country.strip())
        )
        if true_capital and claimed_capital.lower().strip() != true_capital.lower().strip():
            score += 0.9
            evidence += [country, claimed_capital, true_capital]
            reason_parts.append(
                f"Entity mismatch: capital of {country} claimed as '{claimed_capital}', "
                f"expected '{true_capital}'."
            )

    role_patterns = [
        re.compile(
            r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)\s+(?:was|is)\s+(?:a |an |the )?([a-zA-Z][\w\s]{2,40})",
            re.IGNORECASE,
        ),
        re.compile(
            r"question:\s*who\s+(?:was|is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)+)\?\s*answer:\s*([a-zA-Z][\w\s]{2,40})",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for role_pattern in role_patterns:
        for match in role_pattern.finditer(text):
            claimed_role_raw = match.group(2).strip()

            if _BIRTH_DEATH.match(claimed_role_raw):
                continue

            person_raw = match.group(1).strip()
            claimed_role = claimed_role_raw.lower()
            person_key = person_raw.lower()

            if person_key not in ROLE_CONTRADICTIONS:
                continue

            for bad in ROLE_CONTRADICTIONS[person_key]:
                if bad in claimed_role:
                    score += 0.85
                    evidence += [person_raw, bad]
                    reason_parts.append(
                        f"Entity-role conflict: '{person_raw}' described as "
                        f"'{claimed_role_raw[:50]}' which conflicts with known identity."
                    )
                    break

    flag = 1 if score > 0 else 0
    return {
        "module": "entity",
        "flag": flag,
        "score": min(round(score, 4), 1.0),
        "reason": " | ".join(reason_parts) if reason_parts else "No entity issue detected.",
        "evidence": evidence,
        "entities_found": extract_entities_spacy(text),
    }