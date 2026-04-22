# src/modules/knowledge_checker.py
import re
from typing import Dict, Any, List, Set, Tuple


STOPWORDS: Set[str] = {
    "the", "is", "was", "are", "in", "on", "at", "of", "a", "an", "and", "to",
    "for", "with", "that", "this", "it", "as", "by", "from", "be", "or", "which",
    "who", "what", "when", "where", "why", "how", "into", "onto", "than", "then",
    "also", "called", "about", "after", "before", "been", "being", "have", "has",
    "had", "does", "did", "their", "there", "they", "them", "its", "his", "her",
    "him", "she", "he", "you", "your", "our", "ours", "but", "however", "because",
    "while", "during", "through", "over", "under", "between", "among", "within",
    "without", "very", "more", "most", "some", "such", "only", "many", "much",
    "other", "another", "same", "each", "both", "all", "any", "few", "own",
    "part", "based", "first", "second", "third", "popular", "famous", "known",
    "named", "kind", "type", "form", "made", "make", "used", "using",
    "one", "two", "three", "four", "five",
}

SECTION_LABELS = ["knowledge:", "question:", "answer:"]


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_section(text: str, start_label: str, end_labels: List[str]) -> str:
    end_pattern = "|".join(re.escape(lbl) for lbl in end_labels) if end_labels else r"$"
    pattern = re.compile(
        rf"{start_label}\s*(.*?)(?=(?:{end_pattern})\s*|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_fields(text: str) -> Dict[str, str]:
    raw = text.replace("\r", "\n")
    knowledge = _extract_section(raw, r"knowledge:", ["question:", "answer:"])
    question = _extract_section(raw, r"question:", ["answer:"])
    answer = _extract_section(raw, r"answer:", [])

    return {
        "knowledge": knowledge.strip(),
        "question": question.strip(),
        "answer": answer.strip(),
    }


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    result: List[str] = []

    for w in words:
        w = w.strip("-'")
        if len(w) < 3:
            continue
        if w in STOPWORDS:
            continue
        result.append(w)

    return result


def extract_capitalized_phrases(text: str) -> List[str]:
    phrases = re.findall(
        r"\b([A-Z][a-zA-Z'&.-]*(?:\s+(?:[A-Z][a-zA-Z'&.-]*|of|for|and|the|in)){0,5})\b",
        text,
    )

    cleaned: List[str] = []
    seen = set()

    for p in phrases:
        p = re.sub(r"\s+", " ", p).strip(" .,:;!?\"'")
        if len(p) < 3:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    return cleaned


def _token_overlap_ratio(answer_tokens: Set[str], knowledge_tokens: Set[str]) -> float:
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & knowledge_tokens) / max(len(answer_tokens), 1)


def _contains_direct_negation_conflict(answer_lower: str, knowledge_lower: str) -> bool:
    negation_patterns = [
        (" not ", " "),
        (" never ", " "),
        (" no ", " "),
        (" without ", " with "),
    ]

    for neg, pos in negation_patterns:
        if neg in answer_lower:
            candidate = re.sub(r"\s+", " ", answer_lower.replace(neg, pos)).strip()
            if candidate and candidate in knowledge_lower:
                return True

    return False


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        key = item.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _extract_years(text: str) -> List[str]:
    return re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text)


def _numeric_mismatch(answer: str, knowledge: str) -> Tuple[bool, List[str], str]:
    answer_numbers = _extract_numbers(answer)
    knowledge_numbers = _extract_numbers(knowledge)

    if not answer_numbers or not knowledge_numbers:
        return False, [], ""

    if len(answer_numbers) <= 3 and answer_numbers != knowledge_numbers[:len(answer_numbers)]:
        return (
            True,
            [f"Answer numbers: {answer_numbers}", f"Knowledge numbers: {knowledge_numbers[:5]}"],
            "Answer numeric values differ from knowledge."
        )

    unsupported = [n for n in answer_numbers if n not in knowledge_numbers]
    if unsupported:
        return (
            True,
            [f"Unsupported numbers: {unsupported[:5]}"],
            "Answer includes numeric values not supported by knowledge."
        )

    return False, [], ""


def _year_mismatch(answer: str, knowledge: str) -> Tuple[bool, List[str], str]:
    answer_years = _extract_years(answer)
    knowledge_years = _extract_years(knowledge)

    if not answer_years or not knowledge_years:
        return False, [], ""

    unsupported_years = [y for y in answer_years if y not in knowledge_years]
    if unsupported_years:
        return (
            True,
            [f"Unsupported years: {unsupported_years[:5]}", f"Knowledge years: {knowledge_years[:5]}"],
            "Answer year differs from knowledge."
        )

    return False, [], ""


def check_knowledge(text: str) -> Dict[str, Any]:
    evidence: List[str] = []
    reason_parts: List[str] = []
    score = 0.0

    fields = _extract_fields(text)
    knowledge = _clean_text(fields["knowledge"])
    question = _clean_text(fields["question"])
    answer = _clean_text(fields["answer"])

    if not knowledge or not answer:
        return {
            "module": "knowledge",
            "flag": 0,
            "score": 0.0,
            "reason": "No knowledge context.",
            "evidence": [],
        }

    knowledge_lower = knowledge.lower()
    answer_lower = answer.lower()
    question_lower = question.lower()

    if answer_lower in knowledge_lower:
        return {
            "module": "knowledge",
            "flag": 0,
            "score": 0.0,
            "reason": "Answer supported directly by knowledge.",
            "evidence": [],
        }

    answer_tokens = set(extract_keywords(answer))
    knowledge_tokens = set(extract_keywords(knowledge))
    overlap = answer_tokens & knowledge_tokens
    overlap_ratio = _token_overlap_ratio(answer_tokens, knowledge_tokens)

    answer_entities = extract_capitalized_phrases(answer)
    unsupported_entities = [
        ent for ent in answer_entities
        if ent.lower() not in knowledge_lower
    ]

    if answer_entities and unsupported_entities:
        score += 0.9
        evidence.append(f"Unsupported entities: {unsupported_entities[:3]}")
        reason_parts.append(
            "Answer introduces named entities or phrases not supported by knowledge."
        )

    if overlap_ratio < 0.30 and len(answer_tokens) > 1:
        score += 0.5
        evidence.append(
            f"Low overlap: {len(overlap)}/{len(answer_tokens)} ({overlap_ratio:.2f})"
        )
        reason_parts.append("Answer has weak lexical support from knowledge.")

    if len(answer.split()) <= 3 and answer_lower not in knowledge_lower:
        score += 0.5
        evidence.append("Short answer not found in knowledge")
        reason_parts.append("Short factual answer is not directly supported by knowledge.")

    missing_answer_tokens = [t for t in answer_tokens if t not in knowledge_tokens]
    if len(answer_tokens) <= 5 and len(missing_answer_tokens) >= max(1, len(answer_tokens) // 2):
        score += 0.4
        evidence.append(f"Missing tokens: {missing_answer_tokens[:5]}")
        reason_parts.append("Key answer tokens are not present in knowledge.")

    if _contains_direct_negation_conflict(answer_lower, knowledge_lower):
        score += 0.7
        evidence.append("Negation mismatch")
        reason_parts.append("Answer appears to negate a supported knowledge statement.")

    has_num_mismatch, num_evidence, num_reason = _numeric_mismatch(answer, knowledge)
    if has_num_mismatch:
        score += 0.6
        evidence.extend(num_evidence)
        reason_parts.append(num_reason)

    has_year_mismatch, year_evidence, year_reason = _year_mismatch(answer, knowledge)
    if has_year_mismatch:
        score += 0.5
        evidence.extend(year_evidence)
        reason_parts.append(year_reason)

    if question_lower:
        if "capital" in question_lower and "capital" in knowledge_lower:
            if len(answer_tokens) <= 4 and answer_lower not in knowledge_lower:
                score += 0.5
                evidence.append("Capital-style answer not found in knowledge")
                reason_parts.append(
                    "Answer to capital-related question is not supported by knowledge."
                )

        if any(word in question_lower for word in ["who", "which person", "whose"]):
            if answer_entities and unsupported_entities:
                score += 0.2
                evidence.append("Person/entity answer unsupported")
                reason_parts.append(
                    "Question expects a specific entity, but answer entity is unsupported."
                )

        if "when" in question_lower:
            answer_years = _extract_years(answer)
            knowledge_years = _extract_years(knowledge)
            if answer_years and knowledge_years:
                unsupported = [y for y in answer_years if y not in knowledge_years]
                if unsupported:
                    score += 0.4
                    evidence.append(f"When-question unsupported years: {unsupported[:5]}")
                    reason_parts.append(
                        "Answer to time-related question contains unsupported year."
                    )

        if "how many" in question_lower or "how much" in question_lower:
            answer_numbers = _extract_numbers(answer)
            knowledge_numbers = _extract_numbers(knowledge)
            if answer_numbers and knowledge_numbers:
                unsupported = [n for n in answer_numbers if n not in knowledge_numbers]
                if unsupported:
                    score += 0.4
                    evidence.append(f"Count/value unsupported numbers: {unsupported[:5]}")
                    reason_parts.append(
                        "Answer to quantity-related question contains unsupported number."
                    )

    comparison_patterns = [
        "which", "first", "before", "after", "earlier", "later"
    ]

    relation_patterns = [
        "are both", "same country", "same team",
        "same person", "same year", "same place"
    ]

    is_comparison = any(p in question_lower for p in comparison_patterns)
    is_relation = any(p in question_lower for p in relation_patterns)

    if is_comparison and overlap_ratio < 0.45:
        score += 0.6
        evidence.append("Comparison reasoning weak")
        reason_parts.append(
            "Answer does not correctly use knowledge for comparison."
        )

    if is_relation and overlap_ratio < 0.45:
        score += 0.6
        evidence.append("Relation mismatch")
        reason_parts.append(
            "Answer gives incorrect relation between entities."
        )

    if len(answer_entities) >= 1 and unsupported_entities:
        score += 0.4
        evidence.append("Entity substitution")
        reason_parts.append(
            "Answer contains incorrect entity not supported by knowledge."
        )

    if question_lower:
        selection_patterns = [
            "which", "which person", "which actress", "which novelist",
            "which italian", "which actor", "which footballer", "which was"
        ]

        if any(p in question_lower for p in selection_patterns):
            if len(answer_entities) >= 1:
                mentioned_in_knowledge = [
                    ent for ent in answer_entities if ent.lower() in knowledge_lower
                ]

                if mentioned_in_knowledge and overlap_ratio < 0.50:
                    score += 0.5
                    evidence.append("Wrong candidate selection")
                    reason_parts.append(
                        "Answer appears to select the wrong candidate from the knowledge context."
                    )

    score = min(score, 1.0)
    evidence = _dedupe_preserve_order(evidence)

    return {
        "module": "knowledge",
        "flag": 1 if score >= 0.5 else 0,
        "score": round(score, 4),
        "reason": " ".join(reason_parts) if reason_parts else "No strong knowledge mismatch detected.",
        "evidence": evidence,
    }