# src/pipeline.py


from __future__ import annotations

from typing import Dict, Any

from modules.temporal_checker import check_temporal
from modules.entity_checker import check_entity
from modules.contradiction_checker import check_contradiction
from modules.numerical_checker import check_numerical
from modules.citation_checker import check_citation
from modules.knowledge_checker import check_knowledge

from scoring import aggregate_module_scores
from utils import normalize_text


class RuleBasedHallucinationPipeline:
    def _safe_normalize(self, value: Any) -> str:
        return normalize_text("" if value is None else str(value))

    def _build_combined_text(self, answer: str, question: str = "", knowledge: str = "") -> str:
        parts = []
        if knowledge.strip():
            parts.append(f"Knowledge: {knowledge}")
        if question.strip():
            parts.append(f"Question: {question}")
        parts.append(f"Answer: {answer}")
        return "\n".join(parts)

    def _standardize_module_result(
        self,
        module_name: str,
        raw_result: Any,
    ) -> Dict[str, Any]:
        default_result = {
            "module": module_name,
            "score": 0.0,
            "label": 0,
            "reason": "",
            "evidence": [],
        }

        if raw_result is None:
            return default_result

        if not isinstance(raw_result, dict):
            default_result["reason"] = f"{module_name} returned non-dict result"
            return default_result

        result = dict(default_result)
        result.update(raw_result)

        result["module"] = module_name

        # Safe score normalization
        score = result.get("score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        result["score"] = score

        label = result.get("label", 1 if score > 0 else 0)
        result["label"] = 1 if int(label) == 1 else 0

        result["reason"] = str(result.get("reason", "") or "")

        evidence = result.get("evidence", [])
        if evidence is None:
            evidence = []
        elif not isinstance(evidence, list):
            evidence = [str(evidence)]
        result["evidence"] = evidence

        return result

    def predict(
        self,
        text: str | None = None,
        answer: str | None = None,
        question: str | None = None,
        knowledge: str | None = None,
    ) -> Dict[str, Any]:
        if text is not None and answer is None and question is None and knowledge is None:
            clean_text = self._safe_normalize(text)
            clean_answer = clean_text
            clean_question = ""
            clean_knowledge = ""
        else:
            clean_answer = self._safe_normalize(answer if answer is not None else text)
            clean_question = self._safe_normalize(question)
            clean_knowledge = self._safe_normalize(knowledge)
            clean_text = self._build_combined_text(
                answer=clean_answer,
                question=clean_question,
                knowledge=clean_knowledge,
            )

        raw_module_results = {
            "temporal": check_temporal(clean_text),
            "entity": check_entity(clean_text),
            "contradiction": check_contradiction(clean_text),
            "numerical": check_numerical(clean_text),
            "citation": check_citation(clean_text),
            "knowledge": check_knowledge(clean_text),
        }

        module_results = {
            name: self._standardize_module_result(name, result)
            for name, result in raw_module_results.items()
        }

        final_result = aggregate_module_scores(module_results)

        return {
            "input_text": clean_text,
            "answer": clean_answer,
            "question": clean_question,
            "knowledge": clean_knowledge,
            "module_results": module_results,
            "final_result": final_result,
        }