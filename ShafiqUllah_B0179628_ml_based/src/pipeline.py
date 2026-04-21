# src/pipeline.py


from __future__ import annotations

import pickle
from typing import Dict, Any, Optional

import numpy as np

from feature_pipeline import FeaturePipeline
from models import get_probabilities, load_model
from utils import normalize_text


class MLHallucinationPipeline:

    def __init__(
        self,
        model=None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        threshold: float = 0.5,
        model_name: str = "unknown",
    ):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.threshold = threshold
        self.model_name = model_name

    def predict(
        self,
        text: Optional[str] = None,
        answer: Optional[str] = None,
        question: Optional[str] = None,
        knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        
    
        if answer is None:
            answer = text or ""
        question = question or ""
        knowledge = knowledge or ""

        answer = normalize_text(str(answer))
        question = normalize_text(str(question))
        knowledge = normalize_text(str(knowledge))

        # Extract features
        row = {"text": answer, "question": question, "knowledge": knowledge}
        X = self.feature_pipeline.transform([row])

        # Predict
        prob = float(get_probabilities(self.model, X)[0])
        label = 1 if prob >= self.threshold else 0

        explanation = (
            f"ML classifier ({self.model_name}) predicted hallucination "
            f"with probability {prob:.3f} (threshold={self.threshold})."
            if label == 1
            else f"ML classifier ({self.model_name}) predicted factual "
            f"with probability {prob:.3f} (threshold={self.threshold})."
        )

        return {
            "input_text": answer,
            "answer": answer,
            "question": question,
            "knowledge": knowledge,
            "final_result": {
                "label": label,
                "probability": round(prob, 4),
                "final_score": round(prob, 4),
                "explanation": explanation,
                "model": self.model_name,
                "threshold": self.threshold,
            },
        }

    def predict_row(self, row: dict) -> Dict[str, Any]:
        """Predict from a dataset row dict."""
        answer = normalize_text(str(row.get("text", "")))
        question = normalize_text(str(row.get("question", "")))
        knowledge = normalize_text(str(row.get("knowledge", "")))
        result = self.predict(answer=answer, question=question, knowledge=knowledge)
        return result["final_result"]

    def save(self, model_path: str, pipeline_path: str) -> None:
        """Save model and feature pipeline to disk."""
        import pickle
        from pathlib import Path

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"  Model saved → {model_path}")

        self.feature_pipeline.save(pipeline_path)

    @staticmethod
    def load(model_path: str, pipeline_path: str) -> "MLHallucinationPipeline":
        """Load a trained pipeline from disk."""
        model = load_model(model_path)
        feature_pipeline = FeaturePipeline.load(pipeline_path)
        return MLHallucinationPipeline(
            model=model,
            feature_pipeline=feature_pipeline,
            model_name=model.__class__.__name__,
        )
