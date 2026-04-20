# src/pipeline.py
from __future__ import annotations
import torch
from typing import Dict, Any, Optional
from model import TransformerClassifier, load_model
from utils import format_input, normalize_text
from config import SETTINGS


class DeepLearningHallucinationPipeline:

    def __init__(
        self,
        model: TransformerClassifier,
        tokenizer,
        threshold: float = 0.5,
        model_arch: str = "bert",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.model_arch = model_arch
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        text: Optional[str] = None,
        answer: Optional[str] = None,
        question: Optional[str] = None,
        knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:

        if answer is None:
            answer = text or ""

        answer = normalize_text(str(answer))
        question = normalize_text(str(question or ""))
        knowledge = normalize_text(str(knowledge or ""))

        input_text = format_input(answer, question, knowledge)

        encoding = self.tokenizer(
            input_text,
            max_length=SETTINGS.model.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            prob_hallucinated = float(probs[0, 1].item())

        label = 1 if prob_hallucinated >= self.threshold else 0

        explanation = (
            f"Transformer ({self.model_arch}) predicted hallucination "
            f"with probability {prob_hallucinated:.3f} (threshold={self.threshold})."
            if label == 1
            else
            f"Transformer ({self.model_arch}) predicted factual "
            f"with probability {prob_hallucinated:.3f} (threshold={self.threshold})."
        )

        return {
            "input_text": input_text,
            "answer": answer,
            "question": question,
            "knowledge": knowledge,
            "final_result": {
                "label": label,
                "probability": round(prob_hallucinated, 4),
                "final_score": round(prob_hallucinated, 4),
                "explanation": explanation,
                "model_arch": self.model_arch,
                "threshold": self.threshold,
            },
        }

    def predict_row(self, row: dict) -> Dict[str, Any]:
        answer = normalize_text(str(row.get("text", "")))
        question = normalize_text(str(row.get("question", "")))
        knowledge = normalize_text(str(row.get("knowledge", "")))
        result = self.predict(answer=answer, question=question, knowledge=knowledge)
        return result["final_result"]

    def save(self, model_path: str) -> None:
        from model import save_model as _save
        _save(self.model, model_path)

    @staticmethod
    def load(model_path: str, model_name: str, threshold: float = 0.5) -> "DeepLearningHallucinationPipeline":
        """Load a saved model and build the pipeline."""
        from transformers import AutoTokenizer, DebertaV2Tokenizer
        from model import load_model as _load

        model = _load(model_path)

        if "deberta-v3" in model_name.lower():
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        elif "deberta" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        arch = "bert"
        if "roberta" in model_name.lower():
            arch = "roberta"
        elif "deberta" in model_name.lower():
            arch = "deberta"

        return DeepLearningHallucinationPipeline(
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            model_arch=arch,
        )