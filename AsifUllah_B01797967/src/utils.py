# src/utils.py
import re
import random
import numpy as np


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_input(answer: str, question: str = "", knowledge: str = "") -> str:
    answer   = normalize_text(answer)
    question = normalize_text(question)
    knowledge = normalize_text(knowledge)

    parts = []
    if knowledge:
        parts.append(f"Knowledge: {knowledge}")
    if question:
        parts.append(f"Question: {question}")
    parts.append(f"Answer: {answer}")
    return " [SEP] ".join(parts)
