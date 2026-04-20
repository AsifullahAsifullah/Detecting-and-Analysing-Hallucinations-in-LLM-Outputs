# src/main.py
import sys
import argparse
from pathlib import Path

from pipeline import DeepLearningHallucinationPipeline


EXAMPLES = [
    {
        "answer":    "The capital of Germany is Madrid.",
        "question":  "What is the capital of Germany?",
        "knowledge": "Germany is a country in central Europe. Its capital is Berlin.",
    },
    {
        "answer":    "The capital of Pakistan is Islamabad.",
        "question":  "What is the capital of Pakistan?",
        "knowledge": "Pakistan is a country in South Asia. Its capital city is Islamabad.",
    },
    {
        "answer":    "Albert Einstein was a professional footballer who played for Germany.",
        "question":  "Who was Albert Einstein?",
        "knowledge": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    },
    {
        "answer":    "Napoleon used the internet in 1804 to communicate with his generals.",
        "question":  "How did Napoleon communicate with his generals?",
        "knowledge": "Napoleon lived 1769-1821. The internet was invented in the late 20th century.",
    },
    {
        "answer":    "Shakespeare wrote Hamlet in 1601 using his laptop computer.",
        "question":  "When did Shakespeare write Hamlet?",
        "knowledge": "Shakespeare wrote Hamlet around 1600-1601. Computers did not exist in his time.",
    },
    {
        "answer":    "The Great Wall of China is visible from the Moon with the naked eye.",
        "question":  "Can you see the Great Wall of China from space?",
        "knowledge": "The Great Wall of China is about 5-8 metres wide and is not visible from the Moon.",
    },
    {
        "answer":    "Python was created by Guido van Rossum and first released in 1991.",
        "question":  "Who created Python?",
        "knowledge": "Python is a programming language created by Guido van Rossum, first released in 1991.",
    },
    {
        "answer":    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "question":  "At what temperature does water boil?",
        "knowledge": "Water boils at 100 degrees C at sea level under standard atmospheric pressure.",
    },
    {
        "answer":    "The Earth orbits the Sun once every 365.25 days.",
        "question":  "How long does it take the Earth to orbit the Sun?",
        "knowledge": "The Earth takes approximately 365.25 days to complete one full orbit around the Sun.",
    },
    {
        "answer":    "Scientists have probably confirmed that dark matter is mostly made of cheese, "
                     "although this is still somewhat uncertain and possibly not fully proven.",
        "question":  "What is dark matter made of?",
        "knowledge": "Dark matter is a hypothetical form of matter that does not interact with light.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../models/bert_best.pt")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"ERROR: No trained model found at {args.model_path}")
        print("Please run 'python train.py --arch bert' first.")
        sys.exit(1)

    print(f"Loading transformer pipeline ({args.model_name})...")
    pipeline = DeepLearningHallucinationPipeline.load(
        model_path = args.model_path,
        model_name = args.model_name,
        threshold  = args.threshold,
    )

    print(f"\n{'='*70}")
    print(f"  Deep Learning Hallucination Detector — Demo")
    print(f"  Model: {pipeline.model_arch.upper()}  |  Threshold: {pipeline.threshold}")
    print(f"{'='*70}")

    for i, ex in enumerate(EXAMPLES, 1):
        result = pipeline.predict(
            answer    = ex["answer"],
            question  = ex["question"],
            knowledge = ex["knowledge"],
        )
        fr        = result["final_result"]
        label_str = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
        prob      = fr["probability"]

        print(f"\n[{i:02d}] {label_str}  (prob={prob:.3f})")
        print(f"     Q: {ex['question']}")
        print(f"     A: {ex['answer'][:100]}")


if __name__ == "__main__":
    main()
