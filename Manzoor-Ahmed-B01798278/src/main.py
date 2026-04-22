# src\main.py


import json
from pipeline import RuleBasedHallucinationPipeline


EXAMPLES = [
    # Entity hallucinations
    "The capital of Germany is Madrid.",
    "The capital of Pakistan is Islamabad.",
    "Albert Einstein was a professional footballer.",
    "Nelson Mandela was president of South Africa.",
    # Temporal hallucinations
    "He died in 1991 and published the book in 1998.",
    "Napoleon used the internet in 1804 to communicate with his generals.",
    "Julius Caesar tweeted about his campaigns in 50 BC.",
    "Shakespeare wrote Hamlet using his laptop in 1601.",
    # Contradiction
    "The meeting happened. The meeting never happened.",
    "She always succeeds. She never succeeds.",
    # Numerical
    "A human is 250 years old and weighs 5000 kg.",
    "The patient is 180 years old.",
    "Water boils at 100 degrees Celsius.",
    # Citation
    "This is supported by DOI ABC-123 and arXiv:wrong-id.",
    "The result was confirmed in arXiv:2310.12345.",
    # Clean factual
    "Python was created by Guido van Rossum and first released in 1991.",
]


def main():
    pipeline = RuleBasedHallucinationPipeline()

    for i, text in enumerate(EXAMPLES, start=1):
        result = pipeline.predict(text)
        fr = result["final_result"]
        label_str = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
        print(f"\n{'='*70}")
        print(f"[{i:02d}] {label_str}  (score={fr['final_score']:.3f})")
        print(f"     {text}")
        for t in fr["triggered_modules"]:
            print(f"     ⚠  [{t['module']}] {t['reason'][:80]}")


if __name__ == "__main__":
    main()
