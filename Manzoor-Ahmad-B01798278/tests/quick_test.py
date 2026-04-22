# tests\quick_test.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import RuleBasedHallucinationPipeline


TEST_CASES = [
    # ── Entity ────────────────────────────────────────────────────────────────
    ("The capital of Germany is Berlin.",               0),
    ("The capital of Germany is Madrid.",               1),
    ("The capital of Pakistan is Islamabad.",           0),
    ("The capital of France is Berlin.",                1),
    ("The capital of Australia is Sydney.",             1),  # Canberra
    ("Albert Einstein was a professional footballer.",  1),
    ("Nelson Mandela was president of South Africa.",   0),
    ("Marie Curie was a physicist.",                    0),
    ("Isaac Newton was a famous singer.",               1),

    # ── Temporal ─────────────────────────────────────────────────────────────
    ("He died in 1991 and published the book in 1998.", 1),
    ("She was born in 1990 and died in 1985.",          1),
    ("Napoleon used the internet in 1804.",             1),
    ("Julius Caesar tweeted about his campaigns in 50 BC.", 1),
    ("Shakespeare wrote Hamlet using his laptop in 1601.", 1),
    ("The iPhone was released in 2007.",                0),
    ("Python was created by Guido van Rossum in 1991.", 0),

    # ── Contradiction ─────────────────────────────────────────────────────────
    ("The meeting happened. The meeting never happened.", 1),
    ("She always succeeds. She never succeeds.",          1),
    ("The company was founded. The company was never founded.", 1),
    ("Water is always wet.",                              0),

    # ── Numerical ────────────────────────────────────────────────────────────
    ("A human is 250 years old.",                       1),
    ("The patient is 180 years old.",                   1),
    ("A human is 25 years old.",                        0),
    ("The population is 500 billion according to the report.", 1),
    ("Water boils at 100 degrees Celsius.",             0),

    # ── Citation ─────────────────────────────────────────────────────────────
    ("This is supported by DOI ABC-123.",               1),
    ("This is supported by arXiv:wrong-id.",            1),
    ("The result confirmed in arXiv:2310.12345.",       0),
    ("Published in the Journal of Advanced Studies in Everything.", 1),
    ("Mount Everest is 8849 metres tall.",              0),
]


def run_quick_test():
    pipeline = RuleBasedHallucinationPipeline()
    passed = failed = 0

    print(f"{'TEXT':<58} EXP  PRED  SCORE  STATUS")
    print("─" * 90)

    for text, expected in TEST_CASES:
        result = pipeline.predict(text)
        pred   = result["final_result"]["label"]
        score  = result["final_result"]["final_score"]
        ok     = "✓" if pred == expected else "✗  ← FAIL"

        if pred == expected:
            passed += 1
        else:
            failed += 1

        print(f"{text[:57]:<58} {expected}    {pred}   {score:.3f}  {ok}")

    print()
    print(f"Result: {passed}/{len(TEST_CASES)} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print(" — all passed ✓")


if __name__ == "__main__":
    run_quick_test()
