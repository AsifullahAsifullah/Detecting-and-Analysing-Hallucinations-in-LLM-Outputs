# src/predict_file.py
from pipeline import RuleBasedHallucinationPipeline

INPUTS = [
    {
        "id": "1",
        "text": "Napoleon used the internet in 1804.",
        "question": "",
        "knowledge": ""
    },
    {
        "id": "2",
        "text": "The capital of France is Berlin.",
        "question": "",
        "knowledge": ""
    },
    {
        "id": "3",
        "text": "Water boils at 100 degrees Celsius.",
        "question": "",
        "knowledge": ""
    },
    {
        "id": "4",
        "text": "First for Women was started first.",
        "question": "Which magazine was started first?",
        "knowledge": "Arthur's Magazine (1844) was an American literary periodical."
    },
    {
        "id": "5",
        "text": "Albert Einstein was a professional footballer.",
        "question": "",
        "knowledge": ""
    },
]


def predict_one(pipeline, item_id, text, question="", knowledge=""):
    r = pipeline.predict(answer=text, question=question, knowledge=knowledge)
    fr = r["final_result"]
    label = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
    score = fr["final_score"]

    print(f"\n  {'─'*60}")
    print(f"  [{item_id}] {label}  (score={score})")
    print(f"  Text     : {text}")
    if question:
        print(f"  Question : {question}")
    if knowledge:
        print(f"  Knowledge: {knowledge[:70]}...")

    if fr["triggered_modules"]:
        for t in fr["triggered_modules"]:
            print(f"  ⚠ [{t['module']}] {t['reason'][:70]}")
    else:
        print("  ✓ No hallucination signal detected.")


def run_file_mode(pipeline):
    print("\n" + "="*65)
    print("  MODE: File  —  running hardcoded examples")
    print("="*65)
    for item in INPUTS:
        predict_one(
            pipeline,
            item["id"],
            item["text"],
            item.get("question", ""),
            item.get("knowledge", "")
        )
    print(f"\n  {'─'*60}")
    print(f"  Done. {len(INPUTS)} examples processed.\n")


def run_terminal_mode(pipeline):
    print("\n" + "="*65)
    print("  MODE: Terminal  —  type your own text")
    print("  Type 'quit' or press Enter on empty line to exit.")
    print("="*65)

    count = 0
    while True:
        print()
        text = input("  Enter text to check: ").strip()

        if text.lower() in ("quit", "exit", "q") or text == "":
            print("\n  Exiting. Goodbye!\n")
            break

        count += 1

        question = input("  Question (press Enter to skip): ").strip()
        knowledge = input("  Knowledge (press Enter to skip): ").strip()

        predict_one(pipeline, count, text, question, knowledge)


def main():
    pipeline = RuleBasedHallucinationPipeline()

    print("\n" + "="*65)
    print("  HALLUCINATION DETECTOR — Predict Mode")
    print("="*65)
    print("\n  Choose mode:")
    print("  [1] Run hardcoded examples from file")
    print("  [2] Type your own text in terminal")
    print("  [3] Both")

    choice = input("\n  Your choice (1/2/3): ").strip()

    if choice == "1":
        run_file_mode(pipeline)
    elif choice == "2":
        run_terminal_mode(pipeline)
    elif choice == "3":
        run_file_mode(pipeline)
        run_terminal_mode(pipeline)
    else:
        print("\n  Invalid choice. Running file mode by default.")
        run_file_mode(pipeline)


if __name__ == "__main__":
    main()