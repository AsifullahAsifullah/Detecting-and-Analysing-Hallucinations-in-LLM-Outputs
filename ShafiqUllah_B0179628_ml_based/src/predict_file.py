# src/predict_file.py

import sys
from pathlib import Path
from pipeline import MLHallucinationPipeline


INPUTS = [
    {
        "id": "1",
        "text": "Napoleon used the internet in 1804 to communicate with his generals.",
        "question": "",
        "knowledge": ""
    },
    {
        "id": "2",
        "text": "The capital of France is Berlin.",
        "question": "What is the capital of France?",
        "knowledge": "France is a country in Western Europe. Its capital and largest city is Paris."
    },
    {
        "id": "3",
        "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "question": "At what temperature does water boil?",
        "knowledge": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level."
    },
    {
        "id": "4",
        "text": "First for Women was started first.",
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "knowledge": "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia."
    },
    {
        "id": "5",
        "text": "Albert Einstein was a professional footballer who played for the German national team.",
        "question": "Who was Albert Einstein?",
        "knowledge": "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
    },
    {
        "id": "6",
        "text": "The Great Wall of China is clearly visible from the Moon with the naked eye.",
        "question": "Can you see the Great Wall of China from space?",
        "knowledge": "The Great Wall of China is approximately 5 to 8 metres wide. Astronauts have reported it is not visible from the Moon."
    },
    {
        "id": "7",
        "text": "Python was created by Guido van Rossum and first released in 1991.",
        "question": "Who created the Python programming language?",
        "knowledge": "Python is a high-level programming language created by Guido van Rossum. The first version was released in 1991."
    },
    {
        "id": "8",
        "text": "The moon is made of cheese.",
        "question": "What is the moon made of?",
        "knowledge": "The Moon is a rocky body composed primarily of silicate minerals and metals."
    },
    {
        "id": "9",
        "text": "Shakespeare wrote Hamlet in 1601 using his laptop computer.",
        "question": "When did Shakespeare write Hamlet?",
        "knowledge": "William Shakespeare wrote Hamlet around 1600-1601. Personal computers were not invented until the 20th century."
    },
    {
        "id": "10",
        "text": "The Earth orbits the Sun once every 365.25 days.",
        "question": "How long does it take for the Earth to orbit the Sun?",
        "knowledge": "The Earth takes approximately 365.25 days to complete one orbit around the Sun."
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def predict_one(pipeline, item_id, text, question="", knowledge=""):
    """Run prediction on one item and print a clear result."""
    r = pipeline.predict(answer=text, question=question, knowledge=knowledge)
    fr = r["final_result"]
    label = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
    prob  = fr["probability"]

    RED   = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    colour = RED if fr["label"] == 1 else GREEN

    print(f"\n  {'─' * 62}")
    print(f"  [{item_id}] {colour}{label}{RESET}  (prob={prob:.3f})")
    print(f"  Text     : {text[:100]}")
    if question:
        print(f"  Question : {question[:80]}")
    if knowledge:
        print(f"  Knowledge: {knowledge[:80]}{'...' if len(knowledge) > 80 else ''}")
    print(f"  Model    : {fr['model']}  |  Threshold: {fr['threshold']}")


def run_file_mode(pipeline):
    """Run predictions on the hardcoded INPUTS list."""
    print("\n" + "=" * 65)
    print("  MODE: File  —  running hardcoded examples")
    print("=" * 65)

    for item in INPUTS:
        predict_one(
            pipeline,
            item["id"],
            item["text"],
            item.get("question", ""),
            item.get("knowledge", "")
        )

    print(f"\n  {'─' * 62}")
    print(f"  Done. {len(INPUTS)} examples processed.\n")


def run_terminal_mode(pipeline):
    """Ask user to type text interactively in the terminal."""
    print("\n" + "=" * 65)
    print("  MODE: Terminal  —  type your own text")
    print("  Type 'quit' or press Enter on an empty line to exit.")
    print("=" * 65)

    count = 0
    while True:
        print()
        try:
            text = input("  Text to check: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting. Goodbye!\n")
            break

        if text.lower() in ("quit", "exit", "q") or text == "":
            print("\n  Exiting. Goodbye!\n")
            break

        count += 1
        question  = input("  Question  (Enter to skip): ").strip()
        knowledge = input("  Knowledge (Enter to skip): ").strip()

        predict_one(pipeline, count, text, question, knowledge)


def main():
    model_path    = "../models/best_model.pkl"
    pipeline_path = "../models/feature_pipeline.pkl"

    if not Path(model_path).exists():
        print("\nERROR: No trained model found at ../models/best_model.pkl")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)

    print("\nLoading ML pipeline...")
    pipeline = MLHallucinationPipeline.load(model_path, pipeline_path)
    print(f"Model loaded: {pipeline.model_name}  |  Threshold: {pipeline.threshold}")

    print("\n" + "=" * 65)
    print("  ML HALLUCINATION DETECTOR — Predict Mode")
    print("=" * 65)
    print("\n  Choose mode:")
    print("  [1] Run hardcoded examples from file")
    print("  [2] Type your own text in terminal")
    print("  [3] Both")

    try:
        choice = input("\n  Your choice (1 / 2 / 3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        run_file_mode(pipeline)
    elif choice == "2":
        run_terminal_mode(pipeline)
    elif choice == "3":
        run_file_mode(pipeline)
        run_terminal_mode(pipeline)
    else:
        print("\n  Invalid choice — running file mode by default.")
        run_file_mode(pipeline)


if __name__ == "__main__":
    main()
