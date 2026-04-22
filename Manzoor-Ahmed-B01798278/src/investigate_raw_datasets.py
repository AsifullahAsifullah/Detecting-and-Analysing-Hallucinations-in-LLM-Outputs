# src/investigate_raw_datasets.py

import json
import pandas as pd
from pathlib import Path


def inspect_truthfulqa(path):
    print("\n" + "="*60)
    print("TRUTHFULQA")
    print("="*60)

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nSample:")
    print(df[["Question", "Best Answer"]].head(5))


def inspect_fever(path, n=10):
    print("\n" + "="*60)
    print("FEVER")
    print("="*60)

    labels = {}
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)

            label = item.get("label", "")
            labels[label] = labels.get(label, 0) + 1

            if i < n:
                samples.append(item)

    print("Label distribution:")
    for k, v in labels.items():
        print(f"{k}: {v}")

    print("\nSample claims:")
    for s in samples[:5]:
        print("-"*40)
        print("Label:", s.get("label"))
        print("Claim:", s.get("claim"))


def inspect_halueval(path, n=5):
    print("\n" + "="*60)
    print("HALUEVAL")
    print("="*60)

    count = 0
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            count += 1

            if len(samples) < n:
                samples.append(item)

    print("Rows:", count)

    print("\nSample entries:")
    for s in samples:
        print("-"*60)
        print("Question:", s.get("question"))
        print("Knowledge:", s.get("knowledge")[:100], "...")
        print("Right:", s.get("right_answer"))
        print("Hallucinated:", s.get("hallucinated_answer"))


if __name__ == "__main__":
    inspect_truthfulqa("../data/raw/truthfulqa/TruthfulQA.csv")
    inspect_fever("../data/raw/fever/train.jsonl")
    inspect_halueval("../data/raw/halueval/qa_data.json")