# src/investigate_dataset.py

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def safe_len(x: Any) -> int:
    if x is None:
        return 0
    return len(str(x).strip())


def summarize_lengths(series: pd.Series) -> Dict[str, float]:
    vals = series.fillna("").astype(str).map(len)
    return {
        "min": int(vals.min()) if len(vals) else 0,
        "p25": float(vals.quantile(0.25)) if len(vals) else 0.0,
        "median": float(vals.median()) if len(vals) else 0.0,
        "p75": float(vals.quantile(0.75)) if len(vals) else 0.0,
        "max": int(vals.max()) if len(vals) else 0,
        "mean": round(float(vals.mean()), 2) if len(vals) else 0.0,
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    for col in ["source", "task_type", "text", "question", "knowledge", "label", "hallucination_type"]:
        if col not in df.columns:
            df[col] = ""

    df["text_len"] = df["text"].fillna("").astype(str).map(len)
    df["question_len"] = df["question"].fillna("").astype(str).map(len)
    df["knowledge_len"] = df["knowledge"].fillna("").astype(str).map(len)
    return df


def show_dataset_summary(df: pd.DataFrame) -> None:
    print_section("DATASET OVERVIEW")
    print(f"Total rows: {len(df):,}")

    print("\nSource counts:")
    print(df["source"].value_counts(dropna=False).to_string())

    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False).sort_index().to_string())

    print("\nTask-type counts:")
    print(df["task_type"].value_counts(dropna=False).to_string())

    print("\nSource × label:")
    src_label = pd.crosstab(df["source"], df["label"], dropna=False)
    print(src_label.to_string())

    print("\nSource × task_type:")
    src_task = pd.crosstab(df["source"], df["task_type"], dropna=False)
    print(src_task.to_string())


def show_length_summary(df: pd.DataFrame) -> None:
    print_section("LENGTH STATS")

    print("Overall text length:")
    print(summarize_lengths(df["text"]))

    if "question" in df.columns:
        has_q = df["question"].fillna("").astype(str).str.strip() != ""
        print("\nRows with question:", int(has_q.sum()))
        print("Question length stats:")
        print(summarize_lengths(df.loc[has_q, "question"]))

    if "knowledge" in df.columns:
        has_k = df["knowledge"].fillna("").astype(str).str.strip() != ""
        print("\nRows with knowledge:", int(has_k.sum()))
        print("Knowledge length stats:")
        print(summarize_lengths(df.loc[has_k, "knowledge"]))

    print("\nLength stats by source:")
    for src, g in df.groupby("source"):
        print(f"\n[{src}]")
        print("text      :", summarize_lengths(g["text"]))
        if "question" in g.columns:
            q_mask = g["question"].fillna("").astype(str).str.strip() != ""
            print("question  :", summarize_lengths(g.loc[q_mask, "question"]))
        if "knowledge" in g.columns:
            k_mask = g["knowledge"].fillna("").astype(str).str.strip() != ""
            print("knowledge :", summarize_lengths(g.loc[k_mask, "knowledge"]))


def show_hallucinated_only_summary(df: pd.DataFrame) -> None:
    print_section("HALLUCINATED ROWS ONLY")
    hall = df[df["label"] == 1].copy()
    print(f"Total hallucinated rows: {len(hall):,}")

    print("\nHallucinated rows by source:")
    print(hall["source"].value_counts(dropna=False).to_string())

    print("\nHallucinated text length by source:")
    for src, g in hall.groupby("source"):
        print(f"{src}: {summarize_lengths(g['text'])}")


def load_errors(errors_path: str) -> pd.DataFrame:
    if not Path(errors_path).exists():
        return pd.DataFrame()

    df = pd.read_csv(errors_path)

    for col in ["source", "error_type", "text", "question", "knowledge", "triggered_modules", "score", "label", "prediction"]:
        if col not in df.columns:
            df[col] = ""

    df["text_len"] = df["text"].fillna("").astype(str).map(len)
    df["question_len"] = df["question"].fillna("").astype(str).map(len) if "question" in df.columns else 0
    df["knowledge_len"] = df["knowledge"].fillna("").astype(str).map(len) if "knowledge" in df.columns else 0
    return df


def show_error_summary(err: pd.DataFrame) -> None:
    print_section("ERROR SUMMARY")

    if err.empty:
        print("No errors.csv found or file is empty.")
        return

    print(f"Total error rows: {len(err):,}")

    print("\nError type counts:")
    print(err["error_type"].value_counts(dropna=False).to_string())

    print("\nSource × error_type:")
    src_err = pd.crosstab(err["source"], err["error_type"], dropna=False)
    print(src_err.to_string())

    print("\nTriggered modules (top 20):")
    trig = (
        err["triggered_modules"]
        .fillna("")
        .astype(str)
        .replace("", "[none]")
        .value_counts()
        .head(20)
    )
    print(trig.to_string())

    print("\nAverage score by source and error_type:")
    score_tbl = (
        err.groupby(["source", "error_type"])["score"]
        .mean()
        .round(4)
        .reset_index()
    )
    print(score_tbl.to_string(index=False))

    print("\nFN text length by source:")
    fn = err[err["error_type"] == "FN"].copy()
    if len(fn):
        for src, g in fn.groupby("source"):
            print(f"{src}: {summarize_lengths(g['text'])}")

    print("\nFP text length by source:")
    fp = err[err["error_type"] == "FP"].copy()
    if len(fp):
        for src, g in fp.groupby("source"):
            print(f"{src}: {summarize_lengths(g['text'])}")


def print_samples(err: pd.DataFrame, source: str, error_type: str, n: int = 10) -> None:
    subset = err[(err["source"] == source) & (err["error_type"] == error_type)].head(n)
    print_section(f"SAMPLES — {source} / {error_type} / first {len(subset)}")

    if subset.empty:
        print("No rows found.")
        return

    cols = ["id", "score", "label", "prediction", "triggered_modules", "question", "knowledge", "text"]
    for _, row in subset.iterrows():
        print("-" * 80)
        for c in cols:
            if c in subset.columns:
                val = row.get(c, "")
                print(f"{c}: {val}")


def save_summary_csvs(df: pd.DataFrame, err: pd.DataFrame, out_dir: str = "../results") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df["has_question"] = df["question"].fillna("").astype(str).str.strip() != ""
    df["has_knowledge"] = df["knowledge"].fillna("").astype(str).str.strip() != ""

    dataset_summary = (
        df.groupby(["source", "label"])
        .agg(
            rows=("id", "count"),
            avg_text_len=("text_len", "mean"),
            avg_question_len=("question_len", "mean"),
            avg_knowledge_len=("knowledge_len", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(out / "dataset_summary.csv", index=False)

    if not err.empty:
        error_summary = (
            err.groupby(["source", "error_type"])
            .agg(
                rows=("id", "count"),
                avg_score=("score", "mean"),
                avg_text_len=("text_len", "mean"),
            )
            .reset_index()
        )
        error_summary.to_csv(out / "error_summary.csv", index=False)

    print_section("SAVED SUMMARY FILES")
    print(f"Saved: {out / 'dataset_summary.csv'}")
    if not err.empty:
        print(f"Saved: {out / 'error_summary.csv'}")


def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../data/processed/combined_dataset.json"
    errors_path = sys.argv[2] if len(sys.argv) > 2 else "../results/errors.csv"

    df = load_dataset(dataset_path)
    show_dataset_summary(df)
    show_length_summary(df)
    show_hallucinated_only_summary(df)

    err = load_errors(errors_path)
    show_error_summary(err)

    if not err.empty:
        print_samples(err, "HaluEval", "FN", 10)
        print_samples(err, "FEVER", "FN", 10)
        print_samples(err, "TruthfulQA", "FP", 10)
        print_samples(err, "FEVER", "FP", 10)

    save_summary_csvs(df, err)


if __name__ == "__main__":
    main()