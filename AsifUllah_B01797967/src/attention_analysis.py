# src/attention_analysis.py
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch

from pipeline import DeepLearningHallucinationPipeline
from utils import format_input, normalize_text
from config import SETTINGS


def get_token_attention(
    model,
    tokenizer,
    text: str,
    device: torch.device,
) -> Dict:
    encoding = tokenizer(
        text,
        max_length=SETTINGS.model.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get tokens (non-padding)
    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0].tolist()
    )
    seq_len = int(attention_mask[0].sum().item())
    tokens = tokens[:seq_len]

    # Get attention weights
    with torch.no_grad():
        attentions = model.get_attention_weights(input_ids, attention_mask)

    if attentions is None:
        return {"tokens": tokens, "attention_available": False}

    all_layers = []
    for layer_attn in attentions:
        layer_mean = layer_attn[0].mean(dim=0)[0, :seq_len]
        all_layers.append(layer_mean.cpu())

    # Mean across layers
    stacked = torch.stack(all_layers, dim=0)
    mean_attn = stacked.mean(dim=0)
    mean_attn = mean_attn / (mean_attn.sum() + 1e-9)

    # Top attended tokens (excluding special tokens)
    special = {"[cls]", "[sep]", "<s>", "</s>", "<pad>", "[pad]"}
    scored = [
        (tok.lower(), float(mean_attn[i]))
        for i, tok in enumerate(tokens)
        if tok.lower() not in special
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_tokens = scored[:10]

    return {
        "tokens": tokens,
        "attention_scores": mean_attn.tolist(),
        "top_tokens": top_tokens,
        "attention_available": True,
        "num_layers": len(attentions),
        "num_heads": attentions[0].shape[1],
    }


def analyse_sample(
    pipeline: DeepLearningHallucinationPipeline,
    row: dict,
    idx: int,
) -> Dict:
    """Analyse one sample: predict + extract attention."""
    answer = normalize_text(str(row.get("text", "")))
    question = normalize_text(str(row.get("question", "")))
    knowledge = normalize_text(str(row.get("knowledge", "")))
    true_label = row.get("label", -1)

    # Predict
    result = pipeline.predict(answer=answer, question=question, knowledge=knowledge)
    fr = result["final_result"]
    pred_label = fr["label"]
    prob = fr["probability"]

    # Format input text
    input_text = format_input(answer, question, knowledge)

    # Attention
    attn = get_token_attention(
        pipeline.model,
        pipeline.tokenizer,
        input_text,
        pipeline.device,
    )

    label_str = "HALLUCINATED" if pred_label == 1 else "FACTUAL"
    true_str = "HALLUCINATED" if true_label == 1 else "FACTUAL"
    correct_str = "✓" if pred_label == true_label else "✗"

    print(
        f"\n  [{idx:02d}] Pred={label_str} (prob={prob:.3f})  "
        f"True={true_str}  {correct_str}"
    )
    print(f"       A: {answer[:80]}")
    if attn["attention_available"] and attn["top_tokens"]:
        top_str = ", ".join(f"'{t}'({s:.3f})" for t, s in attn["top_tokens"][:5])
        print(f"       Top attended: {top_str}")

    return {
        "idx": idx,
        "answer": answer[:120],
        "true_label": true_label,
        "pred_label": pred_label,
        "probability": prob,
        "correct": pred_label == true_label,
        "top_tokens": attn.get("top_tokens", []),
        "num_layers": attn.get("num_layers", 0),
        "num_heads": attn.get("num_heads", 0),
    }


def compare_attention_patterns(results: List[Dict]) -> None:
    """Compare top attended tokens for hallucinated vs factual samples."""
    hall_tokens = {}
    fact_tokens = {}

    for r in results:
        bucket = hall_tokens if r["pred_label"] == 1 else fact_tokens
        for tok, score in r.get("top_tokens", []):
            bucket[tok] = bucket.get(tok, 0.0) + score

    def top_n(d, n=10):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

    print("\n  Top tokens attended in HALLUCINATED predictions:")
    for tok, score in top_n(hall_tokens):
        print(f"    '{tok}'  cumulative_attn={score:.3f}")

    print("\n  Top tokens attended in FACTUAL predictions:")
    for tok, score in top_n(fact_tokens):
        print(f"    '{tok}'  cumulative_attn={score:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../models/bert_best.pt")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--dataset", default="../data/processed/combined_dataset.json")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import json as _json

    print("=" * 60)
    print("  Attention-Based Interpretability Analysis")
    print("=" * 60)

    # Load pipeline
    print(f"\nLoading model from {args.model_path}...")
    pipeline = DeepLearningHallucinationPipeline.load(
        model_path=args.model_path,
        model_name=args.model_name,
        threshold=args.threshold,
    )

    # Load sample data
    with open(args.dataset, "r", encoding="utf-8") as f:
        data = _json.load(f)

    # Pick balanced sample
    hall_rows = [r for r in data if r["label"] == 1][:args.n_samples // 2]
    fact_rows = [r for r in data if r["label"] == 0][:args.n_samples // 2]
    sample = hall_rows + fact_rows

    print(
        f"\nAnalysing {len(sample)} samples "
        f"({len(hall_rows)} hallucinated, {len(fact_rows)} factual)..."
    )

    results = []
    for i, row in enumerate(sample, 1):
        r = analyse_sample(pipeline, row, i)
        results.append(r)

    print(f"\n{'─' * 60}")
    compare_attention_patterns(results)

    # Save results
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    correct = sum(1 for r in results if r["correct"])
    summary = {
        "model_arch": pipeline.model_arch,
        "n_samples": len(results),
        "accuracy": round(correct / len(results), 4),
        "sample_results": results,
    }
    out_path = results_dir / "attention_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Accuracy on {len(results)} samples: {correct}/{len(results)}")
    print(f"  Attention analysis saved → {out_path}")
    print("\nNote: For full BertViz visualisations, install bertviz:")
    print("  pip install bertviz")


if __name__ == "__main__":
    main()