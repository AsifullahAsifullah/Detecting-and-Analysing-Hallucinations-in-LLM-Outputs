# src/train.py
import json
import argparse
import time
from pathlib import Path
from collections import Counter

from config import SETTINGS
from utils import set_seed
from model import load_model_and_tokenizer, save_model
from dataset import load_json_dataset, split_dataset, make_dataloader
from trainer import fine_tune, evaluate, get_device
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import torch
import torch.nn as nn


ARCH_MODELS = {
    "bert":    "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
}


def print_split_info(train, val, test):
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        c = Counter(r["label"] for r in split)
        print(f"  {name}: {len(split)} samples  (0={c[0]}, 1={c[1]})")


def run_architecture(
    arch:       str,
    model_name: str,
    train_data: list,
    val_data:   list,
    test_data:  list,
    args,
    results_dir: Path,
    models_dir:  Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Architecture: {arch.upper()}  ({model_name})")
    print(f"{'='*60}")

    set_seed(SETTINGS.training.random_seed)

    print("  Loading model and tokeniser...")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        num_labels   = SETTINGS.model.num_labels,
        dropout_rate = SETTINGS.model.dropout_rate,
    )

    print("  Building dataloaders...")
    train_loader = make_dataloader(
        train_data, tokenizer,
        max_length   = SETTINGS.model.max_seq_length,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = SETTINGS.training.num_workers,
    )
    val_loader = make_dataloader(
        val_data, tokenizer,
        max_length   = SETTINGS.model.max_seq_length,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = SETTINGS.training.num_workers,
    )
    test_loader = make_dataloader(
        test_data, tokenizer,
        max_length   = SETTINGS.model.max_seq_length,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = SETTINGS.training.num_workers,
    )

    # Fine-tune
    history = fine_tune(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        model_name_str = arch,
        learning_rate  = args.learning_rate,
        num_epochs     = args.epochs,
        warmup_ratio   = SETTINGS.training.warmup_ratio,
        weight_decay   = SETTINGS.training.weight_decay,
        patience       = args.patience,
        save_dir       = str(models_dir),
    )

    ckpt_path = str(models_dir / f"{arch}_best.pt")
    from model import load_model as _load_model
    best_model = _load_model(ckpt_path)
    device     = get_device()
    best_model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics, test_probs = evaluate(
        best_model, test_loader, criterion, device
    )

    print(f"\n  [TEST SET — {arch.upper()}]")
    print(f"  Acc={test_metrics['accuracy']}  F1={test_metrics['f1']}  "
          f"ROC-AUC={test_metrics.get('roc_auc', 'N/A')}")

    result = {
        "arch":          arch,
        "model_name":    model_name,
        "training":      history,
        "test_metrics":  test_metrics,
        "test_loss":     round(test_loss, 4),
    }

    # Save individual results
    with open(results_dir / f"{arch}_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer for hallucination detection")
    parser.add_argument("--arch",         default="bert",
                        choices=["bert", "roberta", "deberta", "all"])
    parser.add_argument("--dataset",      default="../data/processed/combined_dataset.json")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--learning-rate",type=float, default=2e-5)
    parser.add_argument("--patience",     type=int,   default=2)
    parser.add_argument("--max-samples",  type=int,   default=0,
                        help="Limit dataset size for quick testing (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Deep Learning Hallucination Detector")
    print("  Transformer Fine-Tuning Pipeline")
    print("=" * 60)

    results_dir = Path("../results")
    models_dir  = Path("../models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\n[1] Loading dataset: {args.dataset}")
    data = load_json_dataset(args.dataset)
    print(f"  Total samples: {len(data)}")

    if args.max_samples and args.max_samples < len(data):
        import random
        random.seed(42)
        random.shuffle(data)
        data = data[:args.max_samples]
        print(f"  Using {len(data)} samples (--max-samples)")

    # Split
    print("\n[2] Splitting dataset (60/20/20)...")
    train_data, val_data, test_data = split_dataset(
        data,
        train_size  = SETTINGS.training.train_size,
        val_size    = SETTINGS.training.val_size,
        random_seed = SETTINGS.training.random_seed,
    )
    print_split_info(train_data, val_data, test_data)

    # Select architectures
    if args.arch == "all":
        archs_to_run = list(ARCH_MODELS.keys())
    else:
        archs_to_run = [args.arch]

    # Train each
    all_results = {}
    for arch in archs_to_run:
        model_name = ARCH_MODELS[arch]
        result = run_architecture(
            arch, model_name,
            train_data, val_data, test_data,
            args, results_dir, models_dir,
        )
        all_results[arch] = result

    # Summary
    print(f"\n{'='*60}")
    print("  ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Arch':<12} {'Val F1':>8} {'Test F1':>8} {'Test AUC':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for arch, r in all_results.items():
        vf1 = r["training"]["best_val_f1"]
        tf1 = r["test_metrics"]["f1"]
        auc = r["test_metrics"].get("roc_auc", 0.0)
        print(f"  {arch:<12} {vf1:>8.4f} {tf1:>8.4f} {auc:>10.4f}")

    # Save combined summary
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTraining results saved → results/training_results.json")
    print("\nNext steps:")
    print("  python evaluation.py  — full evaluation on test set")
    print("  python main.py        — demo predictions")
    print("  python attention_analysis.py  — attention interpretability")


if __name__ == "__main__":
    main()
