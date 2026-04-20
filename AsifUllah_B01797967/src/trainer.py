# src/trainer.py

import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from transformers import get_linear_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model import TransformerClassifier, save_model


def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
) -> Dict[str, float]:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1":       round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def train_one_epoch(
    model:       TransformerClassifier,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler,
    criterion:   nn.Module,
    device:      torch.device,
    epoch:       int,
    total_epochs: int,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        lbls  = labels.detach().cpu().tolist()

        y_true.extend(lbls)
        y_pred.extend(preds)
        y_prob.extend(probs)

        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{len(loader)}  loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(y_true, y_pred, y_prob)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model:     TransformerClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, Dict[str, float], List[float]]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        lbls  = labels.detach().cpu().tolist()

        y_true.extend(lbls)
        y_pred.extend(preds)
        y_prob.extend(probs)

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(y_true, y_pred, y_prob)
    return avg_loss, metrics, y_prob


def fine_tune(
    model:         TransformerClassifier,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    model_name_str: str,
    learning_rate: float = 2e-5,
    num_epochs:    int   = 3,
    warmup_ratio:  float = 0.1,
    weight_decay:  float = 0.01,
    patience:      int   = 2,
    save_dir:      str   = "../models",
) -> Dict:
    device = get_device()
    print(f"  Device: {device}")
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=learning_rate)

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    if HAS_TRANSFORMERS:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_loss = float("inf")
    best_val_f1   = 0.0
    patience_count = 0
    best_epoch    = 0

    print(f"\n  Fine-tuning {model_name_str}")
    print(f"  Epochs: {num_epochs}  |  LR: {learning_rate}  |  "
          f"Warmup steps: {warmup_steps}/{total_steps}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n  --- Epoch {epoch}/{num_epochs} ---")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch, num_epochs
        )
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"  Train: loss={train_loss:.4f}  F1={train_metrics['f1']:.4f}  "
              f"Acc={train_metrics['accuracy']:.4f}")
        print(f"  Val:   loss={val_loss:.4f}    F1={val_metrics['f1']:.4f}  "
              f"Acc={val_metrics['accuracy']:.4f}  ({elapsed:.0f}s)")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1   = val_metrics["f1"]
            best_epoch    = epoch
            patience_count = 0
            ckpt_path = str(Path(save_dir) / f"{model_name_str}_best.pt")
            save_model(model, ckpt_path)
            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    print(f"\n  Best model: epoch {best_epoch}  val_loss={best_val_loss:.4f}  val_F1={best_val_f1:.4f}")

    return {
        "model_name":   model_name_str,
        "history":      history,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1":  best_val_f1,
    }
