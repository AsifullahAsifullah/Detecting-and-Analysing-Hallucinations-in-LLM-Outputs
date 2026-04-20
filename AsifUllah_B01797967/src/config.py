# src/config.py

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    """Transformer model settings."""
    architectures: Dict[str, str] = field(default_factory=lambda: {
        "bert":    "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
    })

    default_model: str = "bert"

    # Tokenisation
    max_seq_length: int = 256
    num_labels: int = 2            # binary: 0=factual, 1=hallucinated

    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Splits: 60% train, 20% val, 20% test
    train_size: float = 0.60
    val_size:   float = 0.20
    test_size:  float = 0.20

    # Per-architecture defaults
    learning_rate: float = 2e-5
    batch_size:    int   = 16
    num_epochs:    int   = 3
    warmup_ratio:  float = 0.1
    weight_decay:  float = 0.01

    # Early stopping
    patience:      int   = 2

    random_seed:   int   = 42
    num_workers:   int   = 0       # 0 = main process (safe for all OS)


@dataclass
class Settings:
    model:    ModelConfig    = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    debug:    bool           = False


SETTINGS = Settings()
