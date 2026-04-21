# src/config.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureConfig:
    """Feature engineering settings."""
    # Lexical
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 3)

    use_sentence_embeddings: bool = False   # set True if sentence-transformers installed
    sbert_model: str = "all-MiniLM-L6-v2"

    use_spacy: bool = False   # set True if spacy + en_core_web_sm installed

    # Confidence / hedging
    hedge_words: List[str] = field(default_factory=lambda: [
        "maybe", "perhaps", "possibly", "probably", "likely", "might",
        "could", "seems", "appears", "suggest", "believe", "think",
        "approximately", "around", "about", "roughly", "unclear",
        "uncertain", "unknown", "allegedly", "reportedly",
    ])
    certainty_words: List[str] = field(default_factory=lambda: [
        "definitely", "certainly", "absolutely", "clearly", "obviously",
        "undoubtedly", "always", "never", "every", "all", "none",
        "guaranteed", "proven", "confirmed", "established",
    ])


@dataclass
class ModelConfig:
    """Model training and evaluation settings."""
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    threshold: float = 0.5

    # Optuna hyperparameter tuning
    n_trials: int = 30
    tune_models: List[str] = field(default_factory=lambda: [
        "random_forest", "xgboost", "lightgbm", "svm"
    ])


@dataclass
class Settings:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    debug: bool = False


SETTINGS = Settings()
