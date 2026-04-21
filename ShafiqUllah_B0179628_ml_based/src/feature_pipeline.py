# src/feature_pipeline.py


import re
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

from modules.lexical_features import extract_lexical_features
from modules.semantic_features import extract_semantic_features
from modules.linguistic_features import extract_linguistic_features
from modules.confidence_features import extract_confidence_features
from utils import normalize_text


def _parse_row(row: dict) -> Tuple[str, str, str]:
    """Extract answer, question, knowledge from a dataset row."""
    answer = normalize_text(str(row.get("text", "")))
    question = normalize_text(str(row.get("question", "")))
    knowledge = normalize_text(str(row.get("knowledge", "")))
    return answer, question, knowledge


def extract_handcrafted_features(
    answer: str,
    question: str = "",
    knowledge: str = "",
) -> Dict[str, float]:
    
    lex = extract_lexical_features(answer)
    sem = extract_semantic_features(answer, question, knowledge)
    ling = extract_linguistic_features(answer)
    conf = extract_confidence_features(answer)

    combined = {}
    combined.update(lex)
    combined.update(sem)
    combined.update(ling)
    combined.update(conf)
    return combined


def features_to_vector(feat_dict: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Convert feature dict to numpy array using given feature name ordering."""
    return np.array([feat_dict.get(name, 0.0) for name in feature_names], dtype=np.float32)


class FeaturePipeline:
    

    def __init__(
        self,
        tfidf_max_features: int = 3000,
        tfidf_ngram_range: tuple = (1, 2),
    ):
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range

        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            stop_words=None,  # keep all words for hallucination signals
        )
        self.scaler = StandardScaler()
        self.feature_names_handcrafted: List[str] = []
        self.is_fitted = False

    def _extract_batch(self, rows: List[dict]) -> Tuple[List[str], np.ndarray]:
        
        texts = []
        handcrafted_rows = []

        for row in rows:
            answer, question, knowledge = _parse_row(row)
            feats = extract_handcrafted_features(answer, question, knowledge)
            handcrafted_rows.append(feats)
            texts.append(answer)

        
        if handcrafted_rows and not self.feature_names_handcrafted:
            self.feature_names_handcrafted = sorted(handcrafted_rows[0].keys())

        hc_matrix = np.array(
            [features_to_vector(f, self.feature_names_handcrafted) for f in handcrafted_rows],
            dtype=np.float32,
        )

        return texts, hc_matrix

    def fit_transform(self, rows: List[dict]) -> np.ndarray:
        """Fit on training data and return feature matrix."""
        print(f"  Extracting handcrafted features for {len(rows)} samples...")
        texts, hc_matrix = self._extract_batch(rows)

        print("  Fitting TF-IDF vectorizer...")
        tfidf_matrix = self.tfidf.fit_transform(texts)

        print("  Fitting StandardScaler on handcrafted features...")
        hc_scaled = self.scaler.fit_transform(hc_matrix)

        print(f"  Handcrafted features: {hc_matrix.shape[1]}")
        print(f"  TF-IDF features: {tfidf_matrix.shape[1]}")

        tfidf_dense = tfidf_matrix.toarray()
        X = np.hstack([hc_scaled, tfidf_dense])
        
        self._all_feature_names = self.get_feature_names()
        print(f"  Total feature vector size: {X.shape[1]}")
        self.is_fitted = True
    
        import pandas as pd
        return pd.DataFrame(X, columns=self._all_feature_names)

    def transform(self, rows: List[dict]) -> np.ndarray:
        """Transform new data using fitted vectorizer and scaler."""
        assert self.is_fitted, "Call fit_transform first."
        texts, hc_matrix = self._extract_batch(rows)
        tfidf_matrix = self.tfidf.transform(texts)
        hc_scaled = self.scaler.transform(hc_matrix)
        tfidf_dense = tfidf_matrix.toarray()
        X = np.hstack([hc_scaled, tfidf_dense])
        import pandas as pd
        return pd.DataFrame(X, columns=self._all_feature_names)

    def transform_single(self, answer: str, question: str = "", knowledge: str = "") -> np.ndarray:
        """Transform a single sample for inference."""
        row = {"text": answer, "question": question, "knowledge": knowledge}
        return self.transform([row])

    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Feature pipeline saved → {path}")

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        """Load a fitted pipeline from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_feature_names(self) -> List[str]:
        """Return all feature names (handcrafted + tfidf)."""
        hc_names = self.feature_names_handcrafted
        tfidf_names = [f"tfidf_{t}" for t in self.tfidf.get_feature_names_out()]
        return hc_names + tfidf_names
