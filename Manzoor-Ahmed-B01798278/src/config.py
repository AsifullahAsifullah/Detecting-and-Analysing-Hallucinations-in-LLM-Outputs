# src/config.py
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Weights:
    knowledge:     float = 0.30
    entity:        float = 0.18
    temporal:      float = 0.12
    contradiction: float = 0.16
    numerical:     float = 0.14
    citation:      float = 0.10


@dataclass
class Thresholds:
    final_label_threshold: float = 0.06


@dataclass
class Settings:
    weights:          Weights    = field(default_factory=Weights)
    thresholds:       Thresholds = field(default_factory=Thresholds)
    enable_spacy_ner: bool       = True
    debug:            bool       = False

    known_facts: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "Germany": {"capital": "Berlin"},
        "France": {"capital": "Paris"},
        "Pakistan": {"capital": "Islamabad"},
        "Italy": {"capital": "Rome"},
        "Spain": {"capital": "Madrid"},
        "Japan": {"capital": "Tokyo"},
        "Albert Einstein": {"profession": "physicist"},
        "Nelson Mandela": {"country_role": "South Africa"},
    })

    plausible_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "percentage": (0.0, 100.0),
        "human_age": (0.0, 130.0),
        "temperature_c": (-100.0, 100.0),
        "earth_population_billion": (0.0, 20.0),
        "human_weight_kg": (1.0, 700.0),
    })


SETTINGS = Settings()