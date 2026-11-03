"""Paths and constants shared by the training scripts and the API.

Everything that can be overridden from the environment is read through a function
(not a module-level constant) so tests and deployments can change it at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reproducibility: one seed and one split definition for every script.
SEED = 42
TEST_SIZE = 0.2

# The dataset after loading with pandas' default parser (see data.py for why 5572, not 5574).
EXPECTED_ROWS = 5572
EXPECTED_HAM = 4825
EXPECTED_SPAM = 747


def data_path() -> Path:
    return Path(os.getenv("SMS_DATA_PATH", REPO_ROOT / "data" / "SMSSpamCollection"))


def models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", REPO_ROOT / "models"))


def results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", REPO_ROOT / "results"))


def pipeline_path() -> Path:
    return models_dir() / "pipeline.pkl"


def pipeline_meta_path() -> Path:
    return models_dir() / "pipeline_meta.json"


def classical_dir() -> Path:
    return models_dir() / "classical"


def distilbert_dir() -> Path:
    return models_dir() / "distilbert"


def article_data_dir() -> Path:
    return Path(os.getenv("ARTICLE_DATA_DIR", REPO_ROOT / "data" / "article_tasks"))


def article_models_dir() -> Path:
    return Path(os.getenv("ARTICLE_MODELS_DIR", models_dir() / "article_tasks"))
