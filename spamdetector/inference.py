"""Model loading and scoring. Everything here returns spam scores on a 0-100 scale.

torch / transformers are imported lazily, so the TF-IDF part of the project (training, API,
tests) works with the light ``requirements.txt`` install alone.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .preprocess import preprocess

log = logging.getLogger("spamdetector")

MAX_LEN = 128  # must match the value used when fine-tuning (scripts/train_distilbert.py)

# Startup sanity check for the fine-tuned transformer.
# The original check only disabled the model if the spam probability on an obvious spam text
# was < 10 %. But the exact failure that bit the original project (saving a model whose
# classification head was never trained) gives ~50 % on EVERYTHING, which passes that check.
# We now require a confident answer in BOTH directions.
SANITY_SPAM_TEXT = "Congratulations! You won a free prize. Click now to claim!"
SANITY_HAM_TEXT = "Hey, are we still meeting at 6 for dinner?"
SANITY_SPAM_MIN = 0.80
SANITY_HAM_MAX = 0.20


class TfidfScorer:
    """TF-IDF + classifier pipeline saved by ``scripts/train_classical.py``."""

    def __init__(self, pipeline, label: str = "TF-IDF classifier"):
        self.pipeline = pipeline
        self.label = label

    @classmethod
    def load(cls, path: Path | None = None, meta_path: Path | None = None) -> "TfidfScorer":
        path = path or config.pipeline_path()
        meta_path = meta_path or config.pipeline_meta_path()
        with open(path, "rb") as f:  # pickle: only ever load files you trained yourself
            pipeline = pickle.load(f)
        label = "TF-IDF classifier"
        if meta_path.exists():
            label = f"TF-IDF ({json.loads(meta_path.read_text())['model']})"
        return cls(pipeline, label)

    def score_batch(self, texts: list[str]) -> np.ndarray:
        clean = [preprocess(t) for t in texts]
        return self.pipeline.predict_proba(clean)[:, 1] * 100.0

    def score(self, text: str) -> float:
        return float(self.score_batch([text])[0])


class TransformerScorer:
    """Fine-tuned DistilBERT saved by ``scripts/train_distilbert.py`` (model + tokenizer)."""

    def __init__(self, model, tokenizer, device: str, max_len: int = MAX_LEN):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    @classmethod
    def load(cls, directory: Path | None = None, device: str | None = None) -> "TransformerScorer":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        directory = directory or config.distilbert_dir()
        if not (directory / "config.json").exists():
            raise FileNotFoundError(f"No DistilBERT model in {directory}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForSequenceClassification.from_pretrained(directory)
        model.to(device)
        model.eval()
        return cls(model, tokenizer, device)

    def score_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size],
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits, dim=1)[:, 1]
            out.append(probs.float().cpu().numpy())
        return np.concatenate(out) * 100.0 if out else np.array([])

    def score(self, text: str) -> float:
        return float(self.score_batch([text])[0])


def run_sanity_check(scorer: TransformerScorer) -> tuple[bool, float, float]:
    """Return (passed, spam_prob_on_spam_text, spam_prob_on_ham_text), probabilities in 0-1."""
    spam_p, ham_p = (float(x) / 100.0 for x in scorer.score_batch([SANITY_SPAM_TEXT, SANITY_HAM_TEXT]))
    return spam_p >= SANITY_SPAM_MIN and ham_p <= SANITY_HAM_MAX, spam_p, ham_p


@dataclass
class Detector:
    """Holds whichever models could be loaded and degrades gracefully when one is missing."""

    tfidf: TfidfScorer | None = None
    transformer: TransformerScorer | None = None
    transformer_note: str = ""

    @classmethod
    def from_dir(cls, use_transformer: bool = True) -> "Detector":
        det = cls()

        try:
            det.tfidf = TfidfScorer.load()
            log.info("TF-IDF pipeline loaded (%s)", det.tfidf.label)
        except FileNotFoundError:
            log.warning("No TF-IDF pipeline at %s: rule-based fallback only. Run scripts/train_classical.py",
                        config.pipeline_path())

        if not use_transformer:
            det.transformer_note = "disabled by configuration"
        elif not (config.distilbert_dir() / "config.json").exists():
            det.transformer_note = f"no model in {config.distilbert_dir()}"
            log.info("DistilBERT skipped: %s", det.transformer_note)
        else:
            try:
                scorer = TransformerScorer.load()
                ok, spam_p, ham_p = run_sanity_check(scorer)
                if ok:
                    det.transformer = scorer
                    log.info("DistilBERT loaded on %s (sanity check passed: spam=%.0f%%, ham=%.0f%%)",
                             scorer.device, spam_p * 100, ham_p * 100)
                else:
                    det.transformer_note = (
                        f"sanity check FAILED (spam text -> {spam_p:.0%}, ham text -> {ham_p:.0%}); "
                        "the classification head is probably untrained. Re-run scripts/train_distilbert.py"
                    )
                    log.error("DistilBERT disabled: %s", det.transformer_note)
            except ImportError:
                det.transformer_note = "torch/transformers not installed (pip install -r requirements-transformer.txt)"
                log.warning("DistilBERT skipped: %s", det.transformer_note)
            except Exception as exc:  # corrupt/incompatible model files
                det.transformer_note = f"failed to load: {exc}"
                log.error("DistilBERT disabled: %s", det.transformer_note)
        return det

    def tfidf_score(self, text: str) -> float | None:
        return self.tfidf.score(text) if self.tfidf else None

    def transformer_score(self, text: str) -> float | None:
        return self.transformer.score(text) if self.transformer else None

    def status(self) -> dict:
        return {
            "status": "ok",
            "tfidf_available": self.tfidf is not None,
            "tfidf_model": self.tfidf.label if self.tfidf else None,
            "transformer_active": self.transformer is not None,
            "transformer_note": self.transformer_note or None,
            "device": self.transformer.device if self.transformer else "cpu",
        }
