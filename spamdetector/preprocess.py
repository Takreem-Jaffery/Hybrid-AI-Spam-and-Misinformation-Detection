"""Text preprocessing for the TF-IDF models.

This single function is imported by BOTH the training scripts and the API. In the original
project it was copy-pasted into the notebook and ``main.py`` with a comment saying the two
copies "MUST match", which is a classic train/serve drift bug waiting to happen.

DistilBERT does NOT use this: it gets the raw text and its own WordPiece tokenizer.
"""

from __future__ import annotations

import re

_URL = re.compile(r"http\S+|www\S+")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def preprocess(text: str) -> str:
    text = text.lower()
    text = _URL.sub(" url ", text)
    text = _DIGITS.sub(" num ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
