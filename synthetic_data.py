"""Tiny synthetic corpus for smoke tests ONLY.

The two classes use different word lists on purpose so that a model trained on
this data is trivially separable. Metrics computed on it say NOTHING about real
fake-news detection quality. ``train.py --synthetic`` marks the resulting
metadata with ``"synthetic_data": true``.
"""
from __future__ import annotations

import random

import pandas as pd

FAKE_WORDS = [
    "shocking", "hoax", "conspiracy", "bombshell", "exposed", "outrage", "leaked",
    "unbelievable", "coverup", "secret", "scandal", "traitor", "rigged", "destroyed",
]
REAL_WORDS = [
    "ministry", "committee", "parliament", "budget", "statement", "spokesperson",
    "agreement", "officials", "regulator", "proposal", "delegation", "treaty",
    "legislation", "report",
]
SHARED_WORDS = [
    "government", "people", "said", "new", "president", "country", "year", "policy",
    "election", "week", "market", "court", "city", "leader", "public", "state",
]


def _make_doc(rng: random.Random, specific: list[str], n_words: int = 45) -> str:
    words = [rng.choice(specific) if rng.random() < 0.55 else rng.choice(SHARED_WORDS)
             for _ in range(n_words)]
    words.append(str(rng.randint(1000, 99999)))  # keeps documents unique
    return " ".join(words) + "."


def make_synthetic_dataframe(n_per_class: int = 150, seed: int = 0) -> pd.DataFrame:
    """Return columns ``title, text, label, date`` with ``n_per_class`` rows per class."""
    rng = random.Random(seed)
    rows = []
    for label, vocab in (("FAKE", FAKE_WORDS), ("REAL", REAL_WORDS)):
        for i in range(n_per_class):
            date = pd.Timestamp("2016-01-01") + pd.Timedelta(days=rng.randint(0, 700))
            rows.append({
                "title": " ".join(rng.choice(vocab + SHARED_WORDS) for _ in range(6)).capitalize(),
                "text": _make_doc(rng, vocab),
                "label": label,
                "date": date.strftime("%B %d, %Y"),
            })
    rng.shuffle(rows)
    return pd.DataFrame(rows)
