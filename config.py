"""Central configuration for NewsGuard.

Values can be overridden with environment variables (see ``.env.example``).
Nothing in this file is a secret; do not add credentials here.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is only a convenience for local development
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
APP_NAME = "NewsGuard"
MODEL_VERSION = "1.0.0"

DISCLAIMER = (
    "Model output only. The prediction shows which labelled training examples the "
    "text resembles. It is not a fact-check and does not verify the article, its "
    "claims or its publisher."
)
SCORE_NOTE = (
    "model_score is the model's uncalibrated confidence in the predicted label "
    "(0-100). It is not the probability that the article is true or false."
)
EXPLANATION_NOTE = (
    "Terms listed are the words/phrases that pushed this linear model toward one "
    "label. They describe the model's behaviour, not causes, and do not prove "
    "that an article is misinformation."
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.environ.get("NEWSGUARD_DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = BASE_DIR / "models"
MODEL_FILENAME = "newsguard_pipeline.joblib"
METADATA_FILENAME = "model_metadata.json"
MODEL_PATH = Path(os.environ.get("NEWSGUARD_MODEL_PATH", MODELS_DIR / MODEL_FILENAME))
METADATA_PATH = Path(os.environ.get("NEWSGUARD_METADATA_PATH", MODELS_DIR / METADATA_FILENAME))
# Output directory used by ``python train.py --synthetic`` (git-ignored).
SYNTHETIC_OUTPUT_DIR = MODELS_DIR / "_synthetic_smoke"

# --------------------------------------------------------------------------- #
# Request / input limits
# --------------------------------------------------------------------------- #
MAX_TITLE_CHARS = _env_int("MAX_TITLE_CHARS", 300)
MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 20_000)
MIN_INPUT_CHARS = _env_int("MIN_INPUT_CHARS", 20)
# Whole-request size cap (bytes). Flask answers 413 above this.
MAX_CONTENT_LENGTH = _env_int("MAX_CONTENT_LENGTH", 262_144)

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = _env_int("PORT", 5000)
DEBUG = _env_bool("FLASK_DEBUG", False)  # keep False in production

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
MIN_ARTICLE_CHARS = 50          # shorter cleaned articles are dropped
NEAR_DUP_PREFIX_CHARS = 300     # near-duplicate key length (alphanumerics only)

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 50_000,     # keeps the artifact and RAM use small
    "min_df": 2,
    "sublinear_tf": True,
    "stop_words": "english",
}

# Explanation size limits
EXPLAIN_N_SUPPORT = 6
EXPLAIN_N_OPPOSE = 4
