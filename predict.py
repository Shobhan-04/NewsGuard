"""Shared prediction service used by BOTH the web UI and the REST API.

``PredictionService`` validates input, loads the saved pipeline safely, returns
the label, an uncalibrated model score (0-100) and an explanation.

Security note: model files are loaded with joblib (pickle). Only load artifacts
you produced yourself with ``train.py``; never load a model file from an
untrusted source.
"""
from __future__ import annotations

import json
import logging
import threading
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.special import expit

import config
from explain import explain_vector
from preprocess import build_text

logger = logging.getLogger("newsguard.predict")


class PredictionError(Exception):
    """Base class for errors the API maps to HTTP status codes."""


class InvalidInputError(PredictionError):
    """The submitted title/text is not acceptable (HTTP 400)."""


class ModelUnavailableError(PredictionError):
    """The model artifact is missing, unreadable or incompatible (HTTP 503)."""


def validate_article(
    title: Any,
    text: Any,
    max_title_chars: int = config.MAX_TITLE_CHARS,
    max_text_chars: int = config.MAX_TEXT_CHARS,
    min_chars: int = config.MIN_INPUT_CHARS,
) -> tuple[str, str]:
    """Return cleaned ``(title, text)`` strings or raise ``InvalidInputError``."""
    for name, value in (("title", title), ("text", text)):
        if value is not None and not isinstance(value, str):
            raise InvalidInputError(f"'{name}' must be a string.")
    title = (title or "").replace("\x00", "").strip()
    text = (text or "").replace("\x00", "").strip()
    if not title and not text:
        raise InvalidInputError("Provide a non-empty 'text' and/or 'title'.")
    if len(title) > max_title_chars:
        raise InvalidInputError(f"'title' is too long (max {max_title_chars} characters).")
    if len(text) > max_text_chars:
        raise InvalidInputError(f"'text' is too long (max {max_text_chars} characters).")
    if len(title) + len(text) < min_chars:
        raise InvalidInputError(f"Input is too short (at least {min_chars} characters are needed).")
    return title, text


class PredictionService:
    """Loads the saved pipeline lazily and answers prediction requests."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        max_title_chars: int = config.MAX_TITLE_CHARS,
        max_text_chars: int = config.MAX_TEXT_CHARS,
        min_chars: int = config.MIN_INPUT_CHARS,
    ) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.metadata_path = Path(metadata_path or config.METADATA_PATH)
        self.max_title_chars = max_title_chars
        self.max_text_chars = max_text_chars
        self.min_chars = min_chars
        self._lock = threading.Lock()
        self._pipeline: Any = None
        self._vectorizer: Any = None
        self._clf: Any = None
        self._feature_names: Any = None
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------ load
    def load(self, force: bool = False) -> None:
        """Load the artifact; raise ``ModelUnavailableError`` on any problem."""
        with self._lock:
            if self._pipeline is not None and not force:
                return
            if not self.model_path.is_file():
                raise ModelUnavailableError("Model artifact not found. Run `python train.py` first.")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline = joblib.load(self.model_path)
                for w in caught:
                    logger.warning("while loading model: %s", w.message)
                vectorizer = pipeline.named_steps["tfidf"]
                clf = pipeline.named_steps["clf"]
                classes = [str(c) for c in clf.classes_]
                if len(classes) != 2 or not hasattr(clf, "coef_"):
                    raise ValueError("expected a binary linear classifier")
                names = vectorizer.get_feature_names_out()
                if len(names) != np.ravel(clf.coef_).shape[0]:
                    raise ValueError("vectorizer/classifier size mismatch")
            except Exception:  # corrupt, incompatible or wrong type
                logger.exception("Failed to load model artifact")
                raise ModelUnavailableError("The model is currently unavailable.") from None
            self._pipeline, self._vectorizer, self._clf = pipeline, vectorizer, clf
            self._feature_names = names
            self._metadata = self._read_metadata()

    def _read_metadata(self) -> dict[str, Any]:
        try:
            with open(self.metadata_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            logger.warning("Model metadata missing or unreadable; using defaults")
            return {}

    def is_available(self) -> bool:
        try:
            self.load()
            return True
        except ModelUnavailableError:
            return False

    def status(self) -> dict[str, Any]:
        """Small, non-sensitive status dictionary (no filesystem paths)."""
        available = self.is_available()
        return {
            "model_available": available,
            "model_version": self.model_version if available else None,
            "synthetic_model": bool(self._metadata.get("synthetic_data")) if available else None,
        }

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("model_version", config.MODEL_VERSION))

    # --------------------------------------------------------------- predict
    def validate(self, title: Any, text: Any) -> tuple[str, str]:
        return validate_article(title, text, self.max_title_chars, self.max_text_chars, self.min_chars)

    def predict(self, title: Any = None, text: Any = None) -> dict[str, Any]:
        """Validate, score and explain one article. Never hard-codes a result."""
        title, text = self.validate(title, text)
        self.load()
        content = build_text(title, text)
        X = self._vectorizer.transform([content])
        if X.nnz == 0:
            raise InvalidInputError(
                "The text contains no terms the model recognises (it was trained on English news text).")

        classes = [str(c) for c in self._clf.classes_]
        if hasattr(self._clf, "predict_proba"):
            proba = self._clf.predict_proba(X)[0]
            method = "predict_proba"
        else:  # e.g. LinearSVC: squash the margin; this is NOT a calibrated probability
            p1 = float(expit(self._clf.decision_function(X)[0]))
            proba = np.array([1.0 - p1, p1])
            method = "sigmoid_of_decision_function"
        top = int(np.argmax(proba))
        label = classes[top]

        explanation = explain_vector(self._clf, X, label, self._feature_names)
        return {
            "prediction": label,
            "model_score": round(float(proba[top]) * 100.0, 1),
            "score_type": "model_confidence",
            "score_note": config.SCORE_NOTE,
            "explanation": explanation,
            "explanation_note": config.EXPLANATION_NOTE,
            "model_version": self.model_version,
            "disclaimer": config.DISCLAIMER,
            "metadata": {
                "classes": classes,
                "score_method": method,
                "known_terms": int(X.nnz),
                "input_characters": len(content),
                "trained_on_synthetic_data": bool(self._metadata.get("synthetic_data", False)),
                "trained_at_utc": self._metadata.get("trained_at_utc"),
            },
        }


_default_service: PredictionService | None = None
_default_lock = threading.Lock()


def get_default_service() -> PredictionService:
    """Process-wide service built from ``config`` paths."""
    global _default_service
    with _default_lock:
        if _default_service is None:
            _default_service = PredictionService()
        return _default_service


def predict_article(title: Any = None, text: Any = None) -> dict[str, Any]:
    """Functional entry point: ``predict_article(title, text)``."""
    return get_default_service().predict(title, text)


if __name__ == "__main__":  # quick manual check: python predict.py --text "..."
    import argparse

    parser = argparse.ArgumentParser(description="Score one article with the saved model.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--model-dir", help="folder containing the model + metadata (default: models/)")
    args = parser.parse_args()
    if args.model_dir:
        svc = PredictionService(Path(args.model_dir) / config.MODEL_FILENAME,
                                Path(args.model_dir) / config.METADATA_FILENAME)
    else:
        svc = get_default_service()
    try:
        print(json.dumps(svc.predict(args.title, args.text), indent=2))
    except PredictionError as exc:
        raise SystemExit(f"Error: {exc}")
