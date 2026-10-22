"""Training pipeline for NewsGuard.

Usage (from the project root, virtual environment active)::

    python train.py                         # ISOT in data/True.csv + data/Fake.csv
    python train.py --split time            # chronological split (needs a date column)
    python train.py --csv path/to/file.csv  # single CSV with title,text,label[,date]
    python train.py --synthetic             # smoke test only; NOT real evidence

Steps: load -> validate -> clean/deduplicate -> split (train/val/test) ->
fit TF-IDF + classifier on TRAIN only -> choose on VALIDATION -> evaluate the
chosen model ONCE on TEST -> save pipeline + metadata. Every number written to
the metadata file is computed here; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from preprocess import build_text, clean_text, near_duplicate_key

logger = logging.getLogger("newsguard.train")

LABEL_MAP = {
    "real": "REAL", "true": "REAL", "reliable": "REAL",
    "fake": "FAKE", "false": "FAKE", "unreliable": "FAKE",
}
CANDIDATES = ("logistic_regression", "linear_svm")


class DatasetError(Exception):
    """Raised for missing files, bad schemas or unusable data."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def normalize_label(value: Any) -> str:
    """Map a raw label to ``REAL`` or ``FAKE``; raise ``ValueError`` otherwise."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("missing label")
    key = str(value).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    raise ValueError(
        f"Unrecognised label {value!r}. Supported (case-insensitive): {sorted(LABEL_MAP)}"
    )


def _find_file(directory: Path, filename: str) -> Path:
    if directory.is_dir():
        for candidate in directory.iterdir():
            if candidate.is_file() and candidate.name.lower() == filename.lower():
                return candidate
    raise DatasetError(
        f"'{filename}' not found in '{directory}'. See data/README.md for how to "
        "obtain the ISOT dataset, or run `python train.py --synthetic` for a smoke test."
    )


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_isot_pair(data_dir: Path | str) -> pd.DataFrame:
    """Load ``True.csv`` (REAL) and ``Fake.csv`` (FAKE); labels come from the file name."""
    data_dir = Path(data_dir)
    frames = []
    for filename, label in (("True.csv", "REAL"), ("Fake.csv", "FAKE")):
        path = _find_file(data_dir, filename)
        df = _lower_columns(pd.read_csv(path))
        missing = [c for c in ("title", "text") if c not in df.columns]
        if missing:
            raise DatasetError(f"{path.name} is missing required column(s): {missing}")
        # Deliberately ignore 'subject' (differs systematically between the two
        # files, so it would leak the label). 'date' is kept only for --split time.
        keep = [c for c in ("title", "text", "date") if c in df.columns]
        part = df[keep].copy()
        part["label"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_single_csv(path: Path | str) -> pd.DataFrame:
    """Load one CSV with ``text`` and ``label`` columns (``title``/``date`` optional)."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"CSV not found: {path}")
    df = _lower_columns(pd.read_csv(path))
    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise DatasetError(f"{path.name} is missing required column(s): {missing}")
    keep = [c for c in ("title", "text", "label", "date") if c in df.columns]
    return df[keep].copy()


# --------------------------------------------------------------------------- #
# Validation, cleaning, deduplication
# --------------------------------------------------------------------------- #
def _class_counts(labels: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def _drop_dupes_by_key(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int, int]:
    """Drop rows whose key appears under conflicting labels, then exact repeats."""
    n_labels = df.groupby(key)["label"].transform("nunique")
    conflicting = n_labels > 1
    n_conflict = int(conflicting.sum())
    df = df.loc[~conflicting]
    repeated = df.duplicated(subset=key, keep="first")
    n_dup = int(repeated.sum())
    return df.loc[~repeated], n_conflict, n_dup


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, normalise labels, combine title+body, remove bad rows and duplicates.

    Returns ``(clean_df, report)``; ``clean_df`` has columns ``content`` (raw
    combined title+body - cleaning happens inside the model pipeline), ``label``
    and optionally ``date``.
    """
    if "label" not in df.columns or "text" not in df.columns:
        raise DatasetError("Dataset needs at least 'text' and 'label' columns.")
    df = df.copy()
    report: dict[str, Any] = {"rows_loaded": int(len(df))}

    for col in ("title", "text"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # labels
    unknown = []
    normalized = []
    for value in df["label"]:
        try:
            normalized.append(normalize_label(value))
        except ValueError:
            normalized.append(None)
            unknown.append(str(value))
    if unknown:
        raise DatasetError(
            f"Unrecognised label values {sorted(set(unknown))[:10]}. "
            f"Supported (case-insensitive): {sorted(LABEL_MAP)}"
        )
    df["label"] = normalized
    report["class_counts_loaded"] = _class_counts(df["label"])

    # missing / empty body
    has_body = df["text"].str.strip() != ""
    report["removed_missing_body"] = int((~has_body).sum())
    df = df.loc[has_body].copy()

    # combined text + cleaned form used for length and duplicate checks
    df["content"] = [build_text(t, x) for t, x in zip(df["title"], df["text"])]
    df["_clean"] = df["content"].map(clean_text)
    long_enough = df["_clean"].str.len() >= config.MIN_ARTICLE_CHARS
    report["removed_too_short"] = int((~long_enough).sum())
    df = df.loc[long_enough].copy()

    # duplicates: exact (after cleaning), then near-duplicate by prefix
    df, n_conflict_exact, n_exact = _drop_dupes_by_key(df, "_clean")
    df = df.copy()
    df["_near"] = df["_clean"].map(lambda s: near_duplicate_key(s, config.NEAR_DUP_PREFIX_CHARS))
    df, n_conflict_near, n_near = _drop_dupes_by_key(df, "_near")
    report["removed_exact_duplicates"] = n_exact
    report["removed_near_duplicates"] = n_near
    report["removed_conflicting_label_duplicates"] = n_conflict_exact + n_conflict_near

    keep = ["content", "label"] + (["date"] if "date" in df.columns else [])
    out = df[keep].reset_index(drop=True)

    counts = _class_counts(out["label"])
    report["rows_final"] = int(len(out))
    report["class_counts_final"] = counts
    if len(counts) == 2:
        report["majority_to_minority_ratio"] = round(max(counts.values()) / min(counts.values()), 3)
    if len(out) == 0 or len(counts) < 2:
        raise DatasetError("After cleaning, fewer than two classes (or no rows) remain.")
    return out, report


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_dataset(
    df: pd.DataFrame,
    strategy: str = "random",
    seed: int = config.RANDOM_STATE,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return train, validation, test frames plus a description of the split."""
    info: dict[str, Any] = {"strategy": strategy, "seed": seed}
    if strategy == "random":
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=seed)
        train, val = train_test_split(
            train_val, test_size=val_size / (1.0 - test_size),
            stratify=train_val["label"], random_state=seed)
    elif strategy == "time":
        if "date" not in df.columns:
            raise DatasetError("--split time needs a 'date' column in the data.")
        dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        dated = df.assign(_date=dates)
        info["rows_dropped_unparseable_date"] = int(dated["_date"].isna().sum())
        dated = dated.dropna(subset=["_date"]).sort_values("_date", kind="stable")
        n = len(dated)
        n_test, n_val = int(round(n * test_size)), int(round(n * val_size))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise DatasetError("Not enough dated rows for a time-based split.")
        train = dated.iloc[:n_train]
        val = dated.iloc[n_train:n_train + n_val]
        test = dated.iloc[n_train + n_val:]
        for name, part in (("train", train), ("validation", val), ("test", test)):
            info[f"{name}_date_range"] = [str(part["_date"].min().date()), str(part["_date"].max().date())]
        train, val, test = (p.drop(columns="_date") for p in (train, val, test))
    else:
        raise DatasetError(f"Unknown split strategy: {strategy!r}")

    for name, part in (("train", train), ("validation", val), ("test", test)):
        counts = _class_counts(part["label"])
        info[f"{name}_size"] = int(len(part))
        info[f"{name}_class_counts"] = counts
        if len(counts) < 2:
            raise DatasetError(
                f"The {name} split contains only one class ({counts}). "
                "Use --split random or a dataset with both classes across time."
            )
    return train, val, test, info


# --------------------------------------------------------------------------- #
# Models and evaluation
# --------------------------------------------------------------------------- #
def build_pipeline(model_name: str, seed: int = config.RANDOM_STATE) -> Pipeline:
    """TF-IDF (cleaning included) followed by a linear classifier."""
    tfidf = TfidfVectorizer(
        preprocessor=clean_text, lowercase=False, dtype=np.float32, **config.TFIDF_PARAMS)
    if model_name == "logistic_regression":
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed)
    elif model_name == "linear_svm":
        clf = LinearSVC(C=1.0, max_iter=5000, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def evaluate(pipeline: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, Any]:
    """Compute accuracy, precision/recall/F1, confusion matrix and ROC-AUC."""
    classes = [str(c) for c in pipeline.classes_]
    y_true = labels.to_numpy()
    y_pred = pipeline.predict(texts)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(texts)[:, 1]
    else:
        scores = pipeline.decision_function(texts)
    try:
        auc: float | None = float(roc_auc_score(y_true == classes[1], scores))
    except ValueError:
        auc = None

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist(),
                             "note": "rows = true label, columns = predicted label"},
        "roc_auc": auc,
        "roc_auc_positive_class": classes[1],
    }


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def train_from_dataframe(
    df: pd.DataFrame,
    output_dir: Path | str,
    split_strategy: str = "random",
    dataset_report: dict[str, Any] | None = None,
    synthetic: bool = False,
    seed: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """Run split -> fit -> select -> test -> save. ``df`` must come from ``prepare_dataset``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test, split_info = split_dataset(df, split_strategy, seed)

    candidates: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    for name in CANDIDATES:
        pipe = build_pipeline(name, seed)
        pipe.fit(train["content"], train["label"])   # TF-IDF is fitted on TRAIN only
        fitted[name] = pipe
        candidates[name] = {
            "params": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                       for k, v in pipe.named_steps["clf"].get_params().items()},
            "validation": evaluate(pipe, val["content"], val["label"]),
        }
        logger.info("%s validation macro-F1 = %.4f", name, candidates[name]["validation"]["macro_f1"])

    # selection uses validation macro-F1 only; ties go to the first (simpler) candidate
    selected = max(CANDIDATES, key=lambda n: candidates[n]["validation"]["macro_f1"])
    best = fitted[selected]
    test_metrics = evaluate(best, test["content"], test["label"])   # test set used once

    classes = [str(c) for c in best.classes_]
    score_method = "predict_proba" if hasattr(best, "predict_proba") else "sigmoid_of_decision_function"
    metadata: dict[str, Any] = {
        "model_name": config.APP_NAME,
        "model_version": config.MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic_data": bool(synthetic),
        "classes": classes,
        "positive_class": classes[1],
        "score_type": "model_confidence",
        "score_method": score_method,
        "selected_model": selected,
        "selection_metric": "validation macro F1",
        "n_features": int(len(best.named_steps["tfidf"].vocabulary_)),
        "dataset": dataset_report or {},
        "split": split_info,
        "candidates": candidates,
        "test": test_metrics,
        "config": {
            "random_state": seed,
            "tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.TFIDF_PARAMS.items()},
            "min_article_chars": config.MIN_ARTICLE_CHARS,
            "near_duplicate_prefix_chars": config.NEAR_DUP_PREFIX_CHARS,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "notes": [
            "Scores describe resemblance to the labelled training data only; this is not fact-checking.",
            "Test metrics come from a single evaluation of the selected model on held-out data.",
            "Random splits can overstate generalisation when articles share sources or topics.",
        ] + (["SYNTHETIC SMOKE-TEST MODEL: metrics are meaningless for real-world quality."] if synthetic else []),
    }

    joblib.dump(best, output_dir / config.MODEL_FILENAME, compress=3)
    with open(output_dir / config.METADATA_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_to_builtin)
    return metadata


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(meta: dict[str, Any], output_dir: Path) -> None:
    ds, test = meta["dataset"], meta["test"]
    print("\n=== NewsGuard training summary ===")
    if meta["synthetic_data"]:
        print("!! SYNTHETIC DATA - smoke test only; these numbers are NOT evidence of model quality !!")
    print(f"rows loaded: {ds.get('rows_loaded')}  ->  final: {ds.get('rows_final')}")
    print(f"class counts (final): {ds.get('class_counts_final')}")
    print(f"removed: missing body={ds.get('removed_missing_body')}, too short={ds.get('removed_too_short')}, "
          f"exact dups={ds.get('removed_exact_duplicates')}, near dups={ds.get('removed_near_duplicates')}, "
          f"conflicting dups={ds.get('removed_conflicting_label_duplicates')}")
    print(f"split ({meta['split']['strategy']}): train={meta['split']['train_size']} "
          f"val={meta['split']['validation_size']} test={meta['split']['test_size']}")
    for name, info in meta["candidates"].items():
        print(f"  validation macro-F1 {name:<20} {info['validation']['macro_f1']:.4f}")
    print(f"selected: {meta['selected_model']}  (score method: {meta['score_method']})")
    print(f"TEST accuracy={test['accuracy']:.4f}  macro-F1={test['macro_f1']:.4f}  roc_auc={test['roc_auc']}")
    print(f"confusion matrix {test['confusion_matrix']['labels']}: {test['confusion_matrix']['matrix']}")
    print(f"saved to: {output_dir}")
    if test["accuracy"] >= 0.99 and not meta["synthetic_data"]:
        print("\nNote: near-perfect scores on ISOT are widely reported and usually reflect differences "
              "between the news sources (style, formatting), not general misinformation detection. "
              "See README > Known limitations.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the NewsGuard model.")
    parser.add_argument("--data-dir", default=str(config.DATA_DIR), help="folder with True.csv and Fake.csv")
    parser.add_argument("--csv", help="single CSV with text,label[,title,date] instead of the ISOT pair")
    parser.add_argument("--split", choices=["random", "time"], default="random")
    parser.add_argument("--output-dir", help="where to save the model (default: models/)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use a tiny synthetic corpus (smoke test only; output goes to models/_synthetic_smoke)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.synthetic:
            from synthetic_data import make_synthetic_dataframe
            raw = make_synthetic_dataframe(n_per_class=300, seed=args.seed)
            output_dir = Path(args.output_dir or config.SYNTHETIC_OUTPUT_DIR)
        else:
            raw = load_single_csv(args.csv) if args.csv else load_isot_pair(args.data_dir)
            output_dir = Path(args.output_dir or config.MODELS_DIR)
        clean, report = prepare_dataset(raw)
        meta = train_from_dataframe(clean, output_dir, args.split, report, args.synthetic, args.seed)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2
    _print_summary(meta, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
