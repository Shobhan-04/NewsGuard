import json

import pandas as pd
import pytest

import config
import train
from synthetic_data import make_synthetic_dataframe


def test_normalize_label():
    assert train.normalize_label(" Fake ") == "FAKE"
    assert train.normalize_label("TRUE") == "REAL"
    for bad in ("maybe", "", None, float("nan")):
        with pytest.raises(ValueError):
            train.normalize_label(bad)


def _article(n: int, extra: str = "") -> str:
    return f"unique article number {n} about local government budget decisions and public services {extra}" * 2


def test_prepare_dataset_cleans_and_deduplicates():
    rows = [
        {"title": "T1", "text": _article(1), "label": "real"},
        {"title": "T1", "text": _article(1), "label": "real"},            # exact duplicate
        {"title": "T2", "text": _article(2), "label": "fake"},
        {"title": "T2 ", "text": _article(2).upper(), "label": "fake"},   # duplicate after cleaning
        {"title": "T3", "text": _article(3), "label": "fake"},
        {"title": "T3", "text": _article(3), "label": "real"},            # same text, conflicting labels
        {"title": "T4", "text": "", "label": "real"},                     # empty body
        {"title": "T5", "text": "too short", "label": "fake"},            # too short
        {"title": "T6", "text": None, "label": "fake"},                   # missing body
        {"title": "T7", "text": _article(7), "label": "REAL"},
        {"title": "T8", "text": _article(8), "label": "FAKE"},
    ]
    out, rep = train.prepare_dataset(pd.DataFrame(rows))
    assert rep["rows_loaded"] == 11
    assert rep["removed_missing_body"] == 2
    assert rep["removed_too_short"] == 1
    assert rep["removed_exact_duplicates"] == 2
    assert rep["removed_conflicting_label_duplicates"] == 2
    assert rep["rows_final"] == len(out) == 4
    assert set(out["label"]) == {"REAL", "FAKE"}
    assert rep["class_counts_final"] == {"FAKE": 2, "REAL": 2}


def test_prepare_dataset_rejects_unknown_labels():
    df = pd.DataFrame({"text": [_article(1)], "label": ["satire"]})
    with pytest.raises(train.DatasetError):
        train.prepare_dataset(df)


def test_prepare_dataset_requires_columns():
    with pytest.raises(train.DatasetError):
        train.prepare_dataset(pd.DataFrame({"title": ["x"]}))


def test_load_isot_pair_labels_from_filenames(tmp_path):
    pd.DataFrame({"title": ["a"], "text": ["real body"], "subject": ["politicsNews"], "date": ["January 1, 2017"]}
                 ).to_csv(tmp_path / "True.csv", index=False)
    pd.DataFrame({"Title": ["b"], "Text": ["fake body"], "subject": ["News"], "date": ["May 5, 2016"]}
                 ).to_csv(tmp_path / "fake.csv", index=False)   # lower-case name + capitalised headers
    df = train.load_isot_pair(tmp_path)
    assert dict(zip(df["title"], df["label"])) == {"a": "REAL", "b": "FAKE"}
    assert "subject" not in df.columns        # label-correlated metadata is not used


def test_load_isot_pair_reports_missing_files_and_columns(tmp_path):
    with pytest.raises(train.DatasetError):
        train.load_isot_pair(tmp_path)
    pd.DataFrame({"headline": ["a"]}).to_csv(tmp_path / "True.csv", index=False)
    pd.DataFrame({"headline": ["a"]}).to_csv(tmp_path / "Fake.csv", index=False)
    with pytest.raises(train.DatasetError, match="missing required"):
        train.load_isot_pair(tmp_path)


def test_random_split_is_disjoint_stratified(synthetic_df):
    clean, _ = train.prepare_dataset(synthetic_df)
    tr, va, te, info = train.split_dataset(clean, "random")
    assert len(tr) + len(va) + len(te) == len(clean)
    assert not set(tr["content"]) & set(te["content"])
    assert not set(tr["content"]) & set(va["content"])
    assert not set(va["content"]) & set(te["content"])
    for part in (tr, va, te):
        assert part["label"].nunique() == 2
    assert info["strategy"] == "random"


def test_time_split_is_chronological(synthetic_df):
    clean, _ = train.prepare_dataset(synthetic_df)
    tr, va, te, info = train.split_dataset(clean, "time")
    parse = lambda s: pd.to_datetime(s, format="mixed")
    assert parse(tr["date"]).max() <= parse(va["date"]).min() <= parse(va["date"]).max() <= parse(te["date"]).min()


def test_time_split_requires_dates():
    df = pd.DataFrame({"content": ["a"] * 4, "label": ["FAKE", "REAL"] * 2})
    with pytest.raises(train.DatasetError):
        train.split_dataset(df, "time")


def test_train_writes_artifacts_and_real_metrics(model_dir):
    meta = json.loads((model_dir / config.METADATA_FILENAME).read_text())
    assert meta["selected_model"] in train.CANDIDATES
    assert set(meta["candidates"]) == set(train.CANDIDATES)
    assert meta["selection_metric"] == "validation macro F1"
    best_val = max(c["validation"]["macro_f1"] for c in meta["candidates"].values())
    assert meta["candidates"][meta["selected_model"]]["validation"]["macro_f1"] == best_val
    cm = meta["test"]["confusion_matrix"]["matrix"]
    assert sum(sum(r) for r in cm) == meta["test"]["n_samples"] == meta["split"]["test_size"]
    assert meta["environment"]["scikit_learn"]


def test_cli_synthetic_smoke(tmp_path, capsys):
    code = train.main(["--synthetic", "--output-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / config.MODEL_FILENAME).is_file()
    assert "SYNTHETIC DATA" in capsys.readouterr().out


def test_cli_reports_missing_dataset(tmp_path, capsys):
    assert train.main(["--data-dir", str(tmp_path)]) == 2
    assert "Dataset error" in capsys.readouterr().err
