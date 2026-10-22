"""Shared fixtures. A tiny SYNTHETIC model is trained into a temp directory once per
test session, so no real dataset and no large model is ever needed."""
from __future__ import annotations

import pytest

import config
from app import create_app
from predict import PredictionService
from synthetic_data import make_synthetic_dataframe
import train

FAKE_SAMPLE = "shocking hoax conspiracy bombshell exposed outrage leaked secret scandal rigged government people"
REAL_SAMPLE = "ministry committee parliament budget statement spokesperson agreement officials regulator government people"


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic_dataframe(n_per_class=120, seed=1)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, synthetic_df):
    out = tmp_path_factory.mktemp("model")
    clean, report = train.prepare_dataset(synthetic_df)
    train.train_from_dataframe(clean, out, "random", report, synthetic=True)
    return out


@pytest.fixture(scope="session")
def svm_model_dir(tmp_path_factory, synthetic_df):
    """Same data but forcing the LinearSVC (no predict_proba) code path."""
    out = tmp_path_factory.mktemp("svm_model")
    import joblib
    clean, _ = train.prepare_dataset(synthetic_df)
    pipe = train.build_pipeline("linear_svm")
    pipe.fit(clean["content"], clean["label"])
    joblib.dump(pipe, out / config.MODEL_FILENAME)
    return out


@pytest.fixture()
def service(model_dir):
    return PredictionService(model_dir / config.MODEL_FILENAME, model_dir / config.METADATA_FILENAME)


@pytest.fixture()
def missing_service(tmp_path):
    return PredictionService(tmp_path / "nope.joblib", tmp_path / "nope.json")


@pytest.fixture()
def app(service):
    return create_app({"TESTING": True}, service=service)


@pytest.fixture()
def client(app):
    return app.test_client()
