import json

import joblib
import pytest

import config
from predict import (InvalidInputError, ModelUnavailableError, PredictionService,
                     validate_article)
from tests.conftest import FAKE_SAMPLE, REAL_SAMPLE


# ------------------------------------------------------------- validation
@pytest.mark.parametrize("title,text", [
    (None, None), ("", ""), ("   ", "\n\t"),
])
def test_validate_rejects_empty(title, text):
    with pytest.raises(InvalidInputError):
        validate_article(title, text)


@pytest.mark.parametrize("title,text", [(1, "valid text goes here ok"), ("t", ["x"]), (None, 3.5), ({}, None)])
def test_validate_rejects_non_strings(title, text):
    with pytest.raises(InvalidInputError):
        validate_article(title, text)


def test_validate_rejects_too_long_and_too_short():
    with pytest.raises(InvalidInputError):
        validate_article(None, "x" * (config.MAX_TEXT_CHARS + 1))
    with pytest.raises(InvalidInputError):
        validate_article("t" * (config.MAX_TITLE_CHARS + 1), "fine body text here")
    with pytest.raises(InvalidInputError):
        validate_article(None, "short")


def test_validate_accepts_title_only_or_text_only():
    assert validate_article("A sufficiently long headline", None) == ("A sufficiently long headline", "")
    assert validate_article(None, "  A sufficiently long body  ")[1] == "A sufficiently long body"


# ----------------------------------------------------------- artifacts
def test_missing_model_is_unavailable(missing_service):
    assert missing_service.is_available() is False
    with pytest.raises(ModelUnavailableError):
        missing_service.predict(None, FAKE_SAMPLE)
    assert missing_service.status()["model_available"] is False


def test_corrupt_model_is_unavailable(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"this is not a model")
    svc = PredictionService(bad, tmp_path / "m.json")
    with pytest.raises(ModelUnavailableError):
        svc.predict(None, FAKE_SAMPLE)


def test_wrong_object_type_is_unavailable(tmp_path):
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a pipeline"}, wrong)
    with pytest.raises(ModelUnavailableError):
        PredictionService(wrong, tmp_path / "m.json").predict(None, FAKE_SAMPLE)


def test_model_artifacts_exist_and_metadata_is_valid(model_dir):
    assert (model_dir / config.MODEL_FILENAME).is_file()
    meta = json.loads((model_dir / config.METADATA_FILENAME).read_text())
    assert meta["synthetic_data"] is True
    assert meta["classes"] == ["FAKE", "REAL"]
    assert 0.0 <= meta["test"]["accuracy"] <= 1.0


# ------------------------------------------------------------ prediction
def test_prediction_structure(service):
    result = service.predict("A headline about government", FAKE_SAMPLE)
    for key in ("prediction", "model_score", "score_type", "score_note", "explanation",
                "explanation_note", "model_version", "disclaimer", "metadata"):
        assert key in result
    assert result["prediction"] in ("FAKE", "REAL")
    assert 0.0 <= result["model_score"] <= 100.0
    assert result["score_type"] == "model_confidence"
    assert "not the probability" in result["score_note"]
    assert isinstance(result["explanation"], list) and len(result["explanation"]) <= 10
    assert result["metadata"]["trained_on_synthetic_data"] is True


def test_prediction_depends_on_input_not_hardcoded(service):
    assert service.predict(None, FAKE_SAMPLE)["prediction"] == "FAKE"
    assert service.predict(None, REAL_SAMPLE)["prediction"] == "REAL"


def test_prediction_is_deterministic(service):
    assert service.predict("T headline here", FAKE_SAMPLE) == service.predict("T headline here", FAKE_SAMPLE)


def test_unknown_vocabulary_is_rejected(service):
    with pytest.raises(InvalidInputError):
        service.predict(None, "zxqv wplk qqqq rrrr tttt yyyy uuuu")


def test_linear_svm_path_without_predict_proba(svm_model_dir):
    svc = PredictionService(svm_model_dir / config.MODEL_FILENAME, svm_model_dir / "absent.json")
    result = svc.predict(None, FAKE_SAMPLE)
    assert result["metadata"]["score_method"] == "sigmoid_of_decision_function"
    assert result["prediction"] == "FAKE"
    assert 50.0 <= result["model_score"] <= 100.0
