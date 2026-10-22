import joblib

import config
from explain import explain_text, explain_vector
from tests.conftest import FAKE_SAMPLE, REAL_SAMPLE


def _pipe(model_dir):
    return joblib.load(model_dir / config.MODEL_FILENAME)


def test_explanation_is_bounded_and_labelled(model_dir):
    items = explain_text(_pipe(model_dir), FAKE_SAMPLE)
    assert 0 < len(items) <= config.EXPLAIN_N_SUPPORT + config.EXPLAIN_N_OPPOSE
    for it in items:
        assert set(it) == {"term", "pushes_toward", "supports_prediction", "contribution"}
        assert it["pushes_toward"] in ("FAKE", "REAL") and it["contribution"] >= 0


def test_terms_push_toward_their_class(model_dir):
    pipe = _pipe(model_dir)
    fake_items = {i["term"]: i for i in explain_text(pipe, "hoax conspiracy shocking", predicted_label="FAKE")}
    real_items = {i["term"]: i for i in explain_text(pipe, "ministry parliament committee", predicted_label="REAL")}
    assert all(i["pushes_toward"] == "FAKE" for i in fake_items.values())
    assert all(i["pushes_toward"] == "REAL" for i in real_items.values())
    assert all(i["supports_prediction"] for i in fake_items.values())


def test_support_flag_flips_with_predicted_label(model_dir):
    pipe = _pipe(model_dir)
    as_fake = explain_text(pipe, "hoax", predicted_label="FAKE")[0]
    as_real = explain_text(pipe, "hoax", predicted_label="REAL")[0]
    assert as_fake["supports_prediction"] is True and as_real["supports_prediction"] is False


def test_empty_and_unknown_vocabulary_give_empty_list(model_dir):
    pipe = _pipe(model_dir)
    assert explain_text(pipe, "") == []
    assert explain_text(pipe, "   ") == []
    assert explain_text(pipe, "zxqv wplk qqqq") == []


def test_explain_vector_none_input(model_dir):
    pipe = _pipe(model_dir)
    assert explain_vector(pipe.named_steps["clf"], None, "FAKE", []) == []
