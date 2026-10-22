import json

import pytest

from app import create_app
from predict import PredictionService
from tests.conftest import FAKE_SAMPLE, REAL_SAMPLE


def post_json(client, payload, **kw):
    return client.post("/predict", data=json.dumps(payload), content_type="application/json", **kw)


# ----------------------------------------------------------------- health
def test_health_ok_and_does_not_leak_paths(client, model_dir):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok" and body["model_available"] is True
    assert str(model_dir) not in resp.get_data(as_text=True)


def test_health_degraded_without_model(missing_service):
    client = create_app({"TESTING": True}, service=missing_service).test_client()
    body = client.get("/health").get_json()
    assert body["status"] == "degraded" and body["model_available"] is False


# ------------------------------------------------------------------ pages
def test_root_serves_html_with_correct_api_route(client):
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200 and "text/html" in resp.content_type
    assert 'data-predict-url="/predict"' in html
    assert "not a fact-checker" in html.lower()


def test_root_returns_api_info_for_json_clients(client):
    resp = client.get("/", headers={"Accept": "application/json"})
    assert resp.status_code == 200 and resp.get_json()["name"] == "NewsGuard"
    assert client.get("/api").get_json()["endpoints"]


def test_static_assets_are_served(client):
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/static/css/style.css").status_code == 200


def test_security_headers(client):
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


# ---------------------------------------------------------------- predict
@pytest.mark.parametrize("payload", [
    {"title": "A headline about government", "text": FAKE_SAMPLE},
    {"text": FAKE_SAMPLE},
    {"title": "shocking hoax exposed conspiracy headline"},
])
def test_predict_success(client, payload):
    resp = post_json(client, payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] in ("FAKE", "REAL")
    assert body["score_type"] == "model_confidence"
    assert isinstance(body["explanation"], list)
    assert body["model_version"]
    assert resp.headers["Cache-Control"] == "no-store"


def test_predict_matches_prediction_service(client, service):
    api = post_json(client, {"title": "Official statement", "text": REAL_SAMPLE}).get_json()
    direct = service.predict("Official statement", REAL_SAMPLE)
    assert api == direct


def test_malformed_json_is_400(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["error"]["code"] == "invalid_json"


def test_wrong_content_type_is_400(client):
    resp = client.post("/predict", data="text=hello", content_type="application/x-www-form-urlencoded")
    assert resp.status_code == 400


@pytest.mark.parametrize("payload", [[], "string", 5, None])
def test_non_object_json_is_400(client, payload):
    assert post_json(client, payload).status_code == 400


@pytest.mark.parametrize("payload", [{}, {"title": "", "text": ""}, {"text": "   "}])
def test_missing_text_is_400(client, payload):
    resp = post_json(client, payload)
    assert resp.status_code == 400 and resp.get_json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize("payload", [{"text": 123}, {"title": ["a"], "text": FAKE_SAMPLE}, {"text": {"a": 1}}])
def test_non_string_fields_are_400(client, payload):
    assert post_json(client, payload).status_code == 400


def test_text_over_character_limit_is_400(client, service):
    resp = post_json(client, {"text": "hoax " * (service.max_text_chars // 4)})
    assert resp.status_code == 400
    assert "too long" in resp.get_json()["error"]["message"]


def test_oversized_payload_is_413(service):
    small = create_app({"TESTING": True, "MAX_CONTENT_LENGTH": 1024}, service=service).test_client()
    resp = post_json(small, {"text": "hoax " * 1000})
    assert resp.status_code == 413
    assert resp.get_json()["error"]["code"] == "payload_too_large"


def test_model_unavailable_is_503(missing_service):
    client = create_app({"TESTING": True}, service=missing_service).test_client()
    resp = post_json(client, {"text": FAKE_SAMPLE})
    assert resp.status_code == 503
    assert resp.get_json()["error"]["code"] == "model_unavailable"


def test_unexpected_error_is_500_without_details(service, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail /etc/passwd")
    monkeypatch.setattr(service, "predict", boom)
    client = create_app({"TESTING": False, "PROPAGATE_EXCEPTIONS": False}, service=service).test_client()
    resp = post_json(client, {"text": FAKE_SAMPLE})
    text = resp.get_data(as_text=True)
    assert resp.status_code == 500
    assert "secret internal detail" not in text and "Traceback" not in text


def test_get_on_predict_is_405_json(client):
    resp = client.get("/predict")
    assert resp.status_code == 405 and "error" in resp.get_json()


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404 and resp.get_json()["error"]["code"] == "not_found"
