"""REST API (Flask Blueprint) - thin layer over the shared ``PredictionService``.

Endpoints
---------
GET  /health   liveness + model availability (no secrets / paths)
GET  /api      API description (``GET /`` returns the same JSON if the client asks for JSON)
POST /predict  JSON ``{"title": "...", "text": "..."}`` -> prediction

Error format: ``{"error": {"code": "...", "message": "..."}}`` with an HTTP status
of 400 (invalid), 413 (too large), 503 (model unavailable) or 500 (unexpected;
details are logged server-side only).
"""
from __future__ import annotations

import logging

from flask import Blueprint, Flask, current_app, jsonify, request
from werkzeug.exceptions import HTTPException

import config
from predict import InvalidInputError, ModelUnavailableError, PredictionService

logger = logging.getLogger("newsguard.api")
api_bp = Blueprint("api", __name__)


def _service() -> PredictionService:
    return current_app.extensions["newsguard_service"]


def error_response(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def api_info_payload() -> dict:
    return {
        "name": config.APP_NAME,
        "description": "Text classifier that reports which labelled examples (REAL/FAKE) an article "
                       "resembles. Not a fact-checking service.",
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Application status and model availability"},
            {"method": "GET", "path": "/api", "description": "This description"},
            {"method": "POST", "path": "/predict",
             "description": "Body: JSON with 'title' and/or 'text'. Returns prediction, model_score, explanation."},
        ],
        "limits": {
            "max_title_chars": _service().max_title_chars,
            "max_text_chars": _service().max_text_chars,
            "max_request_bytes": current_app.config["MAX_CONTENT_LENGTH"],
        },
        "score_note": config.SCORE_NOTE,
        "disclaimer": config.DISCLAIMER,
    }


@api_bp.get("/api")
def api_info():
    return jsonify(api_info_payload())


@api_bp.get("/health")
def health():
    status = _service().status()
    return jsonify({
        "status": "ok" if status["model_available"] else "degraded",
        "model_available": status["model_available"],
        "model_version": status["model_version"],
        "synthetic_model": status["synthetic_model"],
    })


@api_bp.post("/predict")
def predict():
    if not request.is_json:
        return error_response(400, "invalid_content_type", "Content-Type must be application/json.")
    payload = request.get_json(silent=True)  # a too-large body raises 413 here
    if not isinstance(payload, dict):
        return error_response(400, "invalid_json", "Request body must be a valid JSON object.")
    try:
        result = _service().predict(payload.get("title"), payload.get("text"))
    except InvalidInputError as exc:
        return error_response(400, "invalid_input", str(exc))
    except ModelUnavailableError:
        return error_response(503, "model_unavailable", "The model is currently unavailable.")
    return jsonify(result)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        status = exc.code or 500
        if status == 413:
            return error_response(413, "payload_too_large",
                                  f"Request body exceeds {app.config['MAX_CONTENT_LENGTH']} bytes.")
        return error_response(status, (exc.name or "error").lower().replace(" ", "_"), exc.description or exc.name)

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        logger.exception("Unhandled error")  # stack trace stays in the server log
        return error_response(500, "internal_error", "An unexpected error occurred.")
