"""Flask application: web interface + REST API (see ``api.py``).

Run locally:   python app.py
Production:    gunicorn app:app   (see Procfile / render.yaml)

The model is NOT trained here; the saved pipeline is loaded (if present) at start-up.
"""
from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, render_template, request

import config
from api import api_bp, api_info_payload, register_error_handlers
from predict import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


def create_app(config_overrides: dict[str, Any] | None = None,
               service: PredictionService | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.json.sort_keys = False
    if config_overrides:
        app.config.update(config_overrides)

    service = service or PredictionService()
    app.extensions["newsguard_service"] = service
    try:  # warm up so the first request is fast; failure is non-fatal (/health reports it)
        service.load()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Model not loaded at start-up: %s", exc)

    app.register_blueprint(api_bp)
    register_error_handlers(app)

    @app.get("/")
    def index():
        wants_json = (request.accept_mimetypes.quality("application/json")
                      > request.accept_mimetypes.quality("text/html"))
        if wants_json:
            return jsonify(api_info_payload())
        return render_template(
            "index.html",
            app_name=config.APP_NAME,
            disclaimer=config.DISCLAIMER,
            score_note=config.SCORE_NOTE,
            max_title=service.max_title_chars,
            max_text=service.max_text_chars,
            min_chars=service.min_chars,
        )

    @app.after_request
    def _security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        if request.path in ("/predict", "/health"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    return app


app = create_app()

if __name__ == "__main__":
    # Development server for local use only. Production uses gunicorn (see Procfile).
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
