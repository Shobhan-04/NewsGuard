"""Transparent explanations for the linear TF-IDF model.

For a linear classifier the decision value is ``intercept + sum_j x_j * w_j``,
where ``x_j`` is the TF-IDF weight of term ``j`` in this document and ``w_j`` is
the learned coefficient. ``x_j * w_j`` is therefore an exact, additive
per-term contribution (no sampling, no extra dependency such as SHAP).

Binary handling: scikit-learn stores ONE coefficient vector for two classes.
Positive contributions push toward ``classes_[1]`` (REAL), negative toward
``classes_[0]`` (FAKE). We report each term's magnitude and the label it pushes
toward, and flag whether that supports the predicted label.

These contributions describe the model's behaviour. They are not causal
evidence and do not show that an article is misinformation.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import config


def explain_vector(
    classifier: Any,
    X: Any,
    predicted_label: str,
    feature_names: Sequence[str],
    n_support: int = config.EXPLAIN_N_SUPPORT,
    n_oppose: int = config.EXPLAIN_N_OPPOSE,
) -> list[dict[str, Any]]:
    """Return the most influential terms of one TF-IDF row ``X`` (1 x n_features).

    Output is bounded to ``n_support + n_oppose`` items, ordered by contribution
    size within each group (supporting first). Empty input or a document with no
    known vocabulary yields ``[]``.
    """
    if X is None or getattr(X, "shape", (0, 0))[0] != 1 or X.nnz == 0:
        return []
    classes = [str(c) for c in classifier.classes_]
    if len(classes) != 2:
        raise ValueError("explain_vector supports binary classifiers only")
    coef = np.ravel(classifier.coef_)
    if coef.shape[0] != X.shape[1] or len(feature_names) != X.shape[1]:
        raise ValueError("Feature dimension mismatch between vectorizer and classifier")

    row = X.tocsr()
    idx, vals = row.indices, row.data.astype(np.float64)
    contrib = vals * coef[idx]                    # >0 -> classes[1], <0 -> classes[0]
    keep = contrib != 0
    idx, contrib = idx[keep], contrib[keep]

    items = []
    for j, c in zip(idx, contrib):
        toward = classes[1] if c > 0 else classes[0]
        items.append({
            "term": str(feature_names[j]),
            "pushes_toward": toward,
            "supports_prediction": toward == predicted_label,
            "contribution": round(float(abs(c)), 4),
        })
    items.sort(key=lambda d: d["contribution"], reverse=True)
    support = [d for d in items if d["supports_prediction"]][:max(n_support, 0)]
    oppose = [d for d in items if not d["supports_prediction"]][:max(n_oppose, 0)]
    return support + oppose


def explain_text(pipeline: Any, text: str, predicted_label: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Convenience wrapper: transform ``text`` with the pipeline and explain it."""
    if not isinstance(text, str) or not text.strip():
        return []
    vectorizer, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    X = vectorizer.transform([text])
    if predicted_label is None:
        predicted_label = str(pipeline.predict([text])[0])
    return explain_vector(clf, X, predicted_label, vectorizer.get_feature_names_out(), **kwargs)
