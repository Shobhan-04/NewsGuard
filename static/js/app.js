(function () {
  "use strict";
  var form = document.getElementById("analyze-form");
  if (!form) { return; }

  var predictUrl = form.dataset.predictUrl;            // rendered from url_for('api.predict')
  var maxText = parseInt(form.dataset.maxText, 10);
  var minChars = parseInt(form.dataset.minChars, 10);
  var titleEl = document.getElementById("title");
  var textEl = document.getElementById("text");
  var countEl = document.getElementById("char-count");
  var submitBtn = document.getElementById("submit-btn");
  var clearBtn = document.getElementById("clear-btn");
  var statusEl = document.getElementById("status");
  var errorEl = document.getElementById("error");
  var resultEl = document.getElementById("result");
  var TIMEOUT_MS = 30000;

  function updateCount() { countEl.textContent = textEl.value.length + " / " + maxText + " characters"; }

  function showError(message) {
    errorEl.textContent = message;          // textContent only: never inject HTML
    errorEl.hidden = false;
  }
  function clearMessages() { errorEl.hidden = true; errorEl.textContent = ""; statusEl.hidden = true; }
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? "Analysing\u2026" : "Analyse article";
    statusEl.textContent = isLoading ? "Analysing article\u2026" : "";
    statusEl.hidden = !isLoading;
  }

  function fillList(listEl, items) {
    listEl.textContent = "";
    if (!items.length) {
      var none = document.createElement("li");
      none.textContent = "None among the top terms.";
      listEl.appendChild(none);
      return;
    }
    items.forEach(function (item) {
      var li = document.createElement("li");
      var term = document.createElement("span");
      term.className = "term";
      term.textContent = "\u201C" + item.term + "\u201D";
      li.appendChild(term);
      li.appendChild(document.createTextNode(" \u2192 " + item.pushes_toward + " (weight " + item.contribution.toFixed(3) + ")"));
      listEl.appendChild(li);
    });
  }

  function renderResult(data) {
    var badge = document.getElementById("prediction-label");
    badge.textContent = data.prediction;
    badge.className = "label-badge " + data.prediction.toLowerCase();
    document.getElementById("score-value").textContent = data.model_score.toFixed(1);
    document.getElementById("score-bar").style.width = Math.max(0, Math.min(100, data.model_score)) + "%";
    var expl = data.explanation || [];
    fillList(document.getElementById("support-list"), expl.filter(function (e) { return e.supports_prediction; }));
    fillList(document.getElementById("oppose-list"), expl.filter(function (e) { return !e.supports_prediction; }));
    document.getElementById("explanation-note").textContent = data.explanation_note || "";
    var meta = data.metadata || {};
    var metaText = "Model version " + data.model_version + " \u00B7 score method: " + (meta.score_method || "n/a");
    if (meta.trained_on_synthetic_data) { metaText += " \u00B7 WARNING: demo model trained on synthetic data"; }
    document.getElementById("meta").textContent = metaText;
    resultEl.hidden = false;
    resultEl.focus();
  }

  textEl.addEventListener("input", updateCount);

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    clearMessages();
    resultEl.hidden = true;
    var title = titleEl.value.trim();
    var text = textEl.value.trim();
    if (!text) { showError("Please paste the article text."); textEl.focus(); return; }
    if ((title + text).length < minChars) { showError("The text is too short (at least " + minChars + " characters)."); textEl.focus(); return; }

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, TIMEOUT_MS);
    setLoading(true);
    fetch(predictUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ title: title, text: text }),
      signal: controller.signal
    }).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) {
          var msg = body && body.error && body.error.message ? body.error.message : "Request failed (HTTP " + response.status + ").";
          throw new Error(msg);
        }
        return body;
      });
    }).then(renderResult).catch(function (err) {
      showError(err.name === "AbortError" ? "The request timed out. The server may be waking up; please try again." : err.message);
    }).finally(function () {
      clearTimeout(timer);
      setLoading(false);
    });
  });

  clearBtn.addEventListener("click", function () {
    form.reset();
    updateCount();
    clearMessages();
    resultEl.hidden = true;
    titleEl.focus();
  });

  updateCount();
})();
