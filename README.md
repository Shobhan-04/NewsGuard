
# NewsGuard — Fake News Pattern Detection

NewsGuard is an end-to-end NLP and machine-learning demo that classifies a news article as **REAL-like** or **FAKE-like** based on patterns learned from a labelled dataset. It provides a Flask web interface, a REST API, a model-derived score, and explanations showing which terms influenced the prediction.

> **Responsible-use notice:** NewsGuard is not a fact-checker. It does not verify claims, publishers, sources, dates, or external evidence. A prediction only indicates how closely the submitted text resembles examples in the training dataset. The model score is uncalibrated and must not be interpreted as the probability that an article is true or false.

**Live demo:** [[https://newsguard-v6vy.onrender.com](https://newsguard-v6vy.onrender.com)]  
**Health check:** [[https://newsguard-v6vy.onrender.com/health](https://newsguard-v6vy.onrender.com/health)]  
**Repository:** [[Shobhan-04/NewsGuard](https://github.com/Shobhan-04/NewsGuard)]  
**Deployment:** Live on Render (free instance; may spin down after inactivity).

## Highlights

- End-to-end workflow: dataset loading, validation, cleaning, deduplication, splitting, training, evaluation, and model export.
- A shared scikit-learn pipeline using TF-IDF features and a linear classifier.
- Logistic Regression baseline compared with Linear SVM using validation macro-F1.
- Held-out test evaluation with accuracy, macro-F1, ROC-AUC, classification report, and confusion matrix.
- Per-term explanations based on TF-IDF values and linear-model coefficients.
- Flask web interface and JSON REST API powered by one shared prediction service.
- Input validation, request-size limits, structured errors, and a health endpoint.
- Automated tests that use a small temporary synthetic model and do not require the full dataset.
- No database, paid API, or external API key required.

## Evaluation

The following results are from the recorded training run using the **random stratified split**. They describe performance on a held-out subset of the same dataset—not real-world fact-checking accuracy.

| Test metric | Result |
|---|---:|
| Test accuracy | **98.85%** |
| Test macro-F1 | **98.84%** |
| Test ROC-AUC | **0.9989** |
| Test examples | **5,754** |
| Selected model | **Linear SVM** |
| Score method | Sigmoid of decision function (uncalibrated) |

The run loaded 44,898 rows and retained 38,356 after cleaning. The final class counts were 17,452 FAKE and 20,904 REAL. The training pipeline removed rows with missing bodies, very short text, exact duplicates, and near-duplicates before creating the train/validation/test split.

The test confusion matrix, with labels ordered as `FAKE`, `REAL`, was:

| Actual \ Predicted | FAKE | REAL |
|---|---:|---:|
| FAKE | 2,579 | 39 |
| REAL | 27 | 3,109 |

These high scores may partly reflect source and writing-style differences in the dataset. See [Limitations](#limitations-and-responsible-use) before interpreting them.

## How it works

```text
True.csv + Fake.csv
        |
        v
Validate, clean and deduplicate
        |
        v
Train / validation / test split
        |
        v
Compare Logistic Regression and Linear SVM
        |
        v
Select using validation macro-F1
        |
        v
Evaluate once on held-out test data
        |
        v
Save model pipeline + metadata
        |
        +----------------------+
        |                      |
        v                      v
Flask web interface       REST API
        |                      |
        +----------+-----------+
                   v
          Shared PredictionService
                   |
                   v
      Label + model score + explanation
```

### Explanation method

NewsGuard uses the linear model's TF-IDF feature values and coefficients to show terms that pushed the decision toward either class. These are explanations of the model's behaviour, not evidence that a claim is true or false. A highlighted word does not establish misinformation.

The score is derived from the model's decision function using a sigmoid transformation. **It is not calibrated confidence and is not the probability that the article is factually correct.**

## Technology stack

- **Language:** Python 3.11+
- **Machine learning / NLP:** pandas, NumPy, SciPy, scikit-learn, TF-IDF
- **Models:** Logistic Regression, Linear SVM (`LinearSVC`)
- **Web:** Flask, Gunicorn
- **Frontend:** HTML5, CSS3, vanilla JavaScript, Jinja templates
- **Testing:** pytest
- **Model persistence:** joblib

## Repository structure

```text
NewsGuard/
├── app.py                  # Flask web application
├── api.py                  # REST API routes and error handling
├── predict.py              # Shared prediction service and model loading
├── explain.py              # Per-term model explanations
├── preprocess.py           # Shared text construction and cleaning
├── train.py                # Dataset preparation, training and evaluation
├── synthetic_data.py       # Synthetic corpus for smoke tests only
├── config.py               # Configuration and input limits
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development dependencies
├── Procfile                # Production server command
├── render.yaml             # Render Blueprint configuration
├── pytest.ini
├── .env.example
├── data/
│   └── README.md           # Dataset download and attribution notes
├── models/
│   ├── newsguard_pipeline.joblib
│   ├── model_metadata.json
│   └── README.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/app.js
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_explain.py
    ├── test_predict.py
    ├── test_preprocess.py
    └── test_train.py
```

## Dataset

The default training workflow expects two files in `data/`:

- `True.csv` — REAL-labelled articles
- `Fake.csv` — FAKE-labelled articles

The project uses the ISOT Fake News Dataset. Download instructions, attribution, and licensing notes are in [`data/README.md`](data/README.md). The CSV files are intentionally excluded from Git because of their size and separate dataset terms. Check the dataset's current licence and redistribution conditions before reusing or redistributing it.

The labels are dataset labels, not independent verification of each article's claims. In particular, the dataset's REAL/FAKE categories are tied to its sources and collection process.

## Setup

### Windows PowerShell

```powershell
# Run from the project directory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

If PowerShell blocks environment activation, you can use this for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the virtual environment again.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Run tests

```bash
python -m pytest -v
```

The recorded project test run completed with **79 passed**. Tests use temporary test artifacts and do not require the full news dataset or network access.

## Train the model

1. Download the dataset and place `True.csv` and `Fake.csv` in `data/`.
2. Run training:

```bash
python train.py
```

By default, the pipeline uses a random stratified 70/15/15 train/validation/test split. For a chronological evaluation, try:

```bash
python train.py --split time
```

A time-based split is a useful additional check, but it does not eliminate source leakage by itself.

Training compares Logistic Regression and Linear SVM on validation macro-F1, evaluates the selected model on the held-out test split, and saves:

- `models/newsguard_pipeline.joblib`
- `models/model_metadata.json`

To smoke-test the training workflow without the real dataset:

```bash
python train.py --synthetic
python predict.py --model-dir models\_synthetic_smoke --text "ministry committee parliament budget statement officials"
```

**Synthetic mode is for pipeline smoke tests only. Its metrics are meaningless and must not be presented as project evaluation results.** The synthetic output directory is git-ignored.

To print the saved test metrics after training:

```powershell
python -c "import json; m=json.load(open('models/model_metadata.json')); print(json.dumps(m['test'], indent=2))"
```

## Run locally

```bash
python app.py
```

Open [[http://127.0.0.1:5000](http://127.0.0.1:5000).]

The Flask development server is intended for local testing, not production hosting. The app loads the saved model at startup; it does not train a model when the server starts.

## REST API

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_available": true,
  "model_version": "1.0.0",
  "synthetic_model": false
}
```

`status` can be `ok` or `degraded`. The endpoint returns HTTP 200 even when the model is unavailable; check `model_available` to confirm the model loaded.

### API description

```http
GET /api
```

### Predict

```http
POST /predict
Content-Type: application/json
```

Example PowerShell request:

```powershell
$body = @{
    title = "Example headline"
    text  = "Example article body with enough English text for analysis."
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The request accepts a `title`, `text`, or both. Inputs must meet the configured length limits and contain vocabulary the model recognises. The response includes a predicted label, an uncalibrated model score, explanation terms, model version, metadata, and a responsible-use disclaimer.

Typical response fields include:

- `prediction` — `REAL` or `FAKE`
- `model_score` — model-derived score from 0 to 100; not a factual probability
- `score_type` and `score_note` — describe the score's meaning and limitation
- `explanation` — terms pushing toward or against the predicted label
- `model_version`, `metadata`, and `disclaimer`

### Error handling

The API validates request content type, JSON structure, input fields, and input lengths. Expected errors include:

- `400` — invalid JSON, content type, fields, or input
- `413` — request body exceeds the configured limit
- `503` — model unavailable
- `500` — unexpected server error (generic response; details remain in server logs)

Default limits are configurable through environment variables. See `.env.example` and `config.py`.

## Deployment

NewsGuard is deployed on Render.

- **Live application:** [[https://newsguard-v6vy.onrender.com](https://newsguard-v6vy.onrender.com)]
- **Health endpoint:** [[https://newsguard-v6vy.onrender.com/health](https://newsguard-v6vy.onrender.com/health)]
- **Hosting configuration:** [[`render.yaml`](render.yaml)] and [[`Procfile`](Procfile)]

The service uses Gunicorn and loads the committed trained model at startup; it does not retrain the model during deployment. The Render free instance may spin down after inactivity, so the first request after a quiet period can be delayed.

### Verify the live service

Run these commands in PowerShell:

```powershell
$base = "https://newsguard-v6vy.onrender.com"

Invoke-WebRequest "$base/" | Select-Object StatusCode
Invoke-RestMethod "$base/health"
```

The health response should report `status: ok`, `model_available: true`, and `synthetic_model: false`. Also submit a sample article through the web interface to verify the prediction workflow.

The deployed URL is Render-provided; no separate custom domain is configured in this README.

## Limitations and responsible use

- **Not fact-checking:** the system does not compare claims with trusted sources or external evidence.
- **Source/style leakage:** the dataset's classes are associated with different publishers and writing styles. Removing URLs, datelines, and source markers cannot remove all such signals.
- **Random-split optimism:** articles about the same events may appear across train and test sets, inflating apparent generalisation.
- **Dataset-specific labels:** `FAKE` and `REAL` are proxies based on the dataset's labelling and collection method, not fresh claim-by-claim verification.
- **Uncalibrated scores:** the model can be confidently wrong; the score is not the probability that a story is true.
- **Limited coverage:** the dataset is mainly English-language news from a particular period and context. Performance may degrade on new publishers, topics, languages, satire, opinion, or breaking news.
- **Potential bias:** topical and political associations in the source data can influence predictions.
- **Model file safety:** only load model files you trained or otherwise trust. Joblib/pickle-based files can execute code when loaded.

Do not use NewsGuard as the sole basis for publishing decisions, moderation, reputational judgements, or labelling people and publishers.

## Troubleshooting

| Problem | What to check |
|---|---|
| `python` not found | Install Python 3.11+; on Windows, try `py -3.11`. |
| PowerShell blocks activation | Use the execution-policy command in [Setup](#setup), or run the environment's Python directly. |
| `ModuleNotFoundError` | Activate `.venv` and run `pip install -r requirements-dev.txt` from the project root. |
| `True.csv` or `Fake.csv` not found | Put both files in `data/`, or pass the appropriate `--data-dir` argument. |
| `/health` reports `model_available: false` | Train the model or check `NEWSGUARD_MODEL_PATH` and `NEWSGUARD_METADATA_PATH`. |
| Prediction returns `503` | Check model loading and package versions; retrain using the pinned dependencies if necessary. |
| Input has no recognised terms | Use English news-like text containing terms represented in the training vocabulary. |
| Port 5000 is already in use | Set `PORT` to another port if supported by your local configuration, or stop the other process. |
| Gunicorn fails on Windows | Expected; use `python app.py` locally. Use Gunicorn on supported deployment hosts. |
| First request on a free host is slow | The service may have been idle or restarted; retry and check the provider's current plan behaviour. |

## Future Improvements

- **Robust Evaluation:** Evaluate using time-based and source-grouped data splits to reduce data leakage and assess generalisation.
- **Cross-Dataset Testing:** Test on independent datasets containing unseen publishers, topics, and time periods.
- **Probability Calibration:** Evaluate score calibration and apply methods such as Platt scaling using validation data, if appropriate.
- **Explainability:** Enhance term-level explanations with SHAP-based feature attribution and global feature-importance analysis.
- **Feature Engineering:** Experiment with readability, sentiment, and alternative text representations to measure their impact on performance.
- **Multilingual Support:** Extend to additional languages using suitable training data and language-specific evaluation.
- **API Reliability:** Add rate limiting, request metrics, structured logging, and operational monitoring.
- **Model Tracking:** Integrate MLflow to track experiments, evaluation metrics, and model versions.

## Author

**Shobhan Satpathy**  
GitHub: [[@Shobhan-04](https://github.com/Shobhan-04)]  
Project: [[NewsGuard](https://github.com/Shobhan-04/NewsGuard)]

**Licence:** See [`LICENSE`](LICENSE) for code licensing. Dataset terms are separate and must be checked independently.
