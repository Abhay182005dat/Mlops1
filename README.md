# 🎯 YT Sentiment Lens — YouTube Comment Sentiment Analyser

> **End-to-end MLOps project** — from raw data to a production-deployed Chrome Extension powered by a LightGBM NLP model, with a fully automated CI/CD pipeline.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-brightgreen?logo=lightgbm)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20EC2%20%7C%20S3-FF9900?logo=amazonaws)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)

---

## 📌 Project Summary

**YT Sentiment Lens** is a full-stack MLOps project that analyses the sentiment of YouTube comments in real time through a Chrome Extension. A user opens any YouTube video, clicks the extension, and instantly sees comment sentiment broken down as positive, neutral, or negative — complete with visual charts, a word cloud, and a sentiment trend graph over time.

This project demonstrates a **production-grade ML workflow**: data ingestion → preprocessing → model training → evaluation → MLflow model registry → automated CI/CD → Docker containerization → AWS ECR deployment.

---

## 🏗️ Architecture Overview

```
YouTube API
     │
     ▼
Chrome Extension (popup.js)
     │  POST /predict_with_timestamps
     ▼
Flask REST API  ◄──── MLflow Model Registry (AWS EC2)
     │                       │
     │               LightGBM Classifier
     │               TF-IDF Vectorizer
     ▼
Sentiment Results + Charts + Word Cloud + Trend Graph
```

**Infrastructure:**
```
GitHub Push
    │
    ▼
GitHub Actions CI/CD
    ├── dvc repro  (runs full ML pipeline)
    ├── pytest     (model loading + performance + API tests)
    ├── promote_model.py  (Staging → Production in MLflow)
    ├── Docker build & tag
    └── Push to AWS ECR
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔴🟡🟢 Real-time Sentiment Classification | Classifies YouTube comments as Positive, Neutral, or Negative |
| 📊 Sentiment Distribution Chart | Pie chart visualization of overall comment sentiment |
| ☁️ Word Cloud | Visual representation of the most discussed topics |
| 📈 Trend Graph | Monthly sentiment percentage over time |
| 💬 Comment Browser | Browse individual comments with their predicted sentiment |
| 🔁 Re-analyze | Re-trigger analysis without reloading the page |

---

## 🧠 ML Pipeline (DVC-Managed)

The pipeline is fully reproducible and version-controlled with **DVC**, backed by **AWS S3** remote storage.

```
data_ingestion
      │
      ▼
data_preprocessing
      │
      ▼
model_building  ◄── params.yaml (hyperparameters)
      │
      ▼
model_evaluation  ──► MLflow Experiment Tracking
      │
      ▼
model_registration  ──► MLflow Model Registry (Staging)
```

**Model:** LightGBM Classifier  
**Features:** TF-IDF with trigrams (`ngram_range: [1,3]`, `max_features: 10,000`)  
**Classes:** Positive (`1`), Neutral (`0`), Negative (`-1`)

### Hyperparameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.25

model_building:
  max_features: 10000
  ngram_range: [1, 3]
  learning_rate: 0.08
  max_depth: 20
  n_estimators: 367
```

---

## 🗂️ Project Structure

```
├── .github/workflows/cicd.yaml   # Full CI/CD pipeline
├── dvc.yaml                      # DVC pipeline stages
├── dvc.lock                      # Reproducibility lock file
├── params.yaml                   # Model & pipeline hyperparameters
├── Dockerfile                    # Flask app containerization
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py     # Fetch & split raw data
│   │   └── data_preprocessing.py # Clean, normalize, lemmatize
│   └── model/
│       ├── model_building.py     # TF-IDF + LightGBM training
│       ├── model_evaluation.py   # MLflow logging & metrics
│       └── register_model.py     # Push model to MLflow registry
│
├── flask_app/
│   └── app.py                    # REST API (predict, charts, wordcloud, trends)
│
├── frontend/
│   ├── manifest.json             # Chrome Extension manifest
│   ├── popup.html                # Extension UI
│   └── popup.js                  # YouTube API + Flask API integration
│
└── scripts/
    ├── test_model_loading.py     # Pytest: model loads from MLflow registry
    ├── test_model_performance.py # Pytest: accuracy/F1 threshold checks
    ├── test_model_signature.py   # Pytest: input/output shape validation
    ├── test_flask_app.py         # Pytest: all Flask API endpoints
    └── promote_model.py          # Staging → Production promotion
```

---

## 🚀 CI/CD Pipeline (GitHub Actions)

Every `git push` triggers the full automated pipeline:

1. **Environment Setup** — Python 3.10, pip cache, dependencies
2. **`dvc repro`** — Runs the full ML pipeline end-to-end
3. **`dvc push`** — Syncs data artifacts to S3
4. **Auto Git Commit** — Commits updated `dvc.lock` back to repo (bot-guarded to prevent infinite loops)
5. **Model Tests** — Loading test, performance threshold test
6. **Model Promotion** — Automatically promotes passing model from Staging → Production in MLflow
7. **Flask API Tests** — End-to-end API tests via pytest
8. **Docker Build & Push** — Builds image and pushes to AWS ECR

---

## 🛠️ Tech Stack

**Machine Learning**
- LightGBM · scikit-learn · NLTK · TF-IDF

**Experiment Tracking & Model Registry**
- MLflow (hosted on AWS EC2) · DVC (data versioning) · AWS S3 (artifact storage)

**Backend**
- Flask · Flask-CORS · Matplotlib · WordCloud · Pandas

**Frontend**
- Chrome Extension (Manifest V3) · Vanilla JS · YouTube Data API v3

**Infrastructure & DevOps**
- Docker · AWS ECR · AWS EC2 · GitHub Actions (CI/CD)

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10
- AWS CLI configured (`aws configure`)
- DVC with S3 support

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/yt-comment-sentiment-analyser.git
cd yt-comment-sentiment-analyser
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull data and run the ML pipeline

```bash
dvc pull          # Download data artifacts from S3
dvc repro         # Reproduce the full pipeline
```

### 3. Run the Flask API

```bash
python flask_app/app.py
# API available at http://localhost:5000
```

### 4. Load the Chrome Extension

1. Open Chrome → `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select the `frontend/` folder
4. Navigate to any YouTube video and click the extension icon

---

## 🧪 Tests

```bash
# Model tests
pytest scripts/test_model_loading.py
pytest scripts/test_model_performance.py

# API tests (requires Flask running)
pytest scripts/test_flask_app.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t yt-comment-analyser .

# Run locally
docker run -p 5000:5000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  yt-comment-analyser
```

---

## 📊 MLflow Tracking

MLflow is hosted on an AWS EC2 instance with S3 as the artifact store.

- **Tracking URI:** `http://<ec2-instance>:5000`
- **Experiment:** `dvc-pipeline-runs`
- **Model Registry:** `yt_chrome_plugin_model`
- **Stages:** Staging → Production (automated via CI/CD)

Logged per run: classification report (precision, recall, F1 per class), confusion matrix, model artifact, TF-IDF vectorizer, all hyperparameters.

---

## 🗺️ Roadmap

- [ ] Spam & troll detection (separate ML model)
- [ ] Comment summarisation using LLM (OpenAI API)
- [ ] Comment categorisation (feedback, concern, praise, etc.)
- [ ] Average comment length analytics
- [ ] Support for multi-language comments

---

## 👤 Author

**Abhay**  
MLOps · Machine Learning · Backend Development

---

## 📄 License

This project is licensed under the MIT License.
