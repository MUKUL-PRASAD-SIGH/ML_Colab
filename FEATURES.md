# NLP Model Comparison Dashboard — Feature Documentation

> **Last Updated:** 2026-03-22  
> **Stack:** Python · NLTK · spaCy · HuggingFace Transformers · scikit-learn · Streamlit · Plotly

---

## 📋 Project Overview

An interactive web dashboard that compares **three NLP sentiment-analysis pipelines** side-by-side in real time. The user types any English sentence and instantly sees:

- Prediction (Positive / Negative)
- Confidence score
- Inference time

from NLTK, spaCy, and a BERT-family Transformer — all visualised with Plotly charts.

---

## 🗂 File Structure

```
Sentiment Analysis/
├── app.py                   # Streamlit dashboard (main entry point)
├── pipelines.py             # Inference wrappers for all 3 models
├── train_models.py          # Offline training script (NLTK + spaCy)
├── nlp_comparison.ipynb     # 📓 Research notebook (EDA + training + evaluation)
├── requirements.txt         # Python dependencies
├── FEATURES.md              # ← This file
└── models/                  # Auto-created by train_models.py OR notebook
    ├── nltk_model.pkl
    ├── nltk_vectorizer.pkl
    ├── spacy_model.pkl
    └── spacy_vectorizer.pkl
```

---

## ✅ Implemented Features

### 1. 🔵 NLTK Pipeline
- **Preprocessing:** word tokenisation → stopword removal → TF-IDF (20k features, 1-2 grams)
- **Model:** Logistic Regression (scikit-learn), trained on SST-2
- **Output:** label, confidence (`predict_proba`), inference time in ms
- **Serialised:** `models/nltk_model.pkl` + `models/nltk_vectorizer.pkl`

### 2. 🟢 spaCy Pipeline
- **Preprocessing:** lemmatisation (`en_core_web_sm`) → stopword/punct removal → TF-IDF
- **Model:** Logistic Regression, trained on SST-2
- **Output:** label, confidence, inference time
- **Serialised:** `models/spacy_model.pkl` + `models/spacy_vectorizer.pkl`

### 3. 🔴 Transformer Pipeline
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace)
- **Preprocessing:** None — tokenised internally by BERT WordPiece tokeniser
- **Lazy-loaded:** fetched/cached automatically on first run
- **Output:** label, score, inference time

### 4. 🖥 Streamlit Dashboard (`app.py`)
- Dark glassmorphism UI (Inter font, indigo/violet palette)
- **Tab 1 — Single Sentence:** text input, example buttons, three pipeline result cards
- **Tab 2 — Batch / CSV:** upload CSV, run all pipelines on every row, download results
- **Tab 3 — How It Works:** pipeline explanations + trade-off table

### 5. 📊 Live Visualisations
- **Confidence Bar Chart** — per-model confidence percentages
- **Inference Time Bar Chart** — speed comparison
- **Radar Chart** — multi-axis confidence spider plot
- **Summary Table** — all results in one DataFrame

### 6. 🗂 Batch Mode
- Upload CSV with `text` column (+ optional `label` column)
- Processes up to 500 rows
- Displays per-model accuracy if ground-truth labels supplied
- CSV download of all results

### 7. ⚙️ Sidebar Status Panel
- Live load status for each pipeline (✅ / ⚠️)
- Cached via `@st.cache_resource` — models load only once per session

### 8. 📓 Research Notebook (`nlp_comparison.ipynb`)
A complete end-to-end Jupyter notebook — use this for exploration, presentations, or to re-train models without the CLI script.

| Section | Contents |
|---------|----------|
| **1 — Setup** | Imports, matplotlib dark theme, directory creation |
| **2 — EDA** | Class distribution, word/char count histograms, sample sentences |
| **3 — NLTK Pipeline** | Preprocessing demo, token reduction table, TF-IDF, LR training, feature weights bar chart, confusion matrix |
| **4 — spaCy Pipeline** | Lemmatisation demo, NLTK vs spaCy side-by-side comparison, TF-IDF, LR training, feature weights, confusion matrix |
| **5 — Transformer** | DistilBERT load, single inference, batch eval on 800-sample subset, confusion matrix |
| **6 — Comparison** | ROC curves (all 3 models), accuracy bar chart, speed benchmark, interactive Plotly dashboard, summary metrics table, disagreement analysis |
| **7 — Live Demo** | `predict_all()` function — runs all 3 pipelines, returns DataFrame |
| **8 — Save Models** | `joblib.dump` → `models/` directory, file size listing |

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Train NLTK + spaCy models
```bash
python train_models.py
```
Expected output:
```
NLTK   accuracy : ~0.83
spaCy  accuracy : ~0.84
Models saved in: ./models/
```

### Step 3 — Launch dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

> **Note:** The Transformer model is downloaded automatically on first run (~270 MB).  
> Subsequent runs use the local HuggingFace cache.

---

## 📈 Model Performance (SST-2 test set)

| Model       | Accuracy | Inference (typical) |
|-------------|----------|---------------------|
| NLTK        | ~83%     | 2–8 ms              |
| spaCy       | ~84%     | 4–12 ms             |
| Transformer | ~91%     | 80–200 ms (CPU)     |

---

## 🔥 Next-Level Additions (Roadmap)

| Feature | Status |
|---------|--------|
| BERT fine-tuning on full IMDb dataset | 🔲 Planned |
| Token-level attention visualisation | 🔲 Planned |
| GPU toggle in sidebar | 🔲 Planned |
| Streamlit Cloud deployment | 🔲 Planned |
| Live accuracy trend chart (session history) | 🔲 Planned |

---

## 💼 Resume Bullet

> Built a production-style NLP dashboard comparing traditional (NLTK, spaCy) and deep-learning (DistilBERT) sentiment-analysis pipelines; surfaced real-time confidence scores, inference latency, and batch accuracy metrics via an interactive Streamlit + Plotly UI.
