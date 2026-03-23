# Topic-Aware AI Dashboard

> **Advanced Hybrid NLP** — Gensim LDA + DistilBERT Transformers  
> An educational masterclass platform that teaches complex AI concepts through an interactive live UI.

---

## 🎯 What This Project Does

Builds an end-to-end hybrid AI system that captures BOTH:

| Component | Tool | Output |
|-----------|------|--------|
| **Topic Discovery** | LDA (Gensim) | "This is 90% Sci-Fi, 10% Action" |
| **Deep Semantic Meaning** | DistilBERT (Transformers) | 768 numbers representing context & tone |
| **Hybrid Fusion** | L2-Norm + Concat | One super-list of 768+K numbers |

---

## 📂 Project Structure

```
Topic-Aware-AI/
├── app.py               🖥️  Streamlit Educational Dashboard (The main app)
├── pipelines.py         ⚙️  All backend logic (data, models, search, train)
├── train_models.py      🏋️  Offline terminal alternative (same as the UI builder)
├── code_dump.py         📜  Master code reference file (code for every step)
├── requirements.txt     📦  All dependencies
├── README.md            📖  This file
├── FEATURES.md          📝  Deep-dive feature documentation
├── data/                💾  Saved vectors (.npy) & raw text data
└── models/              📂  Saved model files (.gensim, .pkl)
```

---

## 🚀 Quick Start (2 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Dashboard
```bash
streamlit run app.py
```

The dashboard includes a **Build Engine tab** that trains the whole system in-browser. No external commands needed!

---

## 📱 Dashboard Tabs

| Tab | Icon | What You'll Learn |
|-----|------|--------------------|
| **The Theory** | 📖 | What LDA does, what BERT does, why we combine them, and the L2 Normalization problem |
| **Build Engine** | ⚙️ | Build the AI live using IMDB, a built-in fast dataset, or your own CSV file |
| **Step-by-Step** | 🔬 | Watch your sentence get turned into numbers in real-time with full explanations |
| **Classify (A)** | 🏋️ | Train Logistic Regression / SVM / MLP on the hybrid features and compare scores |
| **Visualise (B)** | 📉 | See a 2D Galaxy Map of all documents using UMAP |
| **Search (C)** | 🔍 | Search by meaning (not exact words) using Cosine Similarity |

---

## ⚙️ Dataset Options

| Option | Time | Best For |
|--------|------|---------|
| ⚡ Fast Built-In Dataset | ~30 seconds | Quick exploration, first-time testing |
| 🌐 Full IMDB Download | ~3-10 min (80MB) | Production-quality experiments |
| 📁 Custom CSV Upload | Depends on size | Any domain, any text column |

> CSV must have `text` (string) and `label` (integer: 0 or 1) columns.

---

## Alternatively - Offline CLI Mode

```bash
# Run the offline terminal pipeline instead of the UI
python train_models.py

# Then launch the dashboard
streamlit run app.py
```

---

## 🔧 Technology Stack

| Role | Tool |
|------|------|
| Dashboard | Streamlit, Plotly |
| Topic Modelling | Gensim, LdaModel |
| Transformers | HuggingFace, DistilBERT |
| Classifiers | Scikit-Learn |
| Dimensionality Reduction | UMAP-learn (PCA fallback) |
| Storage | NumPy .npy arrays, JSON |

---

## 🧠 Why Hybrid?

- **BERT alone** can't tell you *what* a review is about. It understands *how* it's said.
- **LDA alone** counts keywords. It can't understand sarcasm or context ("not terrible" → positive).
- **Together**: The AI understands both **What the text is about** (LDA) and **How it says it** (BERT). No single model can do both.
