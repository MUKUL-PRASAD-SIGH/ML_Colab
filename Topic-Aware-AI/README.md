# Topic-Aware AI Dashboard

> **Advanced Hybrid NLP** — Gensim LDA + DistilBERT Transformers

A full-stack, educational Streamlit dashboard (mirroring the UI/UX pattern of the Sentiment Analyser). It visualises how topic modelling and deep semantic context combine into a highly effective **Hybrid Vector Representation**.

---

## 🎯 What This Project Does

Builds an end-to-end system that captures BOTH:
| Component | Tool | What It Captures |
|-----------|------|-----------------|
| **Topic Distributions** | LDA (Gensim) | Explainable structure (5 Dimensions) |
| **Semantic Embeddings** | DistilBERT (Transformers) | Deep contextual meaning (768 Dimensions) |
| **Hybrid Vector** | Concatenation | Both meaning + topics (773 Dimensions) |

---

## 📂 Project Structure

```
Topic-Aware-AI/
├── app.py                  # 🖥️ Streamlit Dashboard (Live UI)
├── pipelines.py            # ⚙️ All backend processing, modeling & search logic
├── train_models.py         # 🏋️ Offline pipeline to build models & datasets
├── requirements.txt        # 📦 Dependencies
├── README.md               # 📖 Documentation
├── FEATURES.md             # 📝 Technical features detail
├── data/                   # 💾 Saved vectors (.npy) & raw data
└── models/                 # 📂 Scikit-learn & Gensim (.pkl / .gensim) models
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Offline Pipeline
Generates the base models, vectorises 2,000 IMDB reviews, calculates UMAP projections, and stores everything in `data/` and `models/`.
```bash
python train_models.py
```
*(Takes ~5-10 minutes. Utilises CUDA if available.)*

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📱 Dashboard Features

### 🔬 1. Step-by-Step Pipeline
Type any text and watch exactly how it is processed:
1. **Gensim pre-processing** (stops removed) → BoW → 5-D Topic inference
2. **DistilBERT encoding** → Mean Pooling → 768-D Context inference
3. **Fusion** → L2-normed 773-D Hybrid Vector

### 🏋️ 2. Train Classifier (Feature A)
- Real-time training of `LogisticRegression`, `SVM`, or `MLP` on the 773-D hybrid features.
- Directly compares Hybrid accuracy vs standard BERT-only accuracy.
- Displays detailed classification reports.

### 📉 3. Visualise & Explore (Feature B)
- **UMAP/PCA Projections:** Interactive 2-D scatter plot of the semantic space, color-coded by Dominant Topic.
- **Topic Interperting:** Word clouds/distributions for Gensim's learned topics.

### 🔍 4. Search Engine (Feature C)
- Enter complex, conceptual queries (e.g. *"A mind bending sci-fi about time travel"*).
- Finds visually similar reviews using Cosine Similarity on the fused 773-D vectors.
- Supports sentiment filtering (Positive/Negative only).

---

## 🔧 Technology Stack
- **Dashboard:** Streamlit, Plotly
- **Machine Learning:** Scikit-Learn, Gensim, Transformers (HuggingFace)
- **Vectors:** NumPy (High-performance array storage)
- **Visualisation:** UMAP-learn

---

## 🧠 Why Hybrid?
- **BERT alone** focuses heavily on sequence and local context. It might cluster reviews that *sound* structurally similar but are about completely different movies.
- **LDA alone** groups by high-level semantic themes (e.g., *Comedy vs Action*) but lacks the nuance to detect subtle sentiment or contextual meaning (like sarcasm or double-negatives).
- **Together**, the model understands that a review is *both* "a highly positive sentiment" (BERT) AND "about a sci-fi action movie" (LDA).
