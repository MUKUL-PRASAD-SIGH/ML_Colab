"""
train_models.py
───────────────
Offline training script for the NLP Model Comparison Dashboard.
Trains NLTK + spaCy pipelines on the IMDb / SST-2 dataset and saves:
  • models/nltk_model.pkl
  • models/nltk_vectorizer.pkl
  • models/spacy_model.pkl
  • models/spacy_vectorizer.pkl
Run once before launching the Streamlit dashboard.
"""

import os, re, time, warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── standard libs ──────────────────────────────────────────────────────────
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── datasets ───────────────────────────────────────────────────────────────
print("📦  Loading dataset …")
from datasets import load_dataset
dataset = load_dataset("sst2", split="train")
texts  = dataset["sentence"]
labels = dataset["label"]          # 0 = negative, 1 = positive

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  NLTK PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔵  Training NLTK pipeline …")
import nltk
nltk.download("punkt",        quiet=True)
nltk.download("stopwords",    quiet=True)
nltk.download("punkt_tab",    quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS_EN = set(stopwords.words("english"))

def preprocess_nltk(text: str) -> str:
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 1]
    return " ".join(tokens)

t0                = time.time()
X_train_nltk      = [preprocess_nltk(t) for t in X_train]
X_test_nltk       = [preprocess_nltk(t) for t in X_test]

vec_nltk          = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
X_train_vec_nltk  = vec_nltk.fit_transform(X_train_nltk)
X_test_vec_nltk   = vec_nltk.transform(X_test_nltk)

model_nltk        = LogisticRegression(max_iter=1000, C=1.0)
model_nltk.fit(X_train_vec_nltk, y_train)

preds_nltk        = model_nltk.predict(X_test_vec_nltk)
acc_nltk          = accuracy_score(y_test, preds_nltk)
elapsed_nltk      = time.time() - t0

print(f"    ✅  Accuracy : {acc_nltk:.4f}  |  Time: {elapsed_nltk:.1f}s")
print(classification_report(y_test, preds_nltk, target_names=["Negative", "Positive"]))

joblib.dump(model_nltk,  os.path.join(MODELS_DIR, "nltk_model.pkl"))
joblib.dump(vec_nltk,    os.path.join(MODELS_DIR, "nltk_vectorizer.pkl"))
print("    💾  NLTK model saved.")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  spaCy PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
print("\n🟢  Training spaCy pipeline …")
import spacy
try:
    nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("    spaCy model not found — downloading en_core_web_sm …")
    os.system("python -m spacy download en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_spacy(text: str) -> str:
    doc    = nlp_spacy(text.lower())
    tokens = [
        tok.lemma_ for tok in doc
        if not tok.is_stop and not tok.is_punct and len(tok.text) > 1
    ]
    return " ".join(tokens)

t0                 = time.time()
# batch process for speed
texts_train_spacy  = list(X_train)
texts_test_spacy   = list(X_test)

X_train_spacy      = [preprocess_spacy(t) for t in texts_train_spacy]
X_test_spacy       = [preprocess_spacy(t) for t in texts_test_spacy]

vec_spacy          = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
X_train_vec_spacy  = vec_spacy.fit_transform(X_train_spacy)
X_test_vec_spacy   = vec_spacy.transform(X_test_spacy)

model_spacy        = LogisticRegression(max_iter=1000, C=1.0)
model_spacy.fit(X_train_vec_spacy, y_train)

preds_spacy        = model_spacy.predict(X_test_vec_spacy)
acc_spacy          = accuracy_score(y_test, preds_spacy)
elapsed_spacy      = time.time() - t0

print(f"    ✅  Accuracy : {acc_spacy:.4f}  |  Time: {elapsed_spacy:.1f}s")
print(classification_report(y_test, preds_spacy, target_names=["Negative", "Positive"]))

joblib.dump(model_spacy, os.path.join(MODELS_DIR, "spacy_model.pkl"))
joblib.dump(vec_spacy,   os.path.join(MODELS_DIR, "spacy_vectorizer.pkl"))
print("    💾  spaCy model saved.")

# ── summary ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  TRAINING COMPLETE")
print("="*55)
print(f"  NLTK   accuracy : {acc_nltk:.4f}")
print(f"  spaCy  accuracy : {acc_spacy:.4f}")
print(f"\n  Models saved in: ./{MODELS_DIR}/")
print("  Run the dashboard with:  streamlit run app.py")
print("="*55)
