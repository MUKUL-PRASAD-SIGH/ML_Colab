"""
pipelines.py
────────────
Reusable inference wrappers for each NLP pipeline.
Imported by app.py (Streamlit dashboard).
"""

from __future__ import annotations

import os, re, time, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np

MODELS_DIR = "models"

# ───────────────────────────── lazy singletons ─────────────────────────────
_nltk_ready   = False
_spacy_ready  = False
_trans_ready  = False

_model_nltk   = None
_vec_nltk     = None
_nlp_spacy    = None
_model_spacy  = None
_vec_spacy    = None
_classifier   = None

# ═══════════════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_nltk_pipeline():
    global _model_nltk, _vec_nltk, _nltk_ready
    if _nltk_ready:
        return True
    mpath = os.path.join(MODELS_DIR, "nltk_model.pkl")
    vpath = os.path.join(MODELS_DIR, "nltk_vectorizer.pkl")
    if not (os.path.exists(mpath) and os.path.exists(vpath)):
        return False
    _model_nltk  = joblib.load(mpath)
    _vec_nltk    = joblib.load(vpath)
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    _nltk_ready = True
    return True


def load_spacy_pipeline():
    global _nlp_spacy, _model_spacy, _vec_spacy, _spacy_ready
    if _spacy_ready:
        return True
    mpath = os.path.join(MODELS_DIR, "spacy_model.pkl")
    vpath = os.path.join(MODELS_DIR, "spacy_vectorizer.pkl")
    if not (os.path.exists(mpath) and os.path.exists(vpath)):
        return False
    import spacy
    _nlp_spacy   = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    _model_spacy = joblib.load(mpath)
    _vec_spacy   = joblib.load(vpath)
    _spacy_ready = True
    return True


def load_transformer_pipeline():
    global _classifier, _trans_ready
    if _trans_ready:
        return True
    from transformers import pipeline
    _classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,        # CPU (change to 0 for GPU)
        truncation=True,
        max_length=512,
    )
    _trans_ready = True
    return True


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSORS
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_nltk(text: str) -> str:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    sw        = set(stopwords.words("english"))
    text      = text.lower()
    text      = re.sub(r"[^a-z\s]", "", text)
    tokens    = word_tokenize(text)
    tokens    = [t for t in tokens if t not in sw and len(t) > 1]
    return " ".join(tokens)


def _preprocess_spacy(text: str) -> str:
    doc    = _nlp_spacy(text.lower())
    tokens = [
        tok.lemma_ for tok in doc
        if not tok.is_stop and not tok.is_punct and len(tok.text) > 1
    ]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict_nltk(text: str) -> dict:
    """Returns {label, confidence, time_ms}"""
    t0        = time.perf_counter()
    processed = _preprocess_nltk(text)
    X         = _vec_nltk.transform([processed])
    pred      = _model_nltk.predict(X)[0]
    proba     = _model_nltk.predict_proba(X)[0]
    elapsed   = (time.perf_counter() - t0) * 1000

    label      = "Positive" if pred == 1 else "Negative"
    confidence = float(max(proba))
    return {"label": label, "confidence": round(confidence, 4), "time_ms": round(elapsed, 2)}


def predict_spacy(text: str) -> dict:
    """Returns {label, confidence, time_ms}"""
    t0        = time.perf_counter()
    processed = _preprocess_spacy(text)
    X         = _vec_spacy.transform([processed])
    pred      = _model_spacy.predict(X)[0]
    proba     = _model_spacy.predict_proba(X)[0]
    elapsed   = (time.perf_counter() - t0) * 1000

    label      = "Positive" if pred == 1 else "Negative"
    confidence = float(max(proba))
    return {"label": label, "confidence": round(confidence, 4), "time_ms": round(elapsed, 2)}


def predict_transformer(text: str) -> dict:
    """Returns {label, confidence, time_ms}"""
    t0      = time.perf_counter()
    result  = _classifier(text)[0]
    elapsed = (time.perf_counter() - t0) * 1000

    raw_label  = result["label"]          # 'POSITIVE' / 'NEGATIVE'
    label      = raw_label.capitalize()
    confidence = round(float(result["score"]), 4)
    return {"label": label, "confidence": confidence, "time_ms": round(elapsed, 2)}


def run_all_pipelines(text: str) -> dict:
    """
    Runs all three pipelines and returns a dict keyed by pipeline name.
    Unloaded pipelines return an error entry instead of crashing.
    """
    results = {}
    for name, ready_flag, fn in [
        ("NLTK",        _nltk_ready,  predict_nltk),
        ("spaCy",       _spacy_ready, predict_spacy),
        ("Transformer", _trans_ready, predict_transformer),
    ]:
        if not ready_flag:
            results[name] = {"error": "Model not loaded"}
        else:
            try:
                results[name] = fn(text)
            except Exception as exc:
                results[name] = {"error": str(exc)}
    return results
