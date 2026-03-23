"""
pipelines.py  ─  All backend logic for the Topic-Aware AI Dashboard
====================================================================
Handles:
  - Data loading (IMDB via HuggingFace)
  - Preprocessing (Gensim)
  - LDA topic model (Gensim)
  - DistilBERT embeddings
  - Hybrid feature fusion (BERT + LDA)
  - Classifier training (A) — Logistic Regression / SVM / MLP on hybrid features
  - Visualisation data (B) — UMAP 2-D projections, PCA, topic word clouds
  - Search engine (C) — cosine similarity over hybrid vectors
"""

from __future__ import annotations
import os, time, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Global handles
# ─────────────────────────────────────────────
_lda         = None
_dictionary  = None
_tokenizer   = None
_bert_model  = None
_classifier  = None          # sklearn classifier (hybrid features)

_lda_ready       = False
_bert_ready      = False
_hybrid_ready    = False     # hybrid features on disk
_classifier_ready = False

# ═══════════════════════════════════════════════════════════════════════════
# LOADERS  (idempotent — safe to call multiple times)
# ═══════════════════════════════════════════════════════════════════════════

def load_lda() -> bool:
    global _lda, _dictionary, _lda_ready
    if _lda_ready:
        return True
    dpath = os.path.join(MODELS_DIR, "lda_dictionary.gensim")
    mpath = os.path.join(MODELS_DIR, "lda_model.gensim")
    if not (os.path.exists(dpath) and os.path.exists(mpath)):
        return False
    from gensim import corpora
    from gensim.models import LdaModel
    _dictionary = corpora.Dictionary.load(dpath)
    _lda        = LdaModel.load(mpath)
    _lda_ready  = True
    return True


def load_bert() -> bool:
    global _tokenizer, _bert_model, _bert_ready
    if _bert_ready:
        return True
    from transformers import AutoTokenizer, AutoModel
    _tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    _bert_model.eval()
    _bert_ready = True
    return True


def load_classifier() -> bool:
    global _classifier, _classifier_ready
    if _classifier_ready:
        return True
    mpath = os.path.join(MODELS_DIR, "hybrid_classifier.pkl")
    if not os.path.exists(mpath):
        return False
    _classifier       = joblib.load(mpath)
    _classifier_ready = True
    return True


def _hybrid_files_exist() -> bool:
    return (
        os.path.exists(os.path.join(DATA_DIR, "hybrid_features.npy")) and
        os.path.exists(os.path.join(DATA_DIR, "labels.npy"))          and
        os.path.exists(os.path.join(DATA_DIR, "raw_data.json"))
    )


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_STOP_WORDS = None

def _get_stop_words():
    global CUSTOM_STOP_WORDS
    if CUSTOM_STOP_WORDS is None:
        from gensim.parsing.preprocessing import STOPWORDS
        CUSTOM_STOP_WORDS = STOPWORDS.union({"film", "movie", "br", "one", "would", "could", "like"})
    return CUSTOM_STOP_WORDS


def _preprocess_text(text: str) -> list[str]:
    from gensim.utils import simple_preprocess
    stop = _get_stop_words()
    return [t for t in simple_preprocess(text, deacc=True) if t not in stop and len(t) > 2]


def _mean_pool(last_hidden, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _get_bert_embedding(text: str, max_len: int = 256) -> np.ndarray:
    import torch
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        out = _bert_model(**inputs)
    vec = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
    return vec.squeeze().numpy()


def _get_lda_vector(text: str, num_topics: int = 5) -> np.ndarray:
    tokens = _preprocess_text(text)
    bow    = _dictionary.doc2bow(tokens)
    dist   = _lda.get_document_topics(bow, minimum_probability=0.0)
    vec    = np.zeros(num_topics)
    for tid, prob in dist:
        if tid < num_topics:
            vec[tid] = prob
    return vec


def _make_hybrid(text: str) -> np.ndarray:
    bert  = _get_bert_embedding(text)
    bert  = bert / (np.linalg.norm(bert) + 1e-9)
    lda_v = _get_lda_vector(text)
    lda_v = lda_v / (np.linalg.norm(lda_v) + 1e-9)
    return np.concatenate([bert, lda_v])


def _make_hybrid_steps(text: str) -> dict:
    """Return all intermediate representations for the step-by-step UI."""
    # LDA steps
    stop       = _get_stop_words()
    from gensim.utils import simple_preprocess
    all_tokens = simple_preprocess(text, deacc=True)
    kept_tok   = [t for t in all_tokens if t not in stop and len(t) > 2]
    removed_tok= [t for t in all_tokens if t in stop or len(t) <= 2]

    bow = _dictionary.doc2bow(kept_tok)
    dist= _lda.get_document_topics(bow, minimum_probability=0.0)
    num_topics = _lda.num_topics
    lda_raw = np.zeros(num_topics)
    for tid, prob in sorted(dist, key=lambda x: x[0]):
        lda_raw[tid] = prob

    # Topic top words
    topic_words = {}
    for tid in range(num_topics):
        words = _lda.show_topic(tid, topn=6)
        topic_words[tid] = [(w, float(p)) for w, p in words]

    # BERT steps
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    import torch
    with torch.no_grad():
        out = _bert_model(**inputs)
    bert_raw = _mean_pool(out.last_hidden_state, inputs["attention_mask"]).squeeze().numpy()
    bert_tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    # Normalise & combine
    bert_norm = bert_raw  / (np.linalg.norm(bert_raw)  + 1e-9)
    lda_norm  = lda_raw   / (np.linalg.norm(lda_raw)   + 1e-9)
    hybrid    = np.concatenate([bert_norm, lda_norm])

    return {
        "raw":           text,
        "all_tokens":    all_tokens,
        "kept_tokens":   kept_tok,
        "removed_tokens":removed_tok,
        "bow":           bow[:10],       # first 10 bag-of-words entries
        "lda_dist":      lda_raw.tolist(),
        "topic_words":   topic_words,
        "dominant_topic":int(np.argmax(lda_raw)),
        "bert_tokens":   bert_tokens,
        "bert_dim":      len(bert_raw),
        "hybrid_dim":    len(hybrid),
        "hybrid":        hybrid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE A — CLASSIFIER TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_classifier(model_type: str = "LogisticRegression", progress_cb=None) -> dict:
    """
    Train a classifier on pre-built hybrid features.
    model_type: 'LogisticRegression' | 'SVM' | 'MLP'
    progress_cb(msg, pct) — live Streamlit callback
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.calibration import CalibratedClassifierCV

    def log(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    log("📂  Loading hybrid features …", 0.05)
    X = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))

    log(f"✂️  Splitting: 80% train / 20% test ({len(X)} samples) …", 0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log(f"🏋️  Training {model_type} on {X_train.shape} hybrid features …", 0.35)
    if model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    elif model_type == "SVM":
        clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0))
    else:  # MLP
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=42)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    log("📊  Evaluating …", 0.80)
    preds  = clf.predict(X_test)
    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds,
                                   target_names=["Negative", "Positive"],
                                   output_dict=True)
    cm     = confusion_matrix(y_test, preds).tolist()

    # Compare: BERT-only accuracy (using only first 768 dims)
    log("🔬  Computing BERT-only baseline …", 0.90)
    X_bert = X[:, :768]
    Xb_tr, Xb_te, _, _ = train_test_split(X_bert, y, test_size=0.2, random_state=42)
    clf_bert = LogisticRegression(max_iter=1000, C=1.0).fit(Xb_tr, y_train)
    acc_bert_only = accuracy_score(y_test, clf_bert.predict(Xb_te))

    log("💾  Saving classifier …", 0.95)
    global _classifier, _classifier_ready
    _classifier       = clf
    _classifier_ready = True
    joblib.dump(clf, os.path.join(MODELS_DIR, "hybrid_classifier.pkl"))

    log("✅  Done!", 1.0)
    return {
        "model_type":      model_type,
        "accuracy":        acc,
        "bert_only_acc":   acc_bert_only,
        "improvement":     acc - acc_bert_only,
        "report":          report,
        "confusion_matrix":cm,
        "train_time":      train_time,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "feature_dim":     X.shape[1],
    }


def predict_hybrid(text: str) -> dict:
    """Inference on a single text using the hybrid classifier."""
    if not (_lda_ready and _bert_ready and _classifier_ready):
        return None

    t0     = time.perf_counter()
    steps  = _make_hybrid_steps(text)
    vec    = steps["hybrid"]
    pred   = _classifier.predict([vec])[0]
    proba  = _classifier.predict_proba([vec])[0]
    elapsed= (time.perf_counter() - t0) * 1000

    return {
        "label":      "Positive" if pred == 1 else "Negative",
        "confidence": float(max(proba)),
        "proba_neg":  float(proba[0]),
        "proba_pos":  float(proba[1]),
        "time_ms":    round(elapsed, 2),
        "steps":      steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE B — VISUALISATION DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_topic_words(num_words: int = 10) -> dict[int, list[tuple]]:
    """Return top words per topic."""
    if not _lda_ready:
        return {}
    topics = {}
    for tid in range(_lda.num_topics):
        topics[tid] = [(w, float(p)) for w, p in _lda.show_topic(tid, topn=num_words)]
    return topics


def get_umap_data(n_samples: int = 500, use_hybrid: bool = True) -> dict:
    """
    Compute 2-D UMAP projection of the corpus embeddings.
    Returns dict with x, y, labels, texts, topic_ids arrays.
    """
    import json
    X_full = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    y_full = np.load(os.path.join(DATA_DIR, "labels.npy"))

    with open(os.path.join(DATA_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)

    with open(os.path.join(DATA_DIR, "topic_distributions.json"), "r") as f:
        import json as _j
        topic_dists = np.array(_j.load(f))

    n = min(n_samples, len(X_full))
    idx = np.random.default_rng(42).choice(len(X_full), n, replace=False)

    X_sub    = X_full[idx]
    y_sub    = y_full[idx]
    topics   = np.argmax(topic_dists[idx], axis=1)
    texts    = [raw["texts"][i][:120] for i in idx]

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        coords  = reducer.fit_transform(X_sub)
        method  = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(X_sub)
        method = "PCA"

    return {
        "x":        coords[:, 0].tolist(),
        "y":        coords[:, 1].tolist(),
        "labels":   y_sub.tolist(),
        "topics":   topics.tolist(),
        "texts":    texts,
        "method":   method,
        "n":        n,
    }


def get_pca_variance(n_components: int = 50) -> dict:
    """Return explained variance ratio for PCA on hybrid features."""
    from sklearn.decomposition import PCA
    X = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    return {
        "explained_variance_ratio":   pca.explained_variance_ratio_.tolist(),
        "cumulative_variance":        np.cumsum(pca.explained_variance_ratio_).tolist(),
        "n_components":               n_comp,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE C — SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_corpus_hybrid = None
_corpus_texts  = None
_corpus_labels = None

def load_corpus() -> bool:
    global _corpus_hybrid, _corpus_texts, _corpus_labels
    if _corpus_hybrid is not None:
        return True
    if not _hybrid_files_exist():
        return False
    import json
    _corpus_hybrid = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    _corpus_labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    with open(os.path.join(DATA_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
        _corpus_texts = json.load(f)["texts"]
    return True


def search(query: str, top_k: int = 5, filter_label: int = -1) -> list[dict]:
    """
    Hybrid semantic search.
    filter_label: -1 = all, 0 = Negative only, 1 = Positive only
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = _make_hybrid(query)
    sims      = cosine_similarity([query_vec], _corpus_hybrid)[0]

    if filter_label >= 0:
        mask = (_corpus_labels == filter_label)
        sims[~mask] = -1

    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx, 1):
        results.append({
            "rank":      rank,
            "idx":       int(idx),
            "sim":       float(sims[idx]),
            "label":     int(_corpus_labels[idx]),
            "text":      _corpus_texts[idx],
            "snippet":   _corpus_texts[idx][:300].replace("\n", " "),
        })
    return results


def search_steps(query: str) -> dict:
    """Return step-by-step explanation of how a query is processed for the UI."""
    return _make_hybrid_steps(query)
