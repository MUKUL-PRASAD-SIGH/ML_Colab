"""
pipelines.py  ─  All backend logic & model building for the Educational Dashboard
================================================================================
Gensim-FREE: Uses sklearn NMF + TfidfVectorizer for topic modelling.
Works on Python 3.10 / 3.11 / 3.12 / 3.13 / 3.14+
"""

from __future__ import annotations
import os, re, time, json, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import torch

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
_nmf         = None        # replaces _lda
_tfidf       = None        # replaces _dictionary
_tokenizer   = None
_bert_model  = None
_classifier  = None

_lda_ready        = False  # kept as alias for nmf_ready
_bert_ready       = False
_hybrid_ready     = False
_classifier_ready = False

# ─────────────────────────────────────────────
# Stop words (sklearn built-in, no gensim needed)
# ─────────────────────────────────────────────
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKL_STOP

_EXTRA_STOPS = {"film", "movie", "br", "one", "would", "could", "like", "nan",
                "just", "really", "even", "also", "much", "way", "get", "good",
                "time", "make", "made", "see", "think", "know", "well", "lot"}

CUSTOM_STOP_WORDS = _SKL_STOP.union(_EXTRA_STOPS)


def _simple_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation/digits, keep words >2 chars, drop stop words."""
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in tokens if t not in CUSTOM_STOP_WORDS]


def _get_stop_words():
    return CUSTOM_STOP_WORDS


# ═══════════════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_lda() -> bool:
    """Load saved NMF topic model (called 'lda' for API compatibility)."""
    global _nmf, _tfidf, _lda_ready
    if _lda_ready: return True
    mpath = os.path.join(MODELS_DIR, "nmf_model.pkl")
    vpath = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not (os.path.exists(mpath) and os.path.exists(vpath)):
        return False
    _nmf       = joblib.load(mpath)
    _tfidf     = joblib.load(vpath)
    _lda_ready = True
    return True


def load_bert() -> bool:
    global _tokenizer, _bert_model, _bert_ready
    if _bert_ready: return True
    from transformers import AutoTokenizer, AutoModel
    _tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    _bert_model.eval()
    _bert_ready = True
    return True


def load_classifier() -> bool:
    global _classifier, _classifier_ready
    if _classifier_ready: return True
    mpath = os.path.join(MODELS_DIR, "hybrid_classifier.pkl")
    if not os.path.exists(mpath): return False
    _classifier       = joblib.load(mpath)
    _classifier_ready = True
    return True


def _hybrid_files_exist() -> bool:
    return (
        os.path.exists(os.path.join(DATA_DIR, "hybrid_features.npy")) and
        os.path.exists(os.path.join(DATA_DIR, "labels.npy"))          and
        os.path.exists(os.path.join(DATA_DIR, "raw_data.json"))
    )


def get_fast_dummy_data(n=200):
    pos = [
        "A truly spectacular and thrilling masterpiece.",
        "Absolutely loved every second of it. So funny and sweet.",
        "The best action sci-fi movie I have ever seen.",
        "Such a beautiful, emotional romance.",
        "Incredible acting, gripping storyline, 10/10.",
        "Hilarious comedy that kept me laughing all night.",
        "A phenomenal journey through space and time.",
        "Heartwarming and deeply moving cinema."
    ]
    neg = [
        "Terrible acting and a completely boring plot.",
        "I hated this movie, it was a waste of time.",
        "Awful special effects and terrible directing.",
        "Do not watch this, it is unbelievably bad.",
        "A slow, generic, and uninspired disaster.",
        "Worst comedy ever, didn't laugh once.",
        "Completely ruined the original franchise.",
        "Boring, stupid, and way too long."
    ]
    texts, labels = [], []
    for _ in range(n // 2):
        texts.append(np.random.choice(pos) + " " + np.random.choice(pos))
        labels.append(1)
        texts.append(np.random.choice(neg) + " " + np.random.choice(neg))
        labels.append(0)
    return texts[:n], labels[:n]


# ═══════════════════════════════════════════════════════════════════════════
# CORE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_core_pipeline(n_samples=200, n_topics=3, progress_cb=None, ds_type="fast", custom_texts=None, custom_labels=None):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import AutoTokenizer, AutoModel

    def log(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load Data ──
    if ds_type == "custom" and custom_texts and custom_labels:
        log(f"📂 Preparing {len(custom_texts)} samples from Custom CSV...", 0.05)
        texts  = custom_texts[:n_samples]
        labels = custom_labels[:n_samples]
    elif ds_type == "imdb":
        log("⏳ Downloading IMDB... (May take 1-3 minutes if first time)...", 0.05)
        from datasets import load_dataset
        dataset = load_dataset("imdb", split="train")
        texts   = dataset["text"][:n_samples]
        labels  = dataset["label"][:n_samples]
    else:
        log(f"⚡ Loading Fast Built-In Dataset (Instant)...", 0.05)
        texts, labels = get_fast_dummy_data(n_samples)

    with open(os.path.join(DATA_DIR, "raw_data.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": list(texts), "labels": list(labels)}, f)

    # ── 2. Preprocess for NMF ──
    log("🧹 Cleaning text (removing small words like 'a', 'the')...", 0.15)
    cleaned_texts = [" ".join(_simple_tokenize(t)) for t in texts]

    # ── 3. TF-IDF → NMF (replaces Gensim LDA) ──
    log(f"📚 Training Topic Model to find {n_topics} concepts (NMF)...", 0.30)
    tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9)
    tfidf_matrix = tfidf.fit_transform(cleaned_texts)

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=500, init="nndsvda")
    nmf.fit(tfidf_matrix)

    log("🔢 Extracting Topic percentages for all texts...", 0.45)
    topic_matrix = nmf.transform(tfidf_matrix)  # shape: (n_docs, n_topics)
    # Normalize rows to sum to 1 (like LDA probability distributions)
    row_sums = topic_matrix.sum(axis=1, keepdims=True) + 1e-9
    topic_distributions = (topic_matrix / row_sums).tolist()

    # Save models
    joblib.dump(nmf,   os.path.join(MODELS_DIR, "nmf_model.pkl"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    with open(os.path.join(DATA_DIR, "topic_distributions.json"), "w") as f:
        json.dump(topic_distributions, f)

    global _nmf, _tfidf, _lda_ready
    _tfidf     = tfidf
    _nmf       = nmf
    _lda_ready = True

    # ── 4. BERT Embeddings ──
    log("🤖 Turning text into deep meaning numbers (DistilBERT)...", 0.55)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model     = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    model.eval()

    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    batch_size = 16
    all_embeddings = []
    n_actual = len(texts)
    total_batches = (n_actual // batch_size) + 1

    for b_idx, start in enumerate(range(0, n_actual, batch_size)):
        batch = [str(x) for x in texts[start : start + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        vecs = mean_pool(out.last_hidden_state, inputs["attention_mask"])
        all_embeddings.append(vecs.cpu().numpy())
        if b_idx % max(1, total_batches // 4) == 0:
            log(f"🧠 DistilBERT processed batch {b_idx}/{total_batches}...", 0.60 + (b_idx / total_batches) * 0.25)

    if all_embeddings:
        bert_emb = np.vstack(all_embeddings)
        np.save(os.path.join(DATA_DIR, "bert_embeddings.npy"), bert_emb)
    else:
        bert_emb = np.zeros((0, 768))

    global _tokenizer, _bert_model, _bert_ready
    _tokenizer  = tokenizer
    _bert_model = model
    _bert_ready = True

    # ── 5. Combine Hybrid ──
    log("🔗 Combining Topics and Deep Meaning together...", 0.90)
    lda_vec = np.array(topic_distributions)

    def l2_norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    hybrid_features = np.concatenate([l2_norm(bert_emb), l2_norm(lda_vec)], axis=1)

    np.save(os.path.join(DATA_DIR, "hybrid_features.npy"), hybrid_features)
    np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(labels))

    log(f"✅ Finished! Generated {hybrid_features.shape[0]} arrays of {hybrid_features.shape[1]} numbers.", 1.0)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _get_bert_embedding(text: str, max_len: int = 256) -> np.ndarray:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    with torch.no_grad():
        out = _bert_model(**inputs)
    vec = _mean_pool(out.last_hidden_state, inputs["attention_mask"])
    return vec.squeeze().numpy()


def _get_lda_vector(text: str, num_topics: int = None) -> np.ndarray:
    """Get NMF topic vector for a single text (API-compatible with old gensim version)."""
    if num_topics is None: num_topics = _nmf.n_components
    cleaned = " ".join(_simple_tokenize(text))
    tfidf_vec = _tfidf.transform([cleaned])
    raw = _nmf.transform(tfidf_vec)[0]
    # Normalize to sum to 1
    total = raw.sum() + 1e-9
    return raw / total


def _make_hybrid(text: str) -> np.ndarray:
    bert  = _get_bert_embedding(text)
    bert  = bert  / (np.linalg.norm(bert)  + 1e-9)
    lda_v = _get_lda_vector(text)
    lda_v = lda_v / (np.linalg.norm(lda_v) + 1e-9)
    return np.concatenate([bert, lda_v])


def _make_hybrid_steps(text: str) -> dict:
    """Return detailed step-by-step breakdown for the tutorial."""
    stop       = _get_stop_words()
    all_tokens = re.findall(r"[a-z]{2,}", text.lower())
    kept_tok   = [t for t in all_tokens if t not in stop and len(t) > 2]
    removed_tok= [t for t in all_tokens if t in stop or len(t) <= 2]

    # NMF topic vector
    cleaned  = " ".join(kept_tok)
    tfidf_vec= _tfidf.transform([cleaned])
    lda_raw  = _nmf.transform(tfidf_vec)[0]
    total    = lda_raw.sum() + 1e-9
    lda_raw  = lda_raw / total   # normalized probability-like distribution
    num_topics = _nmf.n_components

    # Topic top-words from NMF components
    feature_names = _tfidf.get_feature_names_out()
    topic_words = {}
    for tid in range(num_topics):
        top_idx = _nmf.components_[tid].argsort()[-6:][::-1]
        topic_words[tid] = [(feature_names[i], float(_nmf.components_[tid][i])) for i in top_idx]

    # Bag-of-words preview (top 10 tfidf entries)
    tfidf_arr = tfidf_vec.toarray()[0]
    top10_idx = tfidf_arr.argsort()[-10:][::-1]
    bow_preview = [(feature_names[i], round(float(tfidf_arr[i]), 4)) for i in top10_idx if tfidf_arr[i] > 0]

    # BERT embedding
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = _bert_model(**inputs)
    bert_raw    = _mean_pool(out.last_hidden_state, inputs["attention_mask"]).squeeze().numpy()
    bert_tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    bert_magnitude = float(np.linalg.norm(bert_raw))
    lda_magnitude  = float(np.linalg.norm(lda_raw))

    bert_norm = bert_raw / (bert_magnitude + 1e-9)
    lda_norm  = lda_raw  / (lda_magnitude  + 1e-9)
    hybrid    = np.concatenate([bert_norm, lda_norm])

    return {
        "raw":             text,
        "all_tokens":      all_tokens,
        "kept_tokens":     kept_tok,
        "removed_tokens":  removed_tok,
        "bow":             bow_preview,
        "lda_dist":        lda_raw.tolist(),
        "lda_normed":      lda_norm.tolist(),
        "topic_words":     topic_words,
        "dominant_topic":  int(np.argmax(lda_raw)),
        "bert_tokens":     bert_tokens,
        "bert_raw_sample": bert_raw[:8].tolist(),
        "bert_norm_sample":bert_norm[:8].tolist(),
        "bert_magnitude":  round(bert_magnitude, 4),
        "lda_magnitude":   round(lda_magnitude, 4),
        "bert_dim":        len(bert_raw),
        "hybrid_dim":      len(hybrid),
        "hybrid_sample":   hybrid[:6].tolist() + ["..."] + hybrid[-3:].tolist(),
        "hybrid":          hybrid,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DOWNSTREAM CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def train_classifier(model_type: str = "LogisticRegression", progress_cb=None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.calibration import CalibratedClassifierCV

    def log(msg, pct):
        if progress_cb: progress_cb(msg, pct)

    log("📂  Loading combined features …", 0.05)
    X = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))

    log(f"✂️  Splitting data: 80% to train, 20% to test ({len(X)} reviews) …", 0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    log(f"🏋️  Teaching the AI ({model_type}) …", 0.35)
    if model_type == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    elif model_type == "SVM":
        clf = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0))
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=42)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    log("📊  Grading the AI's test…", 0.80)
    preds  = clf.predict(X_test)
    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, labels=np.unique(y_test), output_dict=True)
    cm     = confusion_matrix(y_test, preds).tolist()

    log("🔬  Comparing against BERT-only…", 0.90)
    X_bert = X[:, :768]
    Xb_tr, Xb_te, _, _ = train_test_split(X_bert, y, test_size=0.2, random_state=42)
    clf_bert = LogisticRegression(max_iter=1000, C=1.0).fit(Xb_tr, y_train)
    acc_bert_only = accuracy_score(y_test, clf_bert.predict(Xb_te))

    log("💾  Saving …", 0.95)
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
    if not (_lda_ready and _bert_ready and _classifier_ready): return None
    t0     = time.perf_counter()
    steps  = _make_hybrid_steps(text)
    vec    = steps["hybrid"]
    pred   = _classifier.predict([vec])[0]
    proba  = _classifier.predict_proba([vec])[0]
    elapsed= (time.perf_counter() - t0) * 1000

    try:
        label_map = {0: "Negative", 1: "Positive"}
        pos_idx = list(_classifier.classes_).index(1) if 1 in _classifier.classes_ else 1
        neg_idx = list(_classifier.classes_).index(0) if 0 in _classifier.classes_ else 0
        l_str = label_map.get(pred, str(pred))
    except Exception:
        pos_idx, neg_idx = 1, 0
        l_str = str(pred)

    return {
        "label":      l_str,
        "confidence": float(max(proba)),
        "proba_neg":  float(proba[neg_idx]) if len(proba) > neg_idx else 0.5,
        "proba_pos":  float(proba[pos_idx]) if len(proba) > pos_idx else 0.5,
        "time_ms":    round(elapsed, 2),
        "steps":      steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TOPIC WORDS (for UI display)
# ═══════════════════════════════════════════════════════════════════════════

def get_topic_words(num_words: int = 10) -> dict:
    if not _lda_ready: return {}
    feature_names = _tfidf.get_feature_names_out()
    topics = {}
    for tid in range(_nmf.n_components):
        top_idx = _nmf.components_[tid].argsort()[-num_words:][::-1]
        topics[tid] = [(feature_names[i], float(_nmf.components_[tid][i])) for i in top_idx]
    return topics


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATION & SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def get_umap_data(n_samples: int = 500) -> dict:
    X_full = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    y_full = np.load(os.path.join(DATA_DIR, "labels.npy"))
    with open(os.path.join(DATA_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(os.path.join(DATA_DIR, "topic_distributions.json"), "r") as f:
        topic_dists = np.array(json.load(f))

    n = min(n_samples, len(X_full))
    idx = np.random.default_rng(42).choice(len(X_full), n, replace=False)
    X_sub, y_sub = X_full[idx], y_full[idx]
    topics = np.argmax(topic_dists[idx], axis=1)
    texts  = [raw["texts"][i][:120] for i in idx]

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
        "x":      coords[:, 0].tolist(),
        "y":      coords[:, 1].tolist(),
        "labels": y_sub.tolist(),
        "topics": topics.tolist(),
        "texts":  texts,
        "method": method,
        "n":      n,
    }


_corpus_hybrid = None
_corpus_texts  = None
_corpus_labels = None


def load_corpus() -> bool:
    global _corpus_hybrid, _corpus_texts, _corpus_labels
    if _corpus_hybrid is not None: return True
    if not _hybrid_files_exist(): return False
    _corpus_hybrid = np.load(os.path.join(DATA_DIR, "hybrid_features.npy"))
    _corpus_labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    with open(os.path.join(DATA_DIR, "raw_data.json"), "r", encoding="utf-8") as f:
        _corpus_texts = json.load(f)["texts"]
    return True


def search(query: str, top_k: int = 5, filter_label: int = -1) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec = _make_hybrid(query)
    sims      = cosine_similarity([query_vec], _corpus_hybrid)[0]
    if filter_label >= 0:
        mask = (_corpus_labels == filter_label)
        sims[~mask] = -1
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    label_map = {0: "Negative", 1: "Positive"}
    for rank, idx in enumerate(top_idx, 1):
        lbl = int(_corpus_labels[idx])
        results.append({
            "rank": rank, "idx": int(idx), "sim": float(sims[idx]),
            "label": lbl, "label_str": label_map.get(lbl, str(lbl)),
            "text": _corpus_texts[idx],
            "snippet": _corpus_texts[idx][:300].replace("\n", " "),
        })
    return results
