"""
code_dump.py  —  Topic-Aware AI System (Master Code Reference)
==============================================================
Complete reference of all critical code in one file.
Auto-updated when new features are added.
Last updated: 2026-03-23

How to use:  Just read this file to understand what each part does.
"""

# ============================================================
# PART 1 — DATASET SOURCES
# ============================================================
LOAD_FAST_DUMMY = '''
# ⚡ Instant built-in dataset (no internet required)
def get_fast_dummy_data(n=200):
    pos = ["A truly spectacular and thrilling masterpiece.", ...]
    neg = ["Terrible acting and a completely boring plot.", ...]
    texts, labels = [], []
    for _ in range(n // 2):
        texts.append(random_positive_review)
        labels.append(1)
        texts.append(random_negative_review)
        labels.append(0)
    return texts[:n], labels[:n]
'''

LOAD_IMDB = '''
# 🌐 Full IMDB Download from HuggingFace (2000 reviews)
from datasets import load_dataset
dataset = load_dataset("imdb", split="train")
texts  = dataset["text"][:n_samples]
labels = dataset["label"][:n_samples]
with open("data/raw_data.json", "w") as f:
    json.dump({"texts": list(texts), "labels": list(labels)}, f)
'''

LOAD_CSV = '''
# 📁 Custom CSV Upload (must have `text` and `label` columns)
import pandas as pd
df = pd.read_csv("my_custom_data.csv")
df = df.dropna(subset=["text", "label"])
texts  = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
'''

# ============================================================
# PART 2 — GENSIM PREPROCESSING
# ============================================================
PREPROCESS = '''
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

STOP = STOPWORDS.union({"film","movie","br","one","would","could","like"})

# Keeps only meaningful keywords, removes stopwords and short words
processed = [
    [t for t in simple_preprocess(text, deacc=True)
     if t not in STOP and len(t) > 2]
    for text in texts
]
'''

# ============================================================
# PART 3 — LDA TOPIC MODEL
# ============================================================
LDA_TRAIN = '''
from gensim import corpora
from gensim.models import LdaModel

# Build a vocabulary dictionary from cleaned word lists
dictionary = corpora.Dictionary(processed)
dictionary.filter_extremes(no_below=2, no_above=0.9)  # Remove too-rare and too-common words

# Convert each doc to a Bag-of-Words sparse count: [(word_id, count), ...]
corpus = [dictionary.doc2bow(doc) for doc in processed]

# Train LDA — finds hidden topic structure probabilistically
lda = LdaModel(
    corpus=corpus, id2word=dictionary, num_topics=num_topics,
    passes=10, random_state=42, alpha="auto", eta="auto"
)

# Save model + dictionary so we can reload anytime
dictionary.save("models/lda_dictionary.gensim")
lda.save("models/lda_model.gensim")

# Extract per-document topic probability distributions
topic_dists = []
for bow in corpus:
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    vec  = np.zeros(num_topics)
    for topic_id, prob in dist:
        vec[topic_id] = prob
    topic_dists.append(vec.tolist())
'''

# ============================================================
# PART 4 — DISTILBERT EMBEDDINGS (Mean Pooling)
# ============================================================
BERT_EMBED = '''
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model     = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

def mean_pool(last_hidden_state, attention_mask):
    # Expand the mask to match the hidden state shape, then average
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# Process in batches of 16 for memory efficiency
all_embeddings = []
for start in range(0, len(texts), 16):
    batch  = texts[start : start + 16]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        out = model(**inputs)
    vecs = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    all_embeddings.append(vecs.cpu().numpy())

bert_emb = np.vstack(all_embeddings)   # shape → (N, 768)
np.save("data/bert_embeddings.npy", bert_emb)
'''

# ============================================================
# PART 5 — HYBRID FUSION (L2 Normalisation)
# ============================================================
HYBRID_FUSE = '''
import numpy as np

bert    = np.load("data/bert_embeddings.npy")       # (N, 768)
lda_vec = np.array(topic_dists)                      # (N, K)

def l2_norm(X):
    # Squash rows so every vector has length = 1 (unit circle)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# Concatenate normalized vectors side-by-side
hybrid = np.concatenate([l2_norm(bert), l2_norm(lda_vec)], axis=1)
# Result shape: (N, 768+K)

np.save("data/hybrid_features.npy", hybrid)
np.save("data/labels.npy", np.array(labels))
'''

# ============================================================
# PART 6 — DOWNSTREAM CLASSIFIER
# ============================================================
CLASSIFIER_TRAIN = '''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.load("data/hybrid_features.npy")
y = np.load("data/labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple classifier on our rich hybrid feature matrix
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))

# Prove Gensim helps: compare vs just BERT
clf_bert = LogisticRegression(max_iter=1000).fit(X_train[:, :768], y_train)
acc_bert_only = accuracy_score(y_test, clf_bert.predict(X_test[:, :768]))

joblib.dump(clf, "models/hybrid_classifier.pkl")
'''

# ============================================================
# PART 7 — SEMANTIC SEARCH ENGINE
# ============================================================
SEARCH_ENGINE = '''
from sklearn.metrics.pairwise import cosine_similarity

# Convert query to Hybrid Vector using same BERT + LDA pipeline
query_vec = make_hybrid(query_text)  # shape: (768+K,)

# Load all pre-computed corpus vectors
corpus = np.load("data/hybrid_features.npy")

# Cosine similarity: 1.0 = same, 0.0 = unrelated
sims       = cosine_similarity([query_vec], corpus)[0]
top_k_idx  = np.argsort(sims)[::-1][:top_k]

# Results ranked by highest semantic match score
'''

# ============================================================
# PART 8 — UMAP 2D VISUALISATION
# ============================================================
UMAP_VIZ = '''
import umap

X = np.load("data/hybrid_features.npy")   # (N, 768+K)

# UMAP squashes high-dimensional vectors to 2D while preserving structure
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
coords  = reducer.fit_transform(X)         # (N, 2) — just X and Y now!

# Use plotly to create the interactive galaxy map
import plotly.express as px
df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
fig = px.scatter(df, x="x", y="y", color="label", template="plotly_dark")
fig.show()
'''
