"""
train_models.py
───────────────
Offline training pipeline for the Topic-Aware AI System.
Loads dataset, pre-processes, trains LDA model, extracts BERT 
embeddings, combines into hybrid vectors, and saves them.
Run once before launching the Streamlit dashboard (`app.py`).
"""

import os, time, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from datasets import load_dataset
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

# ─── config ────────────────────────────────────────────────────────────────
NUM_SAMPLES = 2000
NUM_TOPICS  = 5
BATCH_SIZE  = 16
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR    = "data"
MODELS_DIR  = "models"
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CUSTOM_STOP = STOPWORDS.union({"film", "movie", "br", "one", "would", "could", "like"})

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n📦  Loading {NUM_SAMPLES} reviews from IMDB …")
dataset = load_dataset("imdb", split="train")
texts  = dataset["text"][:NUM_SAMPLES]
labels = dataset["label"][:NUM_SAMPLES]

with open(os.path.join(DATA_DIR, "raw_data.json"), "w", encoding="utf-8") as f:
    json.dump({"texts": list(texts), "labels": list(labels)}, f)

print(f"    ✅  Saved to {DATA_DIR}/raw_data.json")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESS
# ═══════════════════════════════════════════════════════════════════════════
print("\n🧹  Preprocessing texts for Gensim …")
processed_tokens = []
for t in tqdm(texts, desc="Tokenising"):
    toks = [token for token in simple_preprocess(t, deacc=True)
            if token not in CUSTOM_STOP and len(token) > 2]
    processed_tokens.append(toks)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN LDA
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n📚  Training LDA Model ({NUM_TOPICS} Topics) …")
dictionary = corpora.Dictionary(processed_tokens)
dictionary.filter_extremes(no_below=5, no_above=0.7)
corpus = [dictionary.doc2bow(doc) for doc in processed_tokens]

t0  = time.time()
lda = LdaModel(
    corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS,
    passes=10, random_state=42, alpha="auto", eta="auto"
)
print(f"    ✅  LDA trained. Time: {time.time()-t0:.1f}s")

topic_distributions = []
for bow in corpus:
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    vec  = np.array([prob for _, prob in sorted(dist, key=lambda x: x[0])])
    topic_distributions.append(vec.tolist())

dictionary.save(os.path.join(MODELS_DIR, "lda_dictionary.gensim"))
lda.save(os.path.join(MODELS_DIR, "lda_model.gensim"))
with open(os.path.join(DATA_DIR, "topic_distributions.json"), "w") as f:
    json.dump(topic_distributions, f)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: BERT EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n🤖  Generating DistilBERT Embeddings ({DEVICE}) …")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model     = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
model.eval()

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

all_embeddings = []
for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
    batch = texts[start : start + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    vecs = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    all_embeddings.append(vecs.cpu().numpy())

bert_emb = np.vstack(all_embeddings)
np.save(os.path.join(DATA_DIR, "bert_embeddings.npy"), bert_emb)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: COMBINE HYBRID FEATURES
# ═══════════════════════════════════════════════════════════════════════════
print("\n🔗  Fusing Hybrid Features …")
lda_vec = np.array(topic_distributions)

def l2_norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

hybrid_features = np.concatenate([l2_norm(bert_emb), l2_norm(lda_vec)], axis=1)

np.save(os.path.join(DATA_DIR, "hybrid_features.npy"), hybrid_features)
np.save(os.path.join(DATA_DIR, "labels.npy"), np.array(labels))

print(f"    ✅  Hybrid Vectors Shape: {hybrid_features.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: INITIAL CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
print("\n🏋️  Training initial Logistic Regression classifier …")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(hybrid_features, np.array(labels))
joblib.dump(clf, os.path.join(MODELS_DIR, "hybrid_classifier.pkl"))

print("\n" + "="*55)
print("  OFFLINE PIPELINE COMPLETE")
print("="*55)
print("  Run the dashboard with:  streamlit run app.py")
print("="*55)
