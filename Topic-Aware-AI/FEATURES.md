# Topic-Aware AI — Feature Documentation

> Updated: 2026-03-23 | Reflects the `pipelines.py` + `app.py` Dashboard Architecture

---

## Feature A: LDA Topic Modelling (Gensim)

### What It Does (Plain English)
Gensim reads all your reviews, then groups the vocabulary into **Topics** based on which words keep appearing together. A movie review about spaceships is highly likely to co-occur with words like "planet", "future", "alien", which Gensim learns to bundle into "Topic 2: Sci-Fi", without us ever telling it.

### Technical Implementation
- **Algorithm:** Latent Dirichlet Allocation (Generative Probabilistic Model)
- **Library:** `gensim.models.LdaModel`
- **Preprocessing:** `simple_preprocess` + custom stopword set (removed film/movie/br noise tokens)
- **Vectorisation:** Bag-of-Words (BoW) sparse matrix via `Dictionary.doc2bow`

### Configuration (Dashboard-adjustable)
| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_topics` | 3-5 | Configurable via UI slider |
| `passes` | 10 | Training epochs over corpus |
| `alpha` | `auto` | Asymmetric Dirichlet prior on docs |
| `eta` | `auto` | Asymmetric Dirichlet prior on words |
| `no_below` | 2 | Prune tokens seen in < 2 docs |
| `no_above` | 0.9 | Prune tokens seen in > 90% of docs |

### Output
```
data/topic_distributions.json   # [[0.02, 0.70, ...], ...] N × num_topics
models/lda_model.gensim
models/lda_dictionary.gensim
```

---

## Feature B: DistilBERT Semantic Embeddings

### What It Does (Plain English)
DistilBERT is a miniaturized version of BERT (Google's language model). It reads your sentence word by word but uses **Self-Attention** to look at all other words simultaneously. So when it sees the word "bank", it knows from context whether you mean a riverbank or a financial bank — and encodes that into the numbers. The output is **768 numbers** that represent the full meaning and tone of the sentence.

### Technical Implementation
- **Model:** `distilbert-base-uncased` (HuggingFace)
- **Tokenisation:** WordPiece Subword encoding (handles unknown words by splitting them)
- **Pooling Strategy:** Masked Mean Pooling (superior to `[CLS]` for similarity tasks)

```
tokens → 6-Layer Self-Attention Transformer → last_hidden_state [batch, seq_len, 768]
       → Masked Mean Pool → sentence vector [batch, 768]
```

### Configuration
| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-uncased` |
| Max Sequence Length | 256 tokens |
| Batch size | 16 |
| Device | Auto (CUDA preferred, CPU fallback) |
| Output Dimension | 768 numbers per sentence |

### Output
```
data/bert_embeddings.npy    # float32, shape (N, 768)
```

---

## Feature C: Hybrid Feature Fusion

### What It Does (Plain English)
We glue both lists of numbers together. But we can't just merge them raw — the BERT numbers are huge (up to ±20) while the Gensim numbers are tiny (between 0.0 and 1.0). So first we "normalize" both lists to be exactly the same size (radius of 1 on a unit circle), then concatenate.

### Formula
```
Hybrid(doc) = [ L2_norm(BERT_768) ⊕ L2_norm(LDA_K) ]
            = shape [768 + K]
```

### Implementation
```python
def l2_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

hybrid = np.concatenate([l2_norm(bert_emb), l2_norm(lda_vec)], axis=1)
```

### Output
```
data/hybrid_features.npy   # float32, shape (N, 768+K)
data/labels.npy            # int,     shape (N,)
```

---

## Feature D: Downstream Classifier (Tab A)

### What It Does (Plain English)
We take all the merged number-lists and teach a simple AI (Logistic Regression, SVM, or MLP) to predict Positive vs Negative from them. We then compare the accuracy against using BERT *alone* to prove that adding Gensim actually helps!

### Options
| Classifier | Best For |
|-----------|----------|
| `LogisticRegression` | Fast, interpretable, great baseline |
| `SVM (LinearSVC)` | High accuracy on high-dimensional data |
| `MLP Neural Net` | Complex non-linear relationships |

### Output Metrics
- **Accuracy** (Hybrid vs BERT-only comparison)
- **Precision / Recall / F1** per class (Classification Report)
- **Training time** in seconds

---

## Feature E: UMAP Projection (Tab B)

### What It Does (Plain English)
We have 700+ numbers per review. We can't draw that in 2D! UMAP magically squashes all those numbers down to just X and Y while keeping similar reviews close to each other. The resulting "galaxy map" shows if our system has correctly separated Positive from Negative reviews or discovered conceptual clusters automatically.

### Fallback
If `umap-learn` is not installed, it automatically falls back to **PCA** (Principal Component Analysis).

---

## Feature F: Semantic Search Engine (Tab C)

### What It Does (Plain English)
Normal search engines look for exact matching words. This uses **Cosine Similarity** — it converts your search query into numbers, then finds which reviews have the most similar number-lists. It finds results *by meaning*, so searching "a hilarious comedy" might also find reviews that never use those exact words but have the same vibe.

### Formula
```
Similarity = cos(θ) = (A · B) / (|A| × |B|)
```
A similarity of 1.0 = identical vibes. 0.0 = completely unrelated.

### Features
- Optional Positive/Negative sentiment filter
- Returns Top-K results with Match Score (%)
- Query is converted to Hybrid Vector using the same pipeline
