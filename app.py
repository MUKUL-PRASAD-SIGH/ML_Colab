"""
app.py  ─  NLP Model Comparison Dashboard
──────────────────────────────────────────
Run:  streamlit run app.py
"""

import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─── page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Dashboard · Model Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── dark base ── */
.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #111827 60%, #0d0f1a 100%);
    color: #e2e8f0;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(17,24,39,0.95);
    border-right: 1px solid rgba(99,102,241,0.25);
}

/* ── cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: box-shadow .25s;
}
.card:hover { box-shadow: 0 0 24px rgba(99,102,241,0.2); }

/* ── pipeline badge cards ── */
.badge-nltk  { border-left: 4px solid #60a5fa; }
.badge-spacy { border-left: 4px solid #34d399; }
.badge-bert  { border-left: 4px solid #f472b6; }

/* ── result metric ── */
.metric-label { font-size:.75rem; color:#94a3b8; letter-spacing:.07em; text-transform:uppercase; }
.metric-value { font-size:1.9rem; font-weight:700; line-height:1.1; margin-top:.15rem; }
.positive { color: #4ade80; }
.negative { color: #f87171; }

/* ── confidence pill ── */
.pill {
    display:inline-block;
    padding:.22rem .75rem;
    border-radius:999px;
    font-size:.78rem;
    font-weight:600;
    letter-spacing:.04em;
}
.pill-pos { background:rgba(74,222,128,.15); color:#4ade80; }
.pill-neg { background:rgba(248,113,113,.15); color:#f87171; }

/* ── input box ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important;
}

/* ── buttons ── */
.stButton>button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .55rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton>button:hover { opacity:.88 !important; transform:translateY(-1px) !important; }

/* ── divider ── */
hr { border-color: rgba(99,102,241,.2) !important; }

/* ── table ── */
.dataframe { border-radius:12px !important; overflow:hidden; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* ── section headers ── */
.section-title {
    font-size:1.05rem;
    font-weight:600;
    color:#c7d2fe;
    letter-spacing:.02em;
    margin-bottom:.6rem;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#4f46e5; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  LOAD PIPELINES  (cached so they only load once)
# ════════════════════════════════════════════════════════════════════════════
import pipelines as pl

@st.cache_resource(show_spinner=False)
def _load_nltk():
    return pl.load_nltk_pipeline()

@st.cache_resource(show_spinner=False)
def _load_spacy():
    return pl.load_spacy_pipeline()

@st.cache_resource(show_spinner=False)
def _load_transformer():
    return pl.load_transformer_pipeline()

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 NLP Dashboard")
    st.caption("Model Comparison · Sentiment Analysis")
    st.divider()

    st.markdown("#### ⚙️ Pipeline Status")

    with st.spinner("Loading NLTK …"):
        nltk_ok = _load_nltk()
    st.markdown(
        f"{'✅' if nltk_ok else '⚠️'} **NLTK** (Logistic Regression)"
        + ("" if nltk_ok else "  \n_Run `train_models.py` first_")
    )

    with st.spinner("Loading spaCy …"):
        spacy_ok = _load_spacy()
    st.markdown(
        f"{'✅' if spacy_ok else '⚠️'} **spaCy** (Logistic Regression)"
        + ("" if spacy_ok else "  \n_Run `train_models.py` first_")
    )

    with st.spinner("Loading Transformer …"):
        trans_ok = _load_transformer()
    st.markdown(
        f"{'✅' if trans_ok else '⚠️'} **DistilBERT** (HuggingFace)"
        + ("" if trans_ok else "  \n_Check internet / model cache_")
    )

    st.divider()
    st.markdown("#### 📚 About")
    st.caption(
        "This dashboard compares three NLP approaches to sentiment analysis:\n\n"
        "- **NLTK** · classical tokenisation + TF-IDF\n"
        "- **spaCy** · lemmatisation + TF-IDF\n"
        "- **Transformer** · contextual embeddings (BERT-family)\n\n"
        "Trained on the **SST-2** dataset (~67 k sentences)."
    )

    st.divider()
    st.caption("Built with ❤️ · Streamlit + HuggingFace + scikit-learn")

# ════════════════════════════════════════════════════════════════════════════
#  MAIN  — HERO HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2.2rem 0 1.6rem;">
  <h1 style="font-size:2.4rem;font-weight:800;
             background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin-bottom:.3rem;">
    NLP Model Comparison Dashboard
  </h1>
  <p style="color:#94a3b8;font-size:1.05rem;max-width:560px;margin:auto;">
    Compare traditional NLP pipelines vs.&nbsp;Transformer-based models
    on real-time sentiment analysis — side by side.
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════
tab_single, tab_batch, tab_about = st.tabs(["🔍 Single Sentence", "📂 Batch / CSV", "📖 How It Works"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — SINGLE SENTENCE
# ────────────────────────────────────────────────────────────────────────────
with tab_single:

    # ── example sentences ──────────────────────────────────────────────────
    EXAMPLES = [
        "This movie is not bad at all — I genuinely enjoyed it!",
        "Absolutely terrible. Would not recommend to anyone.",
        "The acting was decent but the plot was quite boring.",
        "One of the best films I've seen this year.",
        "It started well but fell apart by the third act.",
    ]

    col_input, col_ex = st.columns([3, 1])
    with col_input:
        user_text = st.text_area(
            "✏️ Enter a sentence to analyse:",
            height=110,
            placeholder="e.g. 'This movie is absolutely fantastic!'"
        )
    with col_ex:
        st.markdown("<div style='padding-top:1.85rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>💡 Try an example</div>", unsafe_allow_html=True)
        for ex in EXAMPLES:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=f"ex_{ex[:10]}"):
                user_text = ex

    run_btn = st.button("🚀 Analyse", use_container_width=False)

    if run_btn and user_text.strip():

        with st.spinner("Running all pipelines …"):
            results = pl.run_all_pipelines(user_text.strip())

        st.divider()

        # ── result cards ────────────────────────────────────────────────
        st.markdown("#### 📊 Pipeline Results")

        PIPELINE_META = {
            "NLTK":        ("🔵", "badge-nltk",  "#60a5fa"),
            "spaCy":       ("🟢", "badge-spacy", "#34d399"),
            "Transformer": ("🔴", "badge-bert",  "#f472b6"),
        }

        cols = st.columns(3)
        rows_for_table   = []
        confidence_vals  = []
        time_vals        = []

        for idx, (name, res) in enumerate(results.items()):
            icon, badge_cls, accent = PIPELINE_META[name]
            with cols[idx]:
                if "error" in res:
                    st.markdown(f"""
                    <div class='card {badge_cls}'>
                      <div class='section-title'>{icon} {name}</div>
                      <div style='color:#f87171'>⚠️ {res['error']}</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                label      = res["label"]
                conf       = res["confidence"]
                time_ms    = res["time_ms"]
                is_pos     = label == "Positive"
                label_cls  = "positive" if is_pos else "negative"
                pill_cls   = "pill-pos" if is_pos else "pill-neg"
                emoji      = "✅" if is_pos else "❌"

                st.markdown(f"""
                <div class='card {badge_cls}'>
                  <div class='section-title' style='color:{accent}'>{icon} {name}</div>
                  <div class='metric-label'>Prediction</div>
                  <div class='metric-value {label_cls}'>{emoji} {label}</div>
                  <br>
                  <div class='metric-label'>Confidence</div>
                  <div style='font-size:1.4rem;font-weight:700;color:#e2e8f0;margin-top:.1rem'>
                    {conf:.1%} &nbsp;
                    <span class='pill {pill_cls}'>{label}</span>
                  </div>
                  <br>
                  <div class='metric-label'>Inference Time</div>
                  <div style='font-size:1.15rem;font-weight:600;color:#cbd5e1;margin-top:.1rem'>
                    ⏱ {time_ms:.1f} ms
                  </div>
                </div>""", unsafe_allow_html=True)

                rows_for_table.append({
                    "Model": f"{icon} {name}",
                    "Prediction": f"{emoji} {label}",
                    "Confidence": conf,
                    "Time (ms)": time_ms,
                })
                confidence_vals.append((name, conf, accent))
                time_vals.append((name, time_ms, accent))

        # ── charts ──────────────────────────────────────────────────────
        if confidence_vals:
            st.divider()
            st.markdown("#### 📈 Visual Comparison")

            ch1, ch2 = st.columns(2)

            # — confidence bar chart
            with ch1:
                fig_conf = go.Figure()
                for model_name, conf_val, color in confidence_vals:
                    fig_conf.add_trace(go.Bar(
                        x=[model_name],
                        y=[conf_val],
                        name=model_name,
                        marker_color=color,
                        text=[f"{conf_val:.1%}"],
                        textposition="outside",
                    ))
                fig_conf.update_layout(
                    title="Confidence Score",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="rgba(255,255,255,.07)"),
                    font=dict(family="Inter", color="#e2e8f0"),
                    margin=dict(t=40, b=30, l=20, r=20),
                    height=320,
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            # — inference time bar chart
            with ch2:
                fig_time = go.Figure()
                for model_name, t_ms, color in time_vals:
                    fig_time.add_trace(go.Bar(
                        x=[model_name],
                        y=[t_ms],
                        name=model_name,
                        marker_color=color,
                        text=[f"{t_ms:.1f} ms"],
                        textposition="outside",
                    ))
                fig_time.update_layout(
                    title="Inference Time (ms)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    yaxis=dict(gridcolor="rgba(255,255,255,.07)"),
                    font=dict(family="Inter", color="#e2e8f0"),
                    margin=dict(t=40, b=30, l=20, r=20),
                    height=320,
                )
                st.plotly_chart(fig_time, use_container_width=True)

            # — radar chart
            st.markdown("#### 🕸️ Confidence Radar")
            categories = [r["Model"].split()[-1] for r in rows_for_table]
            conf_scores = [r["Confidence"] for r in rows_for_table]
            fig_radar = go.Figure(go.Scatterpolar(
                r=conf_scores + [conf_scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(99,102,241,0.15)',
                line=dict(color='#818cf8', width=2),
                marker=dict(color='#a5b4fc', size=8),
            ))
            fig_radar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,.1)"),
                ),
                font=dict(family="Inter", color="#e2e8f0"),
                margin=dict(t=40, b=40),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # — summary table
            st.divider()
            st.markdown("#### 🗂 Summary Table")
            df = pd.DataFrame(rows_for_table)
            df["Confidence"] = df["Confidence"].map(lambda v: f"{v:.1%}")
            df["Time (ms)"]  = df["Time (ms)"].map(lambda v: f"{v:.1f} ms")
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif run_btn:
        st.warning("⚠️ Please enter a sentence before clicking Analyse.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — BATCH / CSV
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("""
    <div class='card'>
      <div class='section-title'>📂 Batch Evaluation via CSV Upload</div>
      <p style='color:#94a3b8;margin:0'>
        Upload a CSV with a <code>text</code> column (and optionally a <code>label</code> column
        with values 0 / 1 or Negative / Positive).<br>
        All three pipelines will run on every row and you can download the results.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV (max 500 rows)", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        if "text" not in df_raw.columns:
            st.error("❌ CSV must contain a 'text' column.")
        else:
            df_raw = df_raw.head(500)
            st.info(f"Loaded **{len(df_raw)}** rows.  Columns: {list(df_raw.columns)}")

            if st.button("▶️ Run Batch Analysis", key="batch_run"):
                progress   = st.progress(0, text="Analysing …")
                batch_rows = []
                n          = len(df_raw)

                for i, row_text in enumerate(df_raw["text"].astype(str)):
                    res = pl.run_all_pipelines(row_text)
                    entry = {"text": row_text}
                    for model_name, r in res.items():
                        prefix = model_name.lower().replace(" ", "_")
                        if "error" not in r:
                            entry[f"{prefix}_pred"]  = r["label"]
                            entry[f"{prefix}_conf"]  = r["confidence"]
                            entry[f"{prefix}_ms"]    = r["time_ms"]
                    if "label" in df_raw.columns:
                        true_raw  = str(row_text)
                        entry["true_label"] = df_raw["label"].iloc[i]
                    batch_rows.append(entry)
                    progress.progress((i + 1) / n, text=f"Row {i+1}/{n} …")

                progress.empty()
                df_out = pd.DataFrame(batch_rows)
                st.success(f"✅ Done! {len(df_out)} rows processed.")
                st.dataframe(df_out, use_container_width=True, hide_index=True)

                # accuracy if labels present
                if "true_label" in df_out.columns:
                    st.markdown("#### 🎯 Accuracy Comparison")
                    acc_cols = st.columns(3)
                    for idx2, model_pref in enumerate(["nltk", "spacy", "transformer"]):
                        pred_col = f"{model_pref}_pred"
                        if pred_col in df_out.columns:
                            def normalise(v):
                                s = str(v).strip().lower()
                                if s in ("1", "positive"): return "Positive"
                                return "Negative"
                            preds  = df_out[pred_col].map(normalise)
                            truths = df_out["true_label"].map(normalise)
                            acc    = (preds == truths).mean()
                            with acc_cols[idx2]:
                                st.metric(model_pref.upper(), f"{acc:.1%} accuracy")

                # download
                csv_bytes = df_out.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_bytes,
                    file_name="batch_results.csv",
                    mime="text/csv",
                )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class='card badge-nltk'>
      <div class='section-title' style='color:#60a5fa'>🔵 NLTK Pipeline</div>
      <ol style='color:#cbd5e1;line-height:1.8'>
        <li><strong>Tokenisation</strong> via <code>nltk.word_tokenize</code></li>
        <li><strong>Stopword removal</strong> (NLTK English stopwords)</li>
        <li><strong>TF-IDF vectoriser</strong> (20 k features, 1–2 grams)</li>
        <li><strong>Logistic Regression</strong> (trained on SST-2, ~67 k sentences)</li>
      </ol>
    </div>

    <div class='card badge-spacy'>
      <div class='section-title' style='color:#34d399'>🟢 spaCy Pipeline</div>
      <ol style='color:#cbd5e1;line-height:1.8'>
        <li><strong>Lemmatisation</strong> via <code>en_core_web_sm</code></li>
        <li><strong>Stopword + punctuation removal</strong></li>
        <li><strong>TF-IDF vectoriser</strong> (20 k features, 1–2 grams)</li>
        <li><strong>Logistic Regression</strong> (trained on SST-2)</li>
      </ol>
    </div>

    <div class='card badge-bert'>
      <div class='section-title' style='color:#f472b6'>🔴 Transformer Pipeline</div>
      <ol style='color:#cbd5e1;line-height:1.8'>
        <li><strong>No manual preprocessing</strong> — BERT handles tokenisation internally</li>
        <li><strong>DistilBERT</strong> fine-tuned on SST-2 (<code>distilbert-base-uncased-finetuned-sst-2-english</code>)</li>
        <li>Produces <strong>contextual embeddings</strong> that capture negation & nuance</li>
        <li>Higher accuracy, but ~10–100× slower than classical models</li>
      </ol>
    </div>

    <div class='card'>
      <div class='section-title'>⚖️ Trade-off Summary</div>
      <table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:.9rem'>
        <tr style='border-bottom:1px solid rgba(99,102,241,.3)'>
          <th style='padding:.5rem;text-align:left'>Aspect</th>
          <th style='color:#60a5fa;padding:.5rem;text-align:center'>NLTK</th>
          <th style='color:#34d399;padding:.5rem;text-align:center'>spaCy</th>
          <th style='color:#f472b6;padding:.5rem;text-align:center'>Transformer</th>
        </tr>
        <tr>
          <td style='padding:.45rem'>Accuracy (SST-2)</td>
          <td style='text-align:center'>~83%</td>
          <td style='text-align:center'>~84%</td>
          <td style='text-align:center'>~91%</td>
        </tr>
        <tr>
          <td style='padding:.45rem'>Speed</td>
          <td style='text-align:center'>⚡⚡⚡</td>
          <td style='text-align:center'>⚡⚡⚡</td>
          <td style='text-align:center'>⚡</td>
        </tr>
        <tr>
          <td style='padding:.45rem'>Negation handling</td>
          <td style='text-align:center'>❌ Weak</td>
          <td style='text-align:center'>❌ Weak</td>
          <td style='text-align:center'>✅ Strong</td>
        </tr>
        <tr>
          <td style='padding:.45rem'>Model size</td>
          <td style='text-align:center'>Small</td>
          <td style='text-align:center'>Small</td>
          <td style='text-align:center'>~250 MB</td>
        </tr>
        <tr>
          <td style='padding:.45rem'>Interpretability</td>
          <td style='text-align:center'>✅ High</td>
          <td style='text-align:center'>✅ High</td>
          <td style='text-align:center'>⚠️ Black-box</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
