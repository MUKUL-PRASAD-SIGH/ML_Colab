"""
app.py  ─  NLP Model Comparison Dashboard (Educational Edition)
Run:  streamlit run app.py
"""
import os, time, re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Dashboard · Learn How It Works",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]       { font-family:'Inter',sans-serif; }
.stApp                           { background: #0a0d14; color:#e2e8f0; }
section[data-testid="stSidebar"] { background:#0f1420; border-right:1px solid #1e2a40; }

/* ── cards ── */
.card       { background:rgba(255,255,255,.04); border:1px solid rgba(99,102,241,.2);
              border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:.9rem;
              backdrop-filter:blur(6px); }
.card:hover { box-shadow:0 0 28px rgba(99,102,241,.18); transition:.25s; }
.card-blue  { border-left:4px solid #60a5fa; }
.card-green { border-left:4px solid #34d399; }
.card-pink  { border-left:4px solid #f472b6; }
.card-yellow{ border-left:4px solid #fbbf24; }

/* ── step pipeline ── */
.step-box   { background:#111827; border:1px solid #1f2937; border-radius:10px;
              padding:.9rem 1.1rem; margin:.4rem 0; font-size:.88rem; }
.step-title { font-size:.72rem; font-weight:700; letter-spacing:.1em;
              text-transform:uppercase; color:#6366f1; margin-bottom:.35rem; }
.step-number{ display:inline-block; background:#6366f1; color:#fff; border-radius:50%;
              width:22px; height:22px; text-align:center; line-height:22px;
              font-size:.75rem; font-weight:700; margin-right:.5rem; }
.arrow      { text-align:center; color:#4f46e5; font-size:1.3rem; margin:.05rem 0; }

/* ── token pills ── */
.token      { display:inline-block; padding:.18rem .55rem; border-radius:6px;
              font-size:.8rem; font-family:'JetBrains Mono',monospace;
              margin:.12rem .1rem; font-weight:500; }
.tok-keep   { background:rgba(52,211,153,.18); color:#34d399; border:1px solid rgba(52,211,153,.3); }
.tok-stop   { background:rgba(248,113,113,.12); color:#f87171;
              border:1px solid rgba(248,113,113,.25); text-decoration:line-through; }
.tok-lemma  { background:rgba(99,102,241,.2); color:#a5b4fc; border:1px solid rgba(99,102,241,.35); }
.tok-bert   { background:rgba(244,114,182,.15); color:#f9a8d4; border:1px solid rgba(244,114,182,.3); }
.tok-special{ background:rgba(251,191,36,.15); color:#fcd34d; border:1px solid rgba(251,191,36,.3); }

/* ── prediction badge ── */
.pred-pos  { background:rgba(74,222,128,.15); color:#4ade80; border:2px solid #4ade80;
             border-radius:10px; padding:.8rem 1.4rem; text-align:center; }
.pred-neg  { background:rgba(248,113,113,.15); color:#f87171; border:2px solid #f87171;
             border-radius:10px; padding:.8rem 1.4rem; text-align:center; }
.pred-label{ font-size:1.9rem; font-weight:800; }
.pred-conf { font-size:.95rem; color:#94a3b8; margin-top:.25rem; }

/* ── progress log ── */
.log-box    { background:#070b13; border:1px solid #1e2a40; border-radius:10px;
              padding:.8rem 1rem; font-family:'JetBrains Mono',monospace;
              font-size:.8rem; color:#94a3b8; max-height:220px; overflow-y:auto; }
.log-ok     { color:#4ade80; }
.log-step   { color:#818cf8; }

/* ── metric ── */
.big-metric     { font-size:2.2rem; font-weight:800; line-height:1; }
.metric-sub     { font-size:.76rem; color:#64748b; letter-spacing:.07em; text-transform:uppercase; }
.positive       { color:#4ade80; }
.negative       { color:#f87171; }

/* ── tabs ── */
button[data-baseweb="tab"]               { font-size:.9rem!important; font-weight:600!important; }
button[aria-selected="true"]             { color:#818cf8!important; border-bottom:2px solid #6366f1!important; }

/* ── inputs ── */
.stTextArea textarea, .stTextInput input {
    background:rgba(255,255,255,.05)!important;
    border:1px solid rgba(99,102,241,.4)!important;
    border-radius:10px!important; color:#f1f5f9!important; }
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:#818cf8!important;
    box-shadow:0 0 0 3px rgba(99,102,241,.25)!important; }

/* ── buttons ── */
.stButton>button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
    color:#fff!important; border:none!important; border-radius:10px!important;
    padding:.5rem 1.4rem!important; font-weight:600!important; }
.stButton>button:hover { opacity:.85!important; transform:translateY(-1px)!important; }

div[data-testid="stProgress"] > div { background:#6366f1!important; border-radius:8px!important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#4f46e5; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── import pipelines ─────────────────────────────────────────────────────────
import pipelines as pl

# ─── session state defaults ───────────────────────────────────────────────────
for key, val in {
    "nltk_ready": False, "spacy_ready": False,
    "trans_ready": False, "finetuned_ready": False,
    "train_log": [], "last_result": None,
    "nltk_metrics": None, "spacy_metrics": None,
    "ft_metrics": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── always reload from disk on every rerun ──────────────────────────────────
# Module-level flags are the SINGLE source of truth — never stale session_state.
if not pl._nltk_ready:
    pl.load_nltk_pipeline()
st.session_state.nltk_ready = pl._nltk_ready

if not pl._spacy_ready:
    pl.load_spacy_pipeline()
st.session_state.spacy_ready = pl._spacy_ready

# Large models — load on demand but sync if already loaded
st.session_state.trans_ready     = pl._trans_ready
if not pl._finetuned_ready:
    pl.load_finetuned_pipeline()
st.session_state.finetuned_ready = pl._finetuned_ready

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def status_dot(ready: bool) -> str:
    return "🟢" if ready else "🔴"

def conf_bar_html(neg: float, pos: float) -> str:
    return f"""
    <div style='margin:.4rem 0;'>
      <div style='font-size:.75rem;color:#94a3b8;margin-bottom:.2rem'>
        Negative &nbsp;<b style='color:#f87171'>{neg:.1%}</b>
        &nbsp;│&nbsp;
        Positive &nbsp;<b style='color:#4ade80'>{pos:.1%}</b>
      </div>
      <div style='background:#1f2937;border-radius:6px;height:10px;overflow:hidden;'>
        <div style='background:#f87171;width:{neg*100:.1f}%;height:100%;display:inline-block'></div>
        <div style='background:#4ade80;width:{pos*100:.1f}%;height:100%;display:inline-block'></div>
      </div>
    </div>"""

def tokens_html(tokens: list, removed: list = None) -> str:
    removed_set = set(removed) if removed else set()
    parts = []
    for t in tokens:
        cls = "tok-stop" if t in removed_set else "tok-keep"
        parts.append(f"<span class='token {cls}'>{t}</span>")
    return " ".join(parts)

def lemma_html(pairs: list, removed: list) -> str:
    removed_set = set(removed)
    parts = []
    for orig, lemma in pairs:
        if orig in removed_set:
            parts.append(f"<span class='token tok-stop'>{orig}</span>")
        elif orig == lemma:
            parts.append(f"<span class='token tok-keep'>{orig}</span>")
        else:
            parts.append(f"<span class='token tok-lemma'>{orig}→{lemma}</span>")
    return " ".join(parts)

def bert_tokens_html(tokens: list) -> str:
    parts = []
    for t in tokens:
        cls = "tok-special" if t.startswith("[") else ("tok-bert" if t.startswith("##") else "tok-bert")
        display = t if not t.startswith("##") else f"<i>{t}</i>"
        parts.append(f"<span class='token {cls}'>{display}</span>")
    return " ".join(parts)

def feature_chart(features: list, title: str, color: str):
    if not features: return None
    names = [f[0] for f in features[:8]]
    vals  = [f[1] for f in features[:8]]
    bar_colors = ["#4ade80" if v > 0 else "#f87171" for v in vals]
    fig = go.Figure(go.Bar(
        y=names[::-1], x=vals[::-1], orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{v:+.3f}" for v in vals[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=30,b=10,l=10,r=60),
        xaxis=dict(gridcolor="rgba(255,255,255,.07)"),
        title=dict(text=title, font=dict(size=12, color="#c7d2fe")),
        font=dict(family="Inter", color="#e2e8f0"),
    )
    return fig

def prob_gauge(pos_prob: float, model_name: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pos_prob * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563"},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#111827",
            "bordercolor": "#1f2937",
            "steps": [
                {"range": [0, 50],  "color": "rgba(248,113,113,.15)"},
                {"range": [50, 100],"color": "rgba(74,222,128,.15)"},
            ],
            "threshold": {"line": {"color": "#6366f1", "width": 3}, "value": 50},
        },
        title={"text": f"<b>{model_name}</b><br><span style='font-size:.75rem;color:#94a3b8'>Positive probability</span>",
               "font": {"size": 13, "color": "#c7d2fe"}},
    ))
    fig.update_layout(
        height=220, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        font=dict(family="Inter"),
    )
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 NLP Dashboard")
    st.markdown("*Educational Model Comparison*")
    st.divider()

    st.markdown("#### Model Status")
    st.markdown(f"{status_dot(st.session_state.nltk_ready)}  **NLTK** — Logistic Regression")
    st.markdown(f"{status_dot(st.session_state.spacy_ready)}  **spaCy** — Logistic Regression")
    st.markdown(f"{status_dot(st.session_state.trans_ready)}  **DistilBERT** — Pre-trained")
    st.markdown(f"{status_dot(st.session_state.finetuned_ready)}  **Fine-tuned BERT** — Custom")
    st.divider()

    st.markdown("#### 🔑 Legend")
    st.markdown("""
<span class='token tok-keep'>kept token</span>
<span class='token tok-stop'>stopword removed</span>
<span class='token tok-lemma'>original→lemma</span>
<span class='token tok-bert'>bert subword</span>
<span class='token tok-special'>[SPECIAL]</span>
""", unsafe_allow_html=True)
    st.divider()
    st.caption("Built with Streamlit · HuggingFace · scikit-learn · spaCy · NLTK")

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:2rem 0 1.4rem'>
  <h1 style='font-size:2.5rem;font-weight:800;margin-bottom:.3rem;
    background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    NLP Model Comparison
  </h1>
  <p style='color:#64748b;font-size:1.05rem;max-width:600px;margin:auto'>
    See <b style='color:#a5b4fc'>every step</b> of how NLTK, spaCy &amp; DistilBERT process
    your text — from raw input to final prediction.
  </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_analyse, tab_train, tab_batch, tab_compare, tab_about = st.tabs([
    "🔬 Step-by-Step Analyser",
    "🏋️ Train Models",
    "📂 Batch / CSV",
    "📊 Compare Models",
    "📖 How It Works",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ─  STEP-BY-STEP ANALYSER
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyse:

    EXAMPLES = [
        "This movie is not bad at all — I genuinely enjoyed it!",
        "Absolutely terrible. Would not recommend to anyone.",
        "The acting was decent but the plot was quite boring.",
        "One of the best films I've seen this year.",
        "It had its moments, but overall it left me cold.",
    ]

    col_inp, col_ex = st.columns([3, 1])
    with col_inp:
        user_text = st.text_area("✏️ Enter any sentence:", height=100,
            placeholder="e.g. 'This movie is absolutely fantastic!'",
            key="analyse_input")
    with col_ex:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.78rem;color:#6366f1;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.5rem'>💡 Examples</div>", unsafe_allow_html=True)
        for ex in EXAMPLES:
            short = (ex[:42] + "…") if len(ex) > 42 else ex
            if st.button(short, key=f"ex_{ex[:8]}"):
                st.session_state.analyse_input = ex
                st.rerun()

    run_btn = st.button("🚀  Analyse — Show Me Every Step", use_container_width=True)

    if run_btn and user_text.strip():

        any_model = (st.session_state.nltk_ready or
                     st.session_state.spacy_ready or
                     st.session_state.trans_ready)
        if not any_model:
            st.error("⚠️ No models loaded. Go to the **Train Models** tab first.")
            st.stop()

        # ── run all pipelines — use module flags, never stale session_state ─
        results = {}
        with st.spinner("Running pipelines …"):
            if pl._nltk_ready:
                results["NLTK"]             = pl.predict_nltk(user_text.strip())
            if pl._spacy_ready:
                results["spaCy"]            = pl.predict_spacy(user_text.strip())
            if pl._trans_ready:
                results["Transformer"]      = pl.predict_transformer(user_text.strip())
            if pl._finetuned_ready:
                results["Fine-tuned BERT"]  = pl.predict_finetuned(user_text.strip())

        st.session_state.last_result = results

        st.divider()

        # ── top summary cards ──────────────────────────────────────────
        st.markdown("### 📊 Predictions at a Glance")
        META = {
            "NLTK":           ("#60a5fa", "card-blue",   "🔵"),
            "spaCy":          ("#34d399", "card-green",  "🟢"),
            "Transformer":    ("#f472b6", "card-pink",   "🔴"),
            "Fine-tuned BERT":("#a855f7", "card-yellow", "🟣"),
        }
        cols = st.columns(len(results))
        for idx, (name, res) in enumerate(results.items()):
            color, card_cls, icon = META[name]
            with cols[idx]:
                is_pos = res["label"] == "Positive"
                pred_cls = "pred-pos" if is_pos else "pred-neg"
                emoji    = "✅" if is_pos else "❌"
                st.markdown(f"""
                <div class='card {card_cls}'>
                  <div style='font-size:.8rem;font-weight:700;color:{color};letter-spacing:.06em;text-transform:uppercase;margin-bottom:.6rem'>{icon} {name}</div>
                  <div class='{pred_cls}'>
                    <div class='pred-label'>{emoji} {res['label']}</div>
                    <div class='pred-conf'>{res['confidence']:.1%} confidence · {res['time_ms']:.1f} ms</div>
                  </div>
                  {conf_bar_html(res['proba_neg'], res['proba_pos'])}
                </div>""", unsafe_allow_html=True)

        # ── probability gauges ─────────────────────────────────────────
        st.markdown("### 🎯 Probability Gauges")
        gcols = st.columns(len(results))
        gauge_colors = {"NLTK": "#60a5fa", "spaCy": "#34d399", "Transformer": "#f472b6"}
        for idx, (name, res) in enumerate(results.items()):
            with gcols[idx]:
                st.plotly_chart(prob_gauge(res["proba_pos"], name, gauge_colors[name]),
                                use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # STEP-BY-STEP BREAKDOWNS
        # ══════════════════════════════════════════════════════════════
        st.markdown("## 🔍 Step-by-Step Pipeline Breakdown")
        st.caption("Expand each model below to see exactly how it processed your text.")

        # ── NLTK STEPS ────────────────────────────────────────────────
        if "NLTK" in results:
            with st.expander("🔵 NLTK Pipeline — Full Walkthrough", expanded=True):
                s = results["NLTK"]["steps"]
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>Raw Input</div>
                  <code style='color:#e2e8f0;font-size:.92rem'>"{s['raw']}"</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>Lowercase</div>
                  <code style='color:#a5b4fc'>{s['lowercased']}</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>Remove punctuation & special chars (regex: [^a-z\\s])</div>
                  <code style='color:#a5b4fc'>{s['punct_removed']}</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>4</span>Tokenise (nltk.word_tokenize) → {len(s['tokenized'])} tokens</div>
                  {tokens_html(s['tokenized'])}
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>5</span>Remove stopwords & short tokens → {len(s['stops_removed'])} tokens kept</div>
                  <div style='margin-bottom:.3rem;font-size:.78rem;color:#64748b'>
                    Green = kept &nbsp;│&nbsp; Red strikethrough = removed
                  </div>
                  {tokens_html(s['tokenized'], s['removed_words'])}
                  <div style='margin-top:.7rem;font-size:.78rem;color:#64748b'>Final string ↓</div>
                  <code style='color:#4ade80;font-size:.88rem'>"{s['final']}"</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>6</span>TF-IDF Vectorisation</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    20,000-feature vocabulary · 1–2 grams · sparse matrix representation.<br>
                    Each token is scored by how <i>important</i> it is in this doc vs. the corpus.
                  </p>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>7</span>Logistic Regression → Prediction</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    score = Σ (TF-IDF weight × learned coefficient) for each token<br>
                    P(Positive) = sigmoid(score) = <b style='color:#818cf8'>{results['NLTK']['proba_pos']:.4f}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # feature contribution chart
                fig = feature_chart(results["NLTK"]["top_features"],
                                    "Top Token Contributions (TF-IDF × coefficient)", "#60a5fa")
                if fig:
                    st.markdown("**📊 Which tokens pushed the prediction?**")
                    st.markdown("<div style='font-size:.8rem;color:#64748b'>Green bar = pushes toward Positive · Red = toward Negative</div>", unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)

        # ── spaCy STEPS ───────────────────────────────────────────────
        if "spaCy" in results:
            with st.expander("🟢 spaCy Pipeline — Full Walkthrough", expanded=True):
                s = results["spaCy"]["steps"]
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>Raw Input</div>
                  <code style='color:#e2e8f0;font-size:.92rem'>"{s['raw']}"</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>Lowercase</div>
                  <code style='color:#a5b4fc'>{s['lowercased']}</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>spaCy Tokenise → Lemmatise</div>
                  <div style='margin-bottom:.3rem;font-size:.78rem;color:#64748b'>
                    Purple = lemma differs from original &nbsp;│&nbsp; Green = unchanged &nbsp;│&nbsp; Red = removed
                  </div>
                  {lemma_html(s['lemma_pairs'], s['removed_words'])}
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>4</span>Remove stopwords & punctuation → {len(s['stops_removed'])} lemmas kept</div>
                  {"".join(f"<span class='token tok-keep'>{t}</span>" for t in s['stops_removed'])}
                  <div style='margin-top:.7rem;font-size:.78rem;color:#64748b'>Final string ↓</div>
                  <code style='color:#34d399;font-size:.88rem'>"{s['final']}"</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>5</span>TF-IDF Vectorisation (same as NLTK but on lemmas)</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    Lemmatisation reduces vocabulary — "running", "runs", "ran" → all become "run".
                  </p>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>6</span>Logistic Regression → Prediction</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    P(Positive) = sigmoid(score) = <b style='color:#818cf8'>{results['spaCy']['proba_pos']:.4f}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                fig = feature_chart(results["spaCy"]["top_features"],
                                    "Top Lemma Contributions (TF-IDF × coefficient)", "#34d399")
                if fig:
                    st.markdown("**📊 Which lemmas pushed the prediction?**")
                    st.plotly_chart(fig, use_container_width=True)

        # ── BERT STEPS ────────────────────────────────────────────────
        if "Transformer" in results:
            with st.expander("🔴 DistilBERT Pipeline — Full Walkthrough", expanded=True):
                s = results["Transformer"]["steps"]
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>Raw Input (NO manual preprocessing needed)</div>
                  <code style='color:#e2e8f0;font-size:.92rem'>"{s['raw']}"</code>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>WordPiece Tokeniser → {s['n_tokens']} subword tokens</div>
                  <div style='margin-bottom:.4rem;font-size:.78rem;color:#64748b'>
                    Yellow = special tokens ([CLS]/[SEP]) &nbsp;│&nbsp; Pink = word/subword tokens &nbsp;│&nbsp; <i>##prefix</i> = continuation of previous word
                  </div>
                  {bert_tokens_html(s['tokens'])}
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>Token ID Mapping</div>
                  <div style='font-size:.78rem;font-family:"JetBrains Mono",monospace;color:#6366f1;word-break:break-all'>
                    {str(s['ids'][:20])}{'…' if len(s['ids'])>20 else ''}
                  </div>
                  <p style='color:#94a3b8;font-size:.82rem;margin-top:.35rem'>
                    Each token → integer ID in BERT's 30,522-word vocabulary.
                  </p>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>4</span>DistilBERT Encoder (6 layers × 12 attention heads)</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    Each token attends to every other token → <b>contextual embeddings</b>.<br>
                    This is why "not bad" → Positive (BERT sees the negation; NLTK/spaCy miss it).
                  </p>
                </div>
                <div class='arrow'>↓</div>

                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>5</span>[CLS] token → Linear classifier → Softmax</div>
                  <p style='color:#94a3b8;margin:.2rem 0;font-size:.85rem'>
                    P(Negative) = <b style='color:#f87171'>{results['Transformer']['proba_neg']:.4f}</b>
                    &nbsp;│&nbsp;
                    P(Positive) = <b style='color:#4ade80'>{results['Transformer']['proba_pos']:.4f}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

                st.info("💡 **Why no feature chart for BERT?** BERT uses dense contextual vectors — there's no single 'important word' score. The whole sentence context matters simultaneously.", icon="ℹ️")

        if "Fine-tuned BERT" in results and "Transformer" in results:
            with st.expander("🟣 Fine-tuned vs Pre-trained Comparison", expanded=True):
                st.markdown("See how fine-tuning on your dataset changed the model's confidence:")
                c1, c2 = st.columns(2)
                p_base = results['Transformer']['proba_pos']
                p_fine = results['Fine-tuned BERT']['proba_pos']
                c1.metric("DistilBERT (Pre-trained)", f"{p_base:.1%} Positive")
                
                delta = p_fine - p_base
                # Streamlit arrow logic: Positive delta is green, negative is red. 
                # If both are > 0.5 (positive class), higher is better.
                c2.metric("DistilBERT (Fine-tuned)", f"{p_fine:.1%} Positive", delta=f"{delta:+.1%} shift")

    elif run_btn:
        st.warning("⚠️ Please type a sentence first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ─  TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:

    st.markdown("### 🏋️ Train Your Models")
    st.markdown("Use the **SST-2 default dataset** or **upload your own CSV**. Watch every training step live.")

    # ── Dataset source ────────────────────────────────────────────────
    st.markdown("#### 📂 Step 1 — Choose Dataset")
    ds_source = st.radio("Dataset source:", ["SST-2 (auto-download, ~67k reviews)", "Upload my own CSV"],
                         horizontal=True)

    texts_data, labels_data = [], []
    dataset_ready = False

    if ds_source.startswith("SST-2"):
        n_samples = st.slider("Number of training samples:", 1000, 67349, 5000, 500)
        if st.button("⬇️ Load SST-2 Dataset"):
            with st.spinner("Downloading SST-2 …"):
                texts_data, labels_data = pl.load_default_dataset(n_samples)
            st.session_state["train_texts"]  = texts_data
            st.session_state["train_labels"] = labels_data
            st.success(f"✅ Loaded {len(texts_data):,} samples  |  "
                       f"Positive: {sum(labels_data):,}  |  "
                       f"Negative: {len(labels_data)-sum(labels_data):,}")
            dataset_ready = True
    else:
        uploaded = st.file_uploader("Upload CSV (needs `text` + `label` columns, label = 0/1)", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "text" not in df_up.columns or "label" not in df_up.columns:
                st.error("❌ CSV must have columns: `text` and `label` (0 = Negative, 1 = Positive)")
            else:
                df_up = df_up.dropna(subset=["text","label"])
                df_up["label"] = df_up["label"].astype(int)
                texts_data  = df_up["text"].tolist()
                labels_data = df_up["label"].tolist()
                st.session_state["train_texts"]  = texts_data
                st.session_state["train_labels"] = labels_data
                st.success(f"✅ Loaded {len(df_up):,} rows  |  "
                           f"Positive: {sum(labels_data):,}  |  "
                           f"Negative: {len(labels_data)-sum(labels_data):,}")
                dataset_ready = True
                # mini EDA
                with st.expander("👀 Dataset Preview"):
                    st.dataframe(df_up.head(10), use_container_width=True, hide_index=True)
                    wc = df_up["text"].apply(lambda x: len(x.split()))
                    fig_wc = px.histogram(wc, nbins=30, title="Word count distribution",
                                          template="plotly_dark", color_discrete_sequence=["#818cf8"])
                    fig_wc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                         showlegend=False, height=250)
                    st.plotly_chart(fig_wc, use_container_width=True)

    # pull from session if already loaded
    if not dataset_ready and "train_texts" in st.session_state:
        texts_data  = st.session_state["train_texts"]
        labels_data = st.session_state["train_labels"]
        dataset_ready = bool(texts_data)

    st.divider()
    st.markdown("#### 🤖 Step 2 — Train a Model")

    col_nltk_btn, col_spacy_btn = st.columns(2)

    # ─── TRAIN NLTK ───────────────────────────────────────────────────
    with col_nltk_btn:
        st.markdown("""
        <div class='card card-blue'>
          <div style='font-size:.85rem;font-weight:700;color:#60a5fa;letter-spacing:.06em;text-transform:uppercase'>🔵 NLTK Pipeline</div>
          <ol style='color:#94a3b8;font-size:.83rem;margin:.5rem 0 0 1rem;line-height:1.8'>
            <li>Lowercase + remove punctuation</li>
            <li>NLTK tokenise</li>
            <li>Remove stopwords</li>
            <li>TF-IDF (20k, 1-2 grams)</li>
            <li>Logistic Regression</li>
          </ol>
        </div>""", unsafe_allow_html=True)

        train_nltk_btn = st.button("▶️ Train NLTK Model",
                                   disabled=not dataset_ready,
                                   use_container_width=True,
                                   key="train_nltk")

        nltk_prog_bar  = st.empty()
        nltk_step_txt  = st.empty()
        nltk_log_box   = st.empty()

        if train_nltk_btn and dataset_ready:
            log_lines = []
            def nltk_cb(msg, pct):
                log_lines.append(msg)
                nltk_prog_bar.progress(pct, text=f"{pct*100:.0f}%")
                nltk_step_txt.markdown(f"<div style='color:#818cf8;font-size:.9rem'>⚡ {msg}</div>",
                                       unsafe_allow_html=True)
                nltk_log_box.markdown(
                    "<div class='log-box'>" + "<br>".join(
                        f"<span class='log-ok'>{l}</span>" if l.startswith("✅") else
                        f"<span class='log-step'>{l}</span>" for l in log_lines
                    ) + "</div>", unsafe_allow_html=True)

            with st.spinner("Training NLTK model …"):
                metrics = pl.train_nltk_model(texts_data, labels_data, nltk_cb)
            st.session_state.nltk_ready   = pl._nltk_ready
            st.session_state.nltk_metrics = metrics
            nltk_prog_bar.empty(); nltk_step_txt.empty()
            st.success(f"✅ NLTK trained!  Accuracy: **{metrics['accuracy']:.2%}**  |  Vocab: {metrics['vocab_size']:,} features")
            st.rerun()  # ← refresh sidebar status dots immediately

        if st.session_state.nltk_metrics:
            m = st.session_state.nltk_metrics
            r = m["report"]
            ecols = st.columns(3)
            ecols[0].metric("Accuracy",  f"{m['accuracy']:.2%}")
            ecols[1].metric("F1 (Pos)",  f"{r['Positive']['f1-score']:.3f}")
            ecols[2].metric("Vocab size",f"{m['vocab_size']:,}")

    # ─── TRAIN spaCy ─────────────────────────────────────────────────
    with col_spacy_btn:
        st.markdown("""
        <div class='card card-green'>
          <div style='font-size:.85rem;font-weight:700;color:#34d399;letter-spacing:.06em;text-transform:uppercase'>🟢 spaCy Pipeline</div>
          <ol style='color:#94a3b8;font-size:.83rem;margin:.5rem 0 0 1rem;line-height:1.8'>
            <li>Lowercase</li>
            <li>spaCy en_core_web_sm</li>
            <li>Lemmatise all tokens</li>
            <li>Remove stops + punct</li>
            <li>TF-IDF → Logistic Regression</li>
          </ol>
        </div>""", unsafe_allow_html=True)

        train_spacy_btn = st.button("▶️ Train spaCy Model",
                                    disabled=not dataset_ready,
                                    use_container_width=True,
                                    key="train_spacy")

        spacy_prog_bar  = st.empty()
        spacy_step_txt  = st.empty()
        spacy_log_box   = st.empty()

        if train_spacy_btn and dataset_ready:
            log_lines = []
            def spacy_cb(msg, pct):
                log_lines.append(msg)
                spacy_prog_bar.progress(pct, text=f"{pct*100:.0f}%")
                spacy_step_txt.markdown(f"<div style='color:#34d399;font-size:.9rem'>⚡ {msg}</div>",
                                        unsafe_allow_html=True)
                spacy_log_box.markdown(
                    "<div class='log-box'>" + "<br>".join(
                        f"<span class='log-ok'>{l}</span>" if l.startswith("✅") else
                        f"<span class='log-step'>{l}</span>" for l in log_lines
                    ) + "</div>", unsafe_allow_html=True)

            with st.spinner("Training spaCy model …"):
                metrics = pl.train_spacy_model(texts_data, labels_data, spacy_cb)
            st.session_state.spacy_ready   = pl._spacy_ready
            st.session_state.spacy_metrics = metrics
            spacy_prog_bar.empty(); spacy_step_txt.empty()
            st.success(f"✅ spaCy trained!  Accuracy: **{metrics['accuracy']:.2%}**  |  Vocab: {metrics['vocab_size']:,} features")
            st.rerun()  # ← refresh sidebar status dots immediately

        if st.session_state.spacy_metrics:
            m = st.session_state.spacy_metrics
            r = m["report"]
            ecols = st.columns(3)
            ecols[0].metric("Accuracy",  f"{m['accuracy']:.2%}")
            ecols[1].metric("F1 (Pos)",  f"{r['Positive']['f1-score']:.3f}")
            ecols[2].metric("Vocab size",f"{m['vocab_size']:,}")

    st.divider()

    # ─── BERT — no training, but load ─────────────────────────────────
    st.markdown("#### 🔴 DistilBERT — Pre-trained (No Training Needed)")
    st.markdown("""
    <div class='card card-pink'>
      <p style='color:#94a3b8;font-size:.88rem;margin:0'>
        DistilBERT is <b>already fine-tuned on SST-2 by HuggingFace</b>. It downloads automatically (~270 MB) on first use.
        Unlike NLTK/spaCy, re-training BERT requires GPUs and hours of compute — so we use the
        pre-trained checkpoint directly.
      </p>
    </div>""", unsafe_allow_html=True)

    load_bert_btn = st.button("⬇️ Load DistilBERT (downloads ~270 MB if not cached)",
                              use_container_width=True)
    if load_bert_btn:
        with st.spinner("Loading DistilBERT …"):
            ok = pl.load_transformer_pipeline()
        st.session_state.trans_ready = pl._trans_ready   # sync from real module flag
        if ok:
            st.success("✅ DistilBERT loaded and ready! Refreshing …")
            st.rerun()  # ← forces full rerender → sidebar shows 🟢 immediately
        else:
            st.error("❌ Failed to load. Check your internet connection.")

    st.markdown("#### 🟣 DistilBERT — Fine-tune on your data")
    st.write("Train the transformer head specifically on your dataset. This takes time (~30+ sec per 100 samples on CPU).")
    ft_epochs = st.slider("Epochs", 1, 3, 1)
    if st.button("🚀 Fine-tune DistilBERT", type="primary", use_container_width=True):
        if not dataset_ready:
            st.error("⚠️ Please load a dataset first (Step 1).")
        else:
            loss_chart = st.empty()
            ft_step_txt = st.empty()
            
            def ft_cb(msg, pct, extra):
                ft_step_txt.markdown(f"**{msg}**")
                if "losses" in extra and extra["losses"]:
                    # plot live loss curve
                    df_l = pd.DataFrame(extra["losses"])
                    fig = px.line(df_l, x="step", y="loss", title="Training Loss (Live)", template="plotly_dark", height=250)
                    loss_chart.plotly_chart(fig, use_container_width=True)
            
            with st.spinner(f"Fine-tuning for {ft_epochs} epoch(s)..."):
                metrics = pl.finetune_distilbert(
                    st.session_state["train_texts"], 
                    st.session_state["train_labels"], 
                    epochs=ft_epochs,
                    progress_cb=ft_cb
                )
            
            st.session_state.finetuned_ready = pl._finetuned_ready
            st.session_state.ft_metrics = metrics
            st.success(f"✅ Fine-tuning complete! Held-out Accuracy: **{metrics['accuracy']:.2%}**")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ─  BATCH / CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### 📂 Batch Evaluation")
    st.markdown("Upload a CSV with a `text` column and optionally a `label` column (0/1). All loaded models run on every row.")

    batch_file = st.file_uploader("Upload CSV (max 500 rows)", type=["csv"], key="batch_up")

    if batch_file:
        df_b = pd.read_csv(batch_file).head(500)
        if "text" not in df_b.columns:
            st.error("❌ Need a `text` column.")
        else:
            st.info(f"Loaded **{len(df_b)}** rows")
            st.dataframe(df_b.head(5), use_container_width=True, hide_index=True)

            if st.button("▶️ Run Batch Analysis", key="batch_run_btn"):
                prog  = st.progress(0, "Analysing …")
                rows  = []
                n     = len(df_b)
                for i, txt in enumerate(df_b["text"].astype(str)):
                    entry = {"text": txt[:60]}
                    if st.session_state.nltk_ready:
                        r = pl.predict_nltk(txt)
                        entry.update({"nltk_pred": r["label"],
                                       "nltk_conf": f"{r['confidence']:.2%}",
                                       "nltk_ms":   f"{r['time_ms']:.1f}"})
                    if st.session_state.spacy_ready:
                        r = pl.predict_spacy(txt)
                        entry.update({"spacy_pred": r["label"],
                                       "spacy_conf": f"{r['confidence']:.2%}",
                                       "spacy_ms":   f"{r['time_ms']:.1f}"})
                    if st.session_state.trans_ready:
                        r = pl.predict_transformer(txt)
                        entry.update({"bert_pred": r["label"],
                                       "bert_conf": f"{r['confidence']:.2%}",
                                       "bert_ms":   f"{r['time_ms']:.1f}"})
                    if st.session_state.finetuned_ready:
                        r = pl.predict_finetuned(txt)
                        entry.update({"ft_pred": r["label"],
                                       "ft_conf": f"{r['confidence']:.2%}",
                                       "ft_ms":   f"{r['time_ms']:.1f}"})
                    if "label" in df_b.columns:
                        entry["true_label"] = df_b["label"].iloc[i]
                    rows.append(entry)
                    prog.progress((i+1)/n, f"Row {i+1}/{n}")

                prog.empty()
                df_out = pd.DataFrame(rows)
                st.success(f"✅ Done!")
                st.dataframe(df_out, use_container_width=True, hide_index=True)

                # accuracy
                if "true_label" in df_out.columns:
                    st.markdown("#### 🎯 Accuracy vs Ground Truth")
                    acols = st.columns(4)
                    for idx2, (pname, pcol) in enumerate([
                        ("NLTK","nltk_pred"), ("spaCy","spacy_pred"), ("BERT","bert_pred"), ("FT BERT", "ft_pred")
                    ]):
                        if pcol in df_out.columns:
                            norm = lambda v: "Positive" if str(v).strip() in ("1","Positive") else "Negative"
                            preds  = df_out[pcol].map(norm)
                            truths = df_out["true_label"].map(norm)
                            acc    = (preds == truths).mean()
                            acols[idx2].metric(pname, f"{acc:.1%} accuracy")

                st.download_button("⬇️ Download Results",
                                   df_out.to_csv(index=False).encode(),
                                   "batch_results.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ─  COMPARE MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### 📊 Model Comparison")

    if st.session_state.last_result:
        results = st.session_state.last_result

        # bar charts
        names = list(results.keys())
        confs = [results[n]["confidence"] for n in names]
        times = [results[n]["time_ms"]    for n in names]
        ppos  = [results[n]["proba_pos"]  for n in names]
        pneg  = [results[n]["proba_neg"]  for n in names]
        CLRS  = {"NLTK": "#60a5fa", "spaCy": "#34d399", "Transformer": "#f472b6", "Fine-tuned BERT": "#a855f7"}
        colors = [CLRS.get(n, "#ffffff") for n in names]

        fig = make_subplots(rows=1, cols=3,
            subplot_titles=("Confidence", "Inference Time (ms)", "P(Positive) vs P(Negative)"))

        fig.add_trace(go.Bar(x=names, y=confs, marker_color=colors,
            text=[f"{v:.1%}" for v in confs], textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=times, marker_color=colors,
            text=[f"{v:.1f}" for v in times], textposition="outside"), row=1, col=2)
        fig.add_trace(go.Bar(x=names, y=ppos, name="Positive",
            marker_color="#4ade80"), row=1, col=3)
        fig.add_trace(go.Bar(x=names, y=pneg, name="Negative",
            marker_color="#f87171"), row=1, col=3)

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            barmode="group", showlegend=True,
            font=dict(family="Inter", color="#e2e8f0"),
            margin=dict(t=50, b=20),
        )
        fig.update_yaxes(range=[0, 1.2], tickformat=".0%", row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # radar
        st.markdown("#### 🕸️ Confidence Radar")
        fig_r = go.Figure()
        for n in names:
            fig_r.add_trace(go.Scatterpolar(
                r=[results[n]["confidence"], results[n]["proba_pos"]],
                theta=["Confidence", "P(Positive)"],
                fill="toself", name=n,
                line=dict(color=CLRS[n], width=2),
            ))
        fig_r.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(range=[0,1], tickformat=".0%",
                                       gridcolor="rgba(255,255,255,.1)")),
            height=380, font=dict(family="Inter", color="#e2e8f0"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # summary table
        st.markdown("#### 🗂 Summary Table")
        rows = [{"Model": n,
                 "Prediction": results[n]["label"],
                 "Confidence": f"{results[n]['confidence']:.2%}",
                 "P(Positive)": f"{results[n]['proba_pos']:.4f}",
                 "P(Negative)": f"{results[n]['proba_neg']:.4f}",
                 "Time": f"{results[n]['time_ms']:.1f} ms"}
                for n in names]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("👆 Run an analysis in the **Step-by-Step Analyser** tab first — charts will appear here.", icon="💡")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  ─  HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:

    st.markdown("### 📖 How Each Pipeline Works")

    with st.expander("🔵 NLTK Pipeline", expanded=False):
        st.markdown("""
| Step | What happens | Why |
|------|-------------|-----|
| Lowercase | `"Movie"` → `"movie"` | Normalise case |
| Punct removal | `"great!"` → `"great"` | Remove noise |
| Tokenise | `"not bad"` → `["not","bad"]` | Split into words |
| Stopword removal | removes "the","is","not" | Reduce noise … **but "not" is removed!** |
| TF-IDF | word → float weight | Importance score |
| Logistic Regression | weighted sum → sigmoid | Classification |

> ⚠️ **Key weakness:** NLTK removes "not" as a stopword — so *"not bad"* looks the same as *"bad"* → often misclassified as Negative.
        """)

    with st.expander("🟢 spaCy Pipeline", expanded=False):
        st.markdown("""
| Step | What happens | Why |
|------|-------------|-----|
| Lowercase | Normalise | Same as NLTK |
| Lemmatise | `"running"` → `"run"` | Reduce vocabulary |
| Stop/punct removal | Same filter | Smaller vocabulary |
| TF-IDF | Lemma → weight | Importance score |
| Logistic Regression | Classify | Same as NLTK |

> ✅ **Advantage over NLTK:** Lemmatisation reduces vocabulary sparsity — `"films"`, `"film"`, `"filming"` all map to `"film"`.  
> ⚠️ **Same weakness:** Negation is still lost when "not" is removed.
        """)

    with st.expander("🔴 DistilBERT — Transformer", expanded=False):
        st.markdown("""
| Step | What happens | Why |
|------|-------------|-----|
| WordPiece tokenise | `"unbelievable"` → `["un","##believ","##able"]` | Sub-word vocab = handles unknown words |
| Token IDs | Each token → integer ID | Input to neural net |
| 6× Transformer layers | Self-attention across all tokens | Context from every word |
| [CLS] representation | Summary of full sentence | Used for classification |
| Linear + Softmax | → probabilities | Final output |

> ✅ **Key strength:** Attention mechanism means `"not bad"` attends to both "not" and "bad" simultaneously — it knows "not" modifies "bad" → Positive.
        """)

    st.divider()
    st.markdown("### ⚖️ Trade-off Summary")
    st.markdown("""
| Aspect | NLTK | spaCy | DistilBERT | Fine-tuned BERT |
|--------|------|-------|-----------|------------------|
| Accuracy (SST-2) | ~83% | ~84% | ~91% | Local Accuracy |
| Speed (CPU) | ⚡⚡⚡ 2–8 ms | ⚡⚡ 4–12 ms | ⚡ 80–200 ms | ⚡ 80–200 ms |
| Negation handling | ❌ Weak | ❌ Weak | ✅ Strong | ✅ Strong |
| Model size | ~few MB | ~few MB | ~270 MB | ~270 MB |
| Interpretability | ✅ High | ✅ High | ⚠️ Black-box | ⚠️ Black-box |
| Training speed | ✅ Fast | ✅ Fast | ❌ Needs GPU | ❌ Slow on CPU |
| Custom training | ✅ Easy | ✅ Easy | ❌ Complex | ✅ Done via UI |
    """)

    st.divider()
    st.markdown("### 💼 Resume Bullet")
    st.info("""Built an educational NLP comparison dashboard (NLTK · spaCy · DistilBERT) featuring step-by-step pipeline visualisation, live model training with progress callbacks, custom dataset upload, token-level explanations, and interactive Plotly charts — demonstrating accuracy vs. latency trade-offs across traditional and deep-learning approaches.""")
