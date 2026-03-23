"""
app.py  ─  Topic-Aware AI Dashboard (Educational Edition)
Run:  streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Topic-Aware AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS (Mirroring Sentiment Analyser) ──────────────────────────────────────
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
.card-hybrid{ border-left:4px solid #8b5cf6; }

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
.tok-bert   { background:rgba(244,114,182,.15); color:#f9a8d4; border:1px solid rgba(244,114,182,.3); }

/* ── buttons & inputs ── */
.stTextArea textarea, .stTextInput input {
    background:rgba(255,255,255,.05)!important;
    border:1px solid rgba(99,102,241,.4)!important;
    border-radius:10px!important; color:#f1f5f9!important; }
.stButton>button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
    color:#fff!important; border:none!important; border-radius:10px!important;
    padding:.5rem 1.4rem!important; font-weight:600!important; }
.stButton>button:hover { opacity:.85!important; transform:translateY(-1px)!important; }

/* ── log box ── */
.log-box    { background:#070b13; border:1px solid #1e2a40; border-radius:10px;
              padding:.8rem 1rem; font-family:'JetBrains Mono',monospace;
              font-size:.8rem; color:#94a3b8; max-height:220px; overflow-y:auto; }
.log-ok     { color:#4ade80; }
.log-step   { color:#818cf8; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#4f46e5; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── import pipelines ─────────────────────────────────────────────────────────
import pipelines as pl

# ─── always reload logic ─────────────────────────────────────────────────────
st.session_state.lda_ready  = pl.load_lda()
st.session_state.bert_ready = pl.load_bert()
st.session_state.clf_ready  = pl.load_classifier()
st.session_state.corpus_ready = pl.load_corpus()

def status_dot(ready: bool) -> str:
    return "🟢" if ready else "🔴"

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Topic-Aware AI")
    st.markdown("*Advanced Hybrid NLP*")
    st.divider()

    st.markdown("#### System Status")
    st.markdown(f"{status_dot(st.session_state.lda_ready)}  **LDA Topic Model** (Gensim)")
    st.markdown(f"{status_dot(st.session_state.bert_ready)}  **Transformer** (DistilBERT)")
    st.markdown(f"{status_dot(st.session_state.clf_ready)}  **Hybrid Classifier** (A)")
    st.markdown(f"{status_dot(st.session_state.corpus_ready)}  **Semantic Corpus** (C)")
    st.divider()

    st.markdown("#### 🔑 Architecture")
    st.markdown("""
<div style='font-size:0.85rem;color:#94a3b8;line-height:1.5'>
1. <b>BoW + Gensim</b> → 5 Topic Probs<br>
2. <b>DistilBERT</b> → 768-D Context<br>
3. <b>Fusion</b> → 773-D Vector<br>
4. <b>Downstream</b> → Classify / Search 
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.caption("Built with Gensim · Transformers · scikit-learn · Streamlit")

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:2rem 0 1.4rem'>
  <h1 style='font-size:2.5rem;font-weight:800;margin-bottom:.3rem;
    background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    Topic-Aware AI System
  </h1>
  <p style='color:#64748b;font-size:1.05rem;max-width:700px;margin:auto'>
    Combines <b style='color:#60a5fa'>Gensim LDA (Topics)</b> and <b style='color:#f472b6'>DistilBERT (Meaning)</b>
    into a powerful <b>Hybrid Semantic representation</b> for classification, visualisation, and search.
  </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_step, tab_train, tab_vis, tab_search, tab_about = st.tabs([
    "🔬 Step-by-Step",
    "🏋️ Train Classifier (A)",
    "📉 Visualise & Explore (B)",
    "🔍 Search Engine (C)",
    "📖 How It Works"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 ─ STEP-BY-STEP (Hybrid Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
with tab_step:
    st.markdown("### 🔬 Test the Hybrid Pipeline")
    
    col_inp, col_ex = st.columns([3, 1])
    with col_inp:
        user_text = st.text_area("✏️ Enter any sentence:", height=100,
            placeholder="e.g. 'A thrilling psychological horror that kept me on the edge of my seat...'",
            key="step_input")
    with col_ex:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.78rem;color:#6366f1;font-weight:600;text-transform:uppercase'>💡 Examples</div>", unsafe_allow_html=True)
        def set_val(t): st.session_state.step_input = t
        for ex in [
            "A thrilling psychological horror that kept me on the edge.",
            "Absolutely terrible acting and a boring, predictable plot.",
            "Very funny romantic comedy, laughed the whole time."
        ]:
            st.button((ex[:30] + "…"), key=f"ex_{ex[:8]}", on_click=set_val, args=(ex,))

    if st.button("🚀 Process Hybrid Vector", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter a sentence.")
        elif not (st.session_state.lda_ready and st.session_state.bert_ready):
            st.error("⚠️ Models not loaded. Run offline pipeline first (`python run_pipeline.py`).")
        else:
            with st.spinner("Processing through LDA and DistilBERT..."):
                steps = pl._make_hybrid_steps(user_text)
            
            # --- Results ---
            st.success("✅ Hybrid Vector generated!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='card card-blue'>
                   <div style='font-size:.8rem;color:#60a5fa;font-weight:bold'>Gensim LDA</div>
                   <h2 style='margin:0'>5 Dims</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>Topic Distribution</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='card card-pink'>
                   <div style='font-size:.8rem;color:#f472b6;font-weight:bold'>DistilBERT</div>
                   <h2 style='margin:0'>768 Dims</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>Mean Pooled Context</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='card card-hybrid'>
                   <div style='font-size:.8rem;color:#8b5cf6;font-weight:bold'>Hybrid Vector</div>
                   <h2 style='margin:0'>773 Dims</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>L2 Normed + Concatenated</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🧩 Pipeline Breakdown")
            
            # LDA Flow
            with st.expander("🔵 Gensim LDA Walkthrough", expanded=True):
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>Preprocess (Gensim simple_preprocess)</div>
                  <div style="margin-bottom:.3rem"><span style="color:#64748b;font-size:.8rem">Green = kept | Red = stopwords/short</span></div>
                  {" ".join([f"<span class='token tok-keep'>{t}</span>" if t in steps['kept_tokens'] else f"<span class='token tok-stop'>{t}</span>" for t in steps['all_tokens']])}
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>Bag-of-Words (BoW) Representation</div>
                  <code style='color:#a5b4fc'>[ (word_id, count), ... ] → {steps['bow']} ...</code>
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>LDA Topic Inference [5 Dimensions]</div>
                </div>
                """, unsafe_allow_html=True)
                
                lda_df = pd.DataFrame({"Topic": [f"Topic {i}" for i in range(5)], "Probability": steps['lda_dist']})
                fig1 = px.bar(lda_df, x="Probability", y="Topic", orientation='h', template='plotly_dark', color_discrete_sequence=['#60a5fa'])
                fig1.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)
                
                dom = steps['dominant_topic']
                top_words = ", ".join([w[0] for w in steps['topic_words'][dom]])
                st.markdown(f"<div style='color:#94a3b8;font-size:0.9rem'><b>Dominant: Topic {dom}</b> ({top_words})</div>", unsafe_allow_html=True)

            # BERT Flow
            with st.expander("🔴 DistilBERT Walkthrough", expanded=True):
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>WordPiece Tokenisation</div>
                  {" ".join([f"<span class='token tok-bert'>{t}</span>" for t in steps['bert_tokens']])}
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>DistilBERT Encoder (6 Layers)</div>
                  <p style='color:#94a3b8;font-size:.85rem;margin:.2rem 0'>Contextual representations for every token: [batch=1, seq_len={len(steps['bert_tokens'])}, dim=768]</p>
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>Mean Pooling (Masked)</div>
                  <p style='color:#94a3b8;font-size:.85rem;margin:.2rem 0'>Averages all non-padding token embeddings to create a single sentence vector [768 Dimensions]</p>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 ─ TRAIN CLASSIFIER (A)
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    st.markdown("### 🏋️ Train Classifier on Hybrid Features")
    st.markdown("This proves the downstream effectiveness of our **773-D Hybrid Vector**.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("Select Classifier Algorithm:", ["LogisticRegression", "SVM", "MLP"])
        train_btn = st.button("▶️ Train Model", type="primary", use_container_width=True)
        st.markdown("""
        <div class="card card-hybrid" style="margin-top:1rem">
        <ul style="color:#94a3b8;font-size:.85rem;padding-left:1.2rem;margin:0">
          <li>Uses <code>data/hybrid_features.npy</code></li>
          <li>80/20 Train/Test split</li>
          <li>Compares Hybrid (773) vs BERT-only (768)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        prog_bar = st.empty()
        log_txt  = st.empty()
        log_box  = st.empty()
        
    if train_btn:
        if not pl._hybrid_files_exist():
            st.error("⚠️ Hybrid features not found! Please run `python run_pipeline.py` locally first to generate `data/hybrid_features.npy`.")
        else:
            logs = []
            def train_cb(msg, pct):
                logs.append(msg)
                prog_bar.progress(pct, text=f"{pct*100:.0f}%")
                log_txt.markdown(f"<div style='color:#8b5cf6;font-weight:bold'>⚡ {msg}</div>", unsafe_allow_html=True)
                log_box.markdown("<div class='log-box'>" + "<br>".join(
                    f"<span class='log-ok'>{l}</span>" if "✅" in l else f"<span class='log-step'>{l}</span>" for l in logs
                ) + "</div>", unsafe_allow_html=True)
            
            with st.spinner("Training..."):
                res = pl.train_classifier(model_type=model_choice, progress_cb=train_cb)
            
            st.session_state.clf_ready = True
            st.success("✅ Training completed and model saved!")
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybrid Accuracy", f"{res['accuracy']:.2%}")
            # Show change vs baseline
            c2.metric("BERT-only Accuracy", f"{res['bert_only_acc']:.2%}", delta=f"{res['improvement']:+.2%} Hybrid boost" if res['improvement'] != 0 else "0.00%", delta_color="normal")
            c3.metric("Train Time", f"{res['train_time']:.2f}s")
            
            st.markdown("#### Detailed Classification Report")
            df_rep = pd.DataFrame(res['report']).transpose().round(3)
            st.dataframe(df_rep, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 ─ VISUALISE & EXPLORE (B)
# ─────────────────────────────────────────────────────────────────────────────
with tab_vis:
    st.markdown("### 📉 Unsupervised Exploration")
    
    v1, v2 = st.tabs(["🗺️ Embeddings (UMAP/PCA)", "📚 Topic Words"])
    
    with v1:
        st.markdown("See how the documents naturally cluster based on their semantic vectors.")
        if not pl._hybrid_files_exist():
            st.warning("Data not available.")
        else:
            if st.button("Generate UMAP Projection (Takes 5-10s)"):
                with st.spinner("Calculating 2D projection..."):
                    umap_data = pl.get_umap_data(n_samples=1000)
                    
                    df_u = pd.DataFrame({
                        "x": umap_data["x"], "y": umap_data["y"],
                        "Label": ["Positive" if l==1 else "Negative" for l in umap_data["labels"]],
                        "Dominant Topic": [f"Topic {t}" for t in umap_data["topics"]],
                        "Text": umap_data["texts"]
                    })
                    
                    st.markdown(f"**{umap_data['method']} Projection of {umap_data['n']} Samples**")
                    
                    fig = px.scatter(df_u, x="x", y="y", color="Dominant Topic", symbol="Label",
                                     hover_data=["Text"], template="plotly_dark",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0)))
                    fig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

    with v2:
        if not st.session_state.lda_ready:
            st.warning("LDA model not loaded.")
        else:
            st.markdown("#### Interperting Gensim Topics")
            topics = pl.get_topic_words(8)
            t_cols = st.columns(len(topics))
            for t_idx, words in topics.items():
                with t_cols[t_idx % len(t_cols)]:
                    st.markdown(f"""
                    <div class='card card-blue'>
                      <div style='font-size:.85rem;font-weight:bold;color:#60a5fa'>Topic {t_idx}</div>
                      <div style='font-size:.85rem;color:#e2e8f0;margin-top:.5rem'>
                        {'<br>'.join([f"{w} <span style='color:#64748b;float:right'>{p:.3f}</span>" for w,p in words])}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 ─ SEARCH ENGINE (C)
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 Hybrid Semantic Search Engine")
    st.markdown("Search the entire IMDB dataset by **Meaning + Concept**, not just exact keywords.")
    
    if not st.session_state.corpus_ready:
        st.warning("⚠️ Corpus not loaded. (Requires `data/hybrid_features.npy`)")
    else:
        sq_col, filt_col = st.columns([3, 1])
        with sq_col:
            q = st.text_input("Enter search query:", placeholder="e.g. 'A mind bending sci-fi about time travel'")
        with filt_col:
            f_val = st.selectbox("Sentiment Filter", ["All", "Positive Only", "Negative Only"])
            flt_map = {"All": -1, "Positive Only": 1, "Negative Only": 0}

        if st.button("Search", use_container_width=True, type="primary"):
            if q.strip():
                with st.spinner("Searching corpus..."):
                    results = pl.search(q, top_k=5, filter_label=flt_map[f_val])
                
                st.markdown(f"**Top {len(results)} Results for:** `{q}`")
                
                for r in results:
                    emoji = "✅ Positive" if r['label'] == 1 else "❌ Negative"
                    color = "#4ade80" if r['label'] == 1 else "#f87171"
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border-left:3px solid {color}; padding:1rem; margin-bottom:.8rem; border-radius:4px;'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:.5rem;'>
                            <span style='color:{color}; font-weight:bold; font-size:.85rem'>{emoji}</span>
                            <span style='color:#8b5cf6; font-size:.85rem; font-family:monospace'>Sim: {r['sim']:.4f}</span>
                        </div>
                        <div style='color:#e2e8f0; font-size:.95rem; line-height:1.5'>{r['snippet']}...</div>
                    </div>
                    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 ─ HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    with open("README.md", "r") as f:
        st.markdown(f.read())
