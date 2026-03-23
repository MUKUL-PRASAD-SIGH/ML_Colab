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
    page_title="Topic-Aware AI Masterclass",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS (Hotter Educational UI) ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]       { font-family:'Inter',sans-serif; }
.stApp                           { background: #07090e; color:#e2e8f0; }
section[data-testid="stSidebar"] { background:#0a0d14; border-right:1px solid #1e2a40; }

/* ── cards ── */
.card       { background:rgba(255,255,255,.02); border:1px solid rgba(99,102,241,.15);
              border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:.9rem;
              backdrop-filter:blur(8px); transition:0.3s; }
.card:hover { background:rgba(255,255,255,.04); border-color:rgba(99,102,241,.4); box-shadow:0 0 35px rgba(99,102,241,.1); transform:translateY(-2px); }
.card-blue  { border-left:4px solid #60a5fa; }
.card-green { border-left:4px solid #34d399; }
.card-pink  { border-left:4px solid #f472b6; }
.card-yellow{ border-left:4px solid #fbbf24; }
.card-hybrid{ border-left:4px solid #c084fc; background:rgba(192,132,252,0.03); }

/* ── theory box ── */
.theory-box { background:linear-gradient(145deg, #111827, #0f1420); border:1px solid #1f2937;
              border-radius:12px; padding:1.5rem; margin-bottom:1rem; border-left:4px solid #fcd34d; }
.theory-title { color:#fcd34d; font-weight:800; font-size:1.1rem; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.05em; }

/* ── step pipeline ── */
.step-box   { background:#0b0f19; border:1px solid #1f2937; border-radius:10px;
              padding:.9rem 1.1rem; margin:.4rem 0; font-size:.88rem; box-shadow:inset 0 0 15px rgba(0,0,0,0.5); }
.step-title { font-size:.72rem; font-weight:700; letter-spacing:.1em;
              text-transform:uppercase; color:#818cf8; margin-bottom:.35rem; }
.step-number{ display:inline-block; background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; border-radius:50%;
              width:22px; height:22px; text-align:center; line-height:22px;
              font-size:.75rem; font-weight:700; margin-right:.5rem; box-shadow:0 0 10px rgba(99,102,241,0.5); }
.arrow      { text-align:center; color:#4f46e5; font-size:1.3rem; margin:.05rem 0; opacity:0.8; }

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
    background:rgba(255,255,255,.03)!important;
    border:1px solid rgba(99,102,241,.3)!important;
    border-radius:10px!important; color:#f1f5f9!important; }
.stButton>button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;
    color:#fff!important; border:none!important; border-radius:10px!important;
    padding:.5rem 1.4rem!important; font-weight:600!important; box-shadow:0 4px 15px rgba(99,102,241,0.3)!important; }
.stButton>button:hover { opacity:.9!important; transform:translateY(-2px)!important; box-shadow:0 6px 20px rgba(99,102,241,0.5)!important; }

/* ── log box ── */
.log-box    { background:#030509; border:1px solid #1e2a40; border-radius:10px;
              padding:.8rem 1rem; font-family:'JetBrains Mono',monospace;
              font-size:.8rem; color:#94a3b8; max-height:220px; overflow-y:auto; box-shadow:inset 0 0 20px rgba(0,0,0,0.8); }
.log-ok     { color:#4ade80; }
.log-step   { color:#818cf8; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-thumb { background:linear-gradient(180deg,#6366f1,#8b5cf6); border-radius:6px; }
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
    st.markdown("""
<h2 style='background:linear-gradient(90deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-top:0;'>
🧠 Topic-Aware AI
</h2>
<p style='color:#94a3b8;font-size:0.9rem;margin-top:-10px;margin-bottom:20px'>The Ultimate ML Masterclass</p>
""", unsafe_allow_html=True)

    st.markdown("#### Database Status")
    st.markdown(f"{status_dot(st.session_state.lda_ready)}  **Topic Finder** (Gensim)")
    st.markdown(f"{status_dot(st.session_state.bert_ready)}  **Deep Reader** (Transformer)")
    st.markdown(f"{status_dot(st.session_state.corpus_ready)}  **Search Database**")
    st.markdown(f"{status_dot(st.session_state.clf_ready)}  **Trained AI Brain**")
    st.divider()

    st.markdown("#### 🎯 Core Concepts (Simplified)")
    st.markdown("""
<div style='font-size:0.85rem;color:#94a3b8;line-height:1.6'>
<b style='color:#60a5fa'>1. What is a Vector?</b><br>Just a list of numbers! AI converts words into numbers so it can do math on them.<br><br>
<b style='color:#f472b6'>2. What does BERT do?</b><br>It reads sentences and turns the "vibe" and "context" into a list of 768 numbers.<br><br>
<b style='color:#c084fc'>3. What does Gensim do?</b><br>It figures out what topics you're talking about (e.g. Comedy vs Horror) and gives you a tiny list of numbers showing the percentages.
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:1rem 0 1.4rem'>
  <div style='display:inline-block;background:rgba(99,102,241,0.1);color:#818cf8;padding:0.3rem 1rem;border-radius:20px;font-size:0.8rem;font-weight:700;letter-spacing:0.1em;border:1px solid rgba(99,102,241,0.3);margin-bottom:1rem'>
    EASY TO UNDERSTAND MASTERCLASS
  </div>
  <h1 style='font-size:3rem;font-weight:800;margin-bottom:.3rem;line-height:1.2;
    background:linear-gradient(90deg,#818cf8,#c084fc,#f472b6,#fcd34d);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    Topic-Aware AI System
  </h1>
  <p style='color:#64748b;font-size:1.1rem;max-width:800px;margin:auto;margin-top:0.5rem'>
    Watch how AI combines <b>High-Level Topics</b> ("Science Fiction") with <b>Deep Contextual Meaning</b> ("Not bad at all") 
    to create the ultimate super-smart search engine and classifier.
  </p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_theory, tab_build, tab_step, tab_train, tab_vis, tab_search = st.tabs([
    "📖 The Theory (Simple)",
    "⚙️ Build Engine",
    "🔬 Step-by-Step",
    "🏋️ Classify (A)",
    "📉 Visualise (B)",
    "🔍 Search (C)",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 ─ THE THEORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_theory:
    st.markdown("## 🧠 Welcome to the Masterclass")
    st.markdown("Don't worry if you aren't a math genius! Let's break down exactly what this AI is doing in plain English.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='theory-box' style='border-color:#60a5fa'>
          <div class='theory-title' style='color:#60a5fa'>📘 Gensim Topic Model (The Categoriser)</div>
          <p style='color:#94a3b8;font-size:0.95rem;line-height:1.6'>
            <b>What it does:</b> It looks at all the keywords in your text and guesses the category.<br><br>
            If you write <i>"The spaceship lasers were cool"</i>, Gensim sees the words "spaceship" and "lasers" and says:<br>
            <i>"I am 90% sure this belongs to Topic 3 (Sci-Fi), and 10% sure it belongs to Topic 1 (Action)."</i>
          </p>
          <ul style='color:#94a3b8;font-size:0.9rem'>
            <li><b>Output:</b> A tiny list of percentages (e.g. 5 numbers if we have 5 topics).</li>
            <li><b>Why it's awesome:</b> We humans can read the list and instantly know what the AI thinks the text is about.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='theory-box' style='border-color:#f472b6'>
          <div class='theory-title' style='color:#f472b6'>📕 DistilBERT (The Deep Reader)</div>
          <p style='color:#94a3b8;font-size:0.95rem;line-height:1.6'>
             <b>What it does:</b> It reads sentences forwards and backwards to understand context, sarcasm, and tone.<br><br>
             If you write <i>"It was not terrible"</i>, Gensim only sees the word "terrible" and might think it's negative. But BERT sees "not terrible" together and realizes it's actually positive!
          </p>
          <ul style='color:#94a3b8;font-size:0.9rem'>
            <li><b>Output:</b> A massive list of 768 numbers representing deep meaning.</li>
            <li><b>Why it's awesome:</b> It's incredibly smart at understanding how humans actually speak.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='background:rgba(192,132,252,0.1); border-left:4px solid #c084fc; padding:1.5rem; border-radius:8px'>
    <h3 style='color:#c084fc;margin-top:0'>🧬 The Hybrid Fusion Problem</h3>
    <p style='color:#e2e8f0;font-size:1.05rem;line-height:1.6'>
    We want to combine both lists of numbers into ONE super-list. But there's a problem:<br><br>
    The Gensim list only has tiny numbers between 0.0 and 1.0 (percentages).<br>
    The BERT list has massive numbers ranging wildly (-20 to +20).<br><br>
    If we glue them together, the huge BERT numbers will completely crush and hide the tiny Gensim numbers!
    </p>
    <p style='color:#fcd34d;font-size:1.05rem'>
    <b>The Mathematical Solution (L2 Normalisation):</b> We squash both lists down so they perfectly form a circle with a radius of 1. By scaling them to be identical in size, they can share the same space equally!
    </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 ─ BUILD ENGINE (Offline pipeline replacement)
# ─────────────────────────────────────────────────────────────────────────────
with tab_build:
    st.markdown("### ⚙️ Build The Core Engine")
    st.markdown("Choose a dataset below. If this is your first time, use the **Fast Built-In Dataset** to see how it works instantly!")
    
    # Selection of Dataset
    ds_source = st.radio("Step 1: Choose Dataset Source", 
                         [
                          "⚡ Fast Built-In Dataset (Instant, 200 Reviews) - BEST FOR TESTING!", 
                          "🐌 Full IMDB Auto-Download (Takes 1-3 mins to download 80MB, 2000 Reviews)", 
                          "📁 Upload Custom CSV (Requires `text` and `label` columns)"
                         ],
                         horizontal=False)

    ds_type_key = "fast"
    if "IMDB" in ds_source: ds_type_key = "imdb"
    elif "CSV" in ds_source: ds_type_key = "custom"

    custom_text_data = None
    custom_label_data = None
    dataset_valid = True

    if ds_type_key == "custom":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "text" not in df_up.columns or "label" not in df_up.columns:
                st.error("❌ CSV must have columns: `text` and `label` (0 = Negative, 1 = Positive)")
                dataset_valid = False
            else:
                df_up = df_up.dropna(subset=["text","label"])
                custom_text_data  = df_up["text"].astype(str).tolist()
                try:
                    custom_label_data = df_up["label"].astype(int).tolist()
                    st.success(f"✅ Loaded {len(df_up)} rows successfully.")
                    with st.expander("👀 Dataset Preview"):
                        st.dataframe(df_up.head(5))
                except Exception as e:
                    st.error(f"❌ Error parsing labels as integer integers: {e}")
                    dataset_valid = False
        else:
            dataset_valid = False

    st.markdown("---")
    st.markdown("#### Step 2: Architecture Configuration")
    
    col_s, col_t = st.columns(2)
    with col_s: 
        max_s = len(custom_text_data) if custom_text_data else 2000
        n_samp = st.slider("Number of Reviews to process", 50, max_s, min(200, max_s), 50)
    with col_t: 
        n_tops = st.slider("How many Topics should Gensim try to find?", 2, 8, 3, 1)

    build_btn = st.button("🚀 START BUILDING ENGINE (Generates Data & Models)", type="primary", use_container_width=True, disabled=not dataset_valid)
    
    prog_bar = st.empty()
    log_txt  = st.empty()
    log_box  = st.empty()

    if build_btn:
        logs = []
        def build_cb(msg, pct):
            logs.append(msg)
            prog_bar.progress(pct, text=f"Step {int(pct*100)}% Complete")
            log_txt.markdown(f"<div style='color:#c084fc;font-weight:bold;font-size:1.1rem;margin-bottom:10px'>⚡ {msg}</div>", unsafe_allow_html=True)
            log_box.markdown("<div class='log-box'>" + "<br>".join(
                f"<span class='log-ok'>{l}</span>" if "✅" in l else f"<span class='log-step'>{l}</span>" for l in logs
            ) + "</div>", unsafe_allow_html=True)
            
        with st.spinner("Processing... Neural Networks are crunching numbers..."):
            pl.build_core_pipeline(
                n_samples=n_samp, 
                n_topics=n_tops, 
                progress_cb=build_cb,
                ds_type=ds_type_key,
                custom_texts=custom_text_data,
                custom_labels=custom_label_data
            )
            
        st.session_state.lda_ready  = pl.load_lda()
        st.session_state.bert_ready = pl.load_bert()
        st.session_state.corpus_ready = pl.load_corpus()
        
        st.success("🎉 ENGINE BUILT SUCCESSFULLY! Head over to the Step-by-Step tab.")
        st.balloons()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 ─ STEP-BY-STEP (Hybrid Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
with tab_step:
    st.markdown("### 🔬 Test the AI Pipeline Live")
    st.markdown("Write a sentence below and watch exactly how the computer turns your English words into numbers.")
    
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
            "A thrilling action movie with lots of special effects.",
            "Absolutely terrible acting and a completely boring plot.",
            "Very funny romantic comedy, laughed the whole time."
        ]:
            st.button((ex[:30] + "…"), key=f"ex_{ex[:8]}", on_click=set_val, args=(ex,))

    if st.button("🚀 Process into Array", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter a sentence.")
        elif not (st.session_state.lda_ready and st.session_state.bert_ready):
            st.error("⚠️ The AI Engine isn't built yet! Go to the 'Build Engine' tab and click start.")
        else:
            with st.spinner("Doing the math..."):
                steps = pl._make_hybrid_steps(user_text)
            
            st.success("✅ Text successfully converted into a massive list of numbers!")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='card card-blue'>
                   <div style='font-size:.8rem;color:#60a5fa;font-weight:bold;text-transform:uppercase'>Gensim Output</div>
                   <h2 style='margin:0'>{steps['lda_dist'].__len__()} Numbers</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>Showing Topic Percentages</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='card card-pink'>
                   <div style='font-size:.8rem;color:#f472b6;font-weight:bold;text-transform:uppercase'>DistilBERT Output</div>
                   <h2 style='margin:0'>768 Numbers</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>Showing Deep Meaning</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='card card-hybrid' style='box-shadow: 0 0 20px rgba(192,132,252,0.3)'>
                   <div style='font-size:.8rem;color:#c084fc;font-weight:bold;text-transform:uppercase'>Final Merged Array</div>
                   <h2 style='margin:0'>{steps['hybrid_dim']} Numbers</h2>
                   <div style='font-size:.8rem;color:#94a3b8'>Glued perfectly together</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🧩 Exactly how it happened:")
            
            # LDA Flow
            with st.expander("📘 Phase 1: Gensim extracts Topics", expanded=True):
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>Cleaning Text (Removing useless words)</div>
                  <div style="margin-bottom:.3rem"><span style="color:#64748b;font-size:.8rem">Green words are kept. Red words (like 'the', 'and') are thrown away because they don't help find the topic.</span></div>
                  {" ".join([f"<span class='token tok-keep'>{t}</span>" if t in steps['kept_tokens'] else f"<span class='token tok-stop'>{t}</span>" for t in steps['all_tokens']])}
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>Counting Word Frequencies (Bag-of-Words)</div>
                  <code style='color:#a5b4fc;background:#030509;padding:10px;border-radius:6px;display:block'>[ (word_ID, how_many_times), ... ]<br><br>Result: <b>{steps['bow']}</b> ...</code>
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>Gensim Guessing the Topic Percentages</div>
                </div>
                """, unsafe_allow_html=True)
                
                lda_df = pd.DataFrame({"Topic": [f"Topic {i}" for i in range(len(steps['lda_dist']))], "Probability": steps['lda_dist']})
                fig1 = px.bar(lda_df, x="Probability", y="Topic", orientation='h', template='plotly_dark', color_discrete_sequence=['#60a5fa'])
                fig1.update_layout(height=200, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig1, use_container_width=True)
                
                dom = steps['dominant_topic']
                top_words = ", ".join([w[0] for w in steps['topic_words'][dom]])
                st.markdown(f"<div style='background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.3);padding:10px;border-radius:8px;color:#e2e8f0;font-size:0.9rem'><b style='color:#60a5fa'>Winning Topic is: Topic {dom}</b><br>Reason? Because this topic is strongly linked with words like: <i style='color:#94a3b8'>{top_words}</i></div>", unsafe_allow_html=True)

            # BERT Flow
            with st.expander("📕 Phase 2: DistilBERT extracts Deep Meaning", expanded=True):
                st.markdown(f"""
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>1</span>WordPiece Splitting (Instead of deleting words, BERT breaks them into chunks!)</div>
                  <div style="margin-bottom:.3rem"><span style="color:#64748b;font-size:.8rem">Notice the special [CLS] starting tag and [SEP] ending tag.</span></div>
                  {" ".join([f"<span class='token tok-bert'>{t}</span>" for t in steps['bert_tokens']])}
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>2</span>Reading text backwards & forwards (Self-Attention)</div>
                  <p style='color:#94a3b8;font-size:.85rem;margin:.2rem 0'>BERT generates 768 specific numbers for every single word in the sentence based on its neighbors.</p>
                </div>
                <div class='arrow'>↓</div>
                <div class='step-box'>
                  <div class='step-title'><span class='step-number'>3</span>Averaging them out (Mean Pooling)</div>
                  <p style='color:#94a3b8;font-size:.85rem;margin:.2rem 0'>It takes the mathematical average of all those word numbers to create one single master list of 768 numbers representing the whole sentence.</p>
                </div>
                """, unsafe_allow_html=True)

            # Concatenation
            with st.expander("🧬 Phase 3: Merging them together", expanded=True):
                st.markdown("""
                Both the Gensim List and the BERT List are "Normalised" (squashed into a perfect circle shape so they are identical sizes), and then glued side-by-side to make the final **Hybrid Array**.
                """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 ─ TRAIN CLASSIFIER (A)
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    st.markdown("### 🏋️ Train Downstream Classifier")
    st.markdown("We can now feed our ultimate merged lists into a basic AI (like a Logistic Regression) and watch it get amazing scores!")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("Select Classification Algorithm:", ["LogisticRegression", "SVM", "MLP (Neural Net)"])
        train_btn = st.button("▶️ Execute AI Training", type="primary", use_container_width=True)
        st.markdown("""
        <div class="card card-hybrid" style="margin-top:1rem">
        <ul style="color:#94a3b8;font-size:.85rem;padding-left:1.2rem;margin:0;line-height:1.6">
          <li>Teaches itself using 80% of data</li>
          <li>Tests itself on remaining 20%</li>
          <li>Auto-calculates the baseline BERT-only score to prove our Gensim trick actually helped!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        c_prog = st.empty()
        c_log_t  = st.empty()
        c_log_b  = st.empty()
        
    if train_btn:
        if not pl._hybrid_files_exist():
            st.error("⚠️ AI not built yet! Please run the Engine Builder in Tab 2 first.")
        else:
            ml_logs = []
            def t_cb_clean(msg, pct):
                ml_logs.append(msg)
                c_prog.progress(pct, text=f"{int(pct*100)}%")
                c_log_t.markdown(f"<div style='color:#c084fc;font-weight:bold'>⚡ {msg}</div>", unsafe_allow_html=True)
                c_log_b.markdown("<div class='log-box'>" + "<br>".join(
                    f"<span class='log-ok'>{l}</span>" if "✅" in l else f"<span class='log-step'>{l}</span>" for l  in ml_logs
                ) + "</div>", unsafe_allow_html=True)

            algo_map = {"LogisticRegression":"LogisticRegression", "SVM":"SVM", "MLP (Neural Net)":"MLP"}
            
            with st.spinner("Training Mathematical Models..."):
                res = pl.train_classifier(model_type=algo_map[model_choice], progress_cb=t_cb_clean)
            
            st.session_state.clf_ready = True
            st.success("✅ Training completed!")
            
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Hybrid Accuracy", f"{res['accuracy']:.2%}", help="How often it guessed Positive/Negative correctly.")
            
            delta = res['improvement']
            delta_color = "normal" if delta == 0 else "normal"
            c2.metric("BERT-Only Score (Without Gensim)", f"{res['bert_only_acc']:.2%}", delta=f"{delta:+.2%} Boost from Gensim" if delta != 0 else "0.00%", delta_color=delta_color)
            c3.metric("Training Time", f"{res['train_time']:.3f} seconds")
            
            st.markdown("#### Scikit-Learn Report Card")
            df_rep = pd.DataFrame(res['report']).transpose().round(3)
            st.dataframe(df_rep, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 ─ VISUALISE SPACE (B)
# ─────────────────────────────────────────────────────────────────────────────
with tab_vis:
    st.markdown("### 📉 2D Galaxy Map (UMAP)")
    
    v1, v2 = st.tabs(["🗺️ Plot Text in 2D Space", "📚 View Learned Topics"])
    
    with v1:
        st.markdown("""
        How do we visualize a list of **700+ numbers** on a flat screen? We use **UMAP**.
        UMAP magically squashes massive lists of numbers down to just X and Y coordinates, pushing mathematically similar reviews together into clusters (galaxies).
        """)
        if not pl._hybrid_files_exist():
            st.warning("⚠️ Data not available. Run the Build Engine in Tab 2 first.")
        else:
            if st.button("🌌 Generate Galaxy Map"):
                with st.spinner("Calculating X and Y coordinates... (can take 5 seconds)"):
                    umap_data = pl.get_umap_data(n_samples=500)
                    
                    df_u = pd.DataFrame({
                        "x": umap_data["x"], "y": umap_data["y"],
                        "Label": ["Positive" if l==1 else "Negative" for l in umap_data["labels"]],
                        "Dominant Concept": [f"Topic {t}" for t in umap_data["topics"]],
                        "Preview": umap_data["texts"]
                    })
                    
                    st.markdown(f"**{umap_data['method']} Manifold Projection (Plotting {umap_data['n']} Datapoints)**")
                    
                    fig = px.scatter(df_u, x="x", y="y", color="Dominant Concept", symbol="Label",
                                     hover_data=["Preview"], template="plotly_dark",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_traces(marker=dict(size=8, opacity=0.9, line=dict(width=0)))
                    fig.update_layout(height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

    with v2:
        if not st.session_state.lda_ready:
            st.warning("⚠️ Topic model not loaded.")
        else:
            st.markdown("#### The Conceptual Categories Discovered by Gensim")
            st.markdown("The AI automatically grouped these words together without us telling it anything!")
            topics = pl.get_topic_words(8)
            t_cols = st.columns(len(topics))
            for t_idx, words in topics.items():
                with t_cols[t_idx % len(t_cols)]:
                    st.markdown(f"""
                    <div class='card card-blue' style='padding:1rem;height:100%'>
                      <div style='font-size:1.1rem;font-weight:900;color:#60a5fa;margin-bottom:.5rem;border-bottom:1px solid rgba(96,165,250,0.3);padding-bottom:.3rem'>Topic {t_idx}</div>
                      <div style='font-size:.9rem;color:#e2e8f0;'>
                        {'<br>'.join([f"<b>{w}</b> <span style='color:#64748b;float:right'>{p:.3f}</span>" for w,p in words])}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 ─ SEARCH ENGINE (C)
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 The Smart Search Engine")
    st.markdown("""
    Normal search bars only look for exact matching words.  
    This search bar calculates the **Cosine Angle** between your query's number-list and the number-lists of every document in the database, allowing you to search by *vibe* and *concept*.
    """)

    if not st.session_state.corpus_ready:
        st.warning("⚠️ Database missing. (Run Build Engine in Tab 2).")
    else:
        sq_col, filt_col = st.columns([3, 1])
        with sq_col:
            q = st.text_input("Database Query (Search by meaning, not keyword):", placeholder="e.g. 'A mind bending sci-fi about traveling through time'")
        with filt_col:
            f_val = st.selectbox("Metadata Filter", ["All", "Positive Sentiment", "Negative Sentiment", "Other (Custom)"])
            flt_map = {"All": -1, "Positive Sentiment": 1, "Negative Sentiment": 0, "Other (Custom)": 2}

        if st.button("Search Database", use_container_width=True, type="primary"):
            if q.strip():
                with st.spinner("Finding mathematically closest matches..."):
                    results = pl.search(q, top_k=5, filter_label=flt_map[f_val] if f_val != "Other (Custom)" else -1)
                
                st.markdown(f"**Top {len(results)} matches for:** `{q}`")
                
                for r in results:
                    l_str = str(r['label_str']).upper()
                    if l_str == "1" or l_str == "POSITIVE":
                        emoji, color = "✅ POSITIVE VERDICT", "#4ade80"
                    elif l_str == "0" or l_str == "NEGATIVE":
                        emoji, color = "❌ NEGATIVE VERDICT", "#f87171"
                    else:
                        emoji, color = f"🏷️ {l_str}", "#60a5fa"
                        
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border-left:4px solid {color}; padding:1.2rem; margin-bottom:1rem; border-radius:6px; box-shadow:0 4px 10px rgba(0,0,0,0.2)'>
                        <div style='display:flex; justify-content:space-between; margin-bottom:.8rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:.5rem'>
                            <span style='color:{color}; font-weight:800; font-size:.85rem; letter-spacing:0.05em'>{emoji}</span>
                            <span style='color:#c084fc; font-weight:700; font-size:.85rem; font-family:"JetBrains Mono",monospace; background:rgba(192,132,252,0.1); padding:0.2rem 0.6rem; border-radius:4px'>Match Score: {r['sim']*100:.1f}%</span>
                        </div>
                        <div style='color:#e2e8f0; font-size:1rem; line-height:1.6'>{r['snippet']}...</div>
                    </div>
                    """, unsafe_allow_html=True)
