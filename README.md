<!-- Auto-generated sections are between the markers below. Don't manually edit those blocks. -->

<div align="center">

```
███╗   ███╗██╗     
████╗ ████║██║     
██╔████╔██║██║     
██║╚██╔╝██║██║     
██║ ╚═╝ ██║███████╗
╚═╝     ╚═╝╚══════╝
    W O R K B E N C H
```

# 🧪 ML Workbench 

### *A growing collection of hands-on Machine Learning & AI mini-projects — built to learn, built to share.*

[![Projects](https://img.shields.io/badge/Projects-1%20and%20counting-6366f1?style=for-the-badge&logo=rocket&logoColor=white)](.)
[![Deployed](https://img.shields.io/badge/All%20Projects-Live%20%26%20Deployed-22c55e?style=for-the-badge&logo=vercel&logoColor=white)](.)
[![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20Streamlit%20%7C%20HuggingFace-f472b6?style=for-the-badge&logo=python&logoColor=white)](.)
[![Beginner Friendly](https://img.shields.io/badge/Beginner-Friendly%20Docs-fbbf24?style=for-the-badge&logo=bookstack&logoColor=white)](.)

</div>

---

## ⚡ What is this?

This repo is my **personal ML playground** — a single place where I drop every small AI/ML project I build. Each one lives in its own folder, has its own docs, and most importantly — **it's deployed so you can actually try it**, not just read about it.

I know this stuff. I've read the papers, burned the 3AM oil, debugged the shape mismatches. But this isn't just a flex — it's a **learning resource**. I document everything so that **beginners can follow along**, understand what's happening under the hood, and use these apps as interactive sandboxes instead of staring at static notebooks.

> **"Build it. Deploy it. Let people play with it. That's the only way it's real."**

---

## 🚀 The Philosophy

```
📦  One folder  =  One project
📖  Every project  =  Its own README + documentation
🌐  Every project  =  Live deployed & accessible to anyone
🎓  Every project  =  Explained for beginners, built by someone who knows it
```

I'm not here to collect stars. I'm here to **build things that work**, learn deeply, and make sure others can learn from them too. If you're new to ML — pick any project below, try the live demo, and then come back and read how it works. That's the loop.

---

## 📁 Projects

> 🤖 *This section is auto-updated by `update_readme.py` whenever a new project folder is added.*

<!-- PROJECTS_START -->
| # | Project | Description | Stack | Live Demo | Status |
|---|---------|-------------|-------|-----------|--------|
| 01 | [**Sentiment Analyser**](./Sentiment_Analyser/) | Interactive NLP dashboard comparing NLTK, spaCy & RoBERTa — see every pipeline step visualised in real-time. Train models, run batch inference, and fine-tune a Transformer — all from the browser. | Python · Streamlit · HuggingFace · scikit-learn · spaCy · Plotly | [**🌐 Try it live →**](https://sentilyticz.streamlit.app/) | ✅ Live |
| 02 | [**Topic-Aware AI**](./Topic-Aware-AI/) | Educational masterclass dashboard teaching Hybrid NLP concepts — combining sklearn NMF for topic extraction and DistilBERT for deep semantic meaning. Train the brain in-browser. | Python · Streamlit · HuggingFace · scikit-learn | [**🌐 Try it live →**](https://topicawareai.streamlit.app/) | ✅ Live |
<!-- PROJECTS_END -->

---

## 🔬 Project #01 — Sentiment Analyser (NLP Dashboard)

<table>
<tr>
<td width="60%">

### What it does

An **educational NLP comparison dashboard** that takes any sentence and shows you — step by step — how three completely different pipelines process it and arrive at a sentiment prediction.

**Not just a black box.** You see the actual tokenisation, stopword removal, TF-IDF weights, BERT subword splits, probability distributions, and which specific tokens drove the final prediction.

### The models inside

| Model | Approach | Accuracy |
|-------|----------|----------|
| 🔵 **NLTK** | TF-IDF + Logistic Regression | ~83% |
| 🟢 **spaCy** | Lemmatisation + TF-IDF + LR | ~84% |
| 🔴 **RoBERTa** | Pre-trained Transformer | ~93% |
| 🟣 **Fine-tuned RoBERTa** | Your data + gradient descent | 🔥 Custom |

### Key features

- ⚡ Step-by-step pipeline visualisation (every preprocessing step shown)
- 🎯 Interactive probability gauges per model
- 📊 Feature contribution charts (which words drove the prediction)
- 🏋️ Train your own models inside the dashboard
- 📂 Batch CSV inference with live progress
- 🔄 Fine-tune RoBERTa on your own dataset — right in the browser

</td>
<td width="40%" align="center">

### 🌐 Live Demo

**[sentilyticz.streamlit.app](https://sentilyticz.streamlit.app/)**

```
Try it yourself →
Type any sentence.
Watch 4 AI models
race to understand it.
See every single step.
No ML knowledge needed.
```

**Built with:**
```
Streamlit    → UI
HuggingFace  → Transformers
scikit-learn → Classical ML
spaCy        → NLP pipeline
Plotly       → Visualisations
```

[📖 Full docs](./Sentiment_Analyser/README.md)

</td>
</tr>
</table>

---

## 🚀 Project #02 — Topic-Aware AI (Hybrid NLP Masterclass)

<table>
<tr>
<td width="60%">

### What it does

An **educational masterclass dashboard** that teaches complex AI concepts through an interactive live UI. It demonstrates how to combine High-Level Topics with Deep Contextual Meaning to create a powerful hybrid semantic search engine.

**Not just a black box.** You can watch the pipeline train live in the browser, see step-by-step mathematical explanations of how text is translated into vectors, and interact visually with the embeddings using UMAP.

### The models inside

| Model | Approach | Output |
|-------|----------|----------|
| 🔵 **NMF (sklearn)** | Topic Extraction | Probability percentages (e.g. 90% Sci-Fi) |
| 🔴 **DistilBERT** | Pre-trained Transformer | 768-dim semantic vector |
| 🟣 **Hybrid Engine** | L2-Norm + Concat | Combined unified feature space |

### Key features

- ⚡ Build and train the system live in your browser using IMDB, Fast, or Custom datasets
- 🔬 **Step-by-Step Explainer**: Watch sentences translate into numbers with real-time math explanations
- 🎯 **Classify & Search**: Train Logistic Regression / SVM / MLP and search by meaning
- 📊 **Visualise Space**: Interactive 2D UMAP scatter plots and topic word clouds
- 💎 **Premium Design**: Built using the "Obsidian Masterclass" design system

</td>
<td width="40%" align="center">

### 🌐 Live Demo

**[topicawareai.streamlit.app](https://topicawareai.streamlit.app/)**

```
Try it yourself →
Click 'Build Engine'.
Train the AI live.
Understand the math
behind modern NLP.
```

**Built with:**
```
Streamlit    → UI
HuggingFace  → Transformers
scikit-learn → Classical ML & NMF
Plotly/UMAP  → Visualisations
```

[📖 Full docs](./Topic-Aware-AI/README.md)

</td>
</tr>
</table>

---

## 🎓 For Beginners

Every project in this repo is designed to be **beginner-accessible**:

1. **Try the live demo first** — no setup, no install, just open the link
2. **Read the project README** — it explains the concepts, not just the code
3. **Run it locally** — each project has a clean `setup & run` section
4. **Read the code** — it's commented and structured to be readable

You don't need to understand transformers to use the Sentiment Analyser. But after playing with it for 20 minutes, you'll understand more about how NLP works than most textbook chapters will tell you.

---

## 🔧 Running Any Project Locally

```bash
# Clone the repo
git clone <your-repo-url>
cd <project-folder>

# Install dependencies
pip install -r requirements.txt

# Run (each project has specific instructions in its README)
streamlit run app.py        # for Streamlit apps
# or
python app.py               # for Flask/FastAPI apps
# or
npm run dev                 # for Next.js projects
```

---

## 📬 Get In Touch

Found a bug? Want to suggest a project? Just want to say the Sentiment Analyser made you understand NLP better?

> Open an issue, drop a PR, or reach out. This is a living repo and contributions / ideas are always welcome.

---

<div align="center">

*Built with curiosity. Deployed with care. Documented for everyone.*

**⭐ Star the repo if any of these projects helped you learn something new.**

</div>
