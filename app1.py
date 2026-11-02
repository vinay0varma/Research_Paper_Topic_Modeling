import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Research Paper AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 5% 0%,rgba(99,102,241,.22),transparent 30%),radial-gradient(circle at 95% 8%,rgba(236,72,153,.18),transparent 28%),#0b1020}
.block-container{padding-top:2rem;max-width:1450px}
.hero{padding:28px 30px;border-radius:22px;margin-bottom:22px;background:linear-gradient(120deg,#4f46e5,#7c3aed,#db2777);box-shadow:0 12px 35px #4f46e544}
.hero h1{color:white;font-size:2.4rem;margin:0}.hero p{color:#f3e8ff;font-size:1.05rem;margin:8px 0 0}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#4f46e533,#db277722);border:1px solid #a78bfa55;padding:16px;border-radius:16px}
div.stButton>button{border-radius:12px;border:1px solid #a78bfa88;font-weight:600}
div.stButton>button:hover{border-color:#f472b6;color:#f9a8d4}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px;border-color:#a78bfa55}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_papers():
    df = pd.read_csv("papers.csv")
    for col in ["title", "abstract"]:
        if col not in df.columns:
            raise ValueError(f"papers.csv must contain a '{col}' column.")
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df = df[(df.title.str.strip() != "") & (df.abstract.str.strip() != "")]
    return df.reset_index(drop=True)

@st.cache_data
def load_embeddings():
    return np.load("embeddings.npy")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

try:
    papers, embeddings, model = load_papers(), load_embeddings(), load_model()
    if len(papers) != len(embeddings):
        st.error(f"Mismatch: {len(papers)} papers but {len(embeddings)} embeddings. Export both files in the same order.")
        st.stop()
except Exception as e:
    st.error(f"Could not load files: {e}")
    st.info("Keep app1.py, papers.csv, and embeddings.npy in the same folder.")
    st.stop()

has_topics = "topic" in papers.columns
if has_topics:
    papers["topic"] = pd.to_numeric(papers["topic"], errors="coerce").fillna(-1).astype(int)

@st.cache_data
def build_topic_labels(titles, abstracts, topic_ids):
    df = pd.DataFrame({"title": list(titles), "abstract": list(abstracts), "topic": list(topic_ids)})
    labels = {-1: "Outliers / Unassigned"}
    for tid in sorted(t for t in df.topic.unique() if t != -1):
        subset = df[df.topic == tid]
        docs = (subset.title + " " + subset.abstract.str.slice(0, 900)).tolist()
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1200, max_df=0.95)
            mat = vec.fit_transform(docs)
            terms = np.array(vec.get_feature_names_out())
            means = np.asarray(mat.mean(axis=0)).ravel()
            words = terms[np.argsort(means)[::-1][:3]]
            labels[int(tid)] = " · ".join(w.title() for w in words)
        except Exception:
            labels[int(tid)] = str(subset.iloc[0].title)[:45]
    return labels

topic_labels = build_topic_labels(tuple(papers.title), tuple(papers.abstract), tuple(papers.topic)) if has_topics else {}

def topic_label(tid):
    try:
        tid = int(tid)
        return topic_labels.get(tid, f"Research Theme {tid + 1}")
    except Exception:
        return "Unknown topic"

@st.cache_data
def normalized_vectors(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

norm_embeddings = normalized_vectors(embeddings)

def search_papers(query, top_k, threshold, topic_choice):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = norm_embeddings @ q
    results = papers.copy()
    results["similarity_score"] = scores
    results = results[results.similarity_score >= threshold]
    if has_topics and topic_choice != "All topics":
        tid = -1 if topic_choice == "Outliers / Unassigned" else int(topic_choice.split(" | ", 1)[0].replace("Topic ", ""))
        results = results[results.topic == tid]
    results = results.sort_values("similarity_score", ascending=False).head(top_k).copy()
    if has_topics:
        results["topic_name"] = results.topic.map(topic_label)
    results["similarity_score"] = results.similarity_score.round(3)
    return results.reset_index(drop=True)

SUGGESTIONS = [
    "Deep learning for image classification", "Reinforcement learning and robotics",
    "Large language models and NLP", "Graph neural networks",
    "AI for healthcare", "Clustering and unsupervised learning",
    "Computer vision and object detection", "Recommendation systems"
]

st.sidebar.markdown("## 🧭 Research AI")
page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🔎 Smart Search", "🗂️ Paper Explorer", "📊 Topic Analytics"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Suggested ideas")
for item in SUGGESTIONS[:4]:
    st.sidebar.caption("• " + item)
st.sidebar.caption(f"📚 {len(papers):,} papers loaded")

st.markdown("""
<div class="hero"><h1>📚 Research Paper AI</h1>
<p>Discover papers by meaning, explore research themes, and find your next idea.</p></div>
""", unsafe_allow_html=True)

if page == "🏠 Dashboard":
    st.subheader("✨ Your research workspace")
    st.write("Explore your collection with AI-powered semantic search.")
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Research papers", f"{len(papers):,}")
    c2.metric("🧠 Research themes", len(topic_labels) - 1 if has_topics else "N/A")
    c3.metric("🔍 Search engine", "Ready")
    st.markdown("### 🚀 Start with a suggested search")
    for start in range(0, len(SUGGESTIONS), 4):
        cols = st.columns(4)
        for col, suggestion in zip(cols, SUGGESTIONS[start:start + 4]):
            with col:
                if st.button(suggestion, key="dash_" + suggestion, use_container_width=True):
                    st.session_state["search_query"] = suggestion
                    st.session_state["requested_page"] = "🔎 Smart Search"
                    st.rerun()
    st.markdown("### 🧾 Sample papers")
    for _, row in papers.head(5).iterrows():
        with st.expander(row.title):
            st.write(row.abstract[:1200] + ("..." if len(row.abstract) > 1200 else ""))
    if has_topics:
        counts = papers[papers.topic != -1].topic.value_counts().head(8).rename_axis("topic").reset_index(name="papers")
        counts["Theme"] = counts.topic.map(topic_label)
        fig = px.bar(counts.sort_values("papers"), x="papers", y="Theme", orientation="h", color="papers", color_continuous_scale="Plasma", title="Most represented research themes")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔎 Smart Search":
    st.subheader("🔎 Semantic Research Search")
    st.write("Search with a question, concept, or research idea. Results are ranked by meaning, not exact keyword matches.")
    st.markdown("#### 💡 Suggested searches")
    for start in range(0, len(SUGGESTIONS), 4):
        cols = st.columns(4)
        for col, suggestion in zip(cols, SUGGESTIONS[start:start + 4]):
            with col:
                if st.button(suggestion, key="suggest_" + suggestion, use_container_width=True):
                    st.session_state["search_query"] = suggestion
    with st.form("search_form"):
        query = st.text_input("What do you want to research?", key="search_query", placeholder="e.g. How can neural networks detect objects in images?")
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.select_slider("Number of results", options=[3, 5, 10, 15, 20], value=5)
        with c2:
            threshold = st.slider("Minimum similarity score", 0.0, 0.9, 0.20, 0.05, help="Scores are similarity measures, not probabilities.")
        options = ["All topics"]
        if has_topics:
            options += [f"Topic {tid} | {topic_label(tid)}" for tid in sorted(papers.topic.unique()) if tid != -1]
            if -1 in papers.topic.values:
                options.append("Outliers / Unassigned")
        topic_choice = st.selectbox("Research theme filter", options)
        submitted = st.form_submit_button("✨ Find similar papers", type="primary", use_container_width=True)
    if submitted:
        if not query.strip():
            st.warning("Type a research idea or choose a suggestion.")
        else:
            with st.spinner("Ranking related papers..."):
                st.session_state["last_results"] = search_papers(query.strip(), top_k, threshold, topic_choice)
                st.session_state["last_query"] = query.strip()
    if "last_results" in st.session_state:
        results = st.session_state["last_results"]
        st.markdown(f"### 🎯 Results for: {st.session_state.get('last_query', '')}")
        st.caption(f"{len(results)} matching papers shown.")
        if results.empty:
            st.info("No papers matched. Lower the similarity threshold or choose All topics.")
        else:
            st.download_button("⬇️ Download results as CSV", results.to_csv(index=False).encode("utf-8"), "research_search_results.csv", "text/csv")
            for i, row in results.iterrows():
                with st.container(border=True):
                    st.markdown(f"#### {i+1}. {row.title}")
                    c1, c2 = st.columns([1, 3])
                    c1.metric("Similarity", f"{row.similarity_score:.3f}")
                    if has_topics:
                        c2.markdown(f"**🎨 Theme:** {row.get('topic_name', topic_label(row.get('topic', -1)))}")
                    st.progress(float(np.clip(row.similarity_score, 0, 1)))
                    with st.expander("📖 Read abstract"):
                        st.write(row.abstract)
                    if has_topics:
                        st.caption(f"BERTopic topic ID: {row.topic}")

elif page == "🗂️ Paper Explorer":
    st.subheader("🗂️ Paper Explorer")
    search_text = st.text_input("Search titles and abstracts", placeholder="e.g. neural networks")
    filtered = papers.copy()
    if search_text.strip():
        mask = filtered.title.str.contains(search_text, case=False, na=False) | filtered.abstract.str.contains(search_text, case=False, na=False)
        filtered = filtered[mask]
    if has_topics:
        options = ["All topics"] + [f"Topic {tid} | {topic_label(tid)}" for tid in sorted(filtered.topic.unique()) if tid != -1]
        if -1 in filtered.topic.values:
            options.append("Outliers / Unassigned")
        chosen = st.selectbox("Filter by research theme", options)
        if chosen == "Outliers / Unassigned":
            filtered = filtered[filtered.topic == -1]
        elif chosen != "All topics":
            tid = int(chosen.split(" | ", 1)[0].replace("Topic ", ""))
            filtered = filtered[filtered.topic == tid]
    st.caption(f"{len(filtered):,} papers match your filters.")
    view = filtered.copy()
    if has_topics:
        view["topic_name"] = view.topic.map(topic_label)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered papers", view.to_csv(index=False).encode("utf-8"), "filtered_papers.csv", "text/csv")

elif page == "📊 Topic Analytics":
    st.subheader("📊 Research Theme Analytics")
    if not has_topics:
        st.warning("No topic column found in papers.csv.")
    else:
        counts = papers.topic.value_counts().rename_axis("topic").reset_index(name="papers")
        include_outliers = st.toggle("Include outliers / unassigned papers", value=False)
        if not include_outliers:
            counts = counts[counts.topic != -1]
        counts["Theme"] = counts.topic.map(topic_label)
        fig = px.bar(counts.sort_values("papers", ascending=False), x="Theme", y="papers", color="papers", color_continuous_scale="Turbo", hover_data=["topic"], title="Papers in each research theme")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)
        opts = [f"Topic {tid} | {topic_label(tid)}" for tid in sorted(papers.topic.unique()) if tid != -1]
        if -1 in papers.topic.values:
            opts.append("Outliers / Unassigned")
        selected = st.selectbox("Explore a theme", opts)
        tid = -1 if selected == "Outliers / Unassigned" else int(selected.split(" | ", 1)[0].replace("Topic ", ""))
        subset = papers[papers.topic == tid]
        st.info(f"**{topic_label(tid)}** — {len(subset)} papers")
        for _, row in subset.head(10).iterrows():
            with st.expander(row.title):
                st.write(row.abstract)

st.markdown("---")
st.caption("Research Paper AI • Streamlit • Sentence Transformers • BERTopic")
