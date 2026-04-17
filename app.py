"""
STEP 6 — Streamlit Dashboard
Full interactive dashboard for Instagram Comment Sentiment Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import re
import sys
import os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
from groq import Groq

# ─────────────────────────────────────────────
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# BACKEND API KEYS
# ─────────────────────────────────────────────
def _load_apify_token() -> str | None:
    try:
        return st.secrets["APIFY_TOKEN"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("APIFY_TOKEN")


def _load_groq_token() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GROQ_API_KEY")


groq_key_string = _load_groq_token()

if groq_key_string:
    client = Groq(api_key=groq_key_string)
else:
    st.warning("⚠️ Groq API Key not found. Using fallback sentiment analysis.")

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #833ab4, #fd1d1d, #fcb045);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .cluster-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #333;
        height: 100%;
    }
    .cluster-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .keyword-pill {
        display: inline-block;
        background: #333;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
        color: #ccc;
    }
    .summary-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #444;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e0e0e0;
    }
    .step-badge {
        background: #833ab4;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STOP_WORDS = set([
    "the", "a", "an", "is", "it", "this", "that", "and", "or", "to", "of", "in",
    "for", "on", "with", "so", "just", "im", "ive", "i", "me", "my", "you", "your",
    "we", "our", "they", "them", "but", "not", "no", "do", "dont", "be", "was",
    "are", "have", "had", "has", "will", "can", "got", "get", "like", "really",
    "very", "much", "more", "too", "been", "than", "what", "its", "also",
])

CULTURAL_POSITIVE_TERMS = {
    "mashallah", "masha'allah", "masha allah", "alhamdulillah", "subhanallah",
    "allahu akbar", "inshallah", "jazakallah", "barakallah", "tabarak allah",
    "ramadan mubarak", "eid mubarak", "mabrook", "mubarak",
    "ماشاء الله", "الحمد لله", "سبحان الله",
}

ROMAN_URDU_POSITIVE = {
    "acha", "achi", "achay", "bohat", "bohot", "khoobsurat", "pyara", "pyari",
    "zabardast", "umda", "behtareen", "mazedar", "lajawab", "wonderful",
    "acha hai", "bohat acha", "bohat khoob", "kia baat hai", "well done",
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def clean_text_basic(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_and_normalize_roman_urdu(text: str) -> tuple:
    """Detect Roman Urdu and normalize common patterns."""
    text_lower = text.lower()
    
    urdu_indicators = [
        'hai', 'hain', 'tha', 'thi', 'thay', 'ko', 'ne', 'se', 'ka', 'ki', 'ke',
        'acha', 'achi', 'achay', 'bohat', 'bohot', 'yar', 'yaar', 'na', 'kya',
        'islamabad', 'lahore', 'karachi', 'multan', 'acha hai', 'bohat acha'
    ]
    
    urdu_score = sum(1 for word in urdu_indicators if word in text_lower)
    
    if urdu_score >= 2 or any(city in text_lower for city in ['islamabad', 'lahore', 'karachi']):
        text = re.sub(r'islamabad', 'Islamabad (capital city)', text, flags=re.IGNORECASE)
        text = re.sub(r'lahore', 'Lahore (city in Pakistan)', text, flags=re.IGNORECASE)
        text = re.sub(r'karachi', 'Karachi (city in Pakistan)', text, flags=re.IGNORECASE)
        
        hint = " [Note: Roman Urdu comment. Words like 'acha'='good', 'bohat'='very' indicate positive sentiment]"
        
        positive_map = {
            'acha': 'good', 'achi': 'good', 'achay': 'good',
            'bohat': 'very', 'bohot': 'very',
            'khoobsurat': 'beautiful', 'pyara': 'lovely', 'pyari': 'lovely',
            'zabardast': 'excellent', 'umda': 'great', 'behtareen': 'best',
        }
        
        for urdu_word, english_word in positive_map.items():
            if urdu_word in text_lower:
                text = re.sub(rf'\b{urdu_word}\b', f'{english_word}', text, flags=re.IGNORECASE)
        
        return text + hint, True
    
    return text, False


def top_words(texts: list, n: int = 15) -> list:
    all_words = []
    for text in texts:
        words = re.sub(r"[^a-zA-Z\s]", "", str(text).lower()).split()
        all_words.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])
    return [word for word, _ in Counter(all_words).most_common(n)]


def count_emojis(text: str) -> int:
    pattern = re.compile(
        r"[\U00010000-\U0010ffff\U00002600-\U000027BF\U0001F300-\U0001F9FF]+",
        flags=re.UNICODE
    )
    return len(pattern.findall(str(text)))


def groq_sentiment(groq_key: str, texts: list) -> pd.DataFrame:
    import requests as _req
    results = []
    batch_size = 20

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))

        prompt = (
            "You are a multilingual sentiment classifier. "
            "Classify each comment below as exactly one of: Positive, Neutral, or Negative.\n"
            "Rules:\n"
            "- Reply ONLY with a JSON array, no explanation, no markdown.\n"
            "- Each item must have: index (int), sentiment (str), confidence (float 0-1).\n"
            "- Consider cultural expressions (e.g. MashaAllah, Alhamdulillah) as Positive.\n"
            "- IMPORTANT for Roman Urdu: 'acha'='good', 'bohat'='very', city names are NEUTRAL.\n"
            "- Do NOT interpret 'bad' in 'Islamabad' as negative.\n\n"
            f"Comments:\n{numbered}\n\n"
            "Reply format: [{\"index\":1,\"sentiment\":\"Positive\",\"confidence\":0.95}, ...]"
        )

        try:
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.0,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            lookup = {item["index"]: item for item in parsed}
            for i in range(len(batch)):
                item = lookup.get(i + 1, {})
                label = item.get("sentiment", "Neutral")
                if label not in ("Positive", "Neutral", "Negative"):
                    label = "Neutral"
                conf = float(item.get("confidence", 0.5))
                if conf == 0:
                    conf = 0.5
                results.append({
                    "sentiment": label,
                    "confidence": round(conf, 4),
                    "score_pos": round(conf if label == "Positive" else 0.05, 4),
                    "score_neu": round(conf if label == "Neutral" else 0.05, 4),
                    "score_neg": round(conf if label == "Negative" else 0.05, 4),
                })
        except Exception as e:
            for _ in batch:
                results.append({
                    "sentiment": "Neutral", "confidence": 0.5,
                    "score_pos": 0.05, "score_neu": 0.9, "score_neg": 0.05,
                })

    return pd.DataFrame(results)


def textblob_sentiment(texts: list) -> pd.DataFrame:
    from textblob import TextBlob
    results = []
    for text in texts:
        pol = TextBlob(str(text)).sentiment.polarity
        if pol > 0.05:
            label = "Positive"
        elif pol < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({
            "sentiment": label,
            "confidence": round(abs(pol), 4),
            "score_pos": round(max(pol, 0), 4),
            "score_neu": round(1 - abs(pol), 4),
            "score_neg": round(max(-pol, 0), 4),
        })
    return pd.DataFrame(results)


def patch_cultural_sentiment(df: pd.DataFrame, original_texts: list) -> pd.DataFrame:
    df = df.copy()
    for i, text in enumerate(original_texts):
        if i >= len(df):
            break
        text_lower = str(text).lower()
        
        has_cultural = any(term.lower() in text_lower for term in CULTURAL_POSITIVE_TERMS)
        has_urdu_pos = any(word in text_lower for word in ROMAN_URDU_POSITIVE)
        has_city = any(city in text_lower for city in ['islamabad', 'lahore', 'karachi'])
        has_negative = any(neg in text_lower for neg in ["bad", "worst", "ugly", "hate", "terrible"])
        
        if (has_cultural or has_urdu_pos) and not has_negative:
            df.at[i, "sentiment"] = "Positive"
            df.at[i, "confidence"] = 0.94
            df.at[i, "score_pos"] = 0.94
            df.at[i, "score_neu"] = 0.03
            df.at[i, "score_neg"] = 0.03
        elif has_city and not has_negative and df.at[i, "sentiment"] == "Negative":
            df.at[i, "sentiment"] = "Neutral"
            df.at[i, "confidence"] = min(df.at[i, "confidence"] + 0.25, 0.90)
    return df


# ─────────────────────────────────────────────
# MAIN PIPELINE FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data
def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["text", "comment", "body", "content"]:
        if col in df.columns:
            df = df.rename(columns={col: "comment"})
            break
    if "comment" not in df.columns:
        st.error("Could not find comment column. Expected: text, comment, body, or content.")
        st.stop()

    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    df = df.drop_duplicates(subset="comment")
    df["comment_clean"] = df["comment"].apply(clean_text_basic)
    df["word_count"] = df["comment_clean"].str.split().str.len()
    df["emoji_count"] = df["comment"].apply(count_emojis)

    try:
        from langdetect import detect, LangDetectException
        def safe_detect(text):
            try:
                return detect(str(text))
            except LangDetectException:
                return "unknown"
        df["language"] = df["comment"].apply(safe_detect)
    except ImportError:
        df["language"] = "unknown"

    return df.reset_index(drop=True)


@st.cache_data
def run_sentiment(_df: pd.DataFrame) -> pd.DataFrame:
    groq_key = _load_groq_token()
    original_texts = _df["comment"].astype(str).tolist()
    
    processed_texts = []
    for text in original_texts:
        processed, _ = detect_and_normalize_roman_urdu(text)
        processed_texts.append(processed)

    if groq_key:
        with st.spinner("🤖 Running sentiment analysis via Groq..."):
            sent_df = groq_sentiment(groq_key, processed_texts)
        sent_df = patch_cultural_sentiment(sent_df, original_texts)
    else:
        st.info("💡 Using TextBlob fallback sentiment analysis.")
        sent_df = textblob_sentiment(processed_texts)

    return pd.concat([_df.reset_index(drop=True), sent_df], axis=1)


@st.cache_data
def run_clustering(_df: pd.DataFrame, n_clusters: int) -> tuple:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_cols = ["score_pos", "score_neu", "score_neg", "confidence", "word_count", "emoji_count"]
    for col in feature_cols:
        if col not in _df.columns:
            _df[col] = 0

    X = _df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    _df = _df.copy()
    _df["cluster_id"] = km.fit_predict(X_sc)

    text_col = "comment_clean"
    profiles = []

    for cid in sorted(_df["cluster_id"].unique()):
        cdf = _df[_df["cluster_id"] == cid]
        texts = cdf[text_col].astype(str).tolist()

        try:
            tfidf = TfidfVectorizer(max_features=100, stop_words=list(STOP_WORDS), ngram_range=(1, 2), min_df=1)
            mat = tfidf.fit_transform(texts)
            scores = np.asarray(mat.sum(axis=0)).flatten()
            vocab = tfidf.get_feature_names_out()
            top_idx = scores.argsort()[::-1][:8]
            keywords = [vocab[i] for i in top_idx]
        except Exception:
            keywords = top_words(texts, 8)

        counts = cdf["sentiment"].value_counts(normalize=True)
        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral", 0)
        avg_emojis = cdf.get("emoji_count", pd.Series([0])).mean()

        if pos >= 0.65:
            label = "🔥 Highly Engaged Fans" if avg_emojis > 1 else "💬 Loyal Commenters"
        elif neg >= 0.50:
            label = "👎 Critical Audience"
        elif neg >= 0.30:
            label = "😐 Mixed / Divided Viewers"
        elif neu >= 0.60:
            label = "👀 Passive Viewers"
        else:
            label = "🤔 Curious Explorers"

        _df.loc[_df["cluster_id"] == cid, "cluster_label"] = label
        profiles.append({
            "cluster_id": int(cid),
            "label": label,
            "size": len(cdf),
            "pct_of_total": round(len(cdf) / len(_df) * 100, 1),
            "top_keywords": keywords,
            "sentiment_dist": cdf["sentiment"].value_counts().to_dict(),
        })

    return _df, profiles


def generate_summary(df: pd.DataFrame, profiles: list) -> str:
    counts = df["sentiment"].value_counts()
    total = len(df)
    pos_pct = round(counts.get("Positive", 0) / total * 100, 1)
    neg_pct = round(counts.get("Negative", 0) / total * 100, 1)
    neu_pct = round(counts.get("Neutral", 0) / total * 100, 1)
    
    tone = ("overwhelmingly positive" if pos_pct >= 70 else
            "mostly positive" if pos_pct >= 55 else
            "notably critical" if neg_pct >= 40 else
            "largely neutral" if neu_pct >= 50 else "mixed")
    
    pos_words = top_words(df[df["sentiment"] == "Positive"]["comment_clean"].tolist(), 5)
    neg_words = top_words(df[df["sentiment"] == "Negative"]["comment_clean"].tolist(), 5)
    top_cluster = max(profiles, key=lambda c: c["size"]) if profiles else None

    summary = f"Analysis of **{total} comments** reveals a **{tone}** audience response. **{pos_pct}%** Positive · **{neg_pct}%** Negative · **{neu_pct}%** Neutral. "
    if pos_words:
        summary += f"Positive comments revolve around: *{', '.join(pos_words)}*. "
    if neg_words:
        summary += f"Critics focus on: *{', '.join(neg_words)}*. "
    if top_cluster:
        summary += f"The largest segment ({top_cluster['pct_of_total']}%) is **{top_cluster['label']}**."
    return summary


# ─────────────────────────────────────────────
# UI RENDER FUNCTIONS
# ─────────────────────────────────────────────
def render_analysis(df_raw: pd.DataFrame, prefix: str, source_label: str = ""):
    import plotly.express as px

    df_clean = run_cleaning(df_raw)
    df_sent = run_sentiment(df_clean)
    df_final, profiles = run_clustering(df_sent, n_clusters)

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    st.caption(f"📌 Source: {source_label}")

    counts = df_final["sentiment"].value_counts()
    total = len(df_final)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments", f"{total:,}")
    with col2:
        st.metric("✅ Positive", f"{counts.get('Positive', 0)} ({counts.get('Positive', 0)/total*100:.0f}%)")
    with col3:
        st.metric("➖ Neutral", f"{counts.get('Neutral', 0)} ({counts.get('Neutral', 0)/total*100:.0f}%)")
    with col4:
        st.metric("❌ Negative", f"{counts.get('Negative', 0)} ({counts.get('Negative', 0)/total*100:.0f}%)")

    # Sentiment Charts
    st.markdown("#### Sentiment Breakdown")
    pie_data = df_final["sentiment"].value_counts().reset_index()
    pie_data.columns = ["Sentiment", "Count"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(pie_data, names="Sentiment", values="Count", color="Sentiment",
                         color_discrete_map={"Positive": "#00c851", "Neutral": "#ffbb33", "Negative": "#ff4444"},
                         hole=0.4)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True, key=f"{prefix}_pie")

    with col_b:
        fig_bar = px.bar(pie_data, x="Sentiment", y="Count", color="Sentiment",
                         color_discrete_map={"Positive": "#00c851", "Neutral": "#ffbb33", "Negative": "#ff4444"},
                         text="Count")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{prefix}_bar")

    # Audience Segments
    st.markdown("#### 👥 Audience Segments")
    seg_cols = st.columns(min(len(profiles), 4))
    for i, profile in enumerate(profiles):
        with seg_cols[i % len(seg_cols)]:
            st.markdown(f"""
            <div class="cluster-card">
                <div class="cluster-title">{profile['label']}</div>
                <div style="font-size:1.8rem;font-weight:800;margin:0.3rem 0">{profile['size']}</div>
                <div style="color:#aaa;font-size:0.8rem;margin-bottom:0.8rem">comments · {profile['pct_of_total']}% of total</div>
                <div style="margin-bottom:0.5rem;font-size:0.8rem;color:#888">Top Keywords:</div>
                <div>{''.join(f'<span class="keyword-pill">{k}</span>' for k in profile['top_keywords'][:6])}</div>
            </div>
            """, unsafe_allow_html=True)

    # AI Summary
    st.markdown("#### 📝 AI-Generated Summary")
    summary = generate_summary(df_final, profiles)
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    # Download
    st.markdown("#### ⬇️ Download Results")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Download CSV", data=df_final.to_csv(index=False).encode("utf-8"),
                          file_name="sentiment_results.csv", mime="text/csv", key=f"{prefix}_csv")
    with col_dl2:
        st.download_button("📄 Download Report", data=summary.encode("utf-8"),
                          file_name="sentiment_report.txt", key=f"{prefix}_txt")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    n_clusters = st.slider("Audience Clusters (k)", 2, 6, 4)
    st.divider()
    st.markdown("### 📖 Pipeline Steps")
    st.markdown("""
    1. **Collect** — CSV, URL or manual input
    2. **Clean** — deduplicate, filter, detect language
    3. **Sentiment** — Groq LLaMA-3.3-70B
    4. **Cluster** — KMeans + TF-IDF
    5. **Insights** — summary + report
    """)
    st.divider()
    st.markdown("**Model:** `groq/llama-3.3-70b-versatile` 🌍")
    st.caption("Supports English, Urdu (including Roman Urdu), Arabic, Hindi")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Instagram Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Audience intelligence from Instagram Reel comments</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📂 Data Input")

# Create tabs
tab_upload, tab_url, tab_manual = st.tabs(["📁 Upload CSV", "🔗 Paste Reel URL", "✏️ Type / Paste Comments"])

# Tab 1: CSV Upload with Sample Dataset
with tab_upload:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded = st.file_uploader("Upload your CSV", type=["csv"], key="csv_uploader")
    
    with col2:
        st.markdown("##### — or —")
        if st.button("📊 Load Sample Dataset", key="btn_sample", use_container_width=True):
            sample_data = pd.DataFrame({
                "comment": [
                    "Love this content! 🔥 Keep it up!",
                    "MashaAllah beautiful video ❤️",
                    "Islamabad bohat acha hai",
                    "Not impressed, boring content",
                    "This is amazing! Learned so much",
                    "Alhamdulillah very inspiring ✨",
                    "Terrible quality, waste of time",
                    "Best reel ever! Shared with friends",
                    "ماشاءاللہ بہت خوب 😊",
                    "Don't like this at all",
                    "So helpful, thank you! 🙏",
                    "Absolutely brilliant! 👏",
                    "SubhanAllah what a beautiful message",
                    "Mashallah very informative"
                ]
            })
            st.session_state["csv_ready"] = sample_data
            st.success("✅ Sample dataset loaded! Click 'Analyse' below.")
            st.rerun()
    
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_csv)} rows")
        if st.button("▶ Analyse CSV", key="btn_csv"):
            st.session_state["csv_ready"] = df_csv
            st.rerun()
    
    if "csv_ready" in st.session_state:
        render_analysis(st.session_state["csv_ready"], prefix="csv", source_label="CSV / Sample")

# Tab 2: Reel URL
with tab_url:
    reel_url = st.text_input("Instagram Reel URL", placeholder="https://www.instagram.com/reel/...", key="reel_url")
    
    apify_token = _load_apify_token()
    if not apify_token:
        st.warning("⚠️ Apify API token not configured. URL scraping will not work.")
    
    if st.button("🚀 Scrape & Analyse", key="btn_scrape") and reel_url and apify_token:
        import requests, time
        try:
            run_resp = requests.post(
                "https://api.apify.com/v2/acts/apify~instagram-comment-scraper/runs",
                headers={"Authorization": f"Bearer {apify_token}"},
                json={"directUrls": [reel_url], "resultsLimit": 200},
                timeout=30,
            )
            if run_resp.status_code == 201:
                run_id = run_resp.json()["data"]["id"]
                dataset_id = run_resp.json()["data"]["defaultDatasetId"]
                time.sleep(10)
                items = requests.get(
                    f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=json&clean=true",
                    headers={"Authorization": f"Bearer {apify_token}"}, timeout=30,
                ).json()
                if items:
                    st.session_state["url_ready"] = pd.DataFrame(items)
                    st.success(f"✅ Scraped {len(items)} comments!")
                    st.rerun()
                else:
                    st.error("No comments found.")
            else:
                st.error(f"Apify error: {run_resp.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if "url_ready" in st.session_state:
        render_analysis(st.session_state["url_ready"], prefix="url", source_label="Live Scraped Reel")

# Tab 3: Manual Comments
with tab_manual:
    user_input = st.text_area("Enter comments (one per line)", height=200, key="manual_input",
                              placeholder="Love this!\nNot impressed\nMashaAllah great video")
    
    if st.button("🔍 Analyse", key="btn_manual"):
        lines = [l.strip() for l in user_input.strip().splitlines() if l.strip()]
        if lines:
            st.session_state["manual_ready"] = pd.DataFrame({"comment": lines})
            st.success(f"✅ {len(lines)} comments loaded!")
            st.rerun()
    
    if "manual_ready" in st.session_state:
        render_analysis(st.session_state["manual_ready"], prefix="manual", source_label="Manually Entered")

# Clear results button
if any(k in st.session_state for k in ["csv_ready", "url_ready", "manual_ready"]):
    with st.expander("🔄 Clear Results"):
        if st.button("🗑️ Clear all data"):
            for k in ["csv_ready", "url_ready", "manual_ready"]:
                st.session_state.pop(k, None)
            st.rerun()
else:
    st.info("👆 Choose a tab above to get started. Try the 'Load Sample Dataset' button in the CSV tab!")
