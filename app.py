"""
STEP 6 — Streamlit Dashboard
================================
Full interactive dashboard for Instagram Comment Sentiment Analysis.

Run:
    streamlit run dashboard/app.py

Sections:
  1. INPUT    — upload CSV or paste Reel URL (triggers Apify)
  2. ANALYSIS — pie chart, bar chart, word clouds
  3. INSIGHTS — cluster cards, AI summary, download report
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
# BACKEND API KEY  (never shown to the user)
# ─────────────────────────────────────────────
# Priority 1: Streamlit Cloud → add to .streamlit/secrets.toml as:
#   APIFY_TOKEN = "apify_xxxx..."
# Priority 2: local dev → set environment variable APIFY_TOKEN
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

# 1. Call the loader to get the key string
groq_key_string = _load_groq_token()

# 2. Check if the key exists before starting the engine
if groq_key_string:
    # This 'client' is what actually sends your data to Groq
    client = Groq(api_key=groq_key_string)
else:
    st.error("Missing Groq API Key! Please add it to your .env file.")
# ── Add parent dir to path so we can import our pipeline modules ──
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Sentiment Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
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
    /* Metric cards */
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
    .metric-card.positive { border-color: #00c851; }
    .metric-card.neutral  { border-color: #ffbb33; }
    .metric-card.negative { border-color: #ff4444; }
    .metric-number {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    /* Cluster cards */
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
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #444;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #e0e0e0;
    }
    /* Step indicator */
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
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

STOP_WORDS = set([
    "the","a","an","is","it","this","that","and","or","to","of","in",
    "for","on","with","so","just","im","ive","i","me","my","you","your",
    "we","our","they","them","but","not","no","do","dont","be","was",
    "are","have","had","has","will","can","got","get","like","really",
    "very","much","more","too","been","than","what","its","also",
])


def clean_text_basic(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_and_normalize_roman_urdu(text: str) -> tuple:
    """
    Detect Roman Urdu and normalize common patterns.
    Returns (normalized_text, is_roman_urdu)
    """
    text_lower = text.lower()
    
    # Common Roman Urdu indicators
    urdu_indicators = [
        'hai', 'hain', 'tha', 'thi', 'thay', 'ko', 'ne', 'se', 'ka', 'ki', 'ke',
        'acha', 'achi', 'achay', 'bohat', 'bohot', 'yar', 'yaar', 'na', 'kya',
        'islamabad', 'lahore', 'karachi', 'multan', 'acha hai', 'bohat acha'
    ]
    
    # Count Urdu indicators
    urdu_score = sum(1 for word in urdu_indicators if word in text_lower)
    
    # If likely Roman Urdu
    if urdu_score >= 2 or any(city in text_lower for city in ['islamabad', 'lahore', 'karachi']):
        # Fix city names - remove "bad" substring misinterpretation
        text = re.sub(r'islamabad', 'Islamabad (capital city)', text, flags=re.IGNORECASE)
        text = re.sub(r'lahore', 'Lahore (city in Pakistan)', text, flags=re.IGNORECASE)
        text = re.sub(r'karachi', 'Karachi (city in Pakistan)', text, flags=re.IGNORECASE)
        
        # Add context hint for the model
        hint = " [Note: Roman Urdu comment. Words like 'acha'='good', 'bohat'='very', 'khoobsurat'='beautiful' indicate positive sentiment]"
        
        # Map common positive Roman Urdu words to English
        positive_map = {
            'acha': 'good', 'achi': 'good', 'achay': 'good',
            'bohat': 'very', 'bohot': 'very',
            'khoobsurat': 'beautiful', 'pyara': 'lovely', 'pyari': 'lovely',
            'zabardast': 'excellent', 'umda': 'great', 'behtareen': 'best',
            'mazedar': 'enjoyable', 'lajawab': 'unbeatable', 'wonderful': 'wonderful'
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
        "[\U00010000-\U0010ffff\U00002600-\U000027BF\U0001F300-\U0001F9FF]+",
        flags=re.UNICODE
    )
    return len(pattern.findall(str(text)))


@st.cache_data
def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe."""
    # Normalise comment column
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
    df = df[df["comment"].str.split().str.len() >= 1]  # allow single-word cultural terms e.g. Mashallah
    df["comment_clean"] = df["comment"].apply(clean_text_basic)
    df["word_count"]    = df["comment_clean"].str.split().str.len()
    df["emoji_count"]   = df["comment"].apply(count_emojis)

    # ── Language detection (best-effort, silent if langdetect not installed) ──
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


def groq_sentiment(groq_key: str, texts: list) -> pd.DataFrame:
    """
    Use Groq (llama-3.3-70b) to classify sentiment for each comment.
    Sends comments in batches of 20 to keep prompts short and fast.
    Returns a DataFrame with sentiment, confidence, score_pos, score_neu, score_neg.
    """
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
            "- IMPORTANT for Roman Urdu (Urdu written in English letters):\n"
            "  * 'acha', 'achi', 'achay' = GOOD (Positive)\n"
            "  * 'bohat', 'bohot' = VERY (amplifies sentiment)\n"
            "  * 'khoobsurat', 'pyara', 'pyari' = BEAUTIFUL/LOVELY (Positive)\n"
            "  * 'zabardast', 'umda', 'behtareen' = EXCELLENT/GREAT/BEST (Positive)\n"
            "- IMPORTANT: City names like 'Islamabad', 'Lahore', 'Karachi' are NEUTRAL.\n"
            "  Do NOT interpret the 'bad' substring in 'Islamabad' as negative.\n"
            "- Works for any language: English, Urdu (including Roman Urdu), Arabic, Hindi, French, etc.\n\n"
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
            # Strip markdown code fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)

            # Build a lookup by index
            lookup = {item["index"]: item for item in parsed}
            for i in range(len(batch)):
                item = lookup.get(i + 1, {})
                label = item.get("sentiment", "Neutral")
                if label not in ("Positive", "Neutral", "Negative"):
                    label = "Neutral"
                conf = float(item.get("confidence", 0.5))
                if conf == 0:
                    conf = 0.5  # treat 0 as uncertain, not zero confidence
                results.append({
                    "sentiment":  label,
                    "confidence": round(conf, 4),
                    "score_pos":  round(conf if label == "Positive" else 0.05, 4),
                    "score_neu":  round(conf if label == "Neutral"  else 0.05, 4),
                    "score_neg":  round(conf if label == "Negative" else 0.05, 4),
                })
        except Exception:
            # Fallback: neutral for whole batch on any error
            for _ in batch:
                results.append({
                    "sentiment": "Neutral", "confidence": 0.5,
                    "score_pos": 0.05, "score_neu": 0.9, "score_neg": 0.05,
                })

    return pd.DataFrame(results)


# Your existing cultural positive terms (preserved)
CULTURAL_POSITIVE_TERMS = {
    "mashallah", "masha'allah", "masha allah", "alhamdulillah", "subhanallah",
    "allahu akbar", "inshallah", "jazakallah", "barakallah", "tabarak allah",
    "ramadan mubarak", "eid mubarak", "mabrook", "mubarak",
    "ماشاء الله", "الحمد لله", "سبحان الله",
}

# NEW: Roman Urdu positive words
ROMAN_URDU_POSITIVE = {
    "acha", "achi", "achay", "bohat", "bohot", "khoobsurat", "pyara", "pyari",
    "zabardast", "umda", "behtareen", "mazedar", "lajawab", "wonderful",
    "acha hai", "bohat acha", "bohat khoob", "kia baat hai", "well done",
}


def patch_cultural_sentiment(df: pd.DataFrame, original_texts: list) -> pd.DataFrame:
    """Apply cultural positive terms and Roman Urdu fixes."""
    df = df.copy()
    for i, text in enumerate(original_texts):
        if i >= len(df):
            break
            
        text_lower = str(text).lower()
        
        # Check existing cultural terms
        has_cultural_term = any(term.lower() in text_lower for term in CULTURAL_POSITIVE_TERMS)
        
        # NEW: Check Roman Urdu positive words
        has_urdu_positive = any(word in text_lower for word in ROMAN_URDU_POSITIVE)
        
        # Check for city names that might cause false negatives
        has_islamabad = "islamabad" in text_lower
        has_lahore = "lahore" in text_lower
        has_karachi = "karachi" in text_lower
        has_city_name = has_islamabad or has_lahore or has_karachi
        
        # Check for explicit negative words near city names
        negative_indicators = ["bad", "worst", "ugly", "hate", "dislike", "terrible", "awful"]
        has_nearby_negative = any(neg in text_lower for neg in negative_indicators)
        
        # LOGIC: Prioritize cultural and Roman Urdu positives
        if (has_cultural_term or has_urdu_positive) and not has_nearby_negative:
            df.at[i, "sentiment"] = "Positive"
            df.at[i, "confidence"] = 0.94
            df.at[i, "score_pos"] = 0.94
            df.at[i, "score_neu"] = 0.03
            df.at[i, "score_neg"] = 0.03
        elif has_city_name and not has_nearby_negative:
            # City name alone or with positive words should NOT be negative
            current_sentiment = df.at[i, "sentiment"]
            if current_sentiment == "Negative":
                # Check if there are any positive indicators
                positive_indicators = ["good", "nice", "love", "beautiful", "great"] + list(ROMAN_URDU_POSITIVE)
                has_any_positive = any(word in text_lower for word in positive_indicators)
                if has_any_positive or len(text_lower.split()) <= 3:  # Short comment mentioning city
                    df.at[i, "sentiment"] = "Neutral"
                    df.at[i, "confidence"] = min(df.at[i, "confidence"] + 0.25, 0.90)
            
    return df


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
            "sentiment":  label,
            "confidence": round(abs(pol), 4),
            "score_pos":  round(max(pol, 0), 4),
            "score_neu":  round(1 - abs(pol), 4),
            "score_neg":  round(max(-pol, 0), 4),
        })
    return pd.DataFrame(results)


@st.cache_data
def run_sentiment(_df: pd.DataFrame) -> pd.DataFrame:
    groq_key = _load_groq_token()
    original_texts = _df["comment"].astype(str).tolist()
    
    # NEW: Process Roman Urdu before sentiment analysis
    processed_texts = []
    is_urdu_flags = []
    for text in original_texts:
        processed, is_urdu = detect_and_normalize_roman_urdu(text)
        processed_texts.append(processed)
        is_urdu_flags.append(is_urdu)

    if groq_key:
        with st.spinner("🤖 Running multilingual sentiment via Groq (llama-3.3-70b)..."):
            sent_df = groq_sentiment(groq_key, processed_texts)
        
        # Apply cultural and Roman Urdu patches
        sent_df = patch_cultural_sentiment(sent_df, original_texts)
        
        # Additional boost for Roman Urdu comments that are likely positive
        for i, is_urdu in enumerate(is_urdu_flags):
            if is_urdu and sent_df.at[i, "sentiment"] != "Positive":
                text_lower = original_texts[i].lower()
                positive_indicators = list(ROMAN_URDU_POSITIVE) + ["good", "nice", "love", "beautiful", "great"]
                if any(word in text_lower for word in positive_indicators):
                    sent_df.at[i, "sentiment"] = "Positive"
                    sent_df.at[i, "confidence"] = 0.89
                    sent_df.at[i, "score_pos"] = 0.89
                    sent_df.at[i, "score_neu"] = 0.06
                    sent_df.at[i, "score_neg"] = 0.05
                    
    else:
        st.info("💡 GROQ_API_KEY not set — using TextBlob as fallback.")
        try:
            sent_df = textblob_sentiment(processed_texts)
        except ImportError:
            import random
            random.seed(42)
            results = []
            for _ in processed_texts:
                label = random.choices(
                    ["Positive","Neutral","Negative"], weights=[0.6,0.25,0.15]
                )[0]
                results.append({
                    "sentiment":  label,
                    "confidence": round(random.uniform(0.6, 0.95), 4),
                    "score_pos":  0.0, "score_neu": 0.0, "score_neg": 0.0,
                })
            sent_df = pd.DataFrame(results)

    return pd.concat([_df.reset_index(drop=True), sent_df], axis=1)


@st.cache_data
def run_clustering(_df: pd.DataFrame, n_clusters: int) -> tuple:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_cols = ["score_pos","score_neu","score_neg","confidence"]
    for col in ["word_count","emoji_count"]:
        if col in _df.columns:
            feature_cols.append(col)

    for col in feature_cols:
        if col not in _df.columns:
            _df[col] = 0

    X      = _df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    _df    = _df.copy()
    _df["cluster_id"] = km.fit_predict(X_sc)

    text_col = "comment_clean" if "comment_clean" in _df.columns else "comment"
    profiles = []

    for cid in sorted(_df["cluster_id"].unique()):
        cdf      = _df[_df["cluster_id"] == cid]
        texts    = cdf[text_col].astype(str).tolist()

        # TF-IDF keywords
        try:
            tfidf    = TfidfVectorizer(max_features=100, stop_words=list(STOP_WORDS),
                                       ngram_range=(1,2), min_df=1)
            mat      = tfidf.fit_transform(texts)
            scores   = np.asarray(mat.sum(axis=0)).flatten()
            vocab    = tfidf.get_feature_names_out()
            top_idx  = scores.argsort()[::-1][:8]
            keywords = [vocab[i] for i in top_idx]
        except Exception:
            keywords = top_words(texts, 8)

        # Auto-label
        counts = cdf["sentiment"].value_counts(normalize=True)
        pos = counts.get("Positive", 0)
        neg = counts.get("Negative", 0)
        neu = counts.get("Neutral",  0)
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
            "cluster_id":    int(cid),
            "label":         label,
            "size":          len(cdf),
            "pct_of_total":  round(len(cdf)/len(_df)*100, 1),
            "top_keywords":  keywords,
            "sentiment_dist": cdf["sentiment"].value_counts().to_dict(),
        })

    return _df, profiles


def _build_summary_context(df: pd.DataFrame, profiles: list) -> str:
    """Build a structured data context string to feed into the Groq prompt."""
    counts  = df["sentiment"].value_counts()
    total   = len(df)
    pos_pct = round(counts.get("Positive", 0) / total * 100, 1)
    neg_pct = round(counts.get("Negative", 0) / total * 100, 1)
    neu_pct = round(counts.get("Neutral",  0) / total * 100, 1)
    avg_words = round(df["comment_clean"].str.split().str.len().mean(), 1)
    pos_words = top_words(df[df["sentiment"] == "Positive"]["comment_clean"].tolist(), 6)
    neg_words = top_words(df[df["sentiment"] == "Negative"]["comment_clean"].tolist(), 6)

    lines = [
        f"Total comments analysed: {total}",
        f"Sentiment split — Positive: {pos_pct}%, Neutral: {neu_pct}%, Negative: {neg_pct}%",
        f"Average words per comment: {avg_words}",
        f"Top positive keywords: {', '.join(pos_words) if pos_words else 'N/A'}",
        f"Top negative keywords: {', '.join(neg_words) if neg_words else 'N/A'}",
        "",
        "Audience segments:",
    ]
    for p in profiles:
        lines.append(
            f"  - {p['label']}: {p['size']} comments ({p['pct_of_total']}%), "
            f"keywords: {', '.join(p['top_keywords'][:5])}, "
            f"sentiment: {p['sentiment_dist']}"
        )
    return "\n".join(lines)


def generate_summary(df: pd.DataFrame, profiles: list) -> str:
    """
    Generate an AI-powered audience insight summary using Groq (llama-3.3-70b-versatile).
    Falls back to a template-based summary if the GROQ_API_KEY is not configured.
    """
    groq_key = _load_groq_token()

    # ── Groq AI path ──────────────────────────────────────────────────────────
    if groq_key:
        try:
            import requests as _req
            context = _build_summary_context(df, profiles)
            prompt = (
                "You are a social-media analyst. Based on the Instagram comment "
                "analysis data below, write a concise 4–6 sentence audience insight "
                "summary. Highlight the overall sentiment tone, what positive viewers "
                "appreciate, what critics mention, and which audience segment dominates. "
                "Use plain prose — no bullet points, no headers.\n\n"
                f"{context}"
            )
            resp = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.6,
                },
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.warning(f"⚠️ Groq API error — falling back to template summary. ({e})")

    # ── Template fallback (no API key or Groq call failed) ───────────────────
    counts  = df["sentiment"].value_counts()
    total   = len(df)
    pos_pct = round(counts.get("Positive", 0) / total * 100, 1)
    neg_pct = round(counts.get("Negative", 0) / total * 100, 1)
    neu_pct = round(counts.get("Neutral",  0) / total * 100, 1)
    tone = ("overwhelmingly positive" if pos_pct >= 70 else
            "mostly positive"          if pos_pct >= 55 else
            "notably critical"         if neg_pct >= 40 else
            "largely neutral"          if neu_pct >= 50 else "mixed")
    pos_words = top_words(df[df["sentiment"] == "Positive"]["comment_clean"].tolist(), 5)
    neg_words = top_words(df[df["sentiment"] == "Negative"]["comment_clean"].tolist(), 5)
    avg_words = df["comment_clean"].str.split().str.len().mean()
    top_cluster = max(profiles, key=lambda c: c["size"]) if profiles else None

    summary = (
        f"Analysis of **{total} comments** reveals a **{tone}** audience response. "
        f"**{pos_pct}%** Positive · **{neg_pct}%** Negative · **{neu_pct}%** Neutral. "
        f"Viewers average **{avg_words:.0f} words** per comment, indicating "
        f"{'high' if avg_words > 8 else 'brief'} engagement. "
    )
    if pos_words:
        summary += f"Positive comments revolve around: *{', '.join(pos_words)}*. "
    if neg_words:
        summary += f"Critics focus on: *{', '.join(neg_words)}*. "
    if top_cluster:
        summary += (
            f"The largest audience segment ({top_cluster['pct_of_total']}%) "
            f"is **{top_cluster['label']}**, discussing: "
            f"*{', '.join(top_cluster['top_keywords'][:4])}*."
        )
    return summary


def make_wordcloud_fig(words: list, color: str):
    """Generate a simple frequency bar chart as word cloud substitute."""
    import plotly.graph_objects as go
    if not words:
        return None
    sizes = list(range(len(words), 0, -1))
    fig = go.Figure(go.Bar(
        x=words, y=sizes,
        marker_color=color,
        text=words,
        textposition="auto",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


def build_pdf_report(df: pd.DataFrame, profiles: list, summary: str) -> bytes:
    """Build a simple CSV + summary as download bundle."""
    buf = io.StringIO()
    buf.write("INSTAGRAM SENTIMENT ANALYSIS REPORT\n")
    buf.write("="*50 + "\n\n")
    buf.write("SUMMARY\n")
    buf.write("-"*30 + "\n")
    # Strip markdown bold markers for plain text
    plain_summary = re.sub(r"\*\*(.+?)\*\*", r"\1", summary)
    plain_summary = re.sub(r"\*(.+?)\*", r"\1", plain_summary)
    buf.write(plain_summary + "\n\n")
    buf.write("SENTIMENT DISTRIBUTION\n")
    buf.write("-"*30 + "\n")
    counts = df["sentiment"].value_counts()
    total  = len(df)
    for label in ["Positive","Neutral","Negative"]:
        n   = counts.get(label, 0)
        pct = n/total*100
        buf.write(f"{label}: {n} ({pct:.1f}%)\n")
    buf.write("\nCLUSTER PROFILES\n")
    buf.write("-"*30 + "\n")
    for p in profiles:
        buf.write(f"\n{p['label']} ({p['size']} comments, {p['pct_of_total']}%)\n")
        buf.write(f"  Keywords: {', '.join(p['top_keywords'][:6])}\n")
        buf.write(f"  Sentiment: {p['sentiment_dist']}\n")
    return buf.getvalue().encode("utf-8")


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
    3. **Sentiment** — Groq LLaMA-3.3-70B (100+ languages)
    4. **Cluster** — KMeans + TF-IDF
    5. **Insights** — summary + report
    """)
    st.divider()
    st.markdown("**Model:** `groq/llama-3.3-70b-versatile` 🌍")
    st.caption("Supports English, Urdu (including Roman Urdu), Arabic, Hindi, French, Spanish")
