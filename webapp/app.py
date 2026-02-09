# =========================================
# STREAMLIT SENTIMENT ANALYSIS APP (CORRECT)
# - Keeps NLTK WordNet Lemmatization logic
# - Fixes WordNet loading by using local nltk_data/
# - Handles BOTH:
#   1) Pipeline model (vectorizer inside model)
#   2) Separate classifier + TF-IDF vectorizer
# =========================================

import os
import re
import streamlit as st
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")
st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review and predict its sentiment")


# -----------------------------
# Ensure NLTK uses project-local data folder
# -----------------------------
PROJECT_NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(PROJECT_NLTK_DIR, exist_ok=True)
if PROJECT_NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, PROJECT_NLTK_DIR)


@st.cache_resource
def setup_nltk():
    # Download into project folder (not user roaming folders)
    nltk.download("stopwords", download_dir=PROJECT_NLTK_DIR, quiet=True)
    nltk.download("wordnet", download_dir=PROJECT_NLTK_DIR, quiet=True)
    nltk.download("omw-1.4", download_dir=PROJECT_NLTK_DIR, quiet=True)

setup_nltk()


# -----------------------------
# Load model + optional vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts(model_path: str, vectorizer_path: str):
    model = joblib.load(model_path)

    vectorizer = None
    try:
        vectorizer = joblib.load(vectorizer_path)
    except Exception:
        vectorizer = None

    return model, vectorizer


MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)
except Exception as e:
    st.error(f"❌ Could not load model/vectorizer.\n\nError: {e}")
    st.stop()


# -----------------------------
# Text preprocessing (same as your training)
# -----------------------------
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# -----------------------------
# UI
# -----------------------------
review_text = st.text_area(
    "Enter your review here:",
    height=150,
    placeholder="Example: This product quality is amazing and worth the price!"
)

show_debug = st.checkbox("Show debug info", value=False)


# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review text.")
    else:
        clean_text = preprocess(review_text)

        # IMPORTANT LOGIC:
        # If model is Pipeline -> pass raw text
        # If model is classifier only -> vectorize then predict
        try:
            prediction = model.predict([clean_text])[0]
            used = "Pipeline (model includes vectorizer)"
        except Exception:
            if vectorizer is None:
                st.error(
                    "❌ Your model looks like it needs a vectorizer, but `tfidf_vectorizer.pkl` is not loaded.\n\n"
                    "Fix: Ensure `tfidf_vectorizer.pkl` exists in the same folder as app.py."
                )
                st.stop()

            X = vectorizer.transform([clean_text])
            prediction = model.predict(X)[0]
            used = "Classifier + separate TF-IDF vectorizer"

        if int(prediction) == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        if show_debug:
            st.write("Model type:", type(model))
            st.write("Vectorizer loaded:", vectorizer is not None)
            st.write("Prediction method used:", used)
            st.write("Cleaned text:", clean_text)


st.markdown("---")
st.markdown("**Model:** TF-IDF + Logistic Regression")
st.markdown("**Metric Used:** F1-score")
