"""AI Sentiment Analyzer.

Analyzes the sentiment of a paragraph (Positive / Neutral / Negative) using a
pretrained Hugging Face Transformer model that runs locally.

Run with:  streamlit run app.py
"""

import logging

import streamlit as st
import torch
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ("Positive", "Neutral", "Negative")
EMOJIS = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
# RoBERTa can read at most 512 tokens, so longer text is truncated.
MAX_TOKENS = 512


@st.cache_resource(show_spinner="Loading Hugging Face sentiment model...")
def load_model():
    """Load the model once. Streamlit reuses it instead of reloading on every click."""
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,  # Return the probability of every class, not only the best one.
        device=0 if torch.cuda.is_available() else -1,  # GPU if available, else CPU.
    )


def analyze_text(text: str) -> dict:
    """Return the predicted sentiment, its confidence and the score of every class."""
    predictions = load_model()([text], truncation=True, max_length=MAX_TOKENS)[0]
    # The model's labels ("negative", "neutral", "positive") come from its config.
    scores = {prediction["label"].capitalize(): prediction["score"] for prediction in predictions}
    sentiment = max(scores, key=scores.get)
    return {"sentiment": sentiment, "confidence": scores[sentiment], "scores": scores}


def show_result(result: dict) -> None:
    """Display the sentiment in a colored box, then the three class probabilities."""
    sentiment = result["sentiment"]
    message = f"{EMOJIS[sentiment]} **{sentiment.upper()}** — {result['confidence']:.2%} confidence"
    if sentiment == "Positive":
        st.success(message)
    elif sentiment == "Neutral":
        st.warning(message)
    else:
        st.error(message)

    for column, label in zip(st.columns(3), LABELS):
        column.metric(f"{EMOJIS[label]} {label}", f"{result['scores'][label]:.2%}")


st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="🧠")
st.title("🧠 AI Sentiment Analyzer")
st.caption("Analyze the sentiment of any paragraph using Hugging Face Transformers (RoBERTa).")

try:
    load_model()
except Exception:
    logging.exception("Model loading failed")
    st.error("Unable to load the Hugging Face model. Check your internet connection and try again.")
    st.stop()

text = st.text_area(
    "Enter a paragraph to analyze",
    height=200,
    placeholder="Example: I absolutely loved this product. The quality was amazing!",
)
st.caption(f"Characters: {len(text)}")

if st.button("Analyze Sentiment", type="primary"):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                result = analyze_text(text.strip())
        except Exception:
            logging.exception("Inference failed")
            st.error("An error occurred while analyzing the text.")
        else:
            show_result(result)

st.divider()
st.caption(f"Model: {MODEL_NAME} · Runs locally after the first download · For educational purposes")
