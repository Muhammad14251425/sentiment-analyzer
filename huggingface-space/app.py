"""AI Sentiment Analyzer - Gradio version for Hugging Face Spaces (ZeroGPU).

Analyzes the sentiment of a paragraph (Positive / Neutral / Negative) using a
pretrained Hugging Face Transformer model.
"""

import spaces  # Must be imported before torch on ZeroGPU Spaces.

import gradio as gr
import torch
from transformers import pipeline

APP_TITLE = "🧠 AI Sentiment Analyzer"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOJIS = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
# RoBERTa can read at most 512 tokens, so longer text is truncated.
MAX_TOKENS = 512

# Load the model once when the app starts. On ZeroGPU the model is placed on
# "cuda" here, and a real GPU is attached only while predict() runs.
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,  # Return the probability of every class, not only the best one.
    device="cuda" if torch.cuda.is_available() else "cpu",
)


@spaces.GPU(duration=5)  # Inference takes well under a second.
def predict(text: str) -> dict:
    """Run the model and return the probability of each sentiment."""
    predictions = classifier([text], truncation=True, max_length=MAX_TOKENS)[0]
    # The model's labels ("negative", "neutral", "positive") come from its config.
    return {prediction["label"].capitalize(): prediction["score"] for prediction in predictions}


def analyze_text(text: str) -> tuple[str, dict]:
    """Validate the input, then return the verdict and all class probabilities."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text before analyzing.")

    scores = predict(text.strip())
    sentiment = max(scores, key=scores.get)
    verdict = f"## {EMOJIS[sentiment]} {sentiment.upper()} — {scores[sentiment]:.2%} confidence"
    return verdict, scores


with gr.Blocks(title="AI Sentiment Analyzer") as demo:
    gr.Markdown(f"# {APP_TITLE}\nAnalyze the sentiment of any paragraph using Hugging Face Transformers (RoBERTa).")

    text_input = gr.Textbox(
        label="Enter a paragraph to analyze",
        lines=6,
        placeholder="Example: I absolutely loved this product. The quality was amazing!",
    )
    analyze_button = gr.Button("Analyze Sentiment", variant="primary")
    verdict_output = gr.Markdown()
    scores_output = gr.Label(label="Class probabilities", num_top_classes=3)

    gr.Examples(
        examples=[
            "I absolutely love this product. It is fantastic.",
            "The package was delivered this afternoon.",
            "This was terrible and I regret buying it.",
        ],
        inputs=text_input,
    )
    gr.Markdown(f"Model: `{MODEL_NAME}` · For educational purposes")

    analyze_button.click(analyze_text, inputs=text_input, outputs=[verdict_output, scores_output])

demo.launch()
