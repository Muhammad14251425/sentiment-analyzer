---
title: AI Sentiment Analyzer
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Paragraph sentiment analysis with RoBERTa
---

# AI Sentiment Analyzer

Classifies a paragraph as **Positive**, **Neutral** or **Negative** and shows the confidence of each class, using the pretrained model [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) (RoBERTa) with Hugging Face Transformers and Gradio.

Predictions are not 100% accurate. The model can struggle with sarcasm, mixed sentiment and domain-specific language.
