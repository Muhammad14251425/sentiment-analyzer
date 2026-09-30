# AI Sentiment Analyzer

A simple web app that analyzes the sentiment of a paragraph and classifies it as **Positive**, **Neutral** or **Negative**. For each paragraph it shows the predicted sentiment, the confidence score and the probability of all three classes.

It uses the pretrained Hugging Face model [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa Transformer. The model runs locally, so no API key or paid service is needed.

## Technology Stack

- Python
- Hugging Face Transformers (RoBERTa)
- PyTorch
- Streamlit

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

```powershell
streamlit run app.py
```

The app opens at http://localhost:8501.

## How it works

```
Paragraph → Tokenizer → Token IDs → RoBERTa Transformer → Classification Layer → Positive / Neutral / Negative + confidence scores
```

The first run downloads the model (about 500 MB), so it takes a few minutes. After that the model loads from the local cache, and `@st.cache_resource` keeps it in memory while the app is running.

## Limitations

Predictions are not 100% accurate. The model can struggle with sarcasm, irony, mixed or ambiguous sentiment, and domain-specific language. It was trained mainly on English tweets. Paragraphs longer than 512 tokens are truncated.
