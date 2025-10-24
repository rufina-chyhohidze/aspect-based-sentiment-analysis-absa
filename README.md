# Unified Aspect-Based Sentiment Analysis (ABSA) API

A **FastAPI-based service** that unifies multiple approaches to **Aspect-Based Sentiment Analysis (ABSA)** — combining **rule-based**, **opinion-first**, **machine learning (Transformer)**, and **LLM** models into one consistent API.

It detects *aspects* (e.g. “keyboard”, “battery life”) within a text and assigns *sentiments* (positive / negative / neutral) with confidence scores and position spans.

---

##  Features

*  **LexiconABSA** — rule-based method using spaCy + VADER
*  **LexiconOpinionFirstABSA** — opinion-first approach for improved contextual sentiment detection
*  **ML_ABSA** — Transformer-based ABSA (using **PyABSA**, automatically downloads pretrained models)
*  **LLMABSA** — Local LLM-based sentiment reasoning using **Ollama**

All models share a **common interface**, allowing direct comparison and unified API responses.

---

##  Project Structure

Python 3.11

```
src/
│
├── api/
│   └── api_exposed.py             # FastAPI main app (entry point)
│
├── lexicon_absa.py                # Rule-based ABSA
├── lexicon_absa2_opinion_to_sentiment.py  # Opinion-first ABSA
├── ML_ABSA.py                     # Transformer-based ABSA (PyABSA)
├── LLMABSA.py                     # LLM-based ABSA (Ollama)
│
└── requirements.txt               # Dependencies
```

---

##  Installation

### Create and Activate a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Initial Setup (Required Downloads)

Before running the API for the first time, download the necessary NLP models and resources.

### For LexiconABSA and LexiconOpinionFirstABSA

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon stopwords punkt
```

### For ML_ABSA (Transformer-based)

PyABSA will automatically download pretrained models on first use.
If you want to prefetch them manually:

```bash
python -m pyabsa download checkpoint
```

### For LLMABSA (Local Ollama)

1. Install **[Ollama](https://ollama.com/download)**
2. Pull a model, e.g.:

   ```bash
   ollama pull llama3
   ```
3. Start Ollama:

   ```bash
   ollama serve
   ```

> LLMABSA requires Ollama to be running in the background.

---

## Running the API

Run with:

```bash
uvicorn src.api.api_exposed:app --host 0.0.0.0 --port 8000
```

The API will start at:
 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Interactive docs available at:
 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---


## Implementation Notes

* **Models load once at startup** — no reinitialization per request (improves speed).
* **Lifespan-based startup event** ensures analyzers are ready before handling requests.
* **Uniform schema** makes it easy to compare outputs from all analyzers.
* **PyABSA** caches its model after the first download for future runs.

---

## Tech Stack

* **FastAPI** — High-performance API framework
* **spaCy** — NLP parsing (POS tagging, dependency parsing)
* **NLTK (VADER)** — Lexicon-based sentiment
* **PyABSA** — Transformer-based ABSA
* **Ollama** — Local LLM engine
* **Uvicorn** — ASGI server

---

## Authors

**Philipe Souza**
**Rufina Chyhohidze**

---
