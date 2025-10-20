# Aspect-Based Sentiment Analysis (ABSA)

This project explores multiple implementations of **Aspect-Based Sentiment Analysis (ABSA)** for extracting aspects (e.g., *service*, *ice cream*) from text and assigning them sentiment labels (positive, negative, or neutral).

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Model & Data Setup

This project uses **[PyABSA](https://github.com/yangheng95/PyABSA)** for transformer-based ABSA and **[NLTK](https://www.nltk.org/)** for sentence segmentation.

### Automatic Downloads

On first run:

* PyABSA model checkpoints are cached under:

  ```
  src/checkpoints/
  src/checkpoints.json
  ```

* NLTK's tokenizer (`punkt`) is downloaded (if not already present) to:

  ```
  src/doc/punkt/
  ```

Both download automatically once and are ignored in version control via `.gitignore`.

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run an ABSA analyzer**

   For lexicon-based ABSA:

   ```bash
   python src/lexicon_absa.py
   ```

   For transformer-based ABSA:

   ```bash
   python src/absa_model.py
   ```

3. **Output**

   On first execution, the model and tokenizer will download. Example review results will be printed to console.

---

## Notes

* `src/base.py` contains shared interfaces and the `AspectSentiment` data class.
* Ensure `src/doc/punkt/` exists for offline tokenization.
* No manual model setup is needed — all assets are downloaded on demand.

---

## Exercise 1 — Lexicon-Based ABSA (Aspect-First)

This ABSA system identifies **noun-based aspects** using spaCy and assigns sentiment using VADER. It includes:

* **Aspect extraction**: Based on syntactic parsing and noun chunking.
* **Sentiment scoring**: VADER polarity scores adjusted with intensifiers, hedges, and negations.
* **Deduplication**: Uses fuzzy matching to merge similar aspect phrases.
* **Normalization**: Filters vague or non-informative aspects.

### Sample Output

```text
AspectSentiment(aspect='pastry', sentiment='positive', confidence=0.67, text_span=[63, 71])
AspectSentiment(aspect='service', sentiment='positive', confidence=0.72, text_span=[95, 102])
```

---

## Exercise 1 — Lexicon-Based ABSA (Opinion-First)

This version flips the approach: it starts with **subjective words** (adjectives or opinion verbs) and links them to the nearest likely aspect.

### Highlights

* Tracks modifiers like **"very"**, **"somewhat"**, or **"not"**.
* Handles **multi-word collocations** (e.g., `"ice cream"`, `"mint chip"`).
* Uses custom heuristics for associating opinion words with aspect phrases.
* Suitable for **emotional**, **implicit**, or **informal** reviews.

### Sample Output

```text
AspectSentiment(aspect='ice cream', sentiment='positive', confidence=0.85, text_span=[4, 13])
AspectSentiment(aspect='coffee', sentiment='negative', confidence=0.62, text_span=[100, 106])
```

---

## Exercise 2 — Transformer-Based ABSA (PyABSA)

This implementation uses **PyABSA**’s `english_lcf_atepc` model to extract aspects and their sentiment via fine-tuned transformers.

### Features

* High-accuracy ABSA with pretrained deep learning models.
* Sentence tokenization with NLTK.
* GPU/CPU auto-detection.
* Configurable confidence threshold.

### Class: `ABSA`

```python
ABSA(model_name="english_lcf_atepc", device=None, min_confidence=0.3)
```

### Method: `analyze(text: str) -> List[AspectSentiment]`

Returns aspects with:

* `aspect`: Aspect phrase (e.g., "service")
* `sentiment`: Sentiment label (positive, neutral, negative)
* `confidence`: Confidence score (float)
* `text_span`: [start, end] character positions in original text

### Sample Output

```text
AspectSentiment(aspect='ice cream', sentiment='positive', confidence=0.95, text_span=[4, 13])
AspectSentiment(aspect='service', sentiment='negative', confidence=0.88, text_span=[39, 46])
AspectSentiment(aspect='prices', sentiment='negative', confidence=0.86, text_span=[106, 112])
```

---

## License
MIT License. Attribution to original PyABSA and VADER authors is appreciated.


Here’s the **updated `README.md`** with **Exercise 3** using **Ollama and a local LLM**. It fits seamlessly with the previous sections and avoids repeating shared explanations.

---

## Exercise 3 — ABSA via Local LLM (Ollama)

This module performs **Aspect-Based Sentiment Analysis (ABSA)** by prompting a **local large language model (LLM)** through [Ollama](https://ollama.com/). It uses a carefully crafted prompt with few-shot examples and expects strict JSON output from the model.

---

### Overview

* **Approach**: Uses a local LLM (e.g., `llama3`, `mistral`, `phi`) to extract aspects and associated sentiment.
* **Architecture**: Calls Ollama via HTTP `/api/chat`, parses JSON output, and maps to the shared `AspectSentiment` structure.
* **Few-shot prompt**: Demonstrates expected behavior with 2 worked-out examples.
* **Offline-first**: Ollama runs locally; no internet or cloud APIs are needed.

---

### Features

* Prompts the model with a strict JSON schema:

  ```json
  {
    "aspect": str,
    "sentiment": "positive|negative|neutral",
    "confidence": float between 0 and 1,
    "text_span": [start, end]
  }
  ```
* Handles:

  * Malformed or extra text responses (e.g., prose before/after JSON)
  * Missing fields and confidence repair
  * Span correction using fuzzy search
  * Deduplication of repeated aspects
* Configurable:

  * LLM model (`llama3.2` by default)
  * Temperature, retry behavior, timeout, and base URL

---

### Setup

Ensure [Ollama](https://ollama.com/) is installed and running locally.

Start a model (e.g., `llama3`):

```bash
ollama run llama3
```

Then, install required Python dependencies:

```bash
pip install requests
```

> This module assumes the shared base class `ABSAAnalyzer` and the `AspectSentiment` dataclass exist in `src/base.py`.

---

### Example Usage

```python
if __name__ == "__main__":
    analyzer = LLMABSA(model="llama3")
    text = "The affogato was amazing but the wait time was unbearable. I wouldn’t recommend the cupcakes either."
    results = analyzer.analyze(text)

    for r in results:
        print(r)
```

#### Sample Output

```text
AspectSentiment(aspect='affogato', sentiment='positive', confidence=0.87, text_span=[4, 12])
AspectSentiment(aspect='wait time', sentiment='negative', confidence=0.82, text_span=[33, 43])
AspectSentiment(aspect='cupcakes', sentiment='negative', confidence=0.75, text_span=[78, 86])
```

---

### Implementation Notes

* Uses `/api/chat` endpoint from Ollama’s local server (`http://localhost:11434`).
* Prompts include two few-shot examples to improve JSON formatting.
* Confidence values default to `0.0` if not parsable.
* If an aspect span is missing or invalid, it tries to find the **first occurrence** in the input text.

---

### Class: `LLMABSA`

```python
LLMABSA(
    model="llama3.2",
    temperature=0.2,
    max_retries=2,
    timeout_s=60,
    base_url="http://localhost:11434"
)
```

#### Method: `analyze(text: str) -> List[AspectSentiment]`

* Calls the LLM via HTTP and parses the output.
* Returns a list of structured `AspectSentiment` objects.
* Invalid/missing predictions are ignored.

---

### Example Prompt (Few-shot)

```json
Input:
"The pizza was delicious but the service was terrible. The ice cream was just okay."

Output:
[
  {"aspect": "pizza", "sentiment": "positive", "confidence": 0.85, "text_span": [4, 9]},
  {"aspect": "service", "sentiment": "negative", "confidence": 0.88, "text_span": [37, 44]},
  {"aspect": "ice cream", "sentiment": "neutral", "confidence": 0.55, "text_span": [55, 64]}
]
```

---

### Strengths

* Model-agnostic: Works with any local model supported by Ollama.
* Zero-training: No need to fine-tune or maintain external model checkpoints.
* Flexible: Can be adapted to other schema formats, domains, or tasks.

---

### Limitations

* JSON parsing may occasionally fail if the model drifts from the expected format.
* Long or complex reviews might exceed model context length.
* Performance varies based on LLM size and quality (e.g., `llama3` > `mistral`).

---

### File Structure

```
project/
├── src/
│   ├── base.py             # Shared ABSA base class & dataclass
│   ├── lexicon_absa.py     # Exercise 1 (aspect-first)
│   ├── lexicon_absa2_opinion_to_sentiment.py     # Exercise 1 (opinion-first)
│   ├── llm_absa.py       # Exercise 2 (PyABSA transformer-based)
│   ├── llm_3_absa.py         # Exercise 3: Ollama-based local LLM
│   └── doc/punkt/          # NLTK tokenizer
├── models/                 # PyABSA checkpoint cache
├── requirements.txt
└── README.md
```
---
