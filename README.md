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

## File Structure

```
project/
├── src/
│   ├── base.py             # Shared ABSA base class & data structure
│   ├── lexicon_absa.py     # Exercise 1 (aspect-first)
│   ├── lexicon_absa2_opinion_to_sentiment.py     # Exercise 1 (opinion-first)
│   ├── llm_absa.py       # Exercise 2 (PyABSA transformer-based)
│   └── doc/punkt/          # NLTK tokenizer directory
├── models/                 # PyABSA model checkpoint cache
├── requirements.txt
└── README.md
```

---

## License
MIT License. Attribution to original PyABSA and VADER authors is appreciated.
