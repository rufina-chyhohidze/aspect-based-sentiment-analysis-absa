## 🔧 Model & Data Setup

This project uses **[PyABSA](https://github.com/yangheng95/PyABSA)** for Aspect-Based Sentiment Analysis (ABSA) and **[NLTK](https://www.nltk.org/)** for sentence tokenization.

### 📦 Automatic Downloads

On the first run:

* The **PyABSA** model (`multilingual`) will automatically download and be cached under:

  ```
  src/checkpoints/
  src/checkpoints.json
  ```
* The **NLTK** tokenizer (`punkt`) will also download automatically if not already present under:

  ```
  src/doc/punkt/
  ```

> 💡 Both downloads occur only once. After that, the analyzer runs fully offline.

These folders are ignored in version control using `.gitignore` to keep the repository lightweight.
They’ll be recreated automatically when you execute the code.

---

## 🚀 Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ABSA analyzer**

   ```bash
   python src/absa_model.py
   ```

3. **Expected behavior**

   * On the first run, the model and tokenizer will be downloaded.
   * You’ll see printed aspects and their corresponding sentiments for the example review.

---

## 🧠 Notes

* Keep the folder `src/doc/punkt/` in your repository for offline sentence tokenization.
* If running in a restricted environment (no internet), make sure the model checkpoint is pre-downloaded under `src/checkpoints/`.
* For teaching or demo purposes, no manual setup is needed — just run the script and the model handles everything automatically.
