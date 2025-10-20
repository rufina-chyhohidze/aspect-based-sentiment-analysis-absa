"""
Unified ABSA Comparison Display
"""
import pandas as pd

from src.lexicon_absa import LexiconABSA
from src.lexicon_absa2_opinion_to_sentiment import LexiconOpinionFirstABSA
from src.ML_ABSA import ABSA
from src.LLMABSA import LLMABSA


def display_results(analyzer_name, results):
    print(f"\n{'=' * 60}")
    print(f" {analyzer_name} Results")
    print(f"{'=' * 60}")
    if not results:
        print("(No aspects detected)")
        return
    for r in results:
        print(f"- Aspect: {r.aspect:25s}  "
              f"Sentiment: {r.sentiment:9s}  "
              f"Conf: {r.confidence:.3f}  "
              f"Span: {r.text_span}")


def main():
    df = pd.read_csv("../data/Yelp Restaurant Reviews.csv")

    possible_cols = [c for c in df.columns if "review" in c.lower() or "text" in c.lower()]
    if not possible_cols:
        raise ValueError("Couldn't find a 'Review Text' or similar column in the CSV.")

    col_name = possible_cols[0]
    sample_review = df[col_name].dropna().sample(1).iloc[0]

    text = str(sample_review)

    print("\nSampled text for ABSA analysis:\n")
    print(text)
    print("\n" + "=" * 60)
    print("Unified ABSA Demonstration")
    print("=" * 60)
    print(f"Input text:\n{text}\n")

    analyzers = [
        ("LexiconABSA (rule-based)", LexiconABSA()),
        ("LexiconOpinionFirstABSA (opinion-first)", LexiconOpinionFirstABSA()),
        ("PyABSA (Transformer-based)", ABSA()),
        ("LLMABSA (Local Ollama LLM)", LLMABSA()),
    ]

    for name, analyzer in analyzers:
        try:
            results = analyzer.analyze(text)
            display_results(name, results)
        except Exception as e:
            print(f"[{name}] Error: {e}")


if __name__ == "__main__":
    main()
