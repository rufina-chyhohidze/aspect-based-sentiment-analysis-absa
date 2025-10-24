import nltk
import os
from typing import List
from nltk.tokenize import sent_tokenize
import json
from sklearn.metrics import precision_recall_fscore_support


try:
    from src.base import ABSAAnalyzer, AspectSentiment
except ModuleNotFoundError:
    from base import ABSAAnalyzer, AspectSentiment
from pyabsa import ATEPCCheckpointManager

# --------------------------
# NLTK setup
# --------------------------
nltk_data_dir = os.path.join(os.path.dirname(__file__), "doc")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for pkg in ["punkt", "punkt_tab/english"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)


# --------------------------
# ABSA Class
# --------------------------
class ABSA(ABSAAnalyzer):
    def __init__(self, model_name=None, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

        # Use default checkpoint name for automatic download
        if model_name is None:
            model_name = "multilingual"  # PyABSA will download if not present

        print(f"Loading PyABSA model '{model_name}'...")

        self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint="multilingual",
            auto_device=False,
            device="cpu"
        )

    def analyze(self, text: str) -> List[AspectSentiment]:
        sentences = sent_tokenize(text)
        aspect_sentiments = []

        results = self.aspect_extractor.predict(
            sentences,
            print_result=False,
            save_result=False,
            ignore_error=True,
            pred_sentiment=True
        )

        for sentence_result in results:
            aspects = sentence_result.get("aspect", [])
            sentiments = sentence_result.get("sentiment", [])
            confidences = sentence_result.get("confidence", [1.0] * len(aspects))
            positions = sentence_result.get("position", [])

            for aspect, sentiment, confidence, position in zip(aspects, sentiments, confidences, positions):
                if confidence >= self.min_confidence:
                    aspect_sentiments.append(
                        AspectSentiment(
                            aspect=aspect,
                            sentiment=sentiment.lower(),
                            confidence=confidence,
                            text_span=position
                        )
                    )

        return aspect_sentiments


# --------------------------
# Evaluation function
# --------------------------
def evaluate_absa(model: ABSA, dataset_path: str):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true, y_pred = [], []

    for item in data:
        text = item["text"]
        gold_aspects = {(a["aspect"].lower(), a["sentiment"].lower()) for a in item["aspects"]}
        pred_aspects = {(r.aspect.lower(), r.sentiment.lower()) for r in model.analyze(text)}

        for aspect, sentiment in gold_aspects:
            if (aspect, sentiment) in pred_aspects:
                y_true.append(sentiment)
                y_pred.append(sentiment)
            else:
                y_true.append(sentiment)
                y_pred.append("none")

        # Extra predictions not in gold data
        for aspect, sentiment in pred_aspects - gold_aspects:
            y_true.append("none")
            y_pred.append(sentiment)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    absa = ABSA()
    # Adjust this path relative to the file
    test_file_path = os.path.join(os.path.dirname(__file__), "evaluator", "absa_test.json")
    evaluate_absa(absa, test_file_path)
