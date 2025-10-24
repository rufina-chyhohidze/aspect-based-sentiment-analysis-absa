import json
import os
import sys
from sklearn.metrics import precision_recall_fscore_support

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.ML_ABSA import ABSA

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

        # Catch extra predictions not in gold data
        for aspect, sentiment in pred_aspects - gold_aspects:
            y_true.append("none")
            y_pred.append(sentiment)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

if __name__ == "__main__":
    absa = ABSA()
    # Use the path relative to the project root
    test_file_path = os.path.join("src", "evaluator", "absa_test.json")
    evaluate_absa(absa, test_file_path)
