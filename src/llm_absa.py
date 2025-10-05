# absa_model.py
from src.base import ABSAAnalyzer, AspectSentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List


class ABSA(ABSAAnalyzer):
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # inference mode

    def _predict_aspect(self, text: str, aspect: str) -> AspectSentiment:
        """Predict sentiment for a single aspect in text."""

        # Combine text and aspect as required by model
        input_text = f"{text} [SEP] {aspect}"

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # Map numeric label to sentiment
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map.get(pred, "unknown")
        confidence = probs[0, pred].item()

        # Try to locate aspect in text (optional heuristic)
        start = text.lower().find(aspect.lower())
        end = start + len(aspect) if start != -1 else -1
        text_span = [start, end] if start != -1 else []

        return AspectSentiment(
            aspect=aspect,
            sentiment=sentiment,
            confidence=confidence,
            text_span=text_span
        )

    def analyze(self, text: str, aspects: List[str]) -> List[AspectSentiment]:
        """
        Analyze a text for a list of aspects and return sentiments.
        This satisfies the ABSAAnalyzer interface.
        """
        results = []
        for aspect in aspects:
            aspect_sentiment = self._predict_aspect(text, aspect)
            results.append(aspect_sentiment)
        return results


# Example usage
if __name__ == "__main__":
    absa = ABSA()
    text = "The battery life of this phone is amazing but the screen is dim."
    aspects = ["battery life", "screen"]

    results = absa.analyze(text, aspects)
    for r in results:
        print(r)
