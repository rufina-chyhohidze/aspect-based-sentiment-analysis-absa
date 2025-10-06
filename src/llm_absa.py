from src.base import ABSAAnalyzer, AspectSentiment
from pyabsa import AspectTermExtraction as ATEPC
from typing import List
import torch
import nltk
import os
from nltk.tokenize import sent_tokenize

# NLTK setup
nltk_data_dir = os.path.join(os.path.dirname(__file__), "doc")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download punkt only if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Download punkt_tab only if missing
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)


class ABSA(ABSAAnalyzer):
    def __init__(self, model_name="english_lcf_atepc", device=None, min_confidence: float = 0.3):
        """
        Initialize the Aspect-Based Sentiment Analyzer using PyABSA.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_confidence = min_confidence

        # Local cache folder for the model
        self.model_dir = os.path.join(os.path.dirname(__file__), "models", model_name)
        os.makedirs(self.model_dir, exist_ok=True)


        print(f"Loading PyABSA model '{model_name}' on device: {self.device}")
        self.aspect_extractor = ATEPC.AspectExtractor(
            model_name,
            checkpoint_save_path=self.model_dir,  # ✅ cache here
            auto_device=True,  # GPU/CPU auto-detect
            cal_perplexity=True
        )

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspects with sentiments using PyABSA.
        """
        sentences = sent_tokenize(text)
        aspect_sentiments = []

        # PyABSA processes one or multiple sentences at once
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
                    # print(f"Aspect: {aspect}, Sentiment: {sentiment}, Confidence: {confidence:.2f}")

        return aspect_sentiments


# Example usage
if __name__ == "__main__":
    absa = ABSA()
    text = (
        "The ice cream was delicious but the service was very slow. "
        "The restaurant had a cozy atmosphere but the prices were high."
    )
    results = absa.analyze(text)
    for r in results:
        print(r)
