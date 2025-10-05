#implementation2


# absa_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class ABSA:
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # inference mode

    def predict(self, text, aspect):
        """
        Predict sentiment for a given text and aspect.
        Some models expect 'aspect' as part of input, e.g., "text [SEP] aspect"
        """
        # Combine text and aspect as required by this ABSA model
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
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            # Get predicted label
            pred = torch.argmax(probs, dim=1).item()

        # Map numeric label to sentiment
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map.get(pred, "unknown")
        confidence = probs[0, pred].item()

        return {"aspect": aspect, "sentiment": sentiment, "confidence": confidence}


# Example usage
if __name__ == "__main__":
    absa = ABSA()
    text = "The battery life of this phone is amazing but the screen is dim."
    aspects = ["battery life", "screen"]

    for aspect in aspects:
        result = absa.predict(text, aspect)
        print(result)
