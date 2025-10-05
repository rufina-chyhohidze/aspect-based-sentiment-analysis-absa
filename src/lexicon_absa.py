import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.base import ABSAAnalyzer, AspectSentiment


nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
KEY_ASPECTS = {
    "cannoli", "filling", "croissant", "bread", "bakery", "service", "pastry",
    "dessert", "tiramisu", "mascarpone", "sweetness", "cookie", "coffee", "latte",
    "straw", "starbucks", "jam", "fruit", "tart", "delivery", "espresso", "price",
    "employee", "chocolate", "cream", "pie", "crew", "place", "treat", "shake", "banana", "split", "manager", "owner","doughnuts","staff", "shakes","donuts"
}
REVIEWER_BLACKLIST = {"tolerance", "preference", "habit", "emotion", "mood"}
INTENSIFIERS = {"very": 1.5, "too": 1.5, "extremely": 1.7, "really": 1.3, "so": 1.3,"great":1.7,"best": 1.7}
HEDGES = {"slightly": 0.5, "somewhat": 0.7, "a bit": 0.5, "kind of": 0.7}

def clean_aspect(txt):
    return re.sub(r"^(the|my|a|this|an|their)\s+", "", txt.strip().lower())

def is_good_aspect(chunk_text):
    a = clean_aspect(chunk_text)
    if any(bad in a for bad in REVIEWER_BLACKLIST):
        return False
    for term in KEY_ASPECTS:
        if term in a:
            return True
    return False

class LexiconABSA(ABSAAnalyzer):
    def analyze(self, text: str):
        doc = nlp(text)
        sents = list(doc.sents)
        aspect_sentiments = {}
        aspects_with_opinions = set()

        for i, sent in enumerate(sents):
            # Search for aspects in this sentence
            aspects_in_sent = []
            for chunk in sent.noun_chunks:
                chunk_text = clean_aspect(chunk.text)
                if is_good_aspect(chunk.text):
                    aspects_in_sent.append({
                        "text": chunk_text,
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "sent_idx": i
                    })

            # For each aspect found, find opinions (adjectives, negation, intensifier) in [sent-1, sent, sent+1]
            for asp in aspects_in_sent:
                aspect = asp["text"]
                span_start = asp["start"]
                span_end = asp["end"]
                search_range = [max(0, i - 1), min(len(sents) - 1, i + 1)]
                scores = []
                for j in range(search_range[0], search_range[1] + 1):
                    window = sents[j]
                    for token in window:
                        if token.pos_ == "ADJ":
                            score = vader.polarity_scores(token.text)["compound"]
                            # Intensifiers and hedges
                            for child in token.children:
                                if child.dep_ == "advmod":
                                    adv = child.text.lower()
                                    if adv in INTENSIFIERS:
                                        score *= INTENSIFIERS[adv]
                                    elif adv in HEDGES:
                                        score *= HEDGES[adv]
                            # Negation handling
                            prev_tok = doc[token.i - 1] if token.i > 0 else None
                            if prev_tok is not None and prev_tok.text.lower() == "not":
                                score = -score
                            scores.append(score)
                        # Direct "not ADJ" pattern for opinions
                        if (
                            token.text.lower() == "not"
                            and token.head.pos_ == "ADJ"
                            and token.head.i > token.i
                        ):
                            score = vader.polarity_scores(token.head.text)["compound"]
                            scores.append(-score)

                # Aggregate and label
                if scores:
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0
                if avg_score > 0.09:
                    label = "positive"
                elif avg_score < -0.09:
                    label = "negative"
                else:
                    label = "neutral"
                if aspect in aspect_sentiments:
                    # Keep highest-confidence only
                    if abs(avg_score) > abs(aspect_sentiments[aspect]["confidence"]):
                        aspect_sentiments[aspect] = {
                            "sentiment": label,
                            "confidence": round(abs(avg_score), 4),
                            "start": span_start,
                            "end": span_end
                        }
                else:
                    aspect_sentiments[aspect] = {
                        "sentiment": label,
                        "confidence": round(abs(avg_score), 4),
                        "start": span_start,
                        "end": span_end
                    }
                aspects_with_opinions.add(aspect)

        # Fallback: detect general positive/negative sentences for non-specific compliments
        if not aspects_with_opinions:
            for i, sent in enumerate(sents):
                for token in sent:
                    if token.pos_ == "ADJ":
                        score = vader.polarity_scores(token.text)["compound"]
                        if abs(score) > 0.2:
                            # Use "experience" as fallback aspect
                            aspect_sentiments["experience"] = {
                                "sentiment": "positive" if score > 0 else "negative",
                                "confidence": round(abs(score), 4),
                                "start": sent.start_char,
                                "end": sent.end_char
                            }

        # Prepare output
        results = []
        for aspect, info in aspect_sentiments.items():
            # Optionally skip weakly neutral aspects:
            if info["sentiment"] == "neutral" and info["confidence"] < 0.1:
                continue
            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=info["sentiment"],
                    confidence=info["confidence"],
                    text_span=[info["start"], info["end"]]
                )
            )



        return results



if __name__ == "__main__":
    analyzer = LexiconABSA()
    text = "I got the banana cream pie with chocolate ice cream. The crew was amazing!"
    results = analyzer.analyze(text)

    print("Input:", text)
    print("Results:")
    for r in results:
        print(r)