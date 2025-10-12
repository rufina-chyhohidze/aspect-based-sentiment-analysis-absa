import re
from collections import defaultdict
from typing import List

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

from src.base import ABSAAnalyzer, AspectSentiment


nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
FUZZY_THRESHOLD = 85


INTENSIFIERS = {
    "very": 1.5, "too": 1.5, "extremely": 1.7,
    "really": 1.3, "so": 1.3, "great": 1.7, "best": 1.7
}

HEDGES = {
    "slightly": 0.5, "somewhat": 0.7, "a bit": 0.5, "kind of": 0.7
}

NEGATION_WORDS = {"not", "n't", "never", "no"}

# Domain stopwords / meaningless aspects
ASPECT_STOPWORDS = {
    "bit", "time", "las", "nyc", "thing", "stuff", "everything",
    "anything", "something", "place"  # optional, if you want to keep only specific entities
}


# Positive words that VADER often misinterprets
DOMAIN_POS_OVERRIDE = {"crazy", "deliciousness", "heaven", "insane"}
DOMAIN_NEG_OVERRIDE = set()  # if needed later

# --- Domain collocations ---
COLLOCATIONS = [
    "ice cream",
    "red velvet",
    "cinnamon toast crunch",
    "mint chip"
]

def preprocess_collocations(text: str) -> str:
    """Replace known multi-word collocations with underscored versions robustly (handle punctuation and special spaces)."""
    # normalize unicode spaces (e.g., non-breaking spaces)
    text = re.sub(r"\s+", " ", text)

    for phrase in COLLOCATIONS:
        underscored = phrase.replace(" ", "_")
        # allow punctuation or apostrophes after collocation
        pattern = rf"(?i)\b{re.escape(phrase)}\b(?=[\s\.,'!?;:]|$)"
        text = re.sub(pattern, underscored, text)
    return text


def postprocess_aspect(aspect: str) -> str:
    """Restore original spacing after analysis."""
    return aspect.replace("_", " ")

# --- Helper functions ---

def get_sentiment_score(token) -> float:
    """Get sentiment polarity for a token with modifiers, overrides and negation."""
    lemma = token.lemma_.lower()
    if lemma in DOMAIN_POS_OVERRIDE:
        base_score = 0.6
    elif lemma in DOMAIN_NEG_OVERRIDE:
        base_score = -0.6
    else:
        base_score = vader.polarity_scores(token.text)["compound"]

    # intensifiers and hedges
    for child in token.children:
        if child.dep_ == "advmod":
            adv = child.text.lower()
            if adv in INTENSIFIERS:
                base_score *= INTENSIFIERS[adv]
            elif adv in HEDGES:
                base_score *= HEDGES[adv]

    # negation (children)
    for child in token.children:
        if child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS:
            base_score = -base_score

    # negation (previous tokens)
    for prev in token.lefts:
        if prev.text.lower() in NEGATION_WORDS:
            base_score = -base_score

    return base_score
def find_aspect_for_opinion(token):
    doc = token.doc
    colloc_tokens = [c.replace(" ", "_") for c in COLLOCATIONS]

    # Direct collocation match
    if token.text.lower() in colloc_tokens:
        return doc[token.i:token.i+1]

    # Look left/right for collocation
    for offset in range(1, 4):
        if token.i - offset >= 0:
            neighbor = doc[token.i - offset]
            if neighbor.text.lower() in colloc_tokens:
                return doc[neighbor.i:neighbor.i+1]
        if token.i + offset < len(doc):
            neighbor = doc[token.i + offset]
            if neighbor.text.lower() in colloc_tokens:
                return doc[neighbor.i:neighbor.i+1]

    # --- 🆕 Expand head if it contains a collocation compound ---
    head = token.head
    if head.pos_ in {"NOUN", "PROPN"}:
        # check its children for collocation tokens
        for child in head.children:
            if child.text.lower() in colloc_tokens:
                return doc[child.i:child.i+1]
    if head.text.lower() in colloc_tokens:
        return doc[head.i:head.i+1]

    # Helper: longest noun chunk containing a token
    def longest_chunk_containing(t):
        chunks = [c for c in doc.noun_chunks if t.i >= c.start and t.i < c.end]
        return max(chunks, key=lambda c: len(c.text)) if chunks else None

    # Standard chunk search
    if head.pos_ in {"NOUN", "PROPN"}:
        chunk = longest_chunk_containing(head)
        if chunk:
            return chunk

    for child in token.children:
        if child.pos_ in {"NOUN", "PROPN"}:
            chunk = longest_chunk_containing(child)
            if chunk:
                return chunk

    # Fallback: left/right scan
    sent_tokens = list(token.sent)
    idx = sent_tokens.index(token)

    for i in range(idx - 1, -1, -1):
        t = sent_tokens[i]
        if t.pos_ in {"NOUN", "PROPN"} or t.text.lower() in colloc_tokens:
            chunk = longest_chunk_containing(t) or doc[t.i:t.i+1]
            if chunk:
                return chunk

    for i in range(idx + 1, len(sent_tokens)):
        t = sent_tokens[i]
        if t.pos_ in {"NOUN", "PROPN"} or t.text.lower() in colloc_tokens:
            chunk = longest_chunk_containing(t) or doc[t.i:t.i+1]
            if chunk:
                return chunk

    return None





def normalize_aspect(txt: str) -> str:
    """Normalize aspect phrases, keep important multi-word expressions."""
    aspect = txt.lower().strip()
    aspect = aspect.replace("-", " ")            # turn ice-cream into ice cream
    aspect = re.sub(r"[^\w\s]", "", aspect)      # remove punctuation but keep spaces
    aspect = re.sub(r"\s+", " ", aspect).strip() # collapse multiple spaces

    # handle known multi-word domain expressions
    DOMAIN_MULTIWORD_FIXES = {
        "ice cream": "ice cream",
        "icecream": "ice cream",   # for some tokenization cases
    }
    if aspect in DOMAIN_MULTIWORD_FIXES:
        aspect = DOMAIN_MULTIWORD_FIXES[aspect]

    if aspect in ASPECT_STOPWORDS or len(aspect) < 2:
        return ""

    return aspect


# --- ABSA implementation ---
class LexiconOpinionFirstABSA(ABSAAnalyzer):
    def analyze(self, text: str) -> List[AspectSentiment]:
        # Lock collocations
        raw_text = text
        text = preprocess_collocations(text)

        doc = nlp(text)
        aspect_scores = defaultdict(list)

        for token in doc:
            if token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.lemma_.lower() in {"love", "like", "hate", "dislike", "recommend"}):
                score = get_sentiment_score(token)
                if score == 0:
                    continue

                chunk = find_aspect_for_opinion(token)
                if chunk:
                    # restore original spacing after normalization
                    normalized = postprocess_aspect(normalize_aspect(chunk.text))
                    if normalized:
                        aspect_scores[normalized].append(score)

        results = []
        for aspect, scores in aspect_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.05:
                sentiment = "positive"
            elif avg_score < -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            confidence = abs(avg_score)

            start = raw_text.lower().find(aspect)
            end = start + len(aspect)

            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=round(confidence, 4),
                    text_span=[start, end]
                )
            )

        # Fuzzy deduplication remains the same
        final_results = []
        for r in results:
            dup = False
            for fr in final_results:
                if fuzz.ratio(r.aspect, fr.aspect) >= FUZZY_THRESHOLD:
                    if r.confidence > fr.confidence:
                        fr.aspect = r.aspect
                        fr.sentiment = r.sentiment
                        fr.confidence = r.confidence
                        fr.text_span = r.text_span
                    dup = True
                    break
            if not dup:
                final_results.append(r)

        return final_results



if __name__ == "__main__":
    analyzer = LexiconOpinionFirstABSA()
    text = "The ice cream was great but wtf this place is, so bad omg,, and ice cream with apple yummy, i suggest, dont try coffee tho, was awful!"
    results = analyzer.analyze(text)
    print(f"Text: {text}")
    for r in results:
        print(r)
