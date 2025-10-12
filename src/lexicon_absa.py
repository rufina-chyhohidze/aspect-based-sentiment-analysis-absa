import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz
from src.base import ABSAAnalyzer, AspectSentiment
from collections import defaultdict


nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
FUZZY_THRESHOLD = 85  # fuzzy similarity cutoff

KEY_ASPECTS = {
    "cannoli", "filling", "croissant", "bread", "bakery", "service", "pastry",
    "dessert", "tiramisu", "mascarpone", "sweetness", "cookie", "coffee", "latte",
    "straw", "starbucks", "jam", "fruit", "tart", "delivery", "espresso", "price",
    "employee", "chocolate", "ice cream", "pie", "crew", "place", "treat", "shake",
    "banana", "split", "manager", "owner", "doughnuts", "staff", "shakes", "donuts",
    "baguette", "butter", "sandwich", "ham", "cheese", "cashew", "sundae", "cereal",
    "topping", "store", "popcorn", "sorbet", "yogurt", "pizza", "taiyaki", "shop",
    "donut", "cupcake", "waffle", "flat white", "cappuccino", "macaron",
    "creme brulee", "cafe", "affogato", "cheesecake", "honey", "waffle cones","tart","brownie"

    "staff", "worker", "employee", "personnel", "cashier", "barista",
    "waiter", "waitress", "manager", "service", "recommendation", "advice", "help"
}

#sentiment to aspect , the list is smaller than the list of possible aspects
REVIEWER_BLACKLIST = {"tolerance", "preference", "habit", "emotion", "mood"}

INTENSIFIERS = {
    "very": 1.5, "too": 1.5, "extremely": 1.7,
    "really": 1.3, "so": 1.3, "great": 1.7, "best": 1.7
}

NEUTRAL_WORDS = {"ok", "okay", "fine", "average", "mediocre", "alright", "decent"}

HEDGES = {
    "slightly": 0.5, "somewhat": 0.7, "a bit": 0.5, "kind of": 0.7
}

ASPECT_STOP_WORDS = {"some", "other", "few", "several", "many", "various", "lot", "lots"}

GENERIC_POS_ADJECTIVES = {
    "great", "good", "awesome", "best", "amazing", "nice",
    "perfect", "fantastic", "wonderful", "exceptional", "excellent"
}

# --- Helper functions ---

def clean_aspect(txt: str) -> str:
    txt = re.sub(r"^(the|my|a|this|an|their|your|these|those|our)\s+", "", txt.strip().lower())
    words = txt.split()
    while words and words[0] in GENERIC_POS_ADJECTIVES:
        words.pop(0)
    return " ".join(words)

def clean_aspect_text(tokens):
    words = [t.text.lower() for t in tokens if t.text.lower() not in ASPECT_STOP_WORDS]
    while words and words[0] in {"the", "a", "an", "this", "my", "their", "your", "these", "those", "our"}:
        words.pop(0)
    return " ".join(words)

def normalized_aspect(txt: str) -> str:
    doc = nlp(txt.lower())
    lemmas = [t.lemma_ for t in doc if not t.is_stop]
    return " ".join(lemmas)

def is_good_aspect(aspect_text: str) -> bool:
    aspect_doc = nlp(aspect_text)
    lemmas = [token.lemma_.lower().replace("-", " ") for token in aspect_doc]
    cleaned = clean_aspect(" ".join(lemmas))

    if any(bad in cleaned for bad in REVIEWER_BLACKLIST):
        return False

    for term in KEY_ASPECTS:
        if term in cleaned or fuzz.partial_ratio(cleaned, term) >= FUZZY_THRESHOLD:
            return True

    for lemma in lemmas:
        if lemma in KEY_ASPECTS:
            return True

    return False

def extract_aspects(sent):
    aspects = []
    for token in sent:
        if token.pos_ in {"NOUN", "PROPN"}:
            aspect_tokens = []
            adj_found = False

            for left in reversed(list(token.lefts)):
                if left.dep_ == "compound":
                    aspect_tokens.insert(0, left)
                else:
                    break

            for left in reversed(list(token.lefts)):
                if left.dep_ == "amod" and not adj_found:
                    aspect_tokens.insert(0, left)
                    adj_found = True

            aspect_tokens.append(token)

            # Merge consecutive proper nouns or compounds (e.g. San Diego / work sundae)
            next_token = token.nbor(1) if token.i + 1 < len(sent.doc) else None
            while next_token is not None and next_token.pos_ in {"PROPN", "NOUN"}:
                if next_token.idx - aspect_tokens[-1].idx - len(aspect_tokens[-1]) <= 1:
                    aspect_tokens.append(next_token)
                    if next_token.i + 1 < len(sent.doc):
                        next_token = next_token.nbor(1)
                    else:
                        next_token = None
                else:
                    break

            # Prepositional objects and conjunctions
            for right in token.rights:
                if right.dep_ == "prep" and right.text.lower() == "of":
                    for pobj in right.children:
                        if pobj.pos_ in {"NOUN", "PROPN"}:
                            aspects.append((pobj.text.lower(), pobj.idx, pobj.idx + len(pobj)))
                if right.dep_ == "conj" and right.pos_ == "NOUN":
                    aspects.append((right.text.lower(), right.idx, right.idx + len(right)))

            aspect_text = clean_aspect_text(aspect_tokens)
            if is_good_aspect(aspect_text):
                start = aspect_tokens[0].idx
                end = aspect_tokens[-1].idx + len(aspect_tokens[-1])
                aspects.append((aspect_text, start, end))
    return aspects

def get_root_aspect(aspect: str) -> str:
    doc = nlp(aspect)
    nouns = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"}]
    return nouns[-1] if nouns else aspect.lower()

# --- ABSA Analyzer ---

class LexiconABSA(ABSAAnalyzer):
    def analyze(self, text: str):
        doc = nlp(text)
        sents = list(doc.sents)
        aspect_sentiments = {}
        seen_aspects = set()

        for i, sent in enumerate(sents):
            aspects_in_sent = extract_aspects(sent)
            search_range = [max(0, i - 1), min(len(sents) - 1, i + 1)]

            for aspect_text, start_char, end_char in aspects_in_sent:
                norm_asp = normalized_aspect(aspect_text)
                if norm_asp in seen_aspects:
                    continue
                seen_aspects.add(norm_asp)

                scores = []
                for j in range(search_range[0], search_range[1] + 1):
                    window = sents[j]
                    for token in window:
                        # adjective-based sentiment
                        if token.pos_ == "ADJ":
                            if token.lemma_.lower() in NEUTRAL_WORDS:
                                scores.append(0.0)
                                continue
                            aspect_span = set(range(start_char, end_char))
                            token_head_idxs = set(tok.idx for tok in token.head.subtree)
                            if aspect_span.intersection(token_head_idxs):
                                compound = vader.polarity_scores(token.text)["compound"]
                                for child in token.children:
                                    if child.dep_ == "advmod":
                                        adv = child.text.lower()
                                        if adv in INTENSIFIERS:
                                            compound *= INTENSIFIERS[adv]
                                        elif adv in HEDGES:
                                            compound *= HEDGES[adv]
                                prev_tok = doc[token.i - 1] if token.i > 0 else None
                                if prev_tok is not None and prev_tok.text.lower() == "not":
                                    compound = -compound
                                scores.append(compound)
                        # negation pattern
                        if (token.text.lower() == "not" and token.head.pos_ == "ADJ" and token.head.i > token.i):
                            compound = vader.polarity_scores(token.head.text)["compound"]
                            scores.append(-compound)
                        # opinion verbs
                        if token.pos_ == "VERB" and token.lemma_.lower() in {"love", "like", "enjoy", "recommend", "hate", "dislike"}:
                            compound = vader.polarity_scores(token.text)["compound"]
                            scores.append(compound)

                # ---  VADER scoring
                avg_score = sum(scores) / len(scores) if scores else 0.0
                if avg_score >= 0.05:
                    sentiment = "positive"
                elif avg_score <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                confidence = round(abs(avg_score), 4)

                existing = aspect_sentiments.get(norm_asp)
                if existing:
                    if confidence > existing["confidence"]:
                        aspect_sentiments[norm_asp] = {
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "start": start_char,
                            "end": end_char
                        }
                else:
                    aspect_sentiments[norm_asp] = {
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "start": start_char,
                        "end": end_char
                    }

        # Fallback if no aspect sentiments found
        if not aspect_sentiments:
            compound = vader.polarity_scores(text)["compound"]
            sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
            confidence = round(abs(compound), 4)
            aspect_sentiments["experience"] = {
                "sentiment": sentiment,
                "confidence": confidence,
                "start": 0,
                "end": len(text)
            }

        # Convert to AspectSentiment instances
        results = []
        for aspect, info in aspect_sentiments.items():
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

        # Group by root aspect lemma to reduce repetitions
        grouped = defaultdict(list)
        for r in results:
            root = get_root_aspect(r.aspect)
            grouped[root].append(r)

        consolidated_results = []
        for root, group in grouped.items():
            best = max(group, key=lambda x: x.confidence)
            consolidated_results.append(best)

        # Final fuzzy deduplication
        final_results = []
        for r in consolidated_results:
            norm_r = normalized_aspect(r.aspect)
            is_duplicate = False
            for fr in final_results:
                norm_fr = normalized_aspect(fr.aspect)
                if fuzz.ratio(norm_r, norm_fr) >= FUZZY_THRESHOLD:
                    if r.confidence > fr.confidence:
                        fr.aspect = r.aspect
                        fr.sentiment = r.sentiment
                        fr.confidence = r.confidence
                        fr.text_span = r.text_span
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_results.append(r)

        return final_results


# --- Example Run ---
if __name__ == "__main__":
    analyzer = LexiconABSA()
    text = (
        "Natalie is phenomenal!!! She helped us pick out pastries based off our taste preferences. "
        "The service was great, and we’ll definitely come back, and waffles were ok"
    )
    results = analyzer.analyze(text)

    print("Input:", text)
    print("Results:")
    for r in results:
        print(r)
