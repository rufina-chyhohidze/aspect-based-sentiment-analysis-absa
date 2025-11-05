from src.LLMABSA import LLMABSA
import pandas as pd

df = pd.read_csv("../data/Yelp Restaurant Reviews.csv")
sample_reviews = df["Review Text"].sample(3)

analyzer = LLMABSA(model="llama3.1", temperature=0.2)

for i, review in enumerate(sample_reviews, 1):
    print(f"\n--- Review {i} ---")
    print(review)
    for res in analyzer.analyze(review):
        print(f"  • {res.aspect}: {res.sentiment} ({res.confidence})")
