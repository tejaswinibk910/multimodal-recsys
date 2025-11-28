import sys
sys.path.append('src')

from pipeline import RecommendationPipeline
import pandas as pd
import numpy as np

pipeline = RecommendationPipeline()

print("Recommendations based on user history")

# Simulate a user who likes animated movies
test_history = [1, 2355, 364, 2081]  # Toy Story, Aladdin, Lion King, Little Mermaid

movies = pd.read_csv("data/raw/ml-25m/movies.csv")
print("\nUser History:")
for mid in test_history:
    movie = movies[movies['movieId'] == mid]
    if len(movie) > 0:
        print(f"  - {movie.iloc[0]['title']}")

print("\nRecommendations:")
recommendations = pipeline.recommend(test_history, n_candidates=100, top_k=10)

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']}")
    print(f"   Genres: {rec['genres']}")
    print(f"   Score: {rec['score']:.4f}")

print("Compare with/without sequential model")

print("\nWith Sequential Model:")
recs_with_seq = pipeline.recommend(test_history, n_candidates=100, top_k=5, use_sequential=True)
for i, rec in enumerate(recs_with_seq, 1):
    print(f"{i}. {rec['title']} (score: {rec['score']:.4f})")

print("\nWithout Sequential Model (Retrieval Only):")
recs_no_seq = pipeline.recommend(test_history, n_candidates=100, top_k=5, use_sequential=False)
for i, rec in enumerate(recs_no_seq, 1):
    print(f"{i}. {rec['title']} (score: {rec['score']:.4f})")

print("Cold start (no history)")

cold_start_recs = pipeline.recommend([], top_k=5)
print("\nCold Start Recommendations:")
for i, rec in enumerate(cold_start_recs, 1):
    print(f"{i}. {rec['title']}")