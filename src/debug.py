import sys
sys.path.append('src')

from pipeline import RecommendationPipeline
import pandas as pd
import numpy as np

print("Initializing pipeline...")
pipeline = RecommendationPipeline()

# Test history - animated movies
test_history = [1, 2355, 364, 2081]  # Toy Story, Bug's Life, Lion King, Little Mermaid

movies = pd.read_csv("data/raw/ml-25m/movies.csv")
print("\nUser History:")
for mid in test_history:
    movie = movies[movies['movieId'] == mid]
    if len(movie) > 0:
        print(f"  - {movie.iloc[0]['title']} (ID: {mid})")
        print(f"    In vocab: {mid in pipeline.item_to_idx}")

print("\n" + "="*60)
print("Testing Sequential Scores")
print("="*60)

# Get candidates
candidate_ids, candidate_scores = pipeline.retriever.search_by_session(test_history, k=20)

print(f"\nTop 20 candidates from FAISS:")
for i, (cid, score) in enumerate(zip(candidate_ids[:10], candidate_scores[:10]), 1):
    movie = movies[movies['movieId'] == cid]
    in_vocab = cid in pipeline.item_to_idx
    if len(movie) > 0:
        print(f"{i}. {movie.iloc[0]['title']} (ID: {cid}, FAISS: {score:.4f}, in_vocab: {in_vocab})")

# Get sequential scores
seq_scores = pipeline._get_sequential_scores(test_history, candidate_ids)

if seq_scores is not None:
    print(f"\nSequential scores stats:")
    print(f"  Min: {seq_scores.min():.4f}")
    print(f"  Max: {seq_scores.max():.4f}")
    print(f"  Mean: {seq_scores.mean():.4f}")
    print(f"  Std: {seq_scores.std():.4f}")
    
    print(f"\nTop 10 by sequential score:")
    seq_ranked = np.argsort(-seq_scores)[:10]
    for i, idx in enumerate(seq_ranked, 1):
        cid = candidate_ids[idx]
        movie = movies[movies['movieId'] == cid]
        if len(movie) > 0:
            print(f"{i}. {movie.iloc[0]['title']} (seq_score: {seq_scores[idx]:.4f}, faiss: {candidate_scores[idx]:.4f})")
else:
    print("\nSequential scores returned None!")

print("\n" + "="*60)
print("Checking valid history items")
print("="*60)

valid_history = [mid for mid in test_history if mid in pipeline.item_to_idx]
print(f"Valid history items: {len(valid_history)}/{len(test_history)}")
for mid in test_history:
    movie = movies[movies['movieId'] == mid]
    if len(movie) > 0:
        print(f"  {movie.iloc[0]['title']}: {'✓' if mid in pipeline.item_to_idx else '✗'}")