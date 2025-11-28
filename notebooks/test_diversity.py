import sys
sys.path.append('src')

from pipeline import RecommendationPipeline
import pandas as pd

pipeline = RecommendationPipeline()

test_history = [1, 364]  # Toy Story, Lion King

print("="*60)
print("COMPARISON: Standard vs Diversity Re-ranking")
print("="*60)

print("\n1. Standard Ranking (No Diversity):")
recs_standard = pipeline.recommend(test_history, top_k=10, use_diversity=False)
for i, rec in enumerate(recs_standard, 1):
    print(f"{i}. {rec['title']} - {rec['genres']}")

print("\n2. Diversity Re-ranking (lambda=0.5 - Balanced):")
recs_diverse = pipeline.recommend(test_history, top_k=10, use_diversity=True, diversity_lambda=0.5)
for i, rec in enumerate(recs_diverse, 1):
    print(f"{i}. {rec['title']} - {rec['genres']}")

print("\n3. Maximum Diversity (lambda=0.2):")
recs_max_diverse = pipeline.recommend(test_history, top_k=10, use_diversity=True, diversity_lambda=0.2)
for i, rec in enumerate(recs_max_diverse, 1):
    print(f"{i}. {rec['title']} - {rec['genres']}")

# Calculate diversity metrics
print("\n" + "="*60)
print("DIVERSITY METRICS")
print("="*60)

standard_ids = [r['movieId'] for r in recs_standard]
diverse_ids = [r['movieId'] for r in recs_diverse]
max_diverse_ids = [r['movieId'] for r in recs_max_diverse]

standard_metrics = pipeline.diversity_reranker.calculate_diversity_metrics(standard_ids)
diverse_metrics = pipeline.diversity_reranker.calculate_diversity_metrics(diverse_ids)
max_diverse_metrics = pipeline.diversity_reranker.calculate_diversity_metrics(max_diverse_ids)

print(f"\nStandard Ranking:")
print(f"  Avg pairwise similarity: {standard_metrics['avg_similarity']:.4f}")
print(f"  Avg pairwise distance: {standard_metrics['avg_distance']:.4f}")

print(f"\nDiversity Re-ranking (lambda=0.5):")
print(f"  Avg pairwise similarity: {diverse_metrics['avg_similarity']:.4f}")
print(f"  Avg pairwise distance: {diverse_metrics['avg_distance']:.4f}")

print(f"\nMaximum Diversity (lambda=0.2):")
print(f"  Avg pairwise similarity: {max_diverse_metrics['avg_similarity']:.4f}")
print(f"  Avg pairwise distance: {max_diverse_metrics['avg_distance']:.4f}")