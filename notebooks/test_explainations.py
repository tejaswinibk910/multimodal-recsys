import sys
sys.path.append('src')

from pipeline import RecommendationPipeline
import json

pipeline = RecommendationPipeline()

test_history = [1, 364, 296]  # Toy Story, Lion King, Pulp Fiction

print("="*60)
print("RECOMMENDATION EXPLANATIONS")
print("="*60)

print("\n1. SIMPLE EXPLANATIONS:")
print("-"*60)
recs_simple = pipeline.recommend_with_explanations(
    test_history, top_k=5, explanation_method='simple'
)

for i, rec in enumerate(recs_simple, 1):
    print(f"\n{i}. {rec['title']}")
    print(f"   {rec['explanation']['explanation']}")
    print(f"   Confidence: {rec['explanation']['confidence']}")

print("\n\n2. DETAILED EXPLANATIONS:")
print("-"*60)
recs_detailed = pipeline.recommend_with_explanations(
    test_history, top_k=3, explanation_method='detailed'
)

for i, rec in enumerate(recs_detailed, 1):
    print(f"\n{i}. {rec['title']} ({rec['genres']})")
    exp = rec['explanation']
    print(f"   Summary: {exp['summary']}")
    print(f"   Reasons:")
    for reason in exp['reasons']:
        print(f"     • {reason['text']}")
        if 'similar_movies' in reason:
            print(f"       Similar to: {', '.join(reason['similar_movies'])}")

print("\n\n3. TECHNICAL EXPLANATION (for debugging):")
print("-"*60)
rec_id = recs_simple[0]['movieId']
tech_exp = pipeline.explainer.explain_recommendation(
    rec_id, test_history, method='technical'
)
print(json.dumps(tech_exp, indent=2))