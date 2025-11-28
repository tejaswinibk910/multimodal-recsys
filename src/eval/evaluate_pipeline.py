import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pipeline import RecommendationPipeline
import json
from pathlib import Path

def calculate_metrics(predictions, ground_truth, k_list=[5, 10, 20]):
    metrics = {}
    
    for k in k_list:
        hits = []
        ndcgs = []
        mrrs = []
        
        for pred, truth in zip(predictions, ground_truth):
            top_k = pred[:k]
            
            # Hit Rate
            hit = 1 if truth in top_k else 0
            hits.append(hit)
            
            # NDCG
            if truth in top_k:
                rank = top_k.index(truth) + 1
                ndcg = 1.0 / np.log2(rank + 1)
                ndcgs.append(ndcg)
            else:
                ndcgs.append(0.0)
            
            # MRR
            if truth in top_k:
                rank = top_k.index(truth) + 1
                mrrs.append(1.0 / rank)
            else:
                mrrs.append(0.0)
        
        metrics[f'HR@{k}'] = np.mean(hits)
        metrics[f'NDCG@{k}'] = np.mean(ndcgs)
        metrics[f'MRR@{k}'] = np.mean(mrrs)
    
    return metrics

def evaluate_pipeline(pipeline, test_data, method='full', n_samples=None, n_candidates=100):
    """
    Evaluate the pipeline
    method: 'retrieval_only', 'full', or 'sequential_only'
    """
    sequences = test_data['sequences']
    targets = test_data['targets']
    
    if n_samples:
        sequences = sequences[:n_samples]
        targets = targets[:n_samples]
    
    predictions = []
    
    print(f"Evaluating {len(sequences)} test cases with method: {method}")
    
    for seq, target in tqdm(zip(sequences, targets), total=len(sequences)):
        if len(seq) == 0:
            predictions.append([])
            continue
        
        if method == 'retrieval_only':
            recs = pipeline.recommend(seq, n_candidates=n_candidates, top_k=20, use_sequential=False)
        elif method == 'full':
            recs = pipeline.recommend(seq, n_candidates=n_candidates, top_k=20, use_sequential=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        pred_ids = [r['movieId'] for r in recs]
        predictions.append(pred_ids)
    
    metrics = calculate_metrics(predictions, targets, k_list=[5, 10, 20])
    
    return metrics, predictions

def evaluate_coverage(predictions, all_movie_ids):
    """Calculate catalog coverage"""
    recommended_items = set()
    for pred in predictions:
        recommended_items.update(pred)
    
    coverage = len(recommended_items) / len(all_movie_ids)
    return coverage, len(recommended_items)

def evaluate_diversity(predictions):
    """Calculate intra-list diversity (average pairwise distance)"""
    # Simplified: just count unique genres per recommendation list
    # In practice, you'd use embeddings for proper diversity
    diversities = []
    for pred in predictions:
        diversities.append(len(set(pred[:10])))  # Unique items in top-10
    
    return np.mean(diversities)

def main():
    print("="*60)
    print("MULTI-STAGE RECOMMENDATION SYSTEM EVALUATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    with open("data/processed/sequential_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    test_data = data['test']
    all_movie_ids = np.load("data/embeddings/movie_ids.npy")
    
    print(f"Test samples: {len(test_data['sequences']):,}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = RecommendationPipeline()
    
    # Use subset for faster evaluation (or remove for full eval)
    n_samples = 5000  # Evaluate on 5k samples (takes ~15 min)
    # n_samples = None  # Uncomment for full evaluation
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: Retrieval-Only Baseline")
    print("="*60)
    
    retrieval_metrics, retrieval_preds = evaluate_pipeline(
        pipeline, test_data, method='retrieval_only', 
        n_samples=n_samples, n_candidates=100
    )
    
    print("\nRetrieval-Only Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Full Pipeline (Retrieval + Sequential)")
    print("="*60)
    
    full_metrics, full_preds = evaluate_pipeline(
        pipeline, test_data, method='full',
        n_samples=n_samples, n_candidates=100
    )
    
    print("\nFull Pipeline Metrics:")
    for metric, value in full_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print("\nImprovement from adding Sequential Model:")
    for metric in retrieval_metrics.keys():
        retrieval_val = retrieval_metrics[metric]
        full_val = full_metrics[metric]
        improvement = ((full_val - retrieval_val) / retrieval_val) * 100
        print(f"  {metric}: {retrieval_val:.4f} -> {full_val:.4f} ({improvement:+.1f}%)")
    
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    
    # Coverage
    ret_coverage, ret_unique = evaluate_coverage(retrieval_preds, all_movie_ids)
    full_coverage, full_unique = evaluate_coverage(full_preds, all_movie_ids)
    
    print(f"\nCatalog Coverage:")
    print(f"  Retrieval-Only: {ret_coverage:.2%} ({ret_unique}/{len(all_movie_ids)} items)")
    print(f"  Full Pipeline: {full_coverage:.2%} ({full_unique}/{len(all_movie_ids)} items)")
    
    # Diversity
    ret_diversity = evaluate_diversity(retrieval_preds)
    full_diversity = evaluate_diversity(full_preds)
    
    print(f"\nAverage unique items in top-10:")
    print(f"  Retrieval-Only: {ret_diversity:.2f}")
    print(f"  Full Pipeline: {full_diversity:.2f}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        'retrieval_only': retrieval_metrics,
        'full_pipeline': full_metrics,
        'coverage': {
            'retrieval_only': float(ret_coverage),
            'full_pipeline': float(full_coverage)
        },
        'n_test_samples': len(retrieval_preds),
        'n_candidates': 100
    }
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "pipeline_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'pipeline_evaluation.json'}")
    
    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE FOR PORTFOLIO")
    print("="*60)
    
    print("\n| Method | HR@5 | HR@10 | HR@20 | NDCG@10 | NDCG@20 |")
    print("|--------|------|-------|-------|---------|---------|")
    print(f"| Retrieval-Only | {retrieval_metrics['HR@5']:.4f} | {retrieval_metrics['HR@10']:.4f} | {retrieval_metrics['HR@20']:.4f} | {retrieval_metrics['NDCG@10']:.4f} | {retrieval_metrics['NDCG@20']:.4f} |")
    print(f"| Full Pipeline | {full_metrics['HR@5']:.4f} | {full_metrics['HR@10']:.4f} | {full_metrics['HR@20']:.4f} | {full_metrics['NDCG@10']:.4f} | {full_metrics['NDCG@20']:.4f} |")
    
    print("\nDone!")

if __name__ == "__main__":
    main()