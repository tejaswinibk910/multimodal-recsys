import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from pipeline import RecommendationPipeline
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats

class ABTestSimulator:
    def __init__(self, pipeline, test_data):
        self.pipeline = pipeline
        self.test_data = test_data
    
    def simulate_user_engagement(self, recommendations, ground_truth, position_bias=True):
        """
        Simulate whether user engages with recommendations
        Returns: clicked_position, relevance_score
        """
        rec_ids = [r['movieId'] for r in recommendations]
        
        if ground_truth not in rec_ids:
            return None, 0.0
        
        position = rec_ids.index(ground_truth)
        
        # Position bias: users less likely to click lower positions
        if position_bias:
            click_probability = 1.0 / (1 + position * 0.3)
        else:
            click_probability = 1.0
        
        # Simulate click
        clicked = np.random.random() < click_probability
        
        if clicked:
            relevance = 1.0 / np.log2(position + 2)
            return position, relevance
        else:
            return None, 0.0
    
    def run_ab_test(self, variant_a_config, variant_b_config, n_samples=2000):
        """Run A/B test comparing two configurations"""
        sequences = self.test_data['sequences'][:n_samples]
        targets = self.test_data['targets'][:n_samples]
        
        results_a = []
        results_b = []
        
        print(f"\nRunning A/B test:")
        print(f"  Variant A: {variant_a_config['name']}")
        print(f"  Variant B: {variant_b_config['name']}")
        print(f"  Sample size: {n_samples} users")
        
        for seq, target in tqdm(zip(sequences, targets), total=len(sequences)):
            if len(seq) == 0:
                continue
            
            # Variant A
            recs_a = self.pipeline.recommend(
                seq, 
                top_k=10,
                use_sequential=variant_a_config.get('use_sequential', True),
                use_diversity=variant_a_config.get('use_diversity', False),
                diversity_lambda=variant_a_config.get('diversity_lambda', 0.5)
            )
            
            # Variant B
            recs_b = self.pipeline.recommend(
                seq,
                top_k=10,
                use_sequential=variant_b_config.get('use_sequential', True),
                use_diversity=variant_b_config.get('use_diversity', False),
                diversity_lambda=variant_b_config.get('diversity_lambda', 0.5)
            )
            
            # Simulate engagement
            click_pos_a, relevance_a = self.simulate_user_engagement(recs_a, target)
            click_pos_b, relevance_b = self.simulate_user_engagement(recs_b, target)
            
            results_a.append({
                'clicked': click_pos_a is not None,
                'click_position': click_pos_a,
                'relevance': relevance_a
            })
            
            results_b.append({
                'clicked': click_pos_b is not None,
                'click_position': click_pos_b,
                'relevance': relevance_b
            })
        
        return self._analyze_results(results_a, results_b, variant_a_config, variant_b_config)
    
    def _analyze_results(self, results_a, results_b, config_a, config_b):
        """Analyze A/B test results with statistical significance"""
        
        # CTR (Click-Through Rate)
        clicks_a = [r['clicked'] for r in results_a]
        clicks_b = [r['clicked'] for r in results_b]
        ctr_a = np.mean(clicks_a)
        ctr_b = np.mean(clicks_b)
        
        # Statistical significance for CTR
        _, p_value_ctr = stats.ttest_ind(clicks_a, clicks_b)
        
        # Average click position (lower is better)
        positions_a = [r['click_position'] for r in results_a if r['click_position'] is not None]
        positions_b = [r['click_position'] for r in results_b if r['click_position'] is not None]
        avg_pos_a = np.mean(positions_a) if positions_a else None
        avg_pos_b = np.mean(positions_b) if positions_b else None
        
        # Engagement quality (relevance score)
        relevance_a = [r['relevance'] for r in results_a]
        relevance_b = [r['relevance'] for r in results_b]
        avg_rel_a = np.mean(relevance_a)
        avg_rel_b = np.mean(relevance_b)
        
        _, p_value_rel = stats.ttest_ind(relevance_a, relevance_b)
        
        # Calculate improvements
        ctr_improvement = ((ctr_b - ctr_a) / ctr_a * 100) if ctr_a > 0 else 0
        rel_improvement = ((avg_rel_b - avg_rel_a) / avg_rel_a * 100) if avg_rel_a > 0 else 0
        
        return {
            'variant_a': {
                'name': config_a['name'],
                'ctr': ctr_a,
                'avg_click_position': avg_pos_a,
                'avg_relevance': avg_rel_a,
                'n_samples': len(results_a)
            },
            'variant_b': {
                'name': config_b['name'],
                'ctr': ctr_b,
                'avg_click_position': avg_pos_b,
                'avg_relevance': avg_rel_b,
                'n_samples': len(results_b)
            },
            'comparison': {
                'ctr_improvement_pct': ctr_improvement,
                'relevance_improvement_pct': rel_improvement,
                'p_value_ctr': p_value_ctr,
                'p_value_relevance': p_value_rel,
                'statistically_significant': p_value_ctr < 0.05
            }
        }

def main():
    print("A/B TEST SIMULATION")
    
    # Load data
    with open("data/processed/sequential_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Initialize pipeline
    pipeline = RecommendationPipeline()
    
    simulator = ABTestSimulator(pipeline, data['test'])
    
    # Test 1: Retrieval-only vs Full Pipeline
    print("TEST 1: Retrieval-Only vs Full Pipeline")
    
    results_test1 = simulator.run_ab_test(
        variant_a_config={
            'name': 'Retrieval-Only (Control)',
            'use_sequential': False,
            'use_diversity': False
        },
        variant_b_config={
            'name': 'Full Pipeline (Sequential)',
            'use_sequential': True,
            'use_diversity': False
        },
        n_samples=2000
    )
    
    print("RESULTS:")
    print(f"\nControl (Retrieval-Only):")
    print(f"  CTR: {results_test1['variant_a']['ctr']:.2%}")
    print(f"  Avg Click Position: {results_test1['variant_a']['avg_click_position']:.2f}")
    print(f"  Avg Relevance: {results_test1['variant_a']['avg_relevance']:.4f}")
    
    print(f"\nTreatment (Full Pipeline):")
    print(f"  CTR: {results_test1['variant_b']['ctr']:.2%}")
    print(f"  Avg Click Position: {results_test1['variant_b']['avg_click_position']:.2f}")
    print(f"  Avg Relevance: {results_test1['variant_b']['avg_relevance']:.4f}")
    
    print(f"\nImprovement:")
    print(f"  CTR: {results_test1['comparison']['ctr_improvement_pct']:+.1f}%")
    print(f"  Relevance: {results_test1['comparison']['relevance_improvement_pct']:+.1f}%")
    print(f"  P-value (CTR): {results_test1['comparison']['p_value_ctr']:.4f}")
    print(f"  Statistically Significant: {'✓ YES' if results_test1['comparison']['statistically_significant'] else '✗ NO'}")
    
    # Test 2: Standard vs Diversity Re-ranking
    print("TEST 2: Standard vs Diversity Re-ranking")
    
    results_test2 = simulator.run_ab_test(
        variant_a_config={
            'name': 'Standard Ranking',
            'use_sequential': True,
            'use_diversity': False
        },
        variant_b_config={
            'name': 'Diversity Re-ranking',
            'use_sequential': True,
            'use_diversity': True,
            'diversity_lambda': 0.5
        },
        n_samples=2000
    )
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"\nControl (Standard):")
    print(f"  CTR: {results_test2['variant_a']['ctr']:.2%}")
    print(f"  Avg Relevance: {results_test2['variant_a']['avg_relevance']:.4f}")
    
    print(f"\nTreatment (Diversity):")
    print(f"  CTR: {results_test2['variant_b']['ctr']:.2%}")
    print(f"  Avg Relevance: {results_test2['variant_b']['avg_relevance']:.4f}")
    
    print(f"\nImprovement:")
    print(f"  CTR: {results_test2['comparison']['ctr_improvement_pct']:+.1f}%")
    print(f"  Relevance: {results_test2['comparison']['relevance_improvement_pct']:+.1f}%")
    print(f"  Statistically Significant: {'✓ YES' if results_test2['comparison']['statistically_significant'] else '✗ NO'}")
    


if __name__ == "__main__":
    main()