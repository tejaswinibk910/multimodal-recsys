import numpy as np
from typing import List, Dict

class DiversityReranker:
    def __init__(self, embeddings, movie_ids, lambda_param=0.5):
        """
        lambda_param: trade-off between relevance (1.0) and diversity (0.0)
        0.5 = balanced
        """
        self.embeddings = embeddings
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
        self.lambda_param = lambda_param
    
    def rerank_mmr(self, candidate_ids: List[int], candidate_scores: List[float], 
                   top_k: int = 10) -> List[Dict]:
        """
        Maximal Marginal Relevance re-ranking
        Selects items that are relevant but diverse from already selected items
        """
        if len(candidate_ids) == 0:
            return []
        
        # Get embeddings for candidates
        candidate_embs = []
        valid_candidates = []
        valid_scores = []
        
        for cid, score in zip(candidate_ids, candidate_scores):
            if cid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[cid]
                candidate_embs.append(self.embeddings[idx])
                valid_candidates.append(cid)
                valid_scores.append(score)
        
        if len(valid_candidates) == 0:
            return []
        
        candidate_embs = np.array(candidate_embs)
        valid_scores = np.array(valid_scores)
        
        # Normalize embeddings for cosine similarity
        candidate_embs = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
        
        # Normalize scores to [0, 1]
        if valid_scores.max() > valid_scores.min():
            norm_scores = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
        else:
            norm_scores = np.ones_like(valid_scores)
        
        selected_indices = []
        selected_embs = []
        remaining_indices = list(range(len(valid_candidates)))
        
        # MMR algorithm
        for _ in range(min(top_k, len(valid_candidates))):
            if len(selected_indices) == 0:
                # First item: just pick highest relevance
                best_idx = np.argmax(norm_scores)
            else:
                # Calculate MMR scores
                mmr_scores = []
                
                for idx in remaining_indices:
                    relevance = norm_scores[idx]
                    
                    # Max similarity to already selected items
                    if len(selected_embs) > 0:
                        similarities = np.dot(selected_embs, candidate_embs[idx])
                        max_sim = np.max(similarities)
                    else:
                        max_sim = 0
                    
                    # MMR formula
                    mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                    mmr_scores.append(mmr_score)
                
                # Select item with highest MMR score
                best_mmr_idx = np.argmax(mmr_scores)
                best_idx = remaining_indices[best_mmr_idx]
            
            selected_indices.append(best_idx)
            selected_embs.append(candidate_embs[best_idx])
            remaining_indices.remove(best_idx)
            
            if len(remaining_indices) == 0:
                break
        
        # Return reranked results
        reranked = []
        for idx in selected_indices:
            reranked.append({
                'movieId': valid_candidates[idx],
                'score': float(valid_scores[idx]),
                'mmr_rank': len(reranked) + 1
            })
        
        return reranked
    
    def calculate_diversity_metrics(self, movie_ids: List[int]) -> Dict:
        """Calculate diversity metrics for a recommendation list"""
        if len(movie_ids) < 2:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0}
        
        # Get embeddings
        embs = []
        for mid in movie_ids:
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                embs.append(self.embeddings[idx])
        
        if len(embs) < 2:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0}
        
        embs = np.array(embs)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = np.dot(embs[i], embs[j])
                similarities.append(sim)
        
        return {
            'avg_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'avg_distance': float(1 - np.mean(similarities))
        }