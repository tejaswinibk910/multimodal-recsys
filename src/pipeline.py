import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from retrieval.faiss_query import FAISSRetriever
from sequential.sasrec import SASRec
from ranker.diversity_reranker import DiversityReranker
from ranker.explainer import RecommendationExplainer


class RecommendationPipeline:
    def __init__(self, 
                 faiss_index_path="data/embeddings/faiss_index.bin",
                 embeddings_path="data/embeddings/item_embeddings.npy",
                 movie_ids_path="data/embeddings/movie_ids.npy",
                 sasrec_path="data/embeddings/sasrec_best.pt",
                 sequential_data_path="data/processed/sequential_data.pkl",
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        
        print("Loading FAISS retriever...")
        self.retriever = FAISSRetriever(faiss_index_path, embeddings_path, movie_ids_path)
        
        print("Loading sequential data...")
        with open(sequential_data_path, 'rb') as f:
            seq_data = pickle.load(f)
        
        self.item_to_idx = seq_data['item_to_idx']
        self.idx_to_item = seq_data['idx_to_item']
        self.num_items = seq_data['num_items']
        
        print("Loading SASRec model...")
        checkpoint = torch.load(sasrec_path, map_location=device, weights_only=False)
        self.sasrec = SASRec(
            num_items=self.num_items,
            d_model=checkpoint['config']['d_model'],
            n_layers=checkpoint['config']['n_layers'],
            n_heads=checkpoint['config']['n_heads'],
            max_seq_len=checkpoint['config']['max_seq_len'],
            dropout=0.0
        ).to(device)
        self.sasrec.load_state_dict(checkpoint['model_state_dict'])
        self.sasrec.eval()
        
        print("Loading movie metadata...")
        self.movies = pd.read_csv("data/raw/ml-25m/movies.csv")
        
        print("Initializing diversity reranker...")
        embeddings = np.load(embeddings_path)
        movie_ids = np.load(movie_ids_path)
        self.diversity_reranker = DiversityReranker(embeddings, movie_ids, lambda_param=0.5)

        print("Initializing explainer...")
        self.explainer = RecommendationExplainer(
            self.movies, 
            embeddings, 
            movie_ids, 
            self.retriever
        )
        
        print("Pipeline ready!")
    
    def recommend(self, user_history, n_candidates=100, top_k=10, 
                  use_sequential=True, use_diversity=False, diversity_lambda=0.5):
        """
        use_diversity: Apply MMR diversity re-ranking
        diversity_lambda: 1.0 = pure relevance, 0.0 = pure diversity, 0.5 = balanced
        """
        if len(user_history) == 0:
            return self._recommend_popular(top_k)
        
        # Stage 1: Retrieval (FAISS)
        candidate_ids, candidate_scores = self.retriever.search_by_session(
            user_history, k=n_candidates
        )
        
        if candidate_ids is None:
            return self._recommend_popular(top_k)
        
        faiss_scores_norm = candidate_scores
        
        # Stage 2: Sequential scoring (SASRec)
        if use_sequential:
            sequential_scores = self._get_sequential_scores(user_history, candidate_ids)
            
            if sequential_scores is not None and not np.isnan(sequential_scores).any():
                faiss_min, faiss_max = candidate_scores.min(), candidate_scores.max()
                faiss_norm = (candidate_scores - faiss_min) / (faiss_max - faiss_min + 1e-8)
                
                seq_min, seq_max = sequential_scores.min(), sequential_scores.max()
                seq_norm = (sequential_scores - seq_min) / (seq_max - seq_min + 1e-8)
                
                alpha = 0.5
                combined_scores = alpha * faiss_norm + (1 - alpha) * seq_norm
            else:
                combined_scores = candidate_scores
        else:
            combined_scores = candidate_scores
        
        # Filter out history
        mask = ~np.isin(candidate_ids, user_history)
        filtered_ids = candidate_ids[mask]
        filtered_scores = combined_scores[mask]
        
        # Stage 3: Diversity Re-ranking (optional)
        if use_diversity:
            self.diversity_reranker.lambda_param = diversity_lambda
            reranked = self.diversity_reranker.rerank_mmr(
                filtered_ids, filtered_scores, top_k=top_k
            )
            
            results = []
            for item in reranked:
                movie_info = self.movies[self.movies['movieId'] == item['movieId']]
                if len(movie_info) > 0:
                    results.append({
                        'movieId': int(item['movieId']),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'score': float(item['score']),
                        'mmr_rank': item['mmr_rank']
                    })
        else:
            # Standard ranking by score
            if len(filtered_ids) > top_k:
                ranked_indices = np.argsort(-filtered_scores)[:top_k]
                final_recommendations = filtered_ids[ranked_indices]
                final_scores = filtered_scores[ranked_indices]
            else:
                final_recommendations = filtered_ids
                final_scores = filtered_scores
            
            results = []
            for movie_id, score in zip(final_recommendations, final_scores):
                movie_info = self.movies[self.movies['movieId'] == movie_id]
                if len(movie_info) > 0:
                    results.append({
                        'movieId': int(movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'score': float(score)
                    })
        
        return results
    
    def _get_sequential_scores(self, user_history, candidate_ids):
        valid_history = [mid for mid in user_history if mid in self.item_to_idx]
        
        if len(valid_history) == 0:
            return None
        
        seq_indices = [self.item_to_idx[mid] for mid in valid_history]
        
        max_len = 50
        if len(seq_indices) > max_len:
            seq_indices = seq_indices[-max_len:]
        
        seq_len = len(seq_indices)
        if seq_len < max_len:
            seq_indices = [0] * (max_len - seq_len) + seq_indices
        
        seq_tensor = torch.LongTensor([seq_indices]).to(self.device)
        
        with torch.no_grad():
            logits = self.sasrec(seq_tensor).cpu().numpy()[0]
        
        candidate_scores = []
        for cid in candidate_ids:
            if cid in self.item_to_idx:
                idx = self.item_to_idx[cid]
                candidate_scores.append(logits[idx])
            else:
                candidate_scores.append(logits[1:].mean())
        
        candidate_scores = np.array(candidate_scores)
        
        if np.isnan(candidate_scores).any() or np.isinf(candidate_scores).any():
            return None
        
        return candidate_scores
    
    def _recommend_popular(self, top_k):
        popular_movie_ids = self.retriever.movie_ids[:top_k]
        
        results = []
        for movie_id in popular_movie_ids:
            movie_info = self.movies[self.movies['movieId'] == movie_id]
            if len(movie_info) > 0:
                results.append({
                    'movieId': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'score': 1.0
                })
        
        return results
    
    def recommend_with_explanations(self, user_history, n_candidates=100, top_k=10,
                                use_sequential=True, use_diversity=False,
                                explanation_method='simple'):
        recommendations = self.recommend(
            user_history, n_candidates, top_k, 
            use_sequential, use_diversity
        )
        
        explained_recs = self.explainer.generate_batch_explanations(
            recommendations, user_history, method=explanation_method
        )
        
        return explained_recs

    
    def explain_recommendation(self, movie_id, user_history):
        movie_emb = self.retriever.get_embedding_by_movie_id(movie_id)
        if movie_emb is None:
            return "Movie not found in index"
        
        history_similarities = []
        for hist_movie_id in user_history[-5:]:
            hist_emb = self.retriever.get_embedding_by_movie_id(hist_movie_id)
            if hist_emb is not None:
                similarity = np.dot(movie_emb, hist_emb) / (np.linalg.norm(movie_emb) * np.linalg.norm(hist_emb))
                hist_movie = self.movies[self.movies['movieId'] == hist_movie_id].iloc[0]
                history_similarities.append({
                    'movie': hist_movie['title'],
                    'similarity': similarity
                })
        
        history_similarities = sorted(history_similarities, key=lambda x: x['similarity'], reverse=True)
        
        movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
        
        explanation = f"Recommended '{movie_info['title']}' because:\n"
        explanation += f"  Genres: {movie_info['genres']}\n"
        explanation += f"  Most similar to your recent watches:\n"
        for sim in history_similarities[:3]:
            explanation += f"    - {sim['movie']} (similarity: {sim['similarity']:.3f})\n"
        
        return explanation