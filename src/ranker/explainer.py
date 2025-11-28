import numpy as np
import pandas as pd
from typing import List, Dict

class RecommendationExplainer:
    def __init__(self, movies_df, embeddings, movie_ids, retriever):
        self.movies = movies_df
        self.embeddings = embeddings
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
        self.retriever = retriever
    
    def explain_recommendation(self, recommended_movie_id: int, 
                              user_history: List[int], 
                              method: str = 'detailed') -> Dict:
        """
        Generate explanation for why a movie was recommended
        method: 'simple', 'detailed', or 'technical'
        """
        rec_movie = self.movies[self.movies['movieId'] == recommended_movie_id]
        if len(rec_movie) == 0:
            return {"error": "Movie not found"}
        
        rec_movie = rec_movie.iloc[0]
        
        if method == 'simple':
            return self._simple_explanation(rec_movie, user_history)
        elif method == 'detailed':
            return self._detailed_explanation(rec_movie, user_history)
        elif method == 'technical':
            return self._technical_explanation(rec_movie, user_history)
        else:
            return {"error": "Unknown explanation method"}
    
    def _simple_explanation(self, rec_movie, user_history):
        """One-sentence explanation"""
        rec_genres = set(rec_movie['genres'].split('|'))
        
        # Find most similar movie from history
        best_match = self._find_most_similar_history_movie(rec_movie['movieId'], user_history)
        
        if best_match:
            return {
                "explanation": f"Because you watched '{best_match['title']}', we think you'll enjoy '{rec_movie['title']}'",
                "confidence": "high" if best_match['similarity'] > 0.8 else "medium"
            }
        else:
            return {
                "explanation": f"Based on your viewing history, you might enjoy {rec_movie['genres']} movies",
                "confidence": "medium"
            }
    
    def _detailed_explanation(self, rec_movie, user_history):
        """Multi-faceted explanation with reasons"""
        reasons = []
        
        # 1. Genre matching
        rec_genres = set(rec_movie['genres'].split('|'))
        history_genres = self._get_user_genre_preferences(user_history)
        common_genres = rec_genres.intersection(set(history_genres.keys()))
        
        if common_genres:
            top_genre = max(common_genres, key=lambda g: history_genres.get(g, 0))
            reasons.append({
                "type": "genre_match",
                "text": f"Matches your interest in {top_genre} movies",
                "weight": 0.3
            })
        
        # 2. Similar to past watches
        similar_movies = self._find_top_similar_history_movies(rec_movie['movieId'], user_history, top_k=3)
        if similar_movies:
            reasons.append({
                "type": "content_similarity",
                "text": f"Similar to movies you've enjoyed",
                "similar_movies": [m['title'] for m in similar_movies],
                "weight": 0.5
            })
        
        # 3. Sequential pattern
        if len(user_history) >= 2:
            reasons.append({
                "type": "sequential_pattern",
                "text": "Fits your viewing patterns",
                "weight": 0.2
            })
        
        return {
            "movie": rec_movie['title'],
            "genres": rec_movie['genres'],
            "reasons": reasons,
            "summary": self._generate_summary(reasons, rec_movie)
        }
    
    def _technical_explanation(self, rec_movie, user_history):
        """Technical details for debugging"""
        rec_emb = self._get_embedding(rec_movie['movieId'])
        
        # Calculate similarities to history
        history_similarities = []
        for hist_id in user_history[-5:]:
            hist_emb = self._get_embedding(hist_id)
            if hist_emb is not None and rec_emb is not None:
                sim = np.dot(rec_emb, hist_emb)
                hist_movie = self.movies[self.movies['movieId'] == hist_id].iloc[0]
                history_similarities.append({
                    "movie": hist_movie['title'],
                    "similarity": float(sim)
                })
        
        return {
            "movie": rec_movie['title'],
            "embedding_similarity": history_similarities,
            "user_history_length": len(user_history),
            "genres": rec_movie['genres']
        }
    
    def _get_embedding(self, movie_id):
        if movie_id not in self.movie_id_to_idx:
            return None
        idx = self.movie_id_to_idx[movie_id]
        return self.embeddings[idx]
    
    def _find_most_similar_history_movie(self, rec_movie_id, user_history):
        rec_emb = self._get_embedding(rec_movie_id)
        if rec_emb is None:
            return None
        
        best_match = None
        best_sim = -1
        
        for hist_id in user_history:
            hist_emb = self._get_embedding(hist_id)
            if hist_emb is not None:
                sim = np.dot(rec_emb, hist_emb)
                if sim > best_sim:
                    best_sim = sim
                    hist_movie = self.movies[self.movies['movieId'] == hist_id].iloc[0]
                    best_match = {
                        "title": hist_movie['title'],
                        "similarity": float(sim)
                    }
        
        return best_match
    
    def _find_top_similar_history_movies(self, rec_movie_id, user_history, top_k=3):
        rec_emb = self._get_embedding(rec_movie_id)
        if rec_emb is None:
            return []
        
        similarities = []
        for hist_id in user_history:
            hist_emb = self._get_embedding(hist_id)
            if hist_emb is not None:
                sim = np.dot(rec_emb, hist_emb)
                hist_movie = self.movies[self.movies['movieId'] == hist_id].iloc[0]
                similarities.append({
                    "title": hist_movie['title'],
                    "similarity": float(sim),
                    "movieId": int(hist_id)
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _get_user_genre_preferences(self, user_history):
        genre_counts = {}
        for movie_id in user_history:
            movie = self.movies[self.movies['movieId'] == movie_id]
            if len(movie) > 0:
                genres = movie.iloc[0]['genres'].split('|')
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts
    
    def _generate_summary(self, reasons, rec_movie):
        parts = []
        for reason in reasons:
            if reason['type'] == 'genre_match':
                parts.append(reason['text'].lower())
            elif reason['type'] == 'content_similarity':
                parts.append(f"similar to {', '.join(reason['similar_movies'][:2])}")
        
        if parts:
            return f"Recommended because it {' and '.join(parts)}."
        else:
            return f"Recommended based on your viewing history."
    
    def generate_batch_explanations(self, recommendations: List[Dict], 
                                    user_history: List[int],
                                    method: str = 'simple') -> List[Dict]:
        """Generate explanations for a list of recommendations"""
        explained_recs = []
        
        for rec in recommendations:
            explanation = self.explain_recommendation(
                rec['movieId'], 
                user_history, 
                method=method
            )
            
            rec_with_explanation = rec.copy()
            rec_with_explanation['explanation'] = explanation
            explained_recs.append(rec_with_explanation)
        
        return explained_recs