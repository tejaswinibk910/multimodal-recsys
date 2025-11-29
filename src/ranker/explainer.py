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
        """Generate explanation for why a movie was recommended"""
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
        """One-sentence explanation based on genre overlap primarily"""
        rec_genres = set(rec_movie['genres'].split('|'))
        
        # Find history movies with genre overlap
        genre_matches = []
        for hist_id in user_history:
            hist_movie = self.movies[self.movies['movieId'] == hist_id]
            if len(hist_movie) > 0:
                hist_movie = hist_movie.iloc[0]
                hist_genres = set(hist_movie['genres'].split('|'))
                overlap = rec_genres.intersection(hist_genres)
                if overlap:
                    genre_matches.append({
                        'title': hist_movie['title'],
                        'overlap': overlap,
                        'overlap_count': len(overlap)
                    })
        
        # Sort by overlap count
        genre_matches.sort(key=lambda x: x['overlap_count'], reverse=True)
        
        if genre_matches:
            best_match = genre_matches[0]
            common_genres = ', '.join(list(best_match['overlap'])[:2])
            return {
                "explanation": f"You enjoyed {common_genres} movies like '{best_match['title']}'",
                "confidence": "high" if best_match['overlap_count'] >= 2 else "medium"
            }
        else:
            # Fallback: use most common genre in history
            user_genres = self._get_user_genre_preferences(user_history)
            if user_genres:
                top_genre = max(user_genres.keys(), key=lambda g: user_genres[g])
                if top_genre in rec_genres:
                    return {
                        "explanation": f"Matches your interest in {top_genre} movies",
                        "confidence": "medium"
                    }
            
            return {
                "explanation": f"Based on your viewing patterns, you might enjoy this {rec_movie['genres']} film",
                "confidence": "medium"
            }
    
    def _detailed_explanation(self, rec_movie, user_history):
        """Multi-faceted explanation with reasons"""
        reasons = []
        
        # 1. Genre matching with specific movies
        rec_genres = set(rec_movie['genres'].split('|'))
        genre_matches = []
        
        for hist_id in user_history:
            hist_movie = self.movies[self.movies['movieId'] == hist_id]
            if len(hist_movie) > 0:
                hist_movie = hist_movie.iloc[0]
                hist_genres = set(hist_movie['genres'].split('|'))
                overlap = rec_genres.intersection(hist_genres)
                if overlap:
                    genre_matches.append({
                        'title': hist_movie['title'],
                        'genres': list(overlap)
                    })
        
        if genre_matches:
            # Group by genre for cleaner explanation
            genre_to_movies = {}
            for match in genre_matches[:3]:
                for genre in match['genres']:
                    if genre not in genre_to_movies:
                        genre_to_movies[genre] = []
                    genre_to_movies[genre].append(match['title'])
            
            # Pick the most common genre
            top_genre = max(genre_to_movies.keys(), key=lambda g: len(genre_to_movies[g]))
            example_movies = genre_to_movies[top_genre][:2]
            
            reasons.append({
                "type": "genre_match",
                "text": f"You've enjoyed {top_genre} films like {', '.join(example_movies)}",
                "weight": 0.5
            })
        
        # 2. Content similarity (only mention if similarity is high)
        similar_movies = self._find_top_similar_history_movies(rec_movie['movieId'], user_history, top_k=3)
        high_similarity = [m for m in similar_movies if m['similarity'] > 0.75]
        
        if high_similarity:
            reasons.append({
                "type": "content_similarity",
                "text": f"Similar themes and style to movies you've watched",
                "similar_movies": [m['title'] for m in high_similarity],
                "weight": 0.3
            })
        
        # 3. Sequential pattern
        if len(user_history) >= 3:
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
            elif reason['type'] == 'content_similarity' and reason.get('similar_movies'):
                parts.append(f"similar to {', '.join(reason['similar_movies'][:2])}")
        
        if parts:
            return f"Recommended because {parts[0]}."
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