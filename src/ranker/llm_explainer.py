from groq import Groq
import os
from typing import List, Dict
import pandas as pd

class LLMExplainer:
    def __init__(self, movies_df, api_key=None):
        """
        Initialize LLM explainer with Groq API
        Get your API key from: https://console.groq.com/
        """
        self.movies = movies_df
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: No Groq API key found. LLM explanations will be unavailable.")
    
    def format_movie_title(title):
        """Keep original MovieLens title format"""
        return title

    def clean_title_for_explanation(title):
        """Remove year from title for cleaner explanations"""
        match = re.match(r'^(.+?)\s*\((\d{4})\)$', title)
        if match:
            return format_movie_title(match.group(1) + f" ({match.group(2)})")
        return format_movie_title(title)

    
    def explain_recommendation(self, recommended_movie_id: int, 
                              user_history: List[int],
                              style: str = 'friendly') -> str:
        """
        Generate LLM-powered explanation
        style: 'friendly', 'professional', or 'casual'
        """
        if not self.client:
            return "LLM explanations unavailable (no API key)"
        
        rec_movie = self.movies[self.movies['movieId'] == recommended_movie_id]
        if len(rec_movie) == 0:
            return "Movie not found"
        
        rec_movie = rec_movie.iloc[0]
        
        history_movies = []
        for movie_id in user_history[-5:]:
            movie = self.movies[self.movies['movieId'] == movie_id]
            if len(movie) > 0:
                movie = movie.iloc[0]
                history_movies.append({
                    'title': movie['title'],
                    'genres': movie['genres']
                })
        
        prompt = self._create_prompt(rec_movie, history_movies, style)
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _create_prompt(self, rec_movie, history_movies, style):
        """Create prompt for LLM"""
        
        style_instructions = {
            'friendly': "Write in a warm, enthusiastic tone like a friend recommending a movie.",
            'professional': "Write in a professional, analytical tone.",
            'casual': "Write in a casual, conversational tone."
        }
        
        # Format and clean titles
        history_text = "\n".join([
            f"- {clean_title_for_explanation(m['title'])} ({m['genres']})" 
            for m in history_movies
        ])
        
        rec_title_clean = clean_title_for_explanation(rec_movie['title'])
        
        # Remove year for explanation
        rec_title_no_year = re.sub(r'\s*\(\d{4}\)$', '', rec_title_clean)
        
        prompt = f"""You are a movie recommendation system explaining why you recommended a movie.

    User's recent watch history:
    {history_text}

    Recommended movie:
    - Title: {rec_title_clean}
    - Genres: {rec_movie['genres']}

    Task: Write ONE short sentence (15-20 words max) explaining why this movie was recommended.

    {style_instructions.get(style, style_instructions['friendly'])}

    Rules:
    - Use movie titles WITHOUT years (e.g., "Toy Story" not "Toy Story (1995)")
    - Focus on genre connections
    - Start with "Since you enjoyed..." or "You liked..."
    - Keep it under 20 words
    - Be conversational and natural

    Explanation:"""
        
        return prompt
    
    def explain_batch(self, recommendations: List[Dict], 
                     user_history: List[int],
                     style: str = 'friendly') -> List[Dict]:
        """Generate explanations for multiple recommendations"""
        explained_recs = []
        
        for rec in recommendations:
            explanation = self.explain_recommendation(
                rec['movieId'],
                user_history,
                style=style
            )
            
            rec_with_explanation = rec.copy()
            rec_with_explanation['llm_explanation'] = explanation
            explained_recs.append(rec_with_explanation)
        
        return explained_recs