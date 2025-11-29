from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import sys
import re
sys.path.append('src')

from fastapi.responses import FileResponse
import os
from pathlib import Path
from pipeline import RecommendationPipeline
import uvicorn

app = FastAPI(
    title="Multi-Stage Recommender API",
    description="Production-style recommendation system with FAISS retrieval, SASRec sequential model, and diversity re-ranking",
    version="1.0.0"
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="src/serving/static"), name="static")

pipeline = None

def format_movie_title(title):
    """Convert 'Title, The (Year)' to 'The Title (Year)'"""
    # Pattern: "Something, Article (Year)"
    match = re.match(r'^(.+?),\s*(The|A|An)\s*\((\d{4})\)$', title)
    if match:
        main_title = match.group(1)
        article = match.group(2)
        year = match.group(3)
        return f"{article} {main_title} ({year})"
    
    return title

def format_recommendations(recs):
    """Format movie titles in recommendations list"""
    for rec in recs:
        if 'title' in rec:
            rec['title'] = format_movie_title(rec['title'])
    return recs

def format_movie_dict(movie_dict):
    """Format a single movie dictionary"""
    if 'title' in movie_dict:
        movie_dict['title'] = format_movie_title(movie_dict['title'])
    return movie_dict

@app.get("/")
async def read_root():
    return FileResponse("src/serving/static/index.html")

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Loading recommendation pipeline...")
    pipeline = RecommendationPipeline()
    print("Pipeline ready!")

class RecommendationRequest(BaseModel):
    user_history: List[int]
    n_candidates: Optional[int] = 100
    top_k: Optional[int] = 10
    use_sequential: Optional[bool] = True
    use_diversity: Optional[bool] = False
    diversity_lambda: Optional[float] = 0.5

class MovieRecommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[MovieRecommendation]
    total_count: int
    method: str

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized movie recommendations"""
    try:
        recommendations = pipeline.recommend(
            user_history=request.user_history,
            n_candidates=request.n_candidates,
            top_k=request.top_k,
            use_sequential=request.use_sequential,
            use_diversity=request.use_diversity,
            diversity_lambda=request.diversity_lambda
        )
        
        # Format titles
        recommendations = format_recommendations(recommendations)
        
        method = []
        if request.use_sequential:
            method.append("Sequential")
        else:
            method.append("Retrieval-Only")
        
        if request.use_diversity:
            method.append("+ Diversity")
        
        return RecommendationResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            method=" ".join(method)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/explained")
async def get_recommendations_with_explanations(request: RecommendationRequest, 
                                               explanation_level: str = "simple"):
    """Get recommendations with explanations"""
    try:
        recommendations = pipeline.recommend_with_explanations(
            user_history=request.user_history,
            n_candidates=request.n_candidates,
            top_k=request.top_k,
            use_sequential=request.use_sequential,
            use_diversity=request.use_diversity,
            explanation_method=explanation_level
        )
        
        # Format titles
        recommendations = format_recommendations(recommendations)
        
        return {
            "recommendations": recommendations,
            "total_count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/llm")
async def get_recommendations_with_llm_explanations(request: RecommendationRequest, 
                                                    explanation_style: str = "friendly"):
    """Get recommendations with LLM-powered explanations"""
    try:
        recommendations = pipeline.recommend_with_llm_explanations(
            user_history=request.user_history,
            n_candidates=request.n_candidates,
            top_k=request.top_k,
            use_sequential=request.use_sequential,
            use_diversity=request.use_diversity,
            explanation_style=explanation_style
        )
        
        # Format titles
        recommendations = format_recommendations(recommendations)
        
        return {
            "recommendations": recommendations,
            "total_count": len(recommendations),
            "explanation_type": "llm"
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movie/{movie_id}")
async def get_movie_info(movie_id: int):
    """Get movie details"""
    movie = pipeline.movies[pipeline.movies['movieId'] == movie_id]
    if len(movie) == 0:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_data = movie.iloc[0]
    result = {
        "movieId": int(movie_data['movieId']),
        "title": movie_data['title'],
        "genres": movie_data['genres']
    }
    
    return format_movie_dict(result)

@app.get("/search")
async def search_movies(query: str, limit: int = 20):
    """Search for movies by title"""
    matches = pipeline.movies[
        pipeline.movies['title'].str.contains(query, case=False, na=False)
    ].head(limit)
    
    results = []
    for _, movie in matches.iterrows():
        results.append({
            "movieId": int(movie['movieId']),
            "title": movie['title'],
            "genres": movie['genres']
        })
    
    # Format all titles
    results = format_recommendations(results)
    
    return {"results": results, "total": len(results)}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_movies": len(pipeline.movies),
        "movies_with_embeddings": len(pipeline.retriever.movie_ids),
        "model": "SASRec Transformer",
        "embedding_dim": 512,
        "metrics": {
            "hr@10": 0.070,
            "ndcg@10": 0.038,
            "improvement_vs_baseline": "+359%",
            "ctr_improvement_ab_test": "+467%"
        }
    }
@app.get("/poster/{movie_id}")
async def get_poster(movie_id: int):
    """Serve movie poster image"""
    poster_path = Path(f"data/raw/posters/{movie_id}.jpg")
    
    if poster_path.exists():
        return FileResponse(poster_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=404, detail="Poster not found")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)