import pandas as pd
import requests
import time
from pathlib import Path
from tqdm import tqdm
import os

TMDB_API_KEY = "6e07419748d68121eaaf90bbd77c67e7"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def download_movie_data(movie_id, tmdb_id):
    try:
        url = f"{BASE_URL}/movie/{int(tmdb_id)}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "overview": data.get("overview", ""),
                "poster_path": data.get("poster_path", ""),
                "genres": "|".join([g["name"] for g in data.get("genres", [])]),
                "popularity": data.get("popularity", 0)
            }
    except Exception as e:
        print(f"Error fetching movie {tmdb_id}: {e}")
    return None

def download_poster(poster_path, save_path):
    if not poster_path:
        return False
    
    try:
        url = f"{IMAGE_BASE_URL}{poster_path}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading poster: {e}")
    return False

def main():
    links_df = pd.read_csv("data/raw/ml-25m/links.csv")
    movies_df = pd.read_csv("data/raw/ml-25m/movies.csv")
    
    df = links_df.merge(movies_df, on="movieId")
    df = df[df["tmdbId"].notna()].copy()
    
    print(f"Found {len(df)} movies with TMDB IDs")
    
    Path("data/raw/posters").mkdir(exist_ok=True)
    
    # Start with subset for testing
    df = df.head(1000)
    print(f"Downloading metadata and posters for {len(df)} movies...")
    
    movie_metadata = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        movie_id = row["movieId"]
        tmdb_id = row["tmdbId"]
        
        metadata = download_movie_data(movie_id, tmdb_id)
        
        if metadata:
            movie_metadata.append(metadata)
            
            if metadata["poster_path"]:
                poster_save_path = f"data/raw/posters/{movie_id}.jpg"
                download_poster(metadata["poster_path"], poster_save_path)
        
        time.sleep(0.25)
    
    metadata_df = pd.DataFrame(movie_metadata)
    metadata_df.to_csv("data/processed/movie_metadata.csv", index=False)
    print(f"\nDone! Saved metadata for {len(metadata_df)} movies")
    print(f"Posters saved to: data/raw/posters/")

if __name__ == "__main__":
    main()