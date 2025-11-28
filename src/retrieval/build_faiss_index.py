import faiss
import numpy as np
import pandas as pd
from pathlib import Path

def build_index(embeddings, use_gpu=False):
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(embeddings)
    return index

def query_index(index, query_emb, k=10):
    query_emb = query_emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    return indices[0], distances[0]

def main():
    embeddings = np.load("data/embeddings/item_embeddings.npy").astype('float32')
    movie_ids = np.load("data/embeddings/movie_ids.npy")
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    index = build_index(embeddings, use_gpu=False)
    
    print(f"Index built with {index.ntotal} items")
    
    faiss.write_index(index, "data/embeddings/faiss_index.bin")
    print("Index saved to data/embeddings/faiss_index.bin")
    
    movies = pd.read_csv("data/raw/ml-25m/movies.csv")
    
    test_idx = 0
    test_movie_id = movie_ids[test_idx]
    test_emb = embeddings[test_idx]
    
    test_movie = movies[movies['movieId'] == test_movie_id].iloc[0]
    print(f"\nQuery Movie: {test_movie['title']}")
    print(f"Genres: {test_movie['genres']}")
    
    similar_indices, distances = query_index(index, test_emb, k=10)
    
    print("\nTop 10 Similar Movies:")
    for i, (idx, dist) in enumerate(zip(similar_indices, distances)):
        similar_movie_id = movie_ids[idx]
        similar_movie = movies[movies['movieId'] == similar_movie_id].iloc[0]
        print(f"{i+1}. {similar_movie['title']} (similarity: {dist:.4f})")
        print(f"   Genres: {similar_movie['genres']}")

if __name__ == "__main__":
    main()