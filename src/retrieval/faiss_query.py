import faiss
import numpy as np
import pandas as pd

class FAISSRetriever:
    def __init__(self, index_path, embeddings_path, movie_ids_path):
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(embeddings_path)
        self.movie_ids = np.load(movie_ids_path)
        
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
    
    def get_embedding_by_movie_id(self, movie_id):
        if movie_id not in self.movie_id_to_idx:
            return None
        idx = self.movie_id_to_idx[movie_id]
        return self.embeddings[idx]
    
    def search(self, query_emb, k=100):
        query_emb = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        
        retrieved_movie_ids = self.movie_ids[indices[0]]
        scores = distances[0]
        
        return retrieved_movie_ids, scores
    
    def search_by_movie_id(self, movie_id, k=100):
        emb = self.get_embedding_by_movie_id(movie_id)
        if emb is None:
            return None, None
        return self.search(emb, k)
    
    def search_by_session(self, movie_ids, k=100, weight_decay=0.9):
        embeddings = []
        weights = []
        
        for i, movie_id in enumerate(reversed(movie_ids)):
            emb = self.get_embedding_by_movie_id(movie_id)
            if emb is not None:
                embeddings.append(emb)
                weights.append(weight_decay ** i)
        
        if not embeddings:
            return None, None
        
        embeddings = np.array(embeddings)
        weights = np.array(weights).reshape(-1, 1)
        
        query_emb = np.average(embeddings, axis=0, weights=weights.flatten())
        
        return self.search(query_emb, k)