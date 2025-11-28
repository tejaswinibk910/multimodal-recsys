import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

class MetadataEncoder:
    def __init__(self):
        self.genre_encoder = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, metadata_df):
        # Handle both pipe-separated and already split genres
        if 'genres' in metadata_df.columns:
            genres_list = metadata_df['genres'].apply(
                lambda x: x.split('|') if isinstance(x, str) else []
            ).tolist()
        else:
            raise ValueError("'genres' column not found in metadata")
        
        self.genre_encoder.fit(genres_list)
        
        popularity = metadata_df['popularity'].values.reshape(-1, 1)
        self.scaler.fit(popularity)
        
        self.is_fitted = True
        self.embedding_dim = len(self.genre_encoder.classes_) + 1
    
    def encode(self, genres_str, popularity):
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted first")
        
        genres_list = genres_str.split('|') if isinstance(genres_str, str) else []
        genre_vec = self.genre_encoder.transform([genres_list])[0]
        
        pop_normalized = self.scaler.transform([[popularity]])[0]
        
        return np.concatenate([genre_vec, pop_normalized])
    
    def encode_batch(self, metadata_df):
        genres_list = metadata_df['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) else []
        ).tolist()
        genre_matrix = self.genre_encoder.transform(genres_list)
        
        popularity_matrix = self.scaler.transform(
            metadata_df['popularity'].values.reshape(-1, 1)
        )
        
        return np.hstack([genre_matrix, popularity_matrix])