import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import pickle
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from metadata_encoder import MetadataEncoder
from multimodal_projector import MultimodalProjector

def build_item_embeddings(use_pretrained_projector=False):
    movies = pd.read_csv("data/raw/ml-25m/movies.csv")
    metadata = pd.read_csv("data/processed/movie_metadata.csv")
    
    merged = movies.merge(metadata, on='movieId', how='inner', suffixes=('_movie', '_tmdb'))
    print(f"Movies with complete metadata: {len(merged)}")
    
    # Debug: print columns to see what we have
    print(f"Columns in merged dataframe: {merged.columns.tolist()}")
    
    # Use the correct genres column
    if 'genres_tmdb' in merged.columns:
        merged['genres'] = merged['genres_tmdb']
    elif 'genres_movie' in merged.columns:
        merged['genres'] = merged['genres_movie']
    
    print("Encoding Images")
    
    image_encoder = ImageEncoder(model_name='resnet50')
    poster_dir = Path("data/raw/posters")
    
    image_embeddings = []
    valid_movie_ids = []
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        movie_id = row['movieId']
        poster_path = poster_dir / f"{movie_id}.jpg"
        
        if poster_path.exists():
            img_emb = image_encoder.encode_image(poster_path)
            image_embeddings.append(img_emb)
            valid_movie_ids.append(movie_id)
    
    image_embeddings = np.array(image_embeddings)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    merged = merged[merged['movieId'].isin(valid_movie_ids)].reset_index(drop=True)
    
    print("Encoding Text")
    
    text_encoder = TextEncoder()
    
    text_data = merged['title'] + " " + merged['overview'].fillna("")
    text_embeddings = text_encoder.encode_batch(text_data.tolist())
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    print("Encoding Metadata")
    
    metadata_encoder = MetadataEncoder()
    metadata_encoder.fit(merged)
    metadata_embeddings = metadata_encoder.encode_batch(merged)
    print(f"Metadata embeddings shape: {metadata_embeddings.shape}")
    
    print("Combining with Projector")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projector = MultimodalProjector(
        image_dim=image_embeddings.shape[1],
        text_dim=text_embeddings.shape[1],
        metadata_dim=metadata_embeddings.shape[1],
        output_dim=512
    ).to(device)
    
    projector.eval()
    
    final_embeddings = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(merged), batch_size)):
            img_batch = torch.FloatTensor(image_embeddings[i:i+batch_size]).to(device)
            txt_batch = torch.FloatTensor(text_embeddings[i:i+batch_size]).to(device)
            meta_batch = torch.FloatTensor(metadata_embeddings[i:i+batch_size]).to(device)
            
            output = projector(img_batch, txt_batch, meta_batch)
            final_embeddings.append(output.cpu().numpy())
    
    final_embeddings = np.vstack(final_embeddings)
    final_embeddings = final_embeddings / np.linalg.norm(final_embeddings, axis=1, keepdims=True)
    
    print(f"Final embeddings shape: {final_embeddings.shape}")
    
    print("Saving Embeddings")    
    np.save("data/embeddings/item_embeddings.npy", final_embeddings)
    np.save("data/embeddings/movie_ids.npy", merged['movieId'].values)
    
    with open("data/embeddings/metadata_encoder.pkl", "wb") as f:
        pickle.dump(metadata_encoder, f)
    
    torch.save(projector.state_dict(), "data/embeddings/projector.pt")
    
    print(f"Total items with embeddings: {len(final_embeddings)}")
    
    return final_embeddings, merged['movieId'].values

if __name__ == "__main__":
    embeddings, movie_ids = build_item_embeddings()