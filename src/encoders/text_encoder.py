from sentence_transformers import SentenceTransformer
import numpy as np

class TextEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, text):
        if not text or len(text.strip()) == 0:
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def encode_batch(self, texts, batch_size=32):
        processed_texts = []
        for text in texts:
            if not text or len(str(text).strip()) == 0:
                processed_texts.append("")
            else:
                processed_texts.append(str(text))
        
        embeddings = self.model.encode(
            processed_texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings