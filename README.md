# Multi-Stage Movie Recommendation System

A production-style recommendation engine combining FAISS retrieval, Transformer-based sequential modeling (SASRec), and diversity re-ranking with multimodal embeddings.

## Demo

https://github.com/tejaswinibk910/multimodal-recsys/raw/main/docs/demo.mp4

*Full system walkthrough: movie search, watch history building, and AI-powered recommendations with posters*

---

## Key Results

- **+467% CTR Improvement** via A/B testing simulation
- **HR@10: 7.0%** | **NDCG@10: 3.8%**
- **359% improvement** over retrieval-only baseline
- Real-time recommendations with LLM-powered explanations

## Architecture
```
User History → FAISS Retrieval (100 candidates) 
            → SASRec Transformer (sequential scoring)
            → MMR Diversity Re-ranking
            → Top-K Recommendations
```

**Multi-Stage Pipeline:**
1. **Retrieval Stage**: FAISS HNSW index on 512D multimodal embeddings
2. **Sequential Stage**: SASRec Transformer with 2 layers, 4 attention heads
3. **Re-ranking Stage**: Maximal Marginal Relevance (MMR) for diversity

## Technical Stack

**Machine Learning:**
- PyTorch (SASRec Transformer implementation)
- FAISS (efficient similarity search)
- SentenceTransformers (text embeddings)
- ResNet50 (image embeddings)

**Backend:**
- FastAPI (REST API)
- Groq API (LLM explanations)

**Data:**
- MovieLens-25M (25M ratings, 162K movies)
- TMDB metadata (posters, descriptions)
- Final dataset: 993 movies with complete multimodal data

## Performance Metrics

| Method | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Improvement |
|--------|-------|---------|-------|---------|-------------|
| FAISS Retrieval Only | 1.5% | 0.7% | 3.0% | 1.1% | baseline |
| **Full Pipeline** | **7.0%** | **3.8%** | **10.1%** | **4.5%** | **+359%** |

**A/B Test Results (2000 users):**
- CTR: 0.75% → 4.25% (**+467%**)
- Avg Click Position: 3.0 → 2.1
- Statistical Significance: p < 0.0001

## Features

- Multimodal embeddings (image + text + metadata)
- FAISS-based efficient retrieval
- Transformer sequential modeling
- Diversity re-ranking (MMR)
- Explainable recommendations (rule-based + LLM)
- A/B testing simulation
- Interactive web UI with movie posters
- Comprehensive evaluation metrics

## Project Structure
```
rec_system/
├── data/
│   ├── raw/ml-25m/           # MovieLens dataset
│   ├── raw/posters/          # Movie posters (993 images)
│   ├── processed/            # Preprocessed data
│   └── embeddings/           # Item embeddings, FAISS index, models
├── src/
│   ├── encoders/             # Multimodal embedding generation
│   ├── retrieval/            # FAISS index building & querying
│   ├── sequential/           # SASRec model & training
│   ├── ranker/               # Diversity re-ranking & explanations
│   ├── eval/                 # Evaluation & A/B testing
│   └── serving/              # FastAPI + Web UI
├── experiments/results/      # Evaluation results
└── requirements.txt
```

## Setup & Installation
```bash
# Clone repository
git clone https://github.com/tejaswinibk910/multimodal-recsys.git
cd multimodal-recsys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download MovieLens-25M dataset
# Place in data/raw/ml-25m/

# Build embeddings & FAISS index
python src/encoders/build_embeddings.py
python src/retrieval/build_faiss_index.py

# Train SASRec model
python src/sequential/train_sasrec.py

# Run evaluation
python src/eval/evaluate_pipeline.py

# Start web server
python src/serving/api.py
# Open http://localhost:8000
```

## Training Results

**SASRec Model:**
- Training: 4.2M sequences, 10 epochs (~5 hours on GPU)
- Validation HR@10: 19.2% | NDCG@10: 10.0%
- Test HR@10: 16.7% | NDCG@10: 8.7%

**Model Architecture:**
- Embedding dim: 256
- Layers: 2
- Attention heads: 4
- Max sequence length: 50
- Total parameters: 2.1M

## Web Interface

Interactive demo with:
- Movie search with poster previews
- Build watch history
- Toggle sequential model / diversity re-ranking
- AI-powered explanations (Groq API)
- Real-time recommendations with scores

## Key Learnings

1. **Multi-stage architecture is essential** for production systems - can't score millions of items with heavy models
2. **Sequential modeling matters** - capturing temporal patterns yields massive improvements
3. **Multimodal features help** - combining images, text, and metadata creates richer representations
4. **Diversity is valuable** - pure relevance can create echo chambers

## Future Enhancements

- [ ] User-user collaborative filtering
- [ ] Graph Neural Networks (PinSage)
- [ ] Multi-objective optimization (engagement + diversity + fairness)
- [ ] Online learning / continual adaptation
- [ ] Contextual bandits for exploration

## References

- **SASRec**: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- **FAISS**: [Billion-scale similarity search](https://arxiv.org/abs/1702.08734)
- **YouTube RecSys**: [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/)

## License

MIT License

## Author

**Tejaswini BK**
- GitHub: [@tejaswinibk910](https://github.com/tejaswinibk910)
- Project: [multimodal-recsys](https://github.com/tejaswinibk910/multimodal-recsys)

---
