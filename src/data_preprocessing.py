import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from collections import defaultdict

def create_user_sequences(min_items=5, max_items=200):
    ratings = pd.read_csv("data/raw/ml-25m/ratings.csv")
    movie_ids_with_emb = np.load("data/embeddings/movie_ids.npy")
    movie_id_set = set(movie_ids_with_emb)
    
    print(f"Total ratings: {len(ratings):,}")
    print(f"Movies with embeddings: {len(movie_ids_with_emb):,}")
    
    # Filter ratings for movies with embeddings
    ratings = ratings[ratings['movieId'].isin(movie_id_set)].copy()
    print(f"Ratings after filtering: {len(ratings):,}")
    
    # Sort by user and timestamp
    ratings = ratings.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    # Create sequences
    user_sequences = defaultdict(list)
    user_timestamps = defaultdict(list)
    
    for _, row in ratings.iterrows():
        user_sequences[row['userId']].append(row['movieId'])
        user_timestamps[row['userId']].append(row['timestamp'])
    
    # Filter users with sufficient history
    valid_users = [u for u, seq in user_sequences.items() 
                   if min_items <= len(seq) <= max_items]
    
    print(f"Users with {min_items}-{max_items} items: {len(valid_users):,}")
    
    # Build final sequences
    sequences = []
    timestamps = []
    user_ids = []
    
    for user_id in valid_users:
        sequences.append(user_sequences[user_id])
        timestamps.append(user_timestamps[user_id])
        user_ids.append(user_id)
    
    return sequences, timestamps, user_ids, movie_ids_with_emb

def train_val_test_split(sequences, timestamps, user_ids, test_size=0.1, val_size=0.1):
    n_users = len(sequences)
    indices = np.arange(n_users)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_test = int(n_users * test_size)
    n_val = int(n_users * val_size)
    
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test+n_val]
    train_indices = indices[n_test+n_val:]
    
    splits = {}
    for name, idx in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
        splits[name] = {
            'sequences': [sequences[i] for i in idx],
            'timestamps': [timestamps[i] for i in idx],
            'user_ids': [user_ids[i] for i in idx]
        }
    
    return splits

def create_leave_one_out_data(sequences, timestamps, user_ids):
    train_sequences = []
    val_sequences = []
    test_sequences = []
    
    train_targets = []
    val_targets = []
    test_targets = []
    
    for seq, ts, uid in zip(sequences, timestamps, user_ids):
        if len(seq) < 3:
            continue
        
        # Last item for test
        test_seq = seq[:-1]
        test_target = seq[-1]
        
        # Second to last for validation
        val_seq = seq[:-2]
        val_target = seq[-2]
        
        # Rest for training (use all items except last 2 for training)
        train_seq = seq[:-2]
        if len(train_seq) > 0:
            for i in range(1, len(train_seq) + 1):
                train_sequences.append(train_seq[:i])
                if i < len(train_seq):
                    train_targets.append(train_seq[i])
                else:
                    train_targets.append(val_target)
        
        val_sequences.append(val_seq)
        val_targets.append(val_target)
        
        test_sequences.append(test_seq)
        test_targets.append(test_target)
    
    return {
        'train': {'sequences': train_sequences, 'targets': train_targets},
        'val': {'sequences': val_sequences, 'targets': val_targets},
        'test': {'sequences': test_sequences, 'targets': test_targets}
    }

def create_item_id_mapping(movie_ids):
    # Create mapping: movieId -> sequential index (1-indexed, 0 reserved for padding)
    item_to_idx = {mid: idx+1 for idx, mid in enumerate(movie_ids)}
    idx_to_item = {idx+1: mid for idx, mid in enumerate(movie_ids)}
    idx_to_item[0] = 0  # padding
    
    return item_to_idx, idx_to_item

def main():
    print("Sequential Training Data")
    
    sequences, timestamps, user_ids, movie_ids = create_user_sequences(
        min_items=20,
        max_items=200
    )
    
    print(f"\nTotal sequences: {len(sequences):,}")
    print(f"Average sequence length: {np.mean([len(s) for s in sequences]):.1f}")
    
    # Create item mappings
    item_to_idx, idx_to_item = create_item_id_mapping(movie_ids)
    
    print(" Leave-One-Out Split")
    loo_data = create_leave_one_out_data(sequences, timestamps, user_ids)
    
    for split_name, split_data in loo_data.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Sequences: {len(split_data['sequences']):,}")
        print(f"  Avg length: {np.mean([len(s) for s in split_data['sequences']]):.1f}")
    
    # Save everything
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "sequential_data.pkl", "wb") as f:
        pickle.dump({
            'train': loo_data['train'],
            'val': loo_data['val'],
            'test': loo_data['test'],
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item,
            'num_items': len(movie_ids)
        }, f)
    

    # Stats
    print("\nDataset Statistics:")
    print(f"Total items (movies): {len(movie_ids):,}")
    print(f"Total users: {len(sequences):,}")
    print(f"Train samples: {len(loo_data['train']['sequences']):,}")
    print(f"Val samples: {len(loo_data['val']['sequences']):,}")
    print(f"Test samples: {len(loo_data['test']['sequences']):,}")

if __name__ == "__main__":
    main()