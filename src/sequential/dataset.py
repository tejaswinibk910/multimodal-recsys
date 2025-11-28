import torch
from torch.utils.data import Dataset
import numpy as np

class SequentialDataset(Dataset):
    def __init__(self, sequences, targets, item_to_idx, max_len=50):
        self.sequences = sequences
        self.targets = targets
        self.item_to_idx = item_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # Convert movie IDs to indices
        seq_indices = [self.item_to_idx.get(item, 0) for item in seq]
        target_idx = self.item_to_idx.get(target, 0)
        
        # Truncate if too long
        if len(seq_indices) > self.max_len:
            seq_indices = seq_indices[-self.max_len:]
        
        # Pad if too short
        seq_len = len(seq_indices)
        if seq_len < self.max_len:
            seq_indices = [0] * (self.max_len - seq_len) + seq_indices
        
        return {
            'sequence': torch.LongTensor(seq_indices),
            'target': torch.LongTensor([target_idx]),
            'seq_len': torch.LongTensor([seq_len])
        }