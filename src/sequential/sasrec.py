import torch
import torch.nn as nn
import math

class SASRec(nn.Module):
    def __init__(self, num_items, d_model=256, n_layers=2, n_heads=4, max_seq_len=50, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Use pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, seq, return_embedding=False):
        batch_size, seq_len = seq.shape
        
        # Get actual sequence lengths (non-padding)
        seq_lens = (seq != 0).sum(dim=1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        seq_emb = self.item_emb(seq)
        pos_emb = self.pos_emb(positions)
        x = self.dropout(seq_emb + pos_emb)
        
        # Only use causal mask, don't use padding mask to avoid NaN
        # The padding embeddings will just be ignored when we extract the last position
        attn_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer with only causal mask
        hidden = self.transformer(x, mask=attn_mask)
        hidden = self.layer_norm(hidden)
        
        if return_embedding:
            batch_indices = torch.arange(batch_size, device=seq.device)
            last_indices = (seq_lens - 1).clamp(min=0)
            last_hidden = hidden[batch_indices, last_indices]
            return last_hidden
        
        # Get last non-padding position for each sequence
        batch_indices = torch.arange(batch_size, device=seq.device)
        last_indices = (seq_lens - 1).clamp(min=0)
        last_hidden = hidden[batch_indices, last_indices] 
        logits = self.out(last_hidden)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask