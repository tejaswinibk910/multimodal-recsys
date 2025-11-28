import sys
sys.path.append('src')

import torch
from sequential.sasrec import SASRec

# Load a minimal test
checkpoint = torch.load("data/embeddings/sasrec_best.pt", weights_only=False)

model = SASRec(
    num_items=993,
    d_model=256,
    n_layers=2,
    n_heads=4,
    max_seq_len=50,
    dropout=0.0
).to('cuda')

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test sequence
test_seq = torch.LongTensor([[0]*48 + [1, 359]]).to('cuda')

print("Checking intermediate outputs...")

with torch.no_grad():
    batch_size, seq_len = test_seq.shape
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    
    # Check embeddings
    positions = torch.arange(seq_len, device=test_seq.device).unsqueeze(0).expand(batch_size, -1)
    seq_emb = model.item_emb(test_seq)
    pos_emb = model.pos_emb(positions)
    
    print(f"\nItem embeddings: shape={seq_emb.shape}")
    print(f"  Has NaN: {torch.isnan(seq_emb).any().item()}")
    print(f"  Mean: {seq_emb.mean().item():.4f}")
    
    print(f"\nPos embeddings: shape={pos_emb.shape}")
    print(f"  Has NaN: {torch.isnan(pos_emb).any().item()}")
    print(f"  Mean: {pos_emb.mean().item():.4f}")
    
    x = model.dropout(seq_emb + pos_emb)
    print(f"\nAfter dropout + add:")
    print(f"  Has NaN: {torch.isnan(x).any().item()}")
    print(f"  Mean: {x.mean().item():.4f}")
    
    # Check attention mask
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=test_seq.device), diagonal=1).bool()
    print(f"\nAttention mask shape: {attn_mask.shape}")
    print(f"Attention mask (first 5x5):")
    print(attn_mask[:5, :5])
    
    # Check padding mask
    padding_mask = (test_seq == 0)
    print(f"\nPadding mask shape: {padding_mask.shape}")
    print(f"Num padding positions: {padding_mask.sum().item()}")
    
    # Try transformer
    print("\nRunning transformer...")
    try:
        hidden = model.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        print(f"Transformer output shape: {hidden.shape}")
        print(f"  Has NaN: {torch.isnan(hidden).any().item()}")
        print(f"  Mean: {hidden.mean().item():.4f}")
    except Exception as e:
        print(f"Error in transformer: {e}")