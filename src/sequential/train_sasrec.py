import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json

from sasrec import SASRec
from dataset import SequentialDataset

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        seq = batch['sequence'].to(device)
        target = batch['target'].squeeze().to(device)
        
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, k_list=[10, 20]):
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            seq = batch['sequence'].to(device)
            target = batch['target'].squeeze().to(device)
            
            logits = model(seq)
            
            all_targets.append(target.cpu().numpy())
            all_predictions.append(logits.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    metrics = {}
    for k in k_list:
        top_k = np.argsort(-all_predictions, axis=1)[:, :k]
        
        hits = np.any(top_k == all_targets.reshape(-1, 1), axis=1)
        hr = hits.mean()
        
        ranks = []
        for i, target in enumerate(all_targets):
            if target in top_k[i]:
                rank = np.where(top_k[i] == target)[0][0] + 1
                dcg = 1.0 / np.log2(rank + 1)
                ranks.append(dcg)
            else:
                ranks.append(0.0)
        ndcg = np.mean(ranks)
        
        metrics[f'HR@{k}'] = hr
        metrics[f'NDCG@{k}'] = ndcg
    
    return metrics

def main():
    with open("data/processed/sequential_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    item_to_idx = data['item_to_idx']
    num_items = data['num_items']
    
    print(f"Number of items: {num_items}")
    print(f"Train samples: {len(data['train']['sequences']):,}")
    print(f"Val samples: {len(data['val']['sequences']):,}")
    print(f"Test samples: {len(data['test']['sequences']):,}")
    
    # Hyperparameters
    config = {
        'batch_size': 256,
        'max_seq_len': 50,
        'd_model': 256,
        'n_layers': 2,
        'n_heads': 4,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\nUsing device: {config['device']}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Create datasets
    train_dataset = SequentialDataset(
        data['train']['sequences'],
        data['train']['targets'],
        item_to_idx,
        max_len=config['max_seq_len']
    )
    
    val_dataset = SequentialDataset(
        data['val']['sequences'],
        data['val']['targets'],
        item_to_idx,
        max_len=config['max_seq_len']
    )
    
    test_dataset = SequentialDataset(
        data['test']['sequences'],
        data['test']['targets'],
        item_to_idx,
        max_len=config['max_seq_len']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = SASRec(
        num_items=num_items,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    ).to(config['device'])
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    best_val_ndcg = 0
    patience = 3
    patience_counter = 0
    
    print("\nStarting training")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        print(f"Train Loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, config['device'])
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        if val_metrics['NDCG@10'] > best_val_ndcg:
            best_val_ndcg = val_metrics['NDCG@10']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics
            }, 'data/embeddings/sasrec_best.pt')
            print("  Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
    
    print("Evaluating on test set")
    
    checkpoint = torch.load('data/embeddings/sasrec_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, config['device'])
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    with open('data/embeddings/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    

if __name__ == "__main__":
    main()