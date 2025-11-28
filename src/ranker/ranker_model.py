import torch
import torch.nn as nn

class RankerMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[512, 256, 128]):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_sizes
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TransformerRanker(nn.Module):
    def __init__(self, item_emb_dim=512, session_emb_dim=256, context_dim=10, 
                 d_model=256, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.item_proj = nn.Linear(item_emb_dim, d_model)
        self.session_proj = nn.Linear(session_emb_dim, d_model)
        self.context_proj = nn.Linear(context_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, item_emb, session_emb, context_features):
        item_proj = self.item_proj(item_emb).unsqueeze(1)
        session_proj = self.session_proj(session_emb).unsqueeze(1)
        context_proj = self.context_proj(context_features).unsqueeze(1)
        
        combined = torch.cat([session_proj, item_proj, context_proj], dim=1)
        
        output = self.transformer(combined)
        pooled = output.mean(dim=1)
        
        score = self.fc(pooled).squeeze(-1)
        return score