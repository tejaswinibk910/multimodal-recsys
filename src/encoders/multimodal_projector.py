import torch
import torch.nn as nn
import numpy as np

class MultimodalProjector(nn.Module):
    def __init__(self, image_dim, text_dim, metadata_dim, output_dim=512):
        super().__init__()
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, image_emb, text_emb, metadata_emb):
        img_proj = self.image_proj(image_emb)
        txt_proj = self.text_proj(text_emb)
        meta_proj = self.metadata_proj(metadata_emb)
        
        combined = torch.cat([img_proj, txt_proj, meta_proj], dim=-1)
        output = self.fusion(combined)
        
        return output