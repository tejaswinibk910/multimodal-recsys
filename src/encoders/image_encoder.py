import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

class ImageEncoder:
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.embedding_dim = 2048
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.embedding_dim = 512
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def encode_image(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(img_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_batch(self, image_paths, batch_size=32):
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    batch_imgs.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    batch_imgs.append(torch.zeros(3, 224, 224))
            
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                batch_embeddings = batch_embeddings.squeeze().cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)