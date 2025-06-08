import torch
import torch.nn as nn
import sys
import os

# Add BEATs path to Python path
beats_path = "/home/saubhagya23/khushal/DCASE/unilm/beats"  
sys.path.append(beats_path)

if not os.path.exists(beats_path):
    raise FileNotFoundError(f"BEATs path not found: {beats_path}")

from BEATs import BEATs, BEATsConfig

class BEATsWrapper(torch.nn.Module):
    def __init__(self, model_path="models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", device='cpu'):
        super().__init__()
        """
        BEATs wrapper using official Microsoft implementation.
        Args:
            model_path (str): Path to the downloaded BEATs model
            device (str): Device to load the model on
        """
        self.device = device
        
        # Load pre-trained BEATs model (following official example)
        checkpoint = torch.load(model_path, map_location=device)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

        # Prevent automatic mixed precision from affecting this model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, samples), 16kHz mono waveform
        Returns:
            Tensor: (batch, embedding_dim)
        """
        # If input is stereo, convert to mono
        # if x.ndim == 3:
        #     x = x.mean(1)
            
        # Ensure input is on the correct device and dtype
        x = x.float().to(self.device)

        # Convert to half precision if model is in half precision
        # if next(self.model.parameters()).dtype == torch.float16:
        #     x = x.half()

        # Create padding mask (all False for no padding, as in official example)
        padding_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        
        # Extract features using BEATs (following official API)
        with torch.no_grad():
            representation = self.model.extract_features(x, padding_mask=padding_mask)[0]
        
        # Pool across frames (mean pooling)
        embedding = representation.mean(dim=1)  # (batch, 768)
        return embedding.unsqueeze(1)  # Add segment dimension for compatibility
