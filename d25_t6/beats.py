import torch
import torch.nn as nn
import sys
import os

beats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/saubhagya23/khushal/DCASE/unilm/beats'))
sys.path.append(beats_path)

# Verify the path exists before importing
if not os.path.exists(beats_path):
    raise FileNotFoundError(f"BEATs path not found: {beats_path}")

from BEATs import BEATs, BEATsConfig

class BEATsWrapper(nn.Module):
    def __init__(self, checkpoint_path, target_dim=1024):
        """
        Args:
            checkpoint_path (str): Path to BEATs finetuned checkpoint (.pt file).
            target_dim (int): Output dimension to match text embedding (default 1024 for RoBERTa-large).
        """
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        device = next(self.model.parameters()).device
        # Infer output dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 16000, device=device)  # 1 second of audio at 16kHz
            padding_mask = torch.zeros(dummy.shape, dtype=torch.bool, device=device)
            features = self.model.extract_features(dummy, padding_mask=padding_mask)
            last_hidden = features[0][-1]
            if last_hidden.dim() == 3:
                beats_dim = last_hidden.mean(dim=1).shape[-1]
            else:
                beats_dim = last_hidden.shape[-1]

        if beats_dim != target_dim:
            self.proj = nn.Linear(beats_dim, target_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, audio_length), float32, mono, normalized [-1, 1], 16kHz
        Returns:
            torch.Tensor: (batch_size, target_dim)
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0) 

        # print("BEATsWrapper input x.shape:", x.shape)

        embeddings = []
        for i in range(x.size(0)):
            xi = x[i].unsqueeze(0)  # (1, samples)
            padding_mask = torch.zeros(xi.shape, dtype=torch.bool, device=device)
            features = self.model.extract_features(xi, padding_mask=padding_mask)
            last_hidden = features[0][-1]
            if last_hidden.dim() == 3:
                emb = last_hidden.mean(dim=1)
            else:
                emb = last_hidden
            emb = self.proj(emb)
            embeddings.append(emb.squeeze(0))
        embedding = torch.stack(embeddings, dim=0)  # (batch, embed_dim)

        # print("BEATsWrapper output embedding.shape:", embedding.shape)

        return embedding