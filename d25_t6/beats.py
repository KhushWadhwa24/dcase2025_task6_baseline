import torch
import torch.nn as nn
# from hear21passt.base import get_model_passt, AugmentMelSTFT
from transformers import BEATSProcessor, BEATSModel

import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="hearpasst")
# warnings.filterwarnings("ignore", category=UserWarning, module="hearpasst")


class BEATsWrapper(torch.nn.Module):

    def __init__(self, s_patchout_t=15, s_patchout_f=2):
        super().__init__()
        """
        Args:
            device (str): Device used to load the model on
        """
        self.processor = BEATSProcessor.from_pretrained("microsoft/beats")
        self.model = BEATSModel.from_pretrained("microsoft/beats")
        self.device = device
        self.model.to(self.device)
        self.model.eval()  # Usually frozen for feature extraction

        # Optional: Add downsampling layer as mentioned in research papers
        # self.downsample = nn.Conv1d(768, 768, kernel_size=3, stride=3)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (batch, samples), 16kHz mono waveform
        Returns:
            Tensor: (batch, embedding_dim)
        """
        # If input is stereo, convert to mono
        if x.ndim == 3:
            x = x.mean(1)
        
        # Ensure 16kHz sampling rate (you may need to resample in preprocessing)
        batch_size = x.shape[0]
        
        # Process with BEATs
        inputs = self.processor(x.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            feats = outputs.last_hidden_state  # (batch, frames, 768)
        
        # Optional downsampling (as used in research papers)
        feats = feats.transpose(1, 2)  # (batch, 768, frames)
        feats = self.downsample(feats)  # (batch, 768, frames//3)
        feats = feats.transpose(1, 2)  # (batch, frames//3, 768)
        
        # Pool across frames (mean pooling)
        embedding = feats.mean(dim=1)  # (batch, 768)
        return embedding.unsqueeze(1)  # Add segment dimension for compatibility
