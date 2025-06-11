import torch
import torch.nn as nn
from collections import OrderedDict

class AdapterBlock(nn.Module):
    """
    Bottleneck Adapter Block: Down-project -> Nonlinearity -> Up-project.
    Optionally, you can add a residual connection and scaling factor.
    """
    def __init__(self, d_model, mid_dim, scale=1.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(d_model, mid_dim),
            nn.ReLU()
        )
        self.up = nn.Linear(mid_dim, d_model)
        self.scale = scale

        # Recommended initialization (see [5])
        nn.init.kaiming_normal_(self.down[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.down[0].bias, 0)
        nn.init.kaiming_normal_(self.up.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.up.bias, 0)

    def forward(self, x):
        # Standard bottleneck adapter with residual connection and scaling
        h = self.down(x)
        h = self.up(h)
        return x + self.scale * h
