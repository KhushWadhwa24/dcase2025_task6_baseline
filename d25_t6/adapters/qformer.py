import torch
from torch import nn, Tensor
import zeta
from zeta.nn import MultiQueryAttention, SimpleFeedForward
from zeta.nn.attention.cross_attention import CrossAttention

class AudioBlock(nn.Module):
    """
    AudioBlock is a module that performs multi-query attention, cross-attention, and feedforward operations on input tensors.

    Args:
        dim (int): The dimension of the input tensors.
        depth (int): The number of times the operations are applied.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Methods:
        forward(x: Tensor, audio: Tensor) -> Tensor:
            Performs the forward pass of the AudioBlock module.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        depth: int = 2,
        heads: int = 16,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_attn_layers = nn.ModuleList([
            MultiQueryAttention(input_dim, heads, *args, **kwargs) for _ in range(depth)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(dim=input_dim, heads=heads, dropout=dropout, *args, **kwargs) for _ in range(depth)
        ])
        self.ffn_layers = nn.ModuleList([
            SimpleFeedForward(input_dim, input_dim * 4, dropout, *args, **kwargs) for _ in range(depth)
        ])

    def forward(self, x: Tensor, audio: Tensor) -> Tensor:
        for self_attn, cross_attn, ffn in zip(
            self.self_attn_layers, self.cross_attn_layers, self.ffn_layers
        ):
            x, _, _ = self_attn(x)
            x = cross_attn(x, audio)
            x = ffn(x)
        return x