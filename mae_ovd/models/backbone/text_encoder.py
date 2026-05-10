"""
Text Encoding: CLIP 512 -> Linear -> 256 (y).
"""
import torch.nn as nn


class TextEncoder(nn.Module):
    """Project CLIP's original 512 dimensions to 256."""

    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, 512) -> (B, 256)
        return self.proj(x)
