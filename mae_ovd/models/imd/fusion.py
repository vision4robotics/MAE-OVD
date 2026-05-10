"""
F^y_3, F^y_4, F^y_5 resize->20×20 Concat->F_fused (B,448,20,20);
Conv1x1->F_proj (B,256,20,20); Y->Linear->Y_proj (B,256).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """Multi-scale features resized to fused_size, Concat + Conv1x1 projection; Y linearly projected to Y_proj."""

    def __init__(self, channels=(64, 128, 256), fused_size=20, proj_dim=256):
        super().__init__()
        self.fused_channels = sum(channels)  # 448
        self.proj = nn.Conv2d(self.fused_channels, proj_dim, 1)
        self.y_proj_linear = nn.Linear(proj_dim, proj_dim)
        self.fused_size = fused_size
        self.proj_dim = proj_dim

    def forward(self, f3, f4, f5, y):
        """
        f3: (B, 64, H3, W3), f4: (B, 128, H4, W4), f5: (B, 256, H5, W5)
        y: (B, 256)
        return: F_fused (B, 448, fused_size, fused_size), F_proj (B, 256, fused_size, fused_size), Y_proj (B, 256)
        """
        size = (self.fused_size, self.fused_size)
        f3_r = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
        f4_r = F.interpolate(f4, size=size, mode="bilinear", align_corners=False)
        f5_r = F.interpolate(f5, size=size, mode="bilinear", align_corners=False)
        f_fused = torch.cat([f3_r, f4_r, f5_r], dim=1)  # (B, 448, 20, 20)
        f_proj = self.proj(f_fused)  # (B, 256, 20, 20)
        y_proj = self.y_proj_linear(y)  # (B, 256)
        return f_fused, f_proj, y_proj
