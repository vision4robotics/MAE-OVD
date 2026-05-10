"""
Template Extractor: Centroid from M_A (Soft-Argmax differentiable), 7×7 crop (grid_sample) + 2 Conv layers -> X (256×7×7).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_argmax_centroid(m_a):
    """Get centroid (cy, cx) from M_A (B, 1, H, W), each (B, 1), differentiable."""
    B, _, H, W = m_a.shape
    device = m_a.device
    # Normalized weights
    w = m_a / (m_a.sum(dim=(2, 3), keepdim=True) + 1e-8)
    # Grid (H, W), broadcast with w
    grid_y = torch.linspace(0, H - 1, H, device=device, dtype=m_a.dtype).view(1, 1, H, 1)
    grid_x = torch.linspace(0, W - 1, W, device=device, dtype=m_a.dtype).view(1, 1, 1, W)
    cy = (w * grid_y).sum(dim=(2, 3))  # (B, 1)
    cx = (w * grid_x).sum(dim=(2, 3))  # (B, 1)
    return cy, cx


def _make_crop_grid(cy, cx, H, W, crop_size, batch_size, device, dtype):
    """Construct normalized coordinates for crop_size×crop_size centered at (cy, cx) on F_target (H,W) (grid_sample format)."""
    half = (crop_size - 1) / 2.0
    H_eff = max(H, 2)
    W_eff = max(W, 2)
    # cy, cx: (B, 1); pixel rows/cols: (B, 7, 7)
    i = torch.arange(crop_size, device=device, dtype=dtype).view(1, -1, 1)
    j = torch.arange(crop_size, device=device, dtype=dtype).view(1, 1, -1)
    yy = (cy.view(batch_size, 1, 1) - half + i)   # (B, 7, 1)
    xx = (cx.view(batch_size, 1, 1) - half + j)   # (B, 1, 7)
    yy = yy.expand(batch_size, crop_size, crop_size)
    xx = xx.expand(batch_size, crop_size, crop_size)
    y_norm = 2.0 * (yy / (H_eff - 1)) - 1.0
    x_norm = 2.0 * (xx / (W_eff - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)   # (B, 7, 7, 2)
    return grid


class TemplateExtractor(nn.Module):
    """Extract soft-argmax centroid from M_A, crop 7×7 via grid_sample, 2 Conv layers -> X (B, 256, 7, 7)."""

    def __init__(self, in_channels=448, out_channels=256, template_size=7):
        super().__init__()
        self.template_size = template_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f_target, m_a):
        """
        f_target: (B, C, H, W), m_a: (B, 1, H, W)
        return: X (B, 256, 7, 7)
        """
        B, C, H, W = f_target.shape
        cy, cx = _soft_argmax_centroid(m_a)  # (B, 1) each
        grid = _make_crop_grid(
            cy, cx, H, W, self.template_size, B, f_target.device, f_target.dtype
        )
        # grid_sample: grid (B, 7, 7, 2), F_target (B, C, H, W) -> (B, C, 7, 7)
        crop = F.grid_sample(
            f_target, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        x = self.act(self.conv1(crop))
        x = self.conv2(x)
        return x
