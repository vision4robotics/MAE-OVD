"""
T-SSG (Text-guided Spatial-Semantic Gating): Implementation aligned with paper equations (7)-(10).

Paper equations:
  Eq(7):  C̃_ij = [F_i; F_j]          # Channel concatenation
  Eq(8):  γ = Linear(Y)              # Text-guided channel weights
  Eq(9):  M_ij = σ(γ ⊗ Conv1×1(C̃_ij)) # Text-guided spatial gate
  Eq(10): F^y_i = (C̃_ij ⊗ M_ij) ⊕ Conv1×1(C̃_ij)  # Gated fusion + residual

ONNX compatible: Conv2d, Linear, Sigmoid, Mul, Add only; no einsum or DynamicShape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSSG(nn.Module):
    """
    T-SSG (Text-guided Spatial-Semantic Gating): Cross-scale feature modulation with text guidance.

    Fully aligned with paper equations (7)-(10).
    """

    def __init__(self, c_i, c_j, text_dim=256):
        super().__init__()
        self.c_i = c_i
        self.c_j = c_j
        total_channels = c_i + c_j
        # Eq(9): Spatial gate generation: Conv1x1(c_i+c_j -> 1) -> Sigmoid -> M_ij
        self.spatial_conv = nn.Conv2d(total_channels, 1, kernel_size=1, bias=True)
        # Eq(8): Channel weights: Y -> Linear(text_dim -> c_i) -> γ
        self.channel_fc = nn.Linear(text_dim, c_i, bias=True)
        # Eq(10) residual path: Conv1x1(c_i+c_j -> c_i)
        self.residual_proj = nn.Conv2d(total_channels, c_i, kernel_size=1, bias=True)
        # Expand gamma from c_i to c_i+c_j
        self.gamma_expand = nn.Linear(c_i, total_channels, bias=False)

    def forward(self, f_i, f_j, y):
        """
        Args:
            f_i: (B, c_i, H, W) Current scale features
            f_j: (B, c_j, Hj, Wj) Adjacent scale features (top-down or bottom-up)
            y: (B, text_dim) Refined text embedding
        Returns:
            F^y_i: (B, c_i, H, W) Text-enhanced features
        """
        B, _, H, W = f_i.shape

        # 1. Align F_j spatial dimensions
        f_j = F.interpolate(f_j, size=(H, W), mode="bilinear", align_corners=False)

        # Eq(7): C̃_ij = [F_i; F_j] - Channel concatenation
        cat_f = torch.cat([f_i, f_j], dim=1)  # (B, c_i+c_j, H, W)

        # Eq(8): γ = Linear(Y)
        gamma = self.channel_fc(y)  # (B, c_i)
        # Expand gamma to c_i+c_j channels
        gamma_expanded = self.gamma_expand(gamma)  # (B, c_i+c_j)
        gamma_expanded = gamma_expanded.view(B, -1, 1, 1)  # (B, c_i+c_j, 1, 1)

        # Eq(9): M_ij = σ(γ ⊗ Conv1×1(C̃_ij))
        conv_out = self.spatial_conv(cat_f)  # (B, 1, H, W)
        # gamma_expanded is (B, c_i+c_j, 1, 1), broadcastable to (B, c_i+c_j, H, W)
        m_ij = torch.sigmoid(conv_out * gamma_expanded)  # (B, c_i+c_j, H, W)

        # Eq(10): F^y_i = (C̃_ij ⊗ M_ij) ⊕ Conv1×1(C̃_ij)
        gated = cat_f * m_ij  # (B, c_i+c_j, H, W)
        # Residual: Conv1x1(C̃_ij) -> c_i
        residual = self.residual_proj(cat_f)  # (B, c_i, H, W)
        # Final output: take first c_i channels of gated + residual
        out = gated[:, :self.c_i, :, :] + residual  # (B, c_i, H, W)

        return out


class TSSGSimple(nn.Module):
    """
    T-SSG Simplified version: Maintains backward compatibility.
    Equation: F^y_i = (F_i * γ * M_ij) + F_i
    """

    def __init__(self, c_i, c_j, text_dim=256):
        super().__init__()
        self.c_i = c_i
        self.c_j = c_j
        self.spatial_conv = nn.Conv2d(c_i + c_j, 1, 1)
        self.channel_fc = nn.Linear(text_dim, c_i)

    def forward(self, f_i, f_j, y):
        B, _, H, W = f_i.shape
        f_j = F.interpolate(f_j, size=(H, W), mode="bilinear", align_corners=False)
        cat_f = torch.cat([f_i, f_j], dim=1)
        m_ij = torch.sigmoid(self.spatial_conv(cat_f))
        gamma = self.channel_fc(y).view(B, self.c_i, 1, 1)
        out = (f_i * gamma * m_ij) + f_i
        return out
